"""
聊天服务模块
集成 OpenAI API 并通过 WebSocket 进行通信
支持多用户多会话
支持流式 ASR 实时识别
"""
import json
import base64
import tempfile
import os, sys
import asyncio
import numpy as np
from openai import AsyncOpenAI
from dotenv import load_dotenv
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from core.asr import (
    get_audio_stream_service,
    get_offline_asr_service,
    AudioInput,
    ASRConfig,
)
from core.tts.edge import get_tts_provider
from core.storage_service import get_storage_service


# 获取项目根目录（chat_service.py 在 core 目录下）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")

# 加载环境变量（override=True 让 .env 覆盖系统环境变量）
load_dotenv(ENV_FILE, override=True)

# 临时文件目录（优先从环境变量读取，默认为项目目录下的 temp 文件夹）
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(PROJECT_ROOT, "temp"))
os.makedirs(TEMP_DIR, exist_ok=True)


class ChatService:
    """聊天服务类，处理单个 WebSocket 连接的聊天逻辑"""

    def __init__(self, ws, api_key=None, base_url=None):
        """
        初始化聊天服务

        Args:
            ws: WebSocket 连接对象
            api_key: OpenAI API 密钥（优先从环境变量读取）
            base_url: OpenAI API 基础 URL（优先从环境变量读取）
        """
        self.ws = ws
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            logger.warning("未配置 OPENAI_API_KEY，聊天功能可能无法正常工作")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        # 多用户多会话支持
        self.user_id = None
        self.session_id = None
        self.chat_history = []

        # 存储服务
        self.storage = get_storage_service()

        # 存储音频分块数据（用于 audio_chunk 消息）
        self.audio_chunks = {}

        # ========== ASR 服务配置 ==========

        # 流式 ASR 服务实例（用于 audio_stream_start/chunk/end 场景）
        # 初始化时创建，整个会话期间使用同一实例
        self._streaming_asr_service = None

        # 当前流式 ASR 会话 ID
        self._streaming_asr_session_id = None

    async def initialize(self):
        """初始化 WebSocket 连接"""
        # 初始化音频流处理服务（根据配置自动选择 offline/streaming/2pass）
        self._streaming_asr_service = get_audio_stream_service()

        # 根据策略类型显示不同的后端信息
        strategy = self._streaming_asr_service.strategy
        if hasattr(strategy, 'streaming_backend') and hasattr(strategy, 'offline_backend'):
            # TwoPassStrategy
            backend_info = f"Pass1={strategy.streaming_backend.__class__.__name__}, Pass2={strategy.offline_backend.__class__.__name__}"
        elif hasattr(strategy, 'backend'):
            # StreamingStrategy 或 OfflineStrategy
            backend_info = strategy.backend.__class__.__name__
        else:
            backend_info = "unknown"

        logger.info(f"WebSocket 聊天服务已初始化, ASR 后端: {backend_info}")

    async def _handle_init(self, data):
        """
        处理初始化请求

        Args:
            data: 包含 user_id 字段的字典
        """
        self.user_id = data.get("user_id")
        if not self.user_id:
            logger.warning("初始化请求缺少 user_id")
            return

        logger.info(f"用户初始化: {self.user_id}")

        # 确保用户存在
        self.storage.get_or_create_user(self.user_id)

        # 获取用户所有会话
        sessions = self.storage.get_user_sessions(self.user_id)

        # 如果没有会话，创建一个默认会话
        if not sessions:
            self.session_id = self.storage.create_session(self.user_id)
            sessions = self.storage.get_user_sessions(self.user_id)
        else:
            # 使用最新的会话
            self.session_id = sessions[0]["session_id"]

        # 加载会话历史
        self._load_session_history()

        # 返回初始化结果
        reply_data = {
            "type": "init_reply",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "sessions": sessions
        }
        await self.ws.send(json.dumps(reply_data, ensure_ascii=False))
        logger.info(f"初始化完成，当前会话: {self.session_id}")

    async def _handle_get_sessions(self):
        """处理获取会话列表请求"""
        if not self.user_id:
            logger.warning("未初始化用户，无法获取会话列表")
            return

        sessions = self.storage.get_user_sessions(self.user_id)
        reply_data = {
            "type": "sessions_reply",
            "sessions": sessions
        }
        await self.ws.send(json.dumps(reply_data, ensure_ascii=False))

    async def _handle_create_session(self, data):
        """
        处理创建新会话请求

        Args:
            data: 包含 title 字段的字典（可选）
        """
        if not self.user_id:
            logger.warning("未初始化用户，无法创建会话")
            return

        title = data.get("title", "新会话")
        new_session_id = self.storage.create_session(self.user_id, title)

        # 切换到新会话
        self.session_id = new_session_id
        self._load_session_history()

        # 返回创建结果
        reply_data = {
            "type": "create_session_reply",
            "session_id": new_session_id,
            "title": title,
            "success": True
        }
        await self.ws.send(json.dumps(reply_data, ensure_ascii=False))
        logger.info(f"创建新会话: {new_session_id}")

    async def _handle_switch_session(self, data):
        """
        处理切换会话请求

        Args:
            data: 包含 session_id 字段的字典
        """
        if not self.user_id:
            logger.warning("未初始化用户，无法切换会话")
            return

        target_session_id = data.get("session_id")
        if not target_session_id:
            logger.warning("切换会话请求缺少 session_id")
            return

        # 检查会话是否存在
        session = self.storage.get_session(self.user_id, target_session_id)
        if not session:
            logger.warning(f"会话不存在: {target_session_id}")
            return

        # 切换会话
        self.session_id = target_session_id
        self._load_session_history()

        # 返回切换结果
        reply_data = {
            "type": "switch_session_reply",
            "session_id": target_session_id,
            "messages": self.chat_history,
            "success": True
        }
        await self.ws.send(json.dumps(reply_data, ensure_ascii=False))
        logger.info(f"切换到会话: {target_session_id}")

    async def _handle_delete_session(self, data):
        """
        处理删除会话请求

        Args:
            data: 包含 session_id 字段的字典
        """
        if not self.user_id:
            logger.warning("未初始化用户，无法删除会话")
            return

        target_session_id = data.get("session_id")
        if not target_session_id:
            logger.warning("删除会话请求缺少 session_id")
            return

        success = self.storage.delete_session(self.user_id, target_session_id)

        if success:
            # 如果删除的是当前会话，切换到最新的其他会话
            if target_session_id == self.session_id:
                sessions = self.storage.get_user_sessions(self.user_id)
                if sessions:
                    self.session_id = sessions[0]["session_id"]
                    self._load_session_history()
                else:
                    # 创建一个新会话
                    self.session_id = self.storage.create_session(self.user_id)
                    self._load_session_history()

        # 返回删除结果
        reply_data = {
            "type": "delete_session_reply",
            "session_id": target_session_id,
            "success": success,
            "current_session_id": self.session_id
        }
        await self.ws.send(json.dumps(reply_data, ensure_ascii=False))
        logger.info(f"删除会话: {target_session_id}")

    async def _handle_rename_session(self, data):
        """
        处理重命名会话请求

        Args:
            data: 包含 session_id 和 title 字段的字典
        """
        if not self.user_id:
            logger.warning("未初始化用户，无法重命名会话")
            return

        target_session_id = data.get("session_id")
        new_title = data.get("title")

        if not target_session_id or not new_title:
            logger.warning("重命名会话请求参数不完整")
            return

        success = self.storage.rename_session(self.user_id, target_session_id, new_title)

        # 返回重命名结果
        reply_data = {
            "type": "rename_session_reply",
            "session_id": target_session_id,
            "title": new_title,
            "success": success
        }
        await self.ws.send(json.dumps(reply_data, ensure_ascii=False))

    def _load_session_history(self):
        """加载当前会话的历史消息"""
        if not self.user_id or not self.session_id:
            return

        messages = self.storage.get_session_messages(self.user_id, self.session_id)

        # 构建 chat_history，添加系统提示
        self.chat_history = [
            {"role": "system", "content": "你是一个AI助手。回答必须简洁，控制在50字以内。不要使用表情符号，不要废话。"}
        ]

        for msg in messages:
            self.chat_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        logger.info(f"加载会话历史，共 {len(messages)} 条消息")

    async def handle_message(self, message):
        """
        处理来自客户端的消息

        大分类路由：audio 和 text 分开处理
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            # ========== 文本消息路由 ==========
            if msg_type in ("init", "get_sessions", "create_session", "switch_session",
                           "delete_session", "rename_session", "chat", "clear_history"):
                await self._handle_text_message(msg_type, data)

            # ========== 音频消息路由 ==========
            elif msg_type in ("audio", "audio_chunk", "audio_stream_start",
                              "audio_stream_chunk", "audio_stream_end"):
                await self._handle_audio_message(msg_type, data)

            else:
                logger.warning(f"未知消息类型: {msg_type}")

        except json.JSONDecodeError:
            logger.error(f"无效的 JSON 消息: {message}")
        except Exception as e:
            logger.error(f"处理消息时出错: {e}")

    async def _handle_text_message(self, msg_type: str, data: dict):
        """处理文本相关消息"""
        handlers = {
            "init": self._handle_init,
            "get_sessions": lambda _: self._handle_get_sessions(),
            "create_session": self._handle_create_session,
            "switch_session": self._handle_switch_session,
            "delete_session": self._handle_delete_session,
            "rename_session": self._handle_rename_session,
            "chat": self._handle_chat_message,
            "clear_history": lambda _: self._handle_clear_history(),
        }
        handler = handlers.get(msg_type)
        if handler:
            await handler(data)

    async def _handle_audio_message(self, msg_type: str, data: dict):
        """处理音频相关消息"""
        logger.info(f"msg_type: {msg_type}")
        if msg_type == "audio":
            await self._handle_audio_upload(data)
        elif msg_type == "audio_chunk":
            await self._handle_audio_chunk_upload(data)
        elif msg_type == "audio_stream_start":
            await self._handle_audio_stream_start()
        elif msg_type == "audio_stream_chunk":
            await self._handle_audio_stream_chunk(data)
        elif msg_type == "audio_stream_end":
            await self._handle_audio_stream_end()

    async def _handle_chat_message(self, data):
        """
        处理聊天消息

        Args:
            data: 包含 message 字段的字典
        """
        user_message = data.get("message", "").strip()

        if not user_message:
            return

        if not self.user_id or not self.session_id:
            logger.warning("用户未初始化，无法处理聊天消息")
            await self._send_reply("请先初始化会话")
            return

        logger.info(f"[{self.user_id}/{self.session_id}] 收到用户消息: {user_message}")

        # 添加用户消息到历史
        self.chat_history.append({
            "role": "user",
            "content": user_message
        })

        # 保存用户消息到存储
        self.storage.save_message(self.user_id, self.session_id, "user", user_message)

        try:
            # 调用 OpenAI API
            response = await self.client.chat.completions.create(
                model="qwen-plus",
                messages=self.chat_history,
                max_tokens=500,
                temperature=0.7,
            )

            ai_reply = response.choices[0].message.content
            logger.info(f"[{self.user_id}/{self.session_id}] AI 回复: {ai_reply}")

            # 添加 AI 回复到历史
            self.chat_history.append({
                "role": "assistant",
                "content": ai_reply
            })

            # 保存 AI 回复到存储
            self.storage.save_message(self.user_id, self.session_id, "assistant", ai_reply)

            # 文字回复和 TTS 并行发送
            await asyncio.gather(
                self._send_reply(ai_reply),
                self._send_tts_binary(ai_reply)
            )

            # 更新会话列表（标题可能已更新）
            sessions = self.storage.get_user_sessions(self.user_id)
            sessions_reply = {
                "type": "sessions_update",
                "sessions": sessions
            }
            await self.ws.send(json.dumps(sessions_reply, ensure_ascii=False))

        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            await self._send_reply("抱歉，服务暂时不可用，请稍后再试。")

    async def _handle_audio_chunk_upload(self, data):
        """
        处理音频分块上传（收集所有块后进行 ASR）

        Args:
            data: 包含 chunkIndex, totalChunks, data 字段的字典
        """
        chunk_index = data.get("chunkIndex", 0)
        total_chunks = data.get("totalChunks", 0)
        chunk_data = data.get("data", "")

        # 使用当前连接作为会话 ID
        session_id = id(self.ws)

        # 初始化会话
        if session_id not in self.audio_chunks:
            self.audio_chunks[session_id] = {
                "chunks": [None] * total_chunks,
                "total": total_chunks,
                "received": 0
            }

        session = self.audio_chunks[session_id]

        # 存储块
        session["chunks"][chunk_index] = chunk_data
        session["received"] += 1

        logger.info(f"收到音频块 {chunk_index + 1}/{total_chunks}")

        # 检查是否所有块都已接收
        if session["received"] == total_chunks:
            logger.info("所有音频块已接收，开始组装")

            # 组装完整的 base64 音频数据
            base64_audio = "".join(session["chunks"])

            # 清理会话数据
            del self.audio_chunks[session_id]

            # 处理完整的音频数据
            await self._process_audio_data(base64_audio)

    async def _process_audio_data(self, base64_audio):
        """
        处理完整的音频数据

        Args:
            base64_audio: base64 编码的音频数据
        """
        if not base64_audio:
            await self._send_asr_reply(success=False, text="音频数据为空")
            return

        logger.info(f"开始处理音频数据，大小: {len(base64_audio)} 字符")

        # 解码 base64 音频数据
        try:
            audio_data = base64.b64decode(base64_audio)
        except Exception as e:
            logger.error(f"音频数据解码失败: {e}")
            await self._send_asr_reply(success=False, text="音频数据格式错误")
            return

        # 保存到临时文件
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            logger.info(f"音频已保存: {temp_file_path}")

            result = await get_offline_asr_service()._do_transcribe(
                AudioInput(file_path=temp_file_path),
                ASRConfig()
            )
            text = result.text

            if text:
                logger.info(f"ASR 识别成功: {text}")
                await self._send_asr_reply(success=True, text=text)
            else:
                logger.warning("ASR 识别失败")
                await self._send_asr_reply(success=False, text="识别失败")

        except Exception as e:
            logger.error(f"ASR 处理失败: {e}")
            await self._send_asr_reply(success=False, text="识别失败")
        finally:
            # 删除临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def _handle_audio_upload(self, data):
        """
        处理完整音频上传（离线 ASR）

        Args:
            data: 包含 audio 字段的字典（base64 编码的音频数据）
        """
        base64_audio = data.get("audio", "")

        if not base64_audio:
            await self._send_asr_reply(success=False, text="音频数据为空")
            return

        logger.info("收到音频消息，开始 ASR 识别")

        # 解码 base64 音频数据
        try:
            audio_data = base64.b64decode(base64_audio)
        except Exception as e:
            logger.error(f"音频数据解码失败: {e}")
            await self._send_asr_reply(success=False, text="音频数据格式错误")
            return

        # 保存到临时文件
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TEMP_DIR) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            logger.info(f"音频已保存: {temp_file_path}")

            result = await get_offline_asr_service()._do_transcribe(
                AudioInput(file_path=temp_file_path),
                ASRConfig()
            )
            text = result.text

            if text:
                logger.info(f"ASR 识别成功: {text}")
                await self._send_asr_reply(success=True, text=text)
            else:
                logger.warning("ASR 识别失败")
                await self._send_asr_reply(success=False, text="识别失败")

        except Exception as e:
            logger.error(f"ASR 处理失败: {e}")
            await self._send_asr_reply(success=False, text="识别失败")
        finally:
            # 删除临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def _send_asr_reply(self, success, text):
        """
        发送 ASR 识别结果

        Args:
            success: 识别是否成功
            text: 识别的文字或错误信息
        """
        reply_data = {
            "type": "asr_reply",
            "success": success,
            "text": text
        }

        try:
            await self.ws.send(json.dumps(reply_data, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"WebSocket 连接已断开，无法发送 ASR 结果: {e}")

    async def _handle_clear_history(self):
        """处理清空聊天历史请求"""
        logger.info("清空聊天历史")
        self.chat_history = [
            {"role": "system", "content": "你是一个AI助手。回答必须简洁，控制在50字以内。不要使用表情符号，不要废话。"}
        ]

    async def _send_reply(self, message):
        """
        通过 WebSocket 发送回复

        Args:
            message: 回复消息内容
        """
        reply_data = {
            "type": "chat_reply",
            "message": message
        }

        try:
            await self.ws.send(json.dumps(reply_data, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"WebSocket 连接已断开，无法发送消息: {e}")

    async def _send_tts_binary(self, text):
        """
        流式发送 Opus/WebM 音频（边转码边发送）
        使用 MSE 实现前端边收边播

        Args:
            text: 要转换的文本
        """
        try:
            # 发送开始信号（MSE codec 字符串）
            await self.ws.send(json.dumps({
                "type": "tts_stream_start",
                "format": "audio/webm; codecs=opus"
            }, ensure_ascii=False))

            # 使用 Opus 流式转码，直接发送
            tts_provider = get_tts_provider()
            audio_size = 0
            chunk_count = 0
            async for data in tts_provider.stream_to_opus(text):
                audio_size += len(data)
                chunk_count += 1
                await self.ws.send(data)

            logger.info(f"Opus 流式发送完成，总大小: {audio_size} 字节，共 {chunk_count} 块")

            # 发送结束信号
            await self.ws.send(json.dumps({
                "type": "tts_stream_end"
            }, ensure_ascii=False))

        except Exception as e:
            logger.error(f"Opus 流式发送失败: {e}")

    # ─── 流式 ASR 处理方法 ─────────────────────────────────────────────

    async def _handle_audio_stream_start(self):
        """
        处理流式录音开始请求
        创建一个新的流式 ASR 会话
        """
        import uuid
        self._streaming_asr_session_id = uuid.uuid4().hex

        # 使用初始化时创建的服务实例
        await self._streaming_asr_service.start_session(self._streaming_asr_session_id)

        logger.info(f"[{self.user_id}] 流式 ASR 会话开始: {self._streaming_asr_session_id}")

        # 发送确认
        await self.ws.send(json.dumps({
            "type": "audio_stream_start_reply",
            "success": True
        }, ensure_ascii=False))

    async def _handle_audio_stream_chunk(self, data):
        """
        处理流式音频块
        实时进行 ASR 识别并返回 partial 结果

        Args:
            data: 包含音频数据的消息（base64 编码的 float32）
        """
        if not self._streaming_asr_session_id:
            logger.warning("未启动流式 ASR 会话")
            return

        # 获取音频数据（base64 编码的 float32）
        audio_base64 = data.get("data", "")
        if not audio_base64:
            return

        try:
            # 解码 base64 -> bytes -> float32 numpy array
            audio_bytes = base64.b64decode(audio_base64)
            audio_float32 = np.frombuffer(audio_bytes, dtype=np.float32)

            # 使用初始化时创建的服务实例
            result = await self._streaming_asr_service.process_chunk(
                self._streaming_asr_session_id,
                audio_float32
            )

            # 发送结果
            if result.text:
                reply_type = "asr_final" if result.is_sentence_end else "asr_partial"
                await self.ws.send(json.dumps({
                    "type": reply_type,
                    "text": result.text,
                    "is_sentence_end": result.is_sentence_end
                }, ensure_ascii=False))
                logger.debug(f"[{self.user_id}] ASR {reply_type}: {result.text}")

        except Exception as e:
            logger.error(f"流式 ASR 处理失败: {e}")

    async def _handle_audio_stream_end(self):
        """
        处理流式录音结束请求
        获取最终识别结果并发送聊天消息
        """
        if not self._streaming_asr_session_id:
            logger.warning("未启动流式 ASR 会话")
            return

        try:
            # 使用初始化时创建的服务实例
            result = await self._streaming_asr_service.end_session(self._streaming_asr_session_id)
            final_text = result.text

            self._streaming_asr_session_id = None

            logger.info(f"[{self.user_id}] 流式 ASR 最终结果: {final_text}")

            # 发送最终结果
            if final_text:
                await self.ws.send(json.dumps({
                    "type": "asr_final",
                    "text": final_text,
                    "is_sentence_end": True
                }, ensure_ascii=False))

                # 自动发送聊天消息
                await self._handle_chat_message({"message": final_text})

        except Exception as e:
            logger.error(f"流式 ASR 结束失败: {e}")
            self._streaming_asr_session_id = None

    async def close(self):
        """关闭聊天服务"""
        # 关闭 OpenAI client
        await self.client.close()

        # 关闭流式 ASR 服务连接
        if self._streaming_asr_service and hasattr(self._streaming_asr_service, 'close'):
            try:
                await self._streaming_asr_service.close()
            except Exception as e:
                logger.warning(f"关闭流式 ASR 服务失败: {e}")

        logger.info("聊天服务已关闭")