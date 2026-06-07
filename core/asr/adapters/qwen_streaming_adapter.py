"""
Qwen 流式适配器
使用 WebSocket 连接进行流式 ASR
支持实时返回 partial 结果
参照 StreamingASRServiceWS 的实现
"""
import os
import json
import asyncio
import socket
import numpy as np
from typing import Dict, Any, Optional
from urllib.parse import urlparse
from dataclasses import dataclass
import time

import websockets
from loguru import logger

from .streaming_adapter_base import StreamingAdapterBase
from ..interfaces.asr_types import ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig


@dataclass
class WSSession:
    """WebSocket ASR 会话"""
    local_session_id: str       # 本地会话 ID
    ws: websockets.WebSocketClientProtocol  # WebSocket 连接
    partial_text: str           # 当前 partial 文本
    final_text: str             # 最终文本
    is_recording: bool          # 是否正在录音
    created_at: float
    last_seen: float


class QwenStreamingAdapter(StreamingAdapterBase):
    """Qwen 流式适配器（WebSocket）参照 StreamingASRServiceWS"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._ws_url = config.ws_url or os.environ.get("ASR_WS_URL", "ws://127.0.0.1:21590/ws/asr")
        self._sessions: Dict[str, WSSession] = {}
        self._session_lock = asyncio.Lock()
        self._main_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._ws_connected: bool = False

        logger.info(f"Qwen 流式适配器初始化，服务器地址: {self._ws_url}")

    async def _do_initialize(self):
        """
        初始化（延迟加载）
        WebSocket 连接会在 start_streaming_session 时按需创建
        """
        logger.info(f"Qwen 流式适配器已初始化（WebSocket 将在首次使用时连接）")

    async def _connect_main_ws(self) -> websockets.WebSocketClientProtocol:
        """建立主 WebSocket 连接并发送 session_start"""
        if self._main_ws is not None and self._ws_connected:
            return self._main_ws

        try:
            logger.info(f"正在连接 WebSocket ASR 服务: {self._ws_url}")

            # 直连 socket 绕过代理
            parsed = urlparse(self._ws_url)
            host = parsed.hostname
            port = parsed.port or (80 if parsed.scheme == 'ws' else 443)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((host, port))

            self._main_ws = await websockets.connect(
                self._ws_url,
                sock=sock,
                max_size=10 * 1024 * 1024,  # 10MB
                ping_interval=30,
                ping_timeout=10
            )

            # 等待 connected 消息
            msg = await self._main_ws.recv()
            data = json.loads(msg)
            if data.get("type") == "connected":
                logger.info(f"WebSocket 连接成功，session_id: {data.get('session_id')}")

            # 发送面试开始
            await self._main_ws.send(json.dumps({"type": "session_start"}))
            msg = await self._main_ws.recv()
            data = json.loads(msg)
            if data.get("type") == "session_started":
                logger.info("面试会话已开始")

            # 设置 PTT 模式
            await self._main_ws.send(json.dumps({"type": "mode", "mode": "ptt"}))
            msg = await self._main_ws.recv()
            data = json.loads(msg)
            if data.get("type") == "mode_set":
                logger.info("已设置 PTT 模式")

            self._ws_connected = True
            return self._main_ws

        except Exception as e:
            logger.error(f"WebSocket 连接失败: {e}")
            self._ws_connected = False
            raise

    async def _receive_loop(self, session: WSSession):
        """接收 WebSocket 消息的循环（核心！）"""
        try:
            while session.is_recording and session.ws:
                msg = await session.ws.recv()
                if isinstance(msg, bytes):
                    # 二进制消息，忽略
                    continue

                data = json.loads(msg)
                msg_type = data.get("type")

                if msg_type == "partial":
                    # 流式中间结果 ★★★ 关键 ★★★
                    session.partial_text = data.get("text", "")
                    logger.debug(f"[{session.local_session_id}] partial: {session.partial_text}")

                elif msg_type == "record_started":
                    logger.debug(f"[{session.local_session_id}] 录音已开始")

                elif msg_type == "record_result":
                    # 录音结束，最终结果
                    session.final_text = data.get("text", "")
                    session.is_recording = False
                    logger.info(f"[{session.local_session_id}] record_result: {session.final_text}")

                elif msg_type == "error":
                    logger.error(f"[{session.local_session_id}] 错误: {data.get('message')}")
                    session.is_recording = False

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"[{session.local_session_id}] WebSocket 连接已关闭")
            session.is_recording = False
        except Exception as e:
            logger.error(f"[{session.local_session_id}] 接收循环异常: {e}")
            session.is_recording = False

    async def start_streaming_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """开始流式会话（PTT 录音开始）"""
        async with self._session_lock:
            try:
                # 获取主 WebSocket 连接
                ws = await self._connect_main_ws()

                # 发送录音开始
                await ws.send(json.dumps({"type": "record_start"}))

                now = time.time()
                session = WSSession(
                    local_session_id=session_id,
                    ws=ws,
                    partial_text="",
                    final_text="",
                    is_recording=True,
                    created_at=now,
                    last_seen=now
                )
                self._sessions[session_id] = session

                # ★★★ 关键：启动接收循环（后台任务）★★★
                asyncio.create_task(self._receive_loop(session))

                logger.info(f"[{session_id}] Qwen 流式会话已启动（PTT 录音开始）")
                return True

            except Exception as e:
                logger.error(f"[{session_id}] Qwen 流式会话启动失败: {e}")
                return False

    async def infer_streaming(
        self,
        chunk: np.ndarray,
        state: Dict[str, Any]
    ) -> ASRPartialResult:
        """
        WebSocket 流式推理：发送音频块，返回当前识别结果

        Args:
            chunk: 音频块 (float32)
            state: 会话状态（由 Strategy 传入，这里不用，用 self._sessions）

        Returns:
            ASRPartialResult: 识别结果
        """
        # 从 session_id 获取会话（state 中有 session_id，但这里需要从 Strategy 传入）
        # Strategy 的 process_chunk 会传入 session_id，但 infer_streaming 只收到 state
        # 所以我们需要从 self._sessions 获取

        # 找到正在录音的会话
        session = None
        for sid, s in self._sessions.items():
            if s.is_recording:
                session = s
                break

        if not session:
            logger.warning("没有活跃的录音会话")
            return ASRPartialResult(text="", is_final=False, is_sentence_end=False)

        session.last_seen = time.time()

        try:
            # 直接发送二进制音频数据
            await session.ws.send(chunk.tobytes())

            # 返回当前 partial 文本（由 _receive_loop 更新）
            return ASRPartialResult(
                text=session.partial_text,
                is_final=False,
                is_sentence_end=False
            )

        except Exception as e:
            logger.error(f"发送音频失败: {e}")
            return ASRPartialResult(text="", is_final=False, is_sentence_end=False)

    async def end_streaming_session(self, session_id: str, state: Dict[str, Any]) -> ASRResult:
        """结束流式会话（PTT 录音结束）"""
        async with self._session_lock:
            session = self._sessions.get(session_id)
            if not session:
                logger.warning(f"[{session_id}] 会话不存在，无法结束")
                return ASRResult(text="", provider="qwen_streaming")

            try:
                # 发送录音结束
                await session.ws.send(json.dumps({"type": "record_end"}))

                # 等待最终结果（最多等待 5 秒）
                for _ in range(50):
                    if not session.is_recording:
                        break
                    await asyncio.sleep(0.1)

                final_text = session.final_text or session.partial_text
                self._sessions.pop(session_id, None)

                logger.info(f"[{session_id}] Qwen 流式会话已结束，最终结果: {final_text}")
                return ASRResult(text=final_text, provider="qwen_streaming")

            except Exception as e:
                logger.error(f"[{session_id}] Qwen 流式结束失败: {e}")
                self._sessions.pop(session_id, None)
                return ASRResult(text="", provider="qwen_streaming")

    async def close(self):
        """关闭 WebSocket 连接"""
        if self._main_ws and self._ws_connected:
            try:
                await self._main_ws.send(json.dumps({"type": "session_end"}))
                msg = await self._main_ws.recv()
                data = json.loads(msg)
                if data.get("type") == "session_result":
                    logger.info(f"面试会话已结束，共 {len(data.get('records', []))} 条录音")
            except Exception as e:
                logger.warning(f"结束面试会话失败: {e}")

            try:
                await self._main_ws.close()
            except:
                pass

            self._main_ws = None
            self._ws_connected = False
            logger.info("Qwen 流式适配器已关闭")