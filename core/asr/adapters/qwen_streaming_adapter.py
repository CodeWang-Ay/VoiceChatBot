"""
Qwen 流式适配器
使用 WebSocket 连接进行流式 ASR
"""
import os
import json
import asyncio
import socket
import numpy as np
from typing import Dict, Any
from urllib.parse import urlparse

import websockets
from loguru import logger

from .streaming_adapter_base import StreamingAdapterBase
from ..interfaces.asr_types import ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig


class QwenStreamingAdapter(StreamingAdapterBase):
    """Qwen 流式适配器（WebSocket）"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._ws_url = config.ws_url or os.environ.get("ASR_WS_URL", "ws://127.0.0.1:21590/ws/asr")
        self._main_ws = None
        self._ws_connected = False

    async def _do_initialize(self):
        """初始化 WebSocket 连接"""
        await self._connect_main_ws()

    async def _connect_main_ws(self):
        """建立主 WebSocket 连接"""
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
                max_size=10 * 1024 * 1024,
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

    async def start_streaming_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """开始流式会话"""
        if not self._ws_connected:
            await self._connect_main_ws()

        # 发送录音开始
        await self._main_ws.send(json.dumps({"type": "record_start"}))

        state["ws"] = self._main_ws
        state["partial_text"] = ""
        state["is_recording"] = True

        logger.info(f"[{session_id}] Qwen 流式会话已启动")
        return True

    async def infer_streaming(
        self,
        chunk: np.ndarray,
        state: Dict[str, Any]
    ) -> ASRPartialResult:
        """
        WebSocket 流式推理：发送音频块，接收结果

        Args:
            chunk: 音频块 (float32)
            state: 会话状态

        Returns:
            ASRPartialResult: 识别结果
        """
        logger.info(f"qwen_asr infer_streaming...............")
        ws = state.get("ws", self._main_ws)
        if not ws:
            return ASRPartialResult(text="", is_final=False)

        try:
            # 发送音频数据
            await ws.send(chunk.tobytes())

            # 接收结果（非阻塞）
            # WebSocket 服务端可能不立即返回，这里简化处理
            return ASRPartialResult(
                text=state.get("partial_text", ""),
                is_final=False,
                is_sentence_end=False
            )

        except Exception as e:
            logger.error(f"WebSocket 发送失败: {e}")
            return ASRPartialResult(text="", is_final=False)

    async def end_streaming_session(self, session_id: str, state: Dict[str, Any]) -> ASRResult:
        """结束流式会话"""
        ws = state.get("ws", self._main_ws)
        if not ws:
            return ASRResult(text="", provider="qwen_streaming")

        try:
            # 发送录音结束
            await ws.send(json.dumps({"type": "record_end"}))

            # 等待最终结果
            final_text = ""
            for _ in range(50):  # 最多等待 5 秒
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    data = json.loads(msg)

                    if data.get("type") == "record_result":
                        final_text = data.get("text", "")
                        break
                    elif data.get("type") == "partial":
                        state["partial_text"] = data.get("text", "")

                except asyncio.TimeoutError:
                    continue

            logger.info(f"[{session_id}] Qwen 流式最终结果: {final_text}")
            return ASRResult(text=final_text, provider="qwen_streaming")

        except Exception as e:
            logger.error(f"结束会话失败: {e}")
            return ASRResult(text=state.get("partial_text", ""), provider="qwen_streaming")

    async def close(self):
        """关闭 WebSocket 连接"""
        if self._main_ws and self._ws_connected:
            try:
                await self._main_ws.send(json.dumps({"type": "session_end"}))
                msg = await self._main_ws.recv()
            except:
                pass

            try:
                await self._main_ws.close()
            except:
                pass

            self._main_ws = None
            self._ws_connected = False
            logger.info("Qwen 流式适配器已关闭")