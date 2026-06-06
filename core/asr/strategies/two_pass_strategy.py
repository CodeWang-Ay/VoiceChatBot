"""
2-pass ASR 策略
Pass 1: 流式识别（实时显示，低延迟）
Pass 2: 离线识别（录音结束后精校，更准确）
"""
import os
import wave
import time
import asyncio
import numpy as np
from typing import AsyncIterator, Dict, Tuple
from dataclasses import dataclass, field

from loguru import logger

from .base_strategy import ASRModeStrategy
from ..interfaces.asr_types import AudioInput, ASRResult, ASRPartialResult, ASRSessionState
from ..interfaces.asr_config import ASRConfig
from ..adapters.base_adapter import ASRBackendAdapter


@dataclass
class TwoPassSession:
    """2-pass 会话状态"""
    audio_buffer: list = field(default_factory=list)
    stream_state: dict = field(default_factory=dict)
    created_at: float = 0.0
    last_seen: float = 0.0


class TwoPassStrategy(ASRModeStrategy):
    """2-pass ASR 策略"""

    def __init__(
        self,
        streaming_backend: ASRBackendAdapter,
        offline_backend: ASRBackendAdapter
    ):
        """
        初始化 2-pass 策略

        Args:
            streaming_backend: 流式后端（Pass 1）
            offline_backend: 离线后端（Pass 2）
        """
        self.streaming_backend = streaming_backend
        self.offline_backend = offline_backend
        self._sessions: Dict[str, TwoPassSession] = {}
        self._session_ttl = 10 * 60

    async def execute(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        """
        离线接口：直接用 Pass 2 后端

        Args:
            audio: 音频输入
            config: 配置

        Returns:
            ASRResult: 识别结果
        """
        await self.offline_backend.initialize()
        return await self.offline_backend.infer(audio.data or audio.file_path)

    async def execute_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: ASRConfig
    ) -> AsyncIterator[ASRPartialResult]:
        """
        2-pass 流式识别

        Args:
            audio_stream: 音频流
            config: 配置

        Returns:
            AsyncIterator[ASRPartialResult]: 流式识别结果
        """
        # 初始化两个后端
        await self.streaming_backend.initialize()
        await self.offline_backend.initialize()

        # 创建临时会话
        import uuid
        session_id = uuid.uuid4().hex

        state = {}
        audio_buffer = []

        # Pass 1: 流式识别
        await self.streaming_backend.start_streaming_session(session_id, state)

        try:
            async for chunk in audio_stream:
                # 解码音频
                if isinstance(chunk, bytes):
                    audio_data = np.frombuffer(chunk, dtype=np.float32)
                else:
                    audio_data = chunk

                # 累积音频（用于 Pass 2）
                audio_buffer.extend(audio_data.tolist())

                # Pass 1: 流式识别
                result = await self.streaming_backend.infer_streaming(audio_data, state)

                # 返回 partial 结果
                yield result

            # Pass 1: 结束流式会话
            stream_result = await self.streaming_backend.end_streaming_session(session_id, state)
            stream_text = stream_result.text

            # Pass 2: 离线识别
            if audio_buffer:
                audio_float32 = np.array(audio_buffer, dtype=np.float32)
                offline_result = await self.offline_backend.infer(audio_float32)
                offline_text = offline_result.text

                # 使用 Pass 2 结果（更准确）
                final_text = offline_text or stream_text
            else:
                final_text = stream_text

            # 返回最终结果
            yield ASRPartialResult(
                text=final_text,
                is_final=True,
                is_sentence_end=True,
                session_id=session_id
            )

        finally:
            self._sessions.pop(session_id, None)

    # ========== 兼容现有接口的方法 ==========

    async def start_session(self, session_id: str, config: ASRConfig) -> bool:
        """开始 2-pass 会话"""
        await self.streaming_backend.initialize()

        state = {
            "audio_buffer": [],
            "stream_state": {},
            "created_at": time.time(),
            "last_seen": time.time()
        }

        success = await self.streaming_backend.start_streaming_session(session_id, state["stream_state"])

        if success:
            self._sessions[session_id] = state
            logger.info(f"[{session_id}] 2-pass ASR 会话已启动")

        return success

    async def process_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        config: ASRConfig
    ) -> Tuple[str, bool]:
        """处理音频块"""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"[{session_id}] 会话不存在")
            return "", False

        session["last_seen"] = time.time()

        # Pass 1: 流式识别
        result = await self.streaming_backend.infer_streaming(audio_chunk, session["stream_state"])

        # 累积音频（用于 Pass 2）
        session["audio_buffer"].extend(audio_chunk.tolist())

        return result.text, result.is_sentence_end

    async def end_session(self, session_id: str, config: ASRConfig) -> Tuple[str, bool]:
        """结束 2-pass 会话"""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"[{session_id}] 会话不存在")
            return "", False

        try:
            # Pass 1: 结束流式会话
            stream_result = await self.streaming_backend.end_streaming_session(
                session_id,
                session["stream_state"]
            )
            stream_text = stream_result.text
            logger.info(f"[{session_id}] Pass 1 结果: {stream_text}")

            # Pass 2: 离线识别
            audio_buffer = session.get("audio_buffer", [])
            if audio_buffer:
                await self.offline_backend.initialize()
                audio_float32 = np.array(audio_buffer, dtype=np.float32)
                offline_result = await self.offline_backend.infer(audio_float32)
                offline_text = offline_result.text
                logger.info(f"[{session_id}] Pass 2 结果: {offline_text}")

                # 使用 Pass 2 结果（更准确）
                final_text = offline_text or stream_text
            else:
                final_text = stream_text

            self._sessions.pop(session_id, None)
            logger.info(f"[{session_id}] 2-pass ASR 会话已结束，最终结果: {final_text}")

            return final_text, True

        except Exception as e:
            logger.error(f"[{session_id}] 2-pass 结束失败: {e}")
            self._sessions.pop(session_id, None)
            return "", False

    async def close(self):
        """关闭两个后端"""
        await self.streaming_backend.close()
        await self.offline_backend.close()