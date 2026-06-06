"""
流式 ASR 策略
按 chunk 推流 → 持续返回 partial / final 结果

支持两种模式：
1. 真流式：后端支持流式时，实时识别
2. 累积模式：后端不支持流式时，累积音频后离线识别
"""
import os
import time
import asyncio
import numpy as np
from typing import AsyncIterator, Dict, Any

from loguru import logger

from .base_strategy import ASRModeStrategy
from ..interfaces.asr_types import AudioInput, ASRResult, ASRPartialResult, ASRSessionState
from ..interfaces.asr_config import ASRConfig
from ..adapters.base_adapter import ASRBackendAdapter


class StreamingStrategy(ASRModeStrategy):
    """流式 ASR 策略（支持真流式和累积模式）"""

    def __init__(self, backend_adapter: ASRBackendAdapter):
        super().__init__(backend_adapter)
        self._sessions: Dict[str, ASRSessionState] = {}
        self._session_ttl = 10 * 60  # 10 分钟

    async def execute(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        """
        离线接口：流式后端可能不支持

        Args:
            audio: 音频输入
            config: 配置

        Returns:
            ASRResult: 识别结果
        """
        if self.backend.is_offline_supported():
            await self.backend.initialize()
            return await self.backend.infer(audio.data or audio.file_path)
        else:
            raise NotImplementedError("流式后端不支持离线推理")

    async def execute_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: ASRConfig
    ) -> AsyncIterator[ASRPartialResult]:
        """
        流式识别：边收边处理

        Args:
            audio_stream: 音频流
            config: 配置

        Returns:
            AsyncIterator[ASRPartialResult]: 流式识别结果
        """
        # 初始化后端
        await self.backend.initialize()

        # 创建临时会话
        import uuid
        session_id = uuid.uuid4().hex

        state = {}
        audio_buffer = []

        # 真流式模式
        if self.backend.is_streaming_supported():
            await self.backend.start_streaming_session(session_id, state)

            try:
                async for chunk in audio_stream:
                    # 解码音频块
                    if isinstance(chunk, bytes):
                        audio_data = np.frombuffer(chunk, dtype=np.float32)
                    else:
                        audio_data = chunk

                    # 累积音频（备用）
                    audio_buffer.extend(audio_data.tolist())

                    # 流式处理
                    result = await self.backend.infer_streaming(audio_data, state)
                    yield result

                # 结束会话
                final_result = await self.backend.end_streaming_session(session_id, state)
                yield ASRPartialResult(
                    text=final_result.text,
                    is_final=True,
                    is_sentence_end=True,
                    session_id=session_id
                )

            finally:
                self._sessions.pop(session_id, None)

        # 累积模式（后端不支持流式）
        else:
            logger.info(f"[{session_id}] 使用累积模式（后端不支持流式）")

            try:
                async for chunk in audio_stream:
                    # 解码音频块
                    if isinstance(chunk, bytes):
                        audio_data = np.frombuffer(chunk, dtype=np.float32)
                    else:
                        audio_data = chunk

                    # 累积音频
                    audio_buffer.extend(audio_data.tolist())

                    # 不返回中间结果
                    yield ASRPartialResult(text="", is_final=False)

                # 结束时调用离线推理
                if audio_buffer:
                    audio_float32 = np.array(audio_buffer, dtype=np.float32)
                    result = await self.backend.infer(audio_float32)
                    yield ASRPartialResult(
                        text=result.text,
                        is_final=True,
                        is_sentence_end=True,
                        session_id=session_id
                    )
                else:
                    yield ASRPartialResult(text="", is_final=True, session_id=session_id)

            finally:
                self._sessions.pop(session_id, None)

    # ========== 兼容现有接口的方法 ==========

    async def start_session(self, session_id: str, config: ASRConfig) -> bool:
        """开始流式会话"""
        await self.backend.initialize()

        state = ASRSessionState(
            session_id=session_id,
            backend_state={},
            created_at=time.time(),
            last_seen=time.time()
        )

        # 真流式模式
        if self.backend.is_streaming_supported():
            success = await self.backend.start_streaming_session(session_id, state.backend_state)
            if success:
                self._sessions[session_id] = state
                logger.info(f"[{session_id}] 流式 ASR 会话已启动")
            return success

        # 累积模式
        else:
            state.backend_state["audio_buffer"] = []
            self._sessions[session_id] = state
            logger.info(f"[{session_id}] 累积模式 ASR 会话已启动")
            return True

    async def process_chunk(
        self,
        session_id: str,
        audio_chunk: np.ndarray,
        config: ASRConfig
    ) -> ASRPartialResult:
        """处理音频块"""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"[{session_id}] 会话不存在")
            return ASRPartialResult(text="", is_final=False)

        session.last_seen = time.time()

        # 真流式模式
        if self.backend.is_streaming_supported():
            result = await self.backend.infer_streaming(audio_chunk, session.backend_state)
            return result

        # 累积模式：只累积音频
        else:
            session.backend_state["audio_buffer"].extend(audio_chunk.tolist())
            return ASRPartialResult(text="", is_final=False)

    async def end_session(self, session_id: str, config: ASRConfig) -> ASRResult:
        """结束流式会话"""
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"[{session_id}] 会话不存在，无法调用 infer")
            return ASRResult(text="", provider=self.backend.__class__.__name__)

        # 真流式模式
        if self.backend.is_streaming_supported():
            result = await self.backend.end_streaming_session(session_id, session.backend_state)
            self._sessions.pop(session_id, None)
            logger.info(f"[{session_id}] 流式 ASR 会话已结束")
            return result

        # 累积模式：调用离线推理
        else:
            audio_buffer = session.backend_state.get("audio_buffer", [])
            logger.info(f"[{session_id}] 累积模式准备调用 infer, 音频长度: {len(audio_buffer)} samples")

            if audio_buffer:
                audio_float32 = np.array(audio_buffer, dtype=np.float32)
                logger.info(f"[{session_id}] 调用 backend.infer...")
                result = await self.backend.infer(audio_float32)
                self._sessions.pop(session_id, None)
                logger.info(f"[{session_id}] 累积模式 ASR 会话已结束，结果: {result.text}")
                return result
            else:
                logger.warning(f"[{session_id}] audio_buffer 为空，无法调用 infer")
                self._sessions.pop(session_id, None)
                return ASRResult(text="", provider=self.backend.__class__.__name__)
                return ASRResult(text="", provider=self.backend.__class__.__name__)