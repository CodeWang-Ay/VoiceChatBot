"""
ASR 统一服务接口
对外暴露的统一接口，屏蔽内部差异
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

from .asr_types import AudioInput, ASRResult, ASRPartialResult, ASRSessionState
from .asr_config import ASRConfig


class ASRService(ABC):
    """ASR 统一服务接口"""

    @abstractmethod
    async def transcribe(self, audio: AudioInput, config: ASRConfig = None) -> ASRResult:
        """
        离线识别：输入完整音频，返回识别结果

        Args:
            audio: 音频输入（文件路径或音频数据）
            config: 配置（可选，默认使用服务配置）

        Returns:
            ASRResult: 识别结果
        """
        pass

    @abstractmethod
    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: ASRConfig = None
    ) -> AsyncIterator[ASRPartialResult]:
        """
        流式识别：输入音频流，返回中间结果

        Args:
            audio_stream: 音频流（每次返回一个 chunk）
            config: 配置（可选）

        Returns:
            AsyncIterator[ASRPartialResult]: 流式识别结果
        """
        pass

    @abstractmethod
    async def start_session(self, session_id: str, config: ASRConfig = None) -> bool:
        """
        开始流式会话

        Args:
            session_id: 会话 ID
            config: 配置（可选，默认使用服务配置）

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    async def process_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        config: ASRConfig = None
    ) -> ASRPartialResult:
        """
        处理音频块

        Args:
            session_id: 会话 ID
            audio_chunk: 音频块数据
            config: 配置（可选，默认使用服务配置）

        Returns:
            ASRPartialResult: 识别结果
        """
        pass

    @abstractmethod
    async def end_session(self, session_id: str, config: ASRConfig = None) -> ASRResult:
        """
        结束流式会话，返回最终结果

        Args:
            session_id: 会话 ID
            config: 配置（可选，默认使用服务配置）

        Returns:
            ASRResult: 最终识别结果
        """
        pass

    @abstractmethod
    async def close(self):
        """关闭服务连接"""
        pass


class ASRServiceBase(ASRService):
    """ASR 服务基类，提供默认实现"""

    def __init__(self, config: ASRConfig):
        self.config = config
        self._sessions: dict = {}  # session_id -> ASRSessionState

    async def transcribe(self, audio: AudioInput, config: ASRConfig = None) -> ASRResult:
        """默认实现：调用子类的 _do_transcribe"""
        return await self._do_transcribe(audio, config or self.config)

    async def transcribe_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: ASRConfig = None
    ) -> AsyncIterator[ASRPartialResult]:
        """默认实现：使用 start/process/end 组合"""
        import uuid
        session_id = uuid.uuid4().hex

        await self.start_session(session_id, config)

        try:
            async for chunk in audio_stream:
                result = await self.process_chunk(session_id, chunk, config)
                yield result

            final = await self.end_session(session_id, config)
            yield ASRPartialResult(
                text=final.text,
                is_final=True,
                is_sentence_end=True,
                session_id=session_id
            )
        finally:
            self._sessions.pop(session_id, None)

    async def _do_transcribe(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        """子类实现离线识别"""
        raise NotImplementedError("子类需实现 _do_transcribe")

    async def _do_start_session(self, session_id: str, config: ASRConfig) -> bool:
        """子类实现开始会话"""
        raise NotImplementedError("子类需实现 _do_start_session")

    async def _do_process_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        config: ASRConfig
    ) -> ASRPartialResult:
        """子类实现处理音频块"""
        raise NotImplementedError("子类需实现 _do_process_chunk")

    async def _do_end_session(self, session_id: str, config: ASRConfig) -> ASRResult:
        """子类实现结束会话"""
        raise NotImplementedError("子类需实现 _do_end_session")

    async def start_session(self, session_id: str, config: ASRConfig = None) -> bool:
        return await self._do_start_session(session_id, config or self.config)

    async def process_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        config: ASRConfig = None
    ) -> ASRPartialResult:
        return await self._do_process_chunk(session_id, audio_chunk, config or self.config)

    async def end_session(self, session_id: str, config: ASRConfig = None) -> ASRResult:
        return await self._do_end_session(session_id, config or self.config)

    async def close(self):
        """默认空实现"""
        pass