"""
ASR 模式策略基类
定义不同识别模式的统一接口
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..interfaces.asr_types import AudioInput, ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig
from ..adapters.base_adapter import ASRBackendAdapter


class ASRModeStrategy(ABC):
    """ASR 模式策略基类"""

    def __init__(self, backend_adapter: ASRBackendAdapter = None):
        self.backend = backend_adapter

    @abstractmethod
    async def execute(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        """
        执行识别

        Args:
            audio: 音频输入
            config: 配置

        Returns:
            ASRResult: 识别结果
        """
        pass

    @abstractmethod
    async def execute_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: ASRConfig
    ) -> AsyncIterator[ASRPartialResult]:
        """
        执行流式识别

        Args:
            audio_stream: 音频流
            config: 配置

        Returns:
            AsyncIterator[ASRPartialResult]: 流式识别结果
        """
        pass

    async def close(self):
        """关闭策略"""
        if self.backend:
            await self.backend.close()