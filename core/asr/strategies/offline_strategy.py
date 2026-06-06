"""
离线 ASR 策略
收完整音频 → 单次推理 → 返回最终结果
"""
import os
import tempfile
import asyncio
import numpy as np
from typing import AsyncIterator

from loguru import logger

from .base_strategy import ASRModeStrategy
from ..interfaces.asr_types import AudioInput, ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig
from ..adapters.base_adapter import ASRBackendAdapter


class OfflineStrategy(ASRModeStrategy):
    """离线 ASR 策略"""

    def __init__(self, backend_adapter: ASRBackendAdapter):
        super().__init__(backend_adapter)

    async def execute(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        """
        离线识别：输入完整音频，单次推理

        Args:
            audio: 音频输入（文件路径或音频数据）
            config: 配置

        Returns:
            ASRResult: 识别结果
        """
        # 初始化后端
        await self.backend.initialize()

        # 获取音频数据
        if audio.file_path:
            # 从文件读取
            audio_data = audio.file_path
        else:
            # 直接使用 numpy 数据
            audio_data = audio.data

        # 调用后端推理
        result = await self.backend.infer(audio_data, {
            "hotword": config.hotword
        })

        return result

    async def execute_stream(
        self,
        audio_stream: AsyncIterator[bytes],
        config: ASRConfig
    ) -> AsyncIterator[ASRPartialResult]:
        """
        流式接口：累积音频后做离线识别

        Args:
            audio_stream: 音频流
            config: 配置

        Returns:
            AsyncIterator[ASRPartialResult]: 识别结果（只有 final）
        """
        # 初始化后端
        await self.backend.initialize()

        # 累积音频
        audio_buffer = []
        async for chunk in audio_stream:
            if isinstance(chunk, bytes):
                audio_data = np.frombuffer(chunk, dtype=np.float32)
                audio_buffer.extend(audio_data.tolist())
            else:
                audio_buffer.extend(chunk.tolist())

        # 累积完成后做离线识别
        if audio_buffer:
            audio_data = np.array(audio_buffer, dtype=np.float32)
            result = await self.backend.infer(audio_data)

            yield ASRPartialResult(
                text=result.text,
                is_final=True,
                is_sentence_end=True
            )
        else:
            yield ASRPartialResult(text="", is_final=True)