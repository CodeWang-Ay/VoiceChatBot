"""
Qwen 离线适配器
使用 OpenAI API 调用 Qwen ASR 模型
"""
import os
import io
import asyncio
import wave
import numpy as np
from typing import Dict, Any

from openai import OpenAI
from loguru import logger

from .offline_adapter_base import OfflineAdapterBase
from ..interfaces.asr_types import ASRResult
from ..interfaces.asr_config import ASRConfig


def convert_to_wav(audio_data: np.ndarray) -> io.BytesIO:
    """
    将音频数据转换为 WAV 格式（16kHz 单声道）
    返回 BytesIO 流，可直接传给 OpenAI API
    """
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)      # 单声道
        wav_file.setsampwidth(2)      # 16-bit
        wav_file.setframerate(16000)  # 16kHz
        # float32 -> int16
        audio_int16 = (audio_data * 32768).astype(np.int16)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    wav_buffer.name = "audio.wav"
    return wav_buffer


class QwenOfflineAdapter(OfflineAdapterBase):
    """Qwen 离线适配器（OpenAI API）"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._client = None
        self._api_key = config.api_key or os.environ.get("ASR_API_KEY", "sk-dummy")
        self._base_url = config.base_url or os.environ.get("ASR_BASE_URL", "http://10.2.5.121:8872/v1")
        self._model = config.model_name or os.environ.get("ASR_MODEL", "/data/shared/Qwen3-ASR")

    async def _do_initialize(self):
        """初始化 OpenAI 客户端"""
        logger.info(f"正在初始化 Qwen 离线客户端: {self._base_url}")
        self._client = await asyncio.to_thread(
            OpenAI,
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=30,
            max_retries=0
        )
        logger.info("Qwen 离线客户端初始化完成")

    async def infer(self, audio: np.ndarray, params: Dict[str, Any] = None) -> ASRResult:
        """
        Qwen 离线推理

        Args:
            audio: 音频数据 (float32, 16kHz) 或文件路径
            params: 参数

        Returns:
            ASRResult: 识别结果
        """
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(f"qwen_asr infer_offline...............")
            # 如果 audio 是文件路径
            if isinstance(audio, str):
                audio_file = open(audio, "rb")
            else:
                # numpy 数组转 WAV
                audio_file = convert_to_wav(audio)

            with audio_file:
                result = await asyncio.to_thread(
                    self._client.audio.transcriptions.create,
                    model=self._model,
                    file=audio_file,
                    response_format="json"
                )

            text = result.text.replace("language Chinese<asr_text>", "")
            logger.info(f"Qwen 离线识别成功: {text}")
            return ASRResult(text=text, provider="qwen_offline")

        except Exception as e:
            logger.error(f"Qwen 离线识别失败: {e}")
            return ASRResult(text="", provider="qwen_offline")

    async def close(self):
        """关闭客户端"""
        if self._client:
            close = getattr(self._client, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        self._initialized = False
        logger.info("Qwen 离线适配器已关闭")