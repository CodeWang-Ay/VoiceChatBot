"""
流式 ASR 适配器基类
继承自 ASRBackendAdapter，专注于流式推理能力
"""
from abc import abstractmethod
import numpy as np
from typing import Dict, Any

from .base_adapter import ASRBackendAdapter
from ..interfaces.asr_types import ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig


class StreamingAdapterBase(ASRBackendAdapter):
    """流式 ASR 适配器基类"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)

    # ========== 流式接口（必须实现） ==========

    @abstractmethod
    async def start_streaming_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """
        开始流式会话

        Args:
            session_id: 会话 ID
            state: 会话状态字典

        Returns:
            bool: 是否成功
        """
        pass

    @abstractmethod
    async def infer_streaming(
        self,
        chunk: np.ndarray,
        state: Dict[str, Any]
    ) -> ASRPartialResult:
        """
        流式推理

        Args:
            chunk: 音频块 (float32, 16kHz)
            state: 会话状态

        Returns:
            ASRPartialResult: 部分识别结果
        """
        pass

    @abstractmethod
    async def end_streaming_session(self, session_id: str, state: Dict[str, Any]) -> ASRResult:
        """
        结束流式会话

        Args:
            session_id: 会话 ID
            state: 会话状态

        Returns:
            ASRResult: 最终识别结果
        """
        pass

    # ========== 能力声明 ==========

    def is_offline_supported(self) -> bool:
        """不支持离线"""
        return False

    def is_streaming_supported(self) -> bool:
        """支持流式"""
        return True