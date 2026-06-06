"""
离线 ASR 适配器基类
继承自 ASRBackendAdapter，专注于离线推理能力
"""
from abc import abstractmethod
import numpy as np
from typing import Dict, Any

from .base_adapter import ASRBackendAdapter
from ..interfaces.asr_types import ASRResult
from ..interfaces.asr_config import ASRConfig


class OfflineAdapterBase(ASRBackendAdapter):
    """离线 ASR 适配器基类"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)

    # ========== 离线接口（必须实现） ==========

    @abstractmethod
    async def infer(self, audio: np.ndarray, params: Dict[str, Any] = None) -> ASRResult:
        """
        离线推理：输入完整音频，返回结果

        Args:
            audio: 音频数据 (float32, 16kHz) 或文件路径
            params: 参数（hotword 等）

        Returns:
            ASRResult: 识别结果
        """
        pass

    # ========== 能力声明 ==========

    def is_offline_supported(self) -> bool:
        """支持离线"""
        return True

    def is_streaming_supported(self) -> bool:
        """不支持流式"""
        return False