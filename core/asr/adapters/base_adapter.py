"""
ASR 后端适配器基类
定义后端统一接口，封装连接管理、协议转换等
"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from ..interfaces.asr_types import ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig


class ASRBackendAdapter(ABC):
    """ASR 后端适配器基类（通用）"""

    def __init__(self, config: ASRConfig):
        self.config = config
        self._initialized = False

    async def initialize(self):
        """初始化后端连接/模型"""
        if not self._initialized:
            await self._do_initialize()
            self._initialized = True

    @abstractmethod
    async def _do_initialize(self):
        """子类实现初始化逻辑"""
        pass

    # ========== 离线接口 ==========

    async def infer(self, audio: np.ndarray, params: Dict[str, Any] = None) -> ASRResult:
        """
        离线推理：输入完整音频，返回结果

        Args:
            audio: 音频数据 (float32, 16kHz) 或文件路径
            params: 后端参数

        Returns:
            ASRResult: 识别结果
        """
        raise NotImplementedError("该适配器不支持离线推理")

    # ========== 流式接口 ==========

    async def start_streaming_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """
        开始流式会话

        Args:
            session_id: 会话 ID
            state: 会话状态

        Returns:
            bool: 是否成功
        """
        raise NotImplementedError("该适配器不支持流式推理")

    async def infer_streaming(
        self,
        chunk: np.ndarray,
        state: Dict[str, Any]
    ) -> ASRPartialResult:
        """
        流式推理：输入音频块，返回中间结果

        Args:
            chunk: 音频块 (float32)
            state: 会话状态

        Returns:
            ASRPartialResult: 识别结果
        """
        raise NotImplementedError("该适配器不支持流式推理")

    async def end_streaming_session(self, session_id: str, state: Dict[str, Any]) -> ASRResult:
        """
        结束流式会话

        Args:
            session_id: 会话 ID
            state: 会话状态

        Returns:
            ASRResult: 最终识别结果
        """
        raise NotImplementedError("该适配器不支持流式推理")

    # ========== 生命周期 ==========

    async def close(self):
        """关闭后端连接"""
        self._initialized = False

    # ========== 能力声明 ==========

    def is_streaming_supported(self) -> bool:
        """是否支持流式"""
        return False

    def is_offline_supported(self) -> bool:
        """是否支持离线"""
        return False

    # ========== 兼容别名（统一接口名） ==========

    async def start_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """兼容别名"""
        return await self.start_streaming_session(session_id, state)

    async def end_session(self, session_id: str, state: Dict[str, Any]) -> ASRResult:
        """兼容别名"""
        return await self.end_streaming_session(session_id, state)