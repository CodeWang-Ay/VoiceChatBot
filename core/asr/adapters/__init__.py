"""
ASR 后端适配器层
"""
# 基类
from .base_adapter import ASRBackendAdapter
from .offline_adapter_base import OfflineAdapterBase
from .streaming_adapter_base import StreamingAdapterBase

# FunASR 适配器
from .funasr_offline_adapter import FunASROfflineAdapter
from .funasr_streaming_adapter import FunASRStreamingAdapter

# Qwen 适配器
from .qwen_offline_adapter import QwenOfflineAdapter
from .qwen_streaming_adapter import QwenStreamingAdapter

__all__ = [
    # 基类
    "ASRBackendAdapter",
    "OfflineAdapterBase",
    "StreamingAdapterBase",
    # FunASR
    "FunASROfflineAdapter",
    "FunASRStreamingAdapter",
    # Qwen
    "QwenOfflineAdapter",
    "QwenStreamingAdapter",
]