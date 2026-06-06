"""
ASR 语音识别服务模块
架构：ASRRouter（统一入口） → Strategy → Adapter → Backend

切换点只有一处——改配置不改代码
"""
from .asr_router import ASRRouter, get_asr_router, reset_asr_router
from .factory import (
    ASRServiceFactory,
    get_asr_service,
    get_streaming_asr_service,
    get_offline_asr_service,
    get_two_pass_asr_service,
    get_audio_stream_service,
    preload_streaming_models,
    preload_model,
    # FunASR 便捷函数
    get_funasr_two_pass_service,
    get_funasr_streaming_service,
    get_funasr_offline_service,
)
from .interfaces import (
    AudioInput,
    ASRInput,
    ASRResult,
    ASRPartialResult,
    ASRConfig,
    ASRMode,
    ASRBackend,
    ASRService,
)
from .strategies import (
    ASRModeStrategy,
    OfflineStrategy,
    StreamingStrategy,
    TwoPassStrategy,
)
from .adapters import (
    ASRBackendAdapter,
    OfflineAdapterBase,
    StreamingAdapterBase,
    FunASROfflineAdapter,
    FunASRStreamingAdapter,
    QwenOfflineAdapter,
    QwenStreamingAdapter,
)

# 兼容别名
FunASRAdapter = FunASROfflineAdapter
QwenOpenAIAdapter = QwenOfflineAdapter
QwenWSAdapter = QwenStreamingAdapter

__all__ = [
    # 工厂
    "ASRServiceFactory",
    "get_asr_service",
    "get_streaming_asr_service",
    "get_offline_asr_service",
    "get_two_pass_asr_service",
    "get_audio_stream_service",
    "preload_streaming_models",
    "preload_model",
    # FunASR 便捷函数
    "get_funasr_two_pass_service",
    "get_funasr_streaming_service",
    "get_funasr_offline_service",
    # 接口
    "AudioInput",
    "ASRResult",
    "ASRPartialResult",
    "ASRConfig",
    "ASRMode",
    "ASRBackend",
    "ASRService",
    # 策略
    "ASRModeStrategy",
    "OfflineStrategy",
    "StreamingStrategy",
    "TwoPassStrategy",
    # 适配器基类
    "ASRBackendAdapter",
    "OfflineAdapterBase",
    "StreamingAdapterBase",
    # 适配器（新命名）
    "FunASROfflineAdapter",
    "FunASRStreamingAdapter",
    "QwenOfflineAdapter",
    "QwenStreamingAdapter",
    # 兼容别名
    "FunASRAdapter",
    "QwenOpenAIAdapter",
    "QwenWSAdapter",
]