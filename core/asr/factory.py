"""
ASR 服务工厂
根据配置创建对应的 ASR 服务
"""
import asyncio
from typing import Dict

from loguru import logger

from .interfaces.asr_config import ASRConfig
from .interfaces.asr_service import ASRService, ASRServiceBase
from .interfaces.asr_types import AudioInput, ASRResult, ASRPartialResult
from .strategies.base_strategy import ASRModeStrategy
from .strategies.offline_strategy import OfflineStrategy
from .strategies.streaming_strategy import StreamingStrategy
from .strategies.two_pass_strategy import TwoPassStrategy
from .adapters.base_adapter import ASRBackendAdapter
from .adapters.funasr_offline_adapter import FunASROfflineAdapter
from .adapters.funasr_streaming_adapter import FunASRStreamingAdapter
from .adapters.qwen_offline_adapter import QwenOfflineAdapter
from .adapters.qwen_streaming_adapter import QwenStreamingAdapter


class ASRServiceFactory:
    """ASR 服务工厂"""

    # 后端适配器注册表
    _adapters = {
        "funasr": FunASROfflineAdapter,
        "funasr_offline": FunASROfflineAdapter,
        "funasr_streaming": FunASRStreamingAdapter,
        "qwen": QwenOfflineAdapter,
        "qwen_offline": QwenOfflineAdapter,
        "qwen_streaming": QwenStreamingAdapter,
        "qwen_ws": QwenStreamingAdapter,  # 兼容旧名称
        "qwen_openai": QwenOfflineAdapter,  # 兼容旧名称
        "qwen_http": QwenOfflineAdapter,  # HTTP 和 OpenAI 共用
    }

    # 策略注册表
    _strategies = {
        "offline": OfflineStrategy,
        "streaming": StreamingStrategy,
        "2pass": TwoPassStrategy,
    }

    @classmethod
    def register_adapter(cls, name: str, adapter_class: ASRBackendAdapter):
        """注册新的后端适配器"""
        cls._adapters[name] = adapter_class
        logger.info(f"注册 ASR 后端适配器: {name}")

    @classmethod
    def create_adapter(cls, backend: str, config: ASRConfig) -> ASRBackendAdapter:
        """
        创建后端适配器

        Args:
            backend: 后端名称
            config: 配置

        Returns:
            ASRBackendAdapter: 后端适配器实例
        """
        adapter_class = cls._adapters.get(backend)
        if not adapter_class:
            raise ValueError(f"未知的 ASR 后端: {backend}")

        return adapter_class(config)

    @classmethod
    def create_strategy(cls, mode: str, config: ASRConfig) -> ASRModeStrategy:
        """
        创建模式策略

        Args:
            mode: 模式名称
            config: 配置

        Returns:
            ASRModeStrategy: 策略实例
        """
        if mode == "2pass":
            # 2-pass 需要两个后端
            first_pass_backend = config.first_pass_backend or config.backend or "qwen_ws"
            second_pass_backend = config.second_pass_backend or "qwen_openai"

            streaming_adapter = cls.create_adapter(first_pass_backend, config)
            offline_adapter = cls.create_adapter(second_pass_backend, config)

            return TwoPassStrategy(streaming_adapter, offline_adapter)

        else:
            # 离线或流式模式
            backend = config.backend or "qwen_ws"
            adapter = cls.create_adapter(backend, config)

            strategy_class = cls._strategies.get(mode)
            if not strategy_class:
                raise ValueError(f"未知的 ASR 模式: {mode}")

            return strategy_class(adapter)

    @classmethod
    def create(cls, config: ASRConfig = None) -> ASRService:
        """
        创建 ASR 服务

        Args:
            config: 配置（可选，默认从环境变量读取）

        Returns:
            ASRService: ASR 服务实例
        """
        if config is None:
            config = ASRConfig.from_env()

        strategy = cls.create_strategy(config.mode, config)
        service = ASRServiceImpl(strategy, config)

        logger.info(f"创建 ASR 服务: mode={config.mode}, backend={config.backend}")
        return service

    @classmethod
    def create_from_mode(cls, mode: str, backend: str = None) -> ASRService:
        """
        快捷创建：指定模式和后端

        Args:
            mode: 模式 (offline/streaming/2pass)
            backend: 后端 (可选)

        Returns:
            ASRService: ASR 服务实例
        """
        config = ASRConfig(mode=mode, backend=backend)
        return cls.create(config)


class ASRServiceImpl(ASRServiceBase):
    """ASR 服务实现"""

    def __init__(self, strategy: ASRModeStrategy, config: ASRConfig):
        super().__init__(config)
        self.strategy = strategy

    async def _do_transcribe(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        """离线识别"""
        return await self.strategy.execute(audio, config)

    async def _do_start_session(self, session_id: str, config: ASRConfig) -> bool:
        """开始流式会话"""
        if hasattr(self.strategy, 'start_session'):
            return await self.strategy.start_session(session_id, config)
        else:
            # 默认实现
            self._sessions[session_id] = {"audio_buffer": []}
            return True

    async def _do_process_chunk(
        self,
        session_id: str,
        audio_chunk: bytes,
        config: ASRConfig
    ) -> ASRPartialResult:
        """处理音频块"""
        import numpy as np

        if isinstance(audio_chunk, bytes):
            audio_data = np.frombuffer(audio_chunk, dtype=np.float32)
        else:
            audio_data = audio_chunk

        if hasattr(self.strategy, 'process_chunk'):
            result = await self.strategy.process_chunk(session_id, audio_data, config)
            # 策略返回 ASRPartialResult 或 tuple
            if isinstance(result, ASRPartialResult):
                return result
            else:
                # 兼容旧策略返回 tuple
                text, is_sentence_end = result
                return ASRPartialResult(
                    text=text,
                    is_final=is_sentence_end,
                    is_sentence_end=is_sentence_end,
                    session_id=session_id
                )
        else:
            # 默认实现：累积音频
            session = self._sessions.get(session_id)
            if session:
                session["audio_buffer"].extend(audio_data.tolist())
            return ASRPartialResult(text="", is_final=False, session_id=session_id)

    async def _do_end_session(self, session_id: str, config: ASRConfig) -> ASRResult:
        """结束流式会话"""
        if hasattr(self.strategy, 'end_session'):
            result = await self.strategy.end_session(session_id, config)
            # 策略返回 ASRResult 或 tuple
            if isinstance(result, ASRResult):
                return result
            else:
                # 兼容旧策略返回 tuple
                text, success = result
                return ASRResult(text=text, provider=config.backend or "unknown")
        else:
            # 默认实现
            session = self._sessions.get(session_id)
            if session and session.get("audio_buffer"):
                import numpy as np
                audio = np.array(session["audio_buffer"], dtype=np.float32)
                result = await self.strategy.execute(AudioInput(data=audio), config)
                return result
            return ASRResult(text="", provider=config.backend)

    async def close(self):
        """关闭服务"""
        await self.strategy.close()
        self._sessions.clear()


# ========== 快捷函数 ==========

# 单例缓存
_asr_service_cache: Dict[str, ASRService] = {}


def get_asr_service(mode: str = None, backend: str = None) -> ASRService:
    """
    获取 ASR 服务实例（单例）

    Args:
        mode: 模式 (可选，默认从环境变量读取)
        backend: 后端 (可选)

    Returns:
        ASRService: ASR 服务实例
    """
    cache_key = f"{mode or 'env'}:{backend or 'env'}"

    if cache_key not in _asr_service_cache:
        if mode:
            _asr_service_cache[cache_key] = ASRServiceFactory.create_from_mode(mode, backend)
        else:
            _asr_service_cache[cache_key] = ASRServiceFactory.create()

    return _asr_service_cache[cache_key]


def get_streaming_asr_service() -> ASRService:
    """
    获取流式 ASR 服务（单例）
    只负责流式服务，不处理 offline 或 2-pass
    """
    cache_key = "streaming_service"

    if cache_key not in _asr_service_cache:
        config = ASRConfig.from_env()
        backend = config.backend or config.first_pass_backend or "qwen_streaming"
        adapter = ASRServiceFactory.create_adapter(backend, config)
        strategy = StreamingStrategy(adapter)
        service = ASRServiceImpl(strategy, config)
        _asr_service_cache[cache_key] = service
        logger.info(f"创建流式 ASR 服务: backend={backend}")

    return _asr_service_cache[cache_key]


def get_offline_asr_service() -> ASRService:
    """
    获取离线 ASR 服务（单例）
    只负责离线服务
    """
    cache_key = "offline_service"

    if cache_key not in _asr_service_cache:
        config = ASRConfig.from_env()
        backend = config.backend or config.second_pass_backend or "funasr_offline"
        adapter = ASRServiceFactory.create_adapter(backend, config)
        strategy = OfflineStrategy(adapter)
        service = ASRServiceImpl(strategy, config)
        _asr_service_cache[cache_key] = service
        logger.info(f"创建离线 ASR 服务: backend={backend}")

    return _asr_service_cache[cache_key]


def get_two_pass_asr_service() -> ASRService:
    """
    获取 2-pass ASR 服务（单例）
    只负责 2-pass 服务
    """
    cache_key = "two_pass_service"

    if cache_key not in _asr_service_cache:
        config = ASRConfig.from_env()
        first_pass_backend = config.first_pass_backend or "funasr_streaming"
        second_pass_backend = config.second_pass_backend or "funasr_offline"
        streaming_adapter = ASRServiceFactory.create_adapter(first_pass_backend, config)
        offline_adapter = ASRServiceFactory.create_adapter(second_pass_backend, config)
        strategy = TwoPassStrategy(streaming_adapter, offline_adapter)
        service = ASRServiceImpl(strategy, config)
        _asr_service_cache[cache_key] = service
        logger.info(f"创建 2-pass ASR 服务: first_pass={first_pass_backend}, second_pass={second_pass_backend}")

    return _asr_service_cache[cache_key]


def get_audio_stream_service() -> ASRService:
    """
    获取音频流处理服务（单例）
    根据 ASR_MODE 配置动态选择服务：
    - offline: 累积音频后离线识别（不支持实时 partial）
    - streaming: 实时流式识别
    - 2-pass: 流式识别 + 离线精校

    用于 audio_stream_start/chunk/end 消息处理
    """
    config = ASRConfig.from_env()
    mode = config.mode

    if mode == "offline":
        return get_offline_asr_service()
    elif mode == "2pass":
        return get_two_pass_asr_service()
    else:
        return get_streaming_asr_service()


def preload_streaming_models():
    """
    预加载流式 ASR 模型（同步接口）
    根据 ASR_MODE 配置决定加载哪个服务的流式模型：
    - streaming: 加载 get_streaming_asr_service() 的 backend
    - 2pass: 加载 get_two_pass_asr_service() 的 streaming_backend
    """
    config = ASRConfig.from_env()
    mode = config.mode

    async def _preload():
        if mode == "2pass":
            # 2-pass 模式：加载 two_pass_service 的 streaming_backend
            service = get_two_pass_asr_service()
            await service.strategy.streaming_backend.initialize()
            logger.info(f"2-pass 流式模型已预加载")
        else:
            # streaming 模式：加载 streaming_service 的 backend
            service = get_streaming_asr_service()
            await service.strategy.backend.initialize()
            logger.info(f"流式模型已预加载")

    asyncio.run(_preload())
    logger.info("流式 ASR 服务已初始化")


def preload_model():
    """
    预加载离线 ASR 模型（同步接口）
    根据 ASR_MODE 配置决定加载哪个服务的离线模型：
    - offline: 加载 get_offline_asr_service() 的 backend
    - 2pass: 加载 get_two_pass_asr_service() 的 offline_backend
    """
    config = ASRConfig.from_env()
    mode = config.mode

    async def _preload():
        if mode == "2pass":
            # 2-pass 模式：加载 two_pass_service 的 offline_backend
            service = get_two_pass_asr_service()
            await service.strategy.offline_backend.initialize()
            logger.info(f"2-pass 离线模型已预加载")
        else:
            # offline 模式：加载 offline_service 的 backend
            service = get_offline_asr_service()                 # service 就是这个 ASRServiceImpl 类
            await service.strategy.backend.initialize()
            logger.info(f"离线模型已预加载")

    asyncio.run(_preload())
    logger.info("离线 ASR 服务已初始化")


# ========== FunASR 便捷函数 ==========

def get_funasr_two_pass_service() -> ASRService:
    """
    获取 FunASR 2-pass 服务
    Pass 1: FunASRStreamingAdapter (paraformer-zh-streaming)
    Pass 2: FunASROfflineAdapter (paraformer-zh)
    """
    config = ASRConfig(
        mode="2pass",
        first_pass_backend="funasr_streaming",
        second_pass_backend="funasr_offline"
    )
    return ASRServiceFactory.create(config)


def get_funasr_streaming_service() -> ASRService:
    """获取 FunASR 流式服务"""
    return ASRServiceFactory.create_from_mode("streaming", "funasr_streaming")


def get_funasr_offline_service() -> ASRService:
    """获取 FunASR 离线服务"""
    return ASRServiceFactory.create_from_mode("offline", "funasr_offline")