"""
ASR 路由器
统一入口，根据配置自动路由到对应策略
切换点只有一处——改配置不改代码
"""
import os
import uuid
import asyncio
import numpy as np
from typing import AsyncIterator, Optional

from loguru import logger

from .interfaces.asr_config import ASRConfig
from .interfaces.asr_types import ASRInput, ASRResult, ASRPartialResult
from .strategies.offline_strategy import OfflineStrategy
from .strategies.streaming_strategy import StreamingStrategy
from .strategies.two_pass_strategy import TwoPassStrategy
from .adapters.base_adapter import ASRBackendAdapter
from .adapters.funasr_offline_adapter import FunASROfflineAdapter
from .adapters.funasr_streaming_adapter import FunASRStreamingAdapter
from .adapters.qwen_offline_adapter import QwenOfflineAdapter
from .adapters.qwen_streaming_adapter import QwenStreamingAdapter


class ASRRouter:
    """
    ASR 统一路由器

    核心思路：请求进来时先做"输入分类"，再根据分类结果路由到对应模式

    切换方式（优先级从高到低）：
    1. 运行时动态切换：router.switch_mode("streaming")
    2. 请求级别：请求头 X-ASR-Mode 覆盖全局配置
    3. 环境变量：ASR_MODE=offline
    """

    # 后端适配器注册表
    _adapters = {
        "funasr": FunASROfflineAdapter,
        "funasr_offline": FunASROfflineAdapter,
        "funasr_streaming": FunASRStreamingAdapter,
        "qwen": QwenOfflineAdapter,
        "qwen_offline": QwenOfflineAdapter,
        "qwen_streaming": QwenStreamingAdapter,
        "qwen_ws": QwenStreamingAdapter,
        "qwen_openai": QwenOfflineAdapter,
    }

    def __init__(self, config: ASRConfig = None):
        """
        初始化 ASR 路由器

        Args:
            config: ASR 配置（可选，默认从环境变量读取）
        """
        self.config = config or ASRConfig.from_env()
        self._strategy = None
        self._sessions = {}  # session_id -> state

        # 初始化策略
        self._init_strategy()

    def _init_strategy(self):
        """根据配置初始化策略"""
        mode = self.config.mode
        backend = self.config.backend or "qwen_streaming"

        logger.info(f"ASRRouter 初始化: mode={mode}, backend={backend}")

        if mode == "2pass":
            # 2-pass 需要两个后端
            first_backend = self.config.first_pass_backend or backend
            second_backend = self.config.second_pass_backend or "qwen_offline"

            first_adapter = self._create_adapter(first_backend)
            second_adapter = self._create_adapter(second_backend)

            self._strategy = TwoPassStrategy(first_adapter, second_adapter)

        elif mode == "offline":
            adapter = self._create_adapter(backend)
            self._strategy = OfflineStrategy(adapter)

        else:  # streaming
            adapter = self._create_adapter(backend)
            self._strategy = StreamingStrategy(adapter)

    def _create_adapter(self, backend: str) -> ASRBackendAdapter:
        """创建后端适配器"""
        adapter_class = self._adapters.get(backend)
        if not adapter_class:
            raise ValueError(f"未知的 ASR 后端: {backend}")
        return adapter_class(self.config)

    # ========== 核心接口：run ==========

    async def run(self, inp: ASRInput) -> AsyncIterator[ASRResult]:
        """
        统一处理入口：根据输入类型自动选择处理方式

        Args:
            inp: ASRInput 封装

        Returns:
            AsyncIterator[ASRResult]: 识别结果流
        """
        if inp.is_stream:
            # 流式输入 → 使用 start/process/end 流程
            session_id = uuid.uuid4().hex
            await self.start_session(session_id)

            try:
                async for chunk in inp.source:
                    result = await self.process_chunk(session_id, chunk)
                    if result.text:
                        yield ASRResult(
                            text=result.text,
                            is_final=result.is_sentence_end,
                            provider=self.config.backend
                        )

                # 结束会话，获取最终结果
                final = await self.end_session(session_id)
                yield ASRResult(
                    text=final.text,
                    is_final=True,
                    provider=final.provider
                )

            finally:
                self._sessions.pop(session_id, None)

        elif inp.is_file:
            # 文件路径 → 直接离线推理
            result = await self._strategy.execute(
                ASRInput(source=inp.source),
                self.config
            )
            yield result

        elif inp.is_array:
            # numpy 数组 → 直接离线推理
            result = await self._strategy.execute(
                ASRInput(source=inp.source),
                self.config
            )
            yield result

    # ========== 会话接口（兼容现有代码） ==========

    async def start_session(self, session_id: str = None) -> str:
        """
        开始流式会话

        Args:
            session_id: 会话 ID（可选，自动生成）

        Returns:
            str: 会话 ID
        """
        if session_id is None:
            session_id = uuid.uuid4().hex

        # 初始化策略
        await self._strategy.backend.initialize()

        # 创建会话状态
        state = {
            "audio_buffer": [],
            "backend_state": {},
        }

        # 如果后端支持流式，调用其 start 方法
        if self._strategy.backend.is_streaming_supported():
            await self._strategy.backend.start_streaming_session(session_id, state["backend_state"])

        self._sessions[session_id] = state
        logger.info(f"[{session_id}] ASR 会话已启动, mode={self.config.mode}")

        return session_id

    async def process_chunk(
        self,
        session_id: str,
        chunk: np.ndarray
    ) -> ASRPartialResult:
        """
        处理音频块

        Args:
            session_id: 会话 ID
            chunk: 音频块 (float32)

        Returns:
            ASRPartialResult: 中间结果
        """
        state = self._sessions.get(session_id)
        if not state:
            logger.warning(f"[{session_id}] 会话不存在")
            return ASRPartialResult(text="", is_final=False)

        # 真流式模式
        if self._strategy.backend.is_streaming_supported():
            result = await self._strategy.backend.infer_streaming(chunk, state["backend_state"])
            return result

        # 累积模式
        else:
            state["audio_buffer"].extend(chunk.tolist())
            return ASRPartialResult(text="", is_final=False)

    async def end_session(self, session_id: str) -> ASRResult:
        """
        结束流式会话

        Args:
            session_id: 会话 ID

        Returns:
            ASRResult: 最终结果
        """
        state = self._sessions.get(session_id)
        if not state:
            logger.warning(f"[{session_id}] 会话不存在")
            return ASRResult(text="", provider=self.config.backend)

        # 真流式模式
        if self._strategy.backend.is_streaming_supported():
            result = await self._strategy.backend.end_streaming_session(
                session_id, state["backend_state"]
            )
            self._sessions.pop(session_id, None)
            logger.info(f"[{session_id}] 流式会话已结束: {result.text}")
            return result

        # 累积模式：调用 infer
        else:
            audio_buffer = state.get("audio_buffer", [])
            if audio_buffer:
                audio = np.array(audio_buffer, dtype=np.float32)
                logger.info(f"[{session_id}] 累积模式调用 infer, 音频长度: {len(audio)} samples")
                result = await self._strategy.backend.infer(audio)
                self._sessions.pop(session_id, None)
                logger.info(f"[{session_id}] 累积会话已结束: {result.text}")
                return result
            else:
                self._sessions.pop(session_id, None)
                return ASRResult(text="", provider=self.config.backend)

    # ========== 动态切换 ==========

    def switch_mode(self, mode: str, backend: str = None):
        """
        运行时动态切换模式（无需重启）

        Args:
            mode: 模式 (offline/streaming/2pass)
            backend: 后端（可选）
        """
        self.config.mode = mode
        if backend:
            self.config.backend = backend

        # 重新初始化策略
        self._init_strategy()

        # 清理旧会话
        self._sessions.clear()

        logger.info(f"ASR 模式已切换: mode={mode}, backend={backend}")

    async def close(self):
        """关闭路由器"""
        if self._strategy:
            await self._strategy.close()
        self._sessions.clear()
        logger.info("ASRRouter 已关闭")


# ========== 全局单例 ==========

_router_instance: Optional[ASRRouter] = None


def get_asr_router(config: ASRConfig = None) -> ASRRouter:
    """
    获取 ASR 路由器（单例）

    Args:
        config: 配置（可选，默认从环境变量读取）

    Returns:
        ASRRouter: 路由器实例
    """
    global _router_instance

    if _router_instance is None:
        _router_instance = ASRRouter(config)

    return _router_instance


def reset_asr_router():
    """重置路由器（用于切换配置）"""
    global _router_instance
    if _router_instance:
        asyncio.run(_router_instance.close())
    _router_instance = None