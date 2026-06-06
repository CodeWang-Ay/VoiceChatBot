"""
FunASR 流式适配器
使用 paraformer-zh-streaming + fsmn-vad 实现真正的实时流式识别
参考 StreamingASRService 的正确实现
"""
import os
import time
import asyncio
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass, field

from funasr import AutoModel
from loguru import logger

from .streaming_adapter_base import StreamingAdapterBase
from ..interfaces.asr_types import ASRResult, ASRPartialResult
from ..interfaces.asr_config import ASRConfig


# 音频参数
SAMPLE_RATE = 16000
VAD_CHUNK_MS = 200
VAD_CHUNK_SAMPLES = int(SAMPLE_RATE * VAD_CHUNK_MS / 1000)  # 3200
ASR_CHUNK_MS = 600
ASR_CHUNK_SAMPLES = int(SAMPLE_RATE * ASR_CHUNK_MS / 1000)  # 9600

# ASR chunk 配置
CHUNK_SIZE_CFG = [0, 10, 5]
ENCODER_LOOK_BACK = 4
DECODER_LOOK_BACK = 1


@dataclass
class StreamingSessionState:
    """流式识别会话状态"""
    vad_cache: dict = field(default_factory=dict)
    is_speaking: bool = False
    silence_start: float = 0.0
    asr_cache: dict = field(default_factory=dict)
    asr_pending: list = field(default_factory=list)
    sentence_text: str = ""           # 当前正在识别的句子
    accumulated_text: str = ""        # 累积的所有句子文本（用于多段语音）
    sentence_start_time: float = 0.0
    created_at: float = 0.0
    last_seen: float = 0.0


class FunASRStreamingAdapter(StreamingAdapterBase):
    """FunASR 流式适配器"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._asr_model = None
        self._vad_model = None

    async def _do_initialize(self):
        """初始化流式模型"""
        root_path = os.environ.get("model_root_dir", "")
        asr_path = os.path.join(root_path, "paraformer-zh-streaming")
        vad_path = os.path.join(root_path, "fsmn-vad")

        logger.info(f"正在加载 FunASR 流式模型: {asr_path}")

        self._asr_model = await asyncio.to_thread(
            AutoModel,
            model=asr_path,
            model_revision="v2.0.4",
            disable_update=True
        )

        self._vad_model = await asyncio.to_thread(
            AutoModel,
            model=vad_path,
            model_revision="v2.0.4",
            disable_update=True
        )

        logger.info("FunASR 流式模型加载完成")

    async def start_streaming_session(self, session_id: str, state: Dict[str, Any]) -> bool:
        """初始化流式会话"""
        now = time.time()
        session_state = StreamingSessionState(
            vad_cache={},
            is_speaking=False,
            silence_start=0.0,
            asr_cache={},
            asr_pending=[],
            sentence_text="",
            accumulated_text="",
            sentence_start_time=0.0,
            created_at=now,
            last_seen=now
        )
        # 将 session_state 存入 state
        state["session"] = session_state
        logger.info(f"[{session_id}] FunASR 流式会话已启动")
        return True

    async def infer_streaming(
        self,
        chunk: np.ndarray,
        state: Dict[str, Any]
    ) -> ASRPartialResult:
        """
        流式推理
        累积音频到一定长度才识别，返回累积文本 + 当前句子
        """
        if not self._initialized:
            await self.initialize()

        s: StreamingSessionState = state.get("session")
        if not s:
            logger.warning("会话状态不存在")
            return ASRPartialResult(text="", is_final=False, is_sentence_end=False)

        s.last_seen = time.time()

        # VAD 检测
        vad_speech_start = False
        vad_speech_end = False

        try:
            vad_result = await asyncio.to_thread(
                self._vad_model.generate,
                input=chunk,
                cache=s.vad_cache,
                is_final=False,
                chunk_size=VAD_CHUNK_MS,
                disable_pbar=True
            )

            if vad_result and vad_result[0].get("value"):
                for seg in vad_result[0]["value"]:
                    if seg[0] >= 0:
                        vad_speech_start = True
                    if seg[1] >= 0:
                        vad_speech_end = True

        except Exception as e:
            logger.warning(f"VAD 异常: {e}")

        # 语音开始
        if vad_speech_start and not s.is_speaking:
            s.is_speaking = True
            s.silence_start = 0.0
            s.sentence_start_time = time.time()
            s.sentence_text = ""
            s.asr_cache = {}
            s.asr_pending = []
            logger.debug(f"语音开始")

        # 处理音频
        if s.is_speaking:
            s.asr_pending.extend(chunk.tolist())

            if vad_speech_end:
                # VAD 检测到语音结束，结束当前句子并累积
                logger.debug(f"VAD 断句")
                sentence_text = await self._flush_pending(s, is_final=True)
                if sentence_text:
                    s.accumulated_text += sentence_text

                # 重置当前句子状态，准备接收下一段语音
                s.is_speaking = False
                s.sentence_text = ""
                s.asr_cache = {}
                s.asr_pending = []

                # 返回累积文本
                return ASRPartialResult(
                    text=s.accumulated_text,
                    is_final=False,
                    is_sentence_end=True
                )
            else:
                # 流式识别（partial）- 累积到一定长度才识别
                partial_text = await self._flush_pending(s, is_final=False)
                # 返回累积文本 + 当前句子
                full_text = s.accumulated_text + partial_text
                return ASRPartialResult(
                    text=full_text,
                    is_final=False,
                    is_sentence_end=False
                )

        # 如果有累积文本但当前没有说话，返回累积文本
        if s.accumulated_text:
            return ASRPartialResult(
                text=s.accumulated_text,
                is_final=False,
                is_sentence_end=False
            )

        return ASRPartialResult(text="", is_final=False, is_sentence_end=False)

    async def _flush_pending(self, s: StreamingSessionState, is_final: bool) -> str:
        """
        处理待识别的音频
        累积到 ASR_CHUNK_SAMPLES 才识别，减少识别次数
        """
        if is_final:
            # 最终识别：处理剩余的所有音频
            if s.asr_pending:
                audio_chunk = np.array(s.asr_pending, dtype=np.float32)
                s.asr_pending = []
                new_piece = await self._feed_asr(audio_chunk, s.asr_cache, is_final=True)
            else:
                # 传入空 chunk 触发最终输出
                new_piece = await self._feed_asr(
                    np.zeros(160, dtype=np.float32),
                    s.asr_cache,
                    is_final=True
                )
            if new_piece:
                s.sentence_text += new_piece
        else:
            # 增量识别：累积到一定长度才识别
            while len(s.asr_pending) >= ASR_CHUNK_SAMPLES:
                audio_chunk = np.array(s.asr_pending[:ASR_CHUNK_SAMPLES], dtype=np.float32)
                s.asr_pending = s.asr_pending[ASR_CHUNK_SAMPLES:]
                new_piece = await self._feed_asr(audio_chunk, s.asr_cache, is_final=False)
                if new_piece:
                    s.sentence_text += new_piece

        return s.sentence_text

    async def _feed_asr(
        self,
        chunk: np.ndarray,
        asr_cache: dict,
        is_final: bool
    ) -> str:
        """送入 ASR，返回识别的文字片段"""
        try:
            result = await asyncio.to_thread(
                self._asr_model.generate,
                input=chunk,
                cache=asr_cache,
                is_final=is_final,
                chunk_size=CHUNK_SIZE_CFG,
                encoder_chunk_look_back=ENCODER_LOOK_BACK,
                decoder_chunk_look_back=DECODER_LOOK_BACK,
                disable_pbar=True
            )

            if result:
                text = result[0].get("text", "").strip()
                if text:
                    logger.debug(f"ASR partial: {text}")
                return text

        except Exception as e:
            logger.warning(f"ASR 异常: {e}")

        return ""

    async def end_streaming_session(self, session_id: str, state: Dict[str, Any]) -> ASRResult:
        """结束会话，返回最终结果"""
        s: StreamingSessionState = state.get("session")
        if not s:
            return ASRResult(text="", provider="funasr_streaming")

        # 如果正在说话，结束当前句子并累积
        if s.is_speaking and s.sentence_text:
            sentence_text = await self._flush_pending(s, is_final=True)
            if sentence_text:
                s.accumulated_text += sentence_text

        # 返回累积的所有文本
        final_text = s.accumulated_text

        logger.info(f"[{session_id}] FunASR 流式最终结果: {final_text}")
        return ASRResult(text=final_text, provider="funasr_streaming")

    async def close(self):
        """关闭"""
        self._asr_model = None
        self._vad_model = None
        self._initialized = False
        logger.info("FunASR 流式适配器已关闭")