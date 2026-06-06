"""
FunASR 离线适配器
使用 paraformer-zh 本地模型
"""
import os
import asyncio
import numpy as np
from typing import Dict, Any

from funasr import AutoModel
from loguru import logger

from .offline_adapter_base import OfflineAdapterBase
from ..interfaces.asr_types import ASRResult
from ..interfaces.asr_config import ASRConfig


class FunASROfflineAdapter(OfflineAdapterBase):
    """FunASR 离线适配器"""

    def __init__(self, config: ASRConfig):
        super().__init__(config)
        self._model = None
        self._model_path = os.path.join(
            os.environ.get("model_root_dir", ""),
            "paraformer-zh"
        )
        self._vad_model_path = os.path.join(
            os.environ.get("model_root_dir", ""),
            "fsmn-vad"
        )
        self._punc_model_path = os.path.join(
            os.environ.get("model_root_dir", ""),
            "punc_ct"
        )

    async def _do_initialize(self):
        """初始化 FunASR 模型"""
        logger.info(f"正在加载 FunASR 模型: {self._model_path}")
        self._model = await asyncio.to_thread(
            AutoModel,
            model=self._model_path,
            vad_model=self._vad_model_path,
            punc_model=self._punc_model_path,
            trust_remote_code=True
        )
        logger.info("FunASR 模型加载完成")

    async def infer(self, audio: np.ndarray, params: Dict[str, Any] = None) -> ASRResult:
        """FunASR 离线推理"""
        if not self._initialized:
            await self.initialize()

        params = params or {}
        hotword = params.get("hotword", "")

        try:
            # FunASR 接受文件路径或 numpy 数组
            logger.info(f"funasr_infer_offline..............")
            result = await asyncio.to_thread(
                self._model.generate,
                input=audio,
                hotword=hotword,
                disable_pbar=True
            )

            if result and len(result) > 0:
                text = result[0]["text"]
                logger.info(f"FunASR 识别成功: {text}")
                return ASRResult(text=text, provider="funasr_offline")
            else:
                logger.warning("FunASR 识别结果为空")
                return ASRResult(text="", provider="funasr_offline")

        except Exception as e:
            logger.error(f"FunASR 识别失败: {e}")
            return ASRResult(text="", provider="funasr_offline")

    async def close(self):
        """关闭"""
        self._model = None
        self._initialized = False
        logger.info("FunASR 离线适配器已关闭")