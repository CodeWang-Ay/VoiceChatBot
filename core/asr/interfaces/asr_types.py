"""
ASR 统一类型定义
定义音频输入、识别结果等通用类型
"""
from dataclasses import dataclass, field
from typing import Optional, AsyncIterator, Union
import numpy as np


@dataclass
class AudioInput:
    """音频输入"""
    data: np.ndarray              # 音频数据 (float32, 16kHz)
    file_path: Optional[str] = None  # 音频文件路径（离线模式用）
    format: str = "wav"           # 音频格式


@dataclass
class ASRInput:
    """ASR 统一输入封装"""
    source: Union[np.ndarray, AsyncIterator[np.ndarray], str]  # 音频数据/流/文件路径
    sample_rate: int = 16000
    language: str = "zh"

    @property
    def is_stream(self) -> bool:
        """是否为流式输入"""
        return hasattr(self.source, '__aiter__')

    @property
    def is_file(self) -> bool:
        """是否为文件路径"""
        return isinstance(self.source, str)

    @property
    def is_array(self) -> bool:
        """是否为 numpy 数组（完整音频）"""
        return isinstance(self.source, np.ndarray)


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str                     # 识别文本
    is_final: bool = True         # 是否最终结果
    duration: float = 0.0         # 音频时长（秒）
    confidence: float = 1.0       # 置信度
    provider: str = ""            # 后端名称


@dataclass
class ASRPartialResult:
    """ASR 流式识别中间结果"""
    text: str                     # 当前识别文本
    is_final: bool = False        # 是否最终结果
    is_sentence_end: bool = False # 是否句子结束
    session_id: str = ""          # 会话 ID


@dataclass
class ASRSessionState:
    """ASR 会话状态"""
    session_id: str
    audio_buffer: list = field(default_factory=list)  # 累积音频
    backend_state: dict = field(default_factory=dict)  # 后端内部状态
    created_at: float = 0.0
    last_seen: float = 0.0
    is_active: bool = True