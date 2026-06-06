"""
ASR 配置定义
定义模式、后端等配置选项
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class ASRMode(Enum):
    """ASR 模式"""
    OFFLINE = "offline"       # 离线模式
    STREAMING = "streaming"    # 流式模式
    TWO_PASS = "2pass"        # 2-pass 模式


class ASRBackend(Enum):
    """ASR 后端"""
    FUNASR = "funasr"         # FunASR 本地模型
    QWEN_OPENAI = "qwen_openai"  # Qwen OpenAI API
    QWEN_WS = "qwen_ws"       # Qwen WebSocket
    QWEN_HTTP = "qwen_http"   # Qwen HTTP API


@dataclass
class ASRConfig:
    """ASR 配置"""
    mode: str = "offline"                     # 模式: offline/streaming/2pass
    backend: str = "funasr"                  # 后端: funasr/qwen_openai/qwen_ws

    # 2-pass 专用配置
    first_pass_backend: Optional[str] = None  # 第一遍后端（流式）
    second_pass_backend: Optional[str] = None # 第二遍后端（离线精校）

    # 后端参数
    params: Dict[str, Any] = None             # 后端自定义参数
    hotword: str = ""                         # 热词

    # 连接配置
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    ws_url: Optional[str] = None
    model_path: Optional[str] = None
    model_name: Optional[str] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}

    @classmethod
    def from_env(cls) -> 'ASRConfig':
        """从环境变量创建配置"""
        import os

        mode = os.getenv("ASR_MODE", "streaming")
        backend = os.getenv("ASR_BACKEND", "qwen_ws")

        return cls(
            mode=mode,
            backend=backend,
            first_pass_backend=os.getenv("ASR_FIRST_PASS_BACKEND", "qwen_ws"),
            second_pass_backend=os.getenv("ASR_SECOND_PASS_BACKEND", "qwen_openai"),
            api_key=os.getenv("ASR_API_KEY"),
            base_url=os.getenv("ASR_BASE_URL"),
            ws_url=os.getenv("ASR_WS_URL"),
            model_path=os.getenv("model_root_dir"),
            model_name=os.getenv("ASR_MODEL"),
        )