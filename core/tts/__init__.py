"""
TTS 语音合成服务模块
"""
from .tts_base import TTSBase
from .edge import TTSProvider, get_tts_provider, text_to_speech

__all__ = [
    "TTSBase",
    "TTSProvider",
    "get_tts_provider",
    "text_to_speech",
]