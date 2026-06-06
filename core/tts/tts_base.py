"""
TTS 语音合成服务基类
定义语音合成服务的标准接口
"""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class TTSBase(ABC):
    """TTS 语音合成服务抽象基类"""

    @abstractmethod
    async def text_to_speak(self, text: str, output_file: str | None = None) -> bytes | None:
        """
        将文本转换为语音（异步方法）

        Args:
            text: 要转换的文本
            output_file: 输出文件路径，如果为 None 则返回音频二进制数据

        Returns:
            如果 output_file 为 None，返回音频二进制数据；否则返回 None（音频写入文件）
        """
        pass

    @abstractmethod
    def generate_filename(self, extension: str = ".mp3") -> str:
        """
        生成唯一的音频文件名

        Args:
            extension: 文件扩展名

        Returns:
            生成的文件路径
        """
        pass