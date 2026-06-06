import os
import uuid
import edge_tts
import base64
import asyncio

from loguru import logger
from datetime import datetime
from core.tts.tts_base import TTSBase

class TTSProvider(TTSBase):
    def __init__(self, config, delete_audio_file):
        self.voice = config.get("voice", "zh-CN-XiaoyiNeural")
        self.audio_file_type = "mp3"
        self.output_file = config.get("output_file", "audio_data")  # 默认输出目录
        self.delete_audio_file = delete_audio_file

    def generate_filename(self, extension: str = ".mp3") -> str:
        return os.path.join(
            self.output_file,
            f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}",
        )

    async def text_to_speak(self, text: str, output_file: str | None = None) -> bytes | None:
        try:
            communicate = edge_tts.Communicate(text, voice=self.voice)
            if output_file:
                # 确保目录存在并创建空文件
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, "wb") as f:
                    pass

                # 流式写入音频数据
                with open(output_file, "ab") as f:  # 改为追加模式避免覆盖
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":  # 只处理音频数据块
                            f.write(chunk["data"])
            else:
                # 返回音频二进制数据
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes
        except Exception as e:
            error_msg = f"Edge TTS请求失败: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)  # 抛出异常，让调用方捕获

    async def stream_to_speak(self, text: str):
        """
        流式生成音频数据（异步生成器）- MP3 格式

        Args:
            text: 要转换的文本

        Yields:
            bytes: MP3 音频数据块
        """
        try:
            communicate = edge_tts.Communicate(text, voice=self.voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        except Exception as e:
            error_msg = f"Edge TTS流式请求失败: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)

    async def stream_to_opus(self, text: str):
        """
        流式生成 Opus/WebM 音频数据（异步生成器）
        使用 ffmpeg 实时转码：MP3 → Opus/WebM

        Args:
            text: 要转换的文本

        Yields:
            bytes: Opus/WebM 音频数据块
        """
        try:
            # 获取 ffmpeg 路径（优先从环境变量读取，否则使用系统 PATH）
            ffmpeg_path = os.getenv("FFMPEG_PATH", "ffmpeg")

            # 启动 ffmpeg 进程，将 MP3 转码为 Opus/WebM
            process = await asyncio.create_subprocess_exec(
                ffmpeg_path,
                '-f', 'mp3',           # 输入格式
                '-i', 'pipe:0',        # 从 stdin 读取
                '-c:a', 'libopus',     # Opus 编码
                '-b:a', '32k',         # 比特率 32kbps（语音足够）
                '-f', 'webm',          # 输出 WebM 容器
                '-flush_packets', '1', # 立即刷新输出
                'pipe:1',              # 输出到 stdout
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )

            # 生产者：从 edge_tts 获取 MP3 数据，写入 ffmpeg stdin
            async def write_to_ffmpeg():
                try:
                    communicate = edge_tts.Communicate(text, voice=self.voice)
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            process.stdin.write(chunk["data"])
                            await process.stdin.drain()
                    process.stdin.close()
                except Exception as e:
                    logger.error(f"写入 ffmpeg 失败: {e}")
                    process.stdin.close()

            # 消费者：从 ffmpeg stdout 读取 Opus 数据，yield 给调用者
            async def read_from_ffmpeg():
                try:
                    while True:
                        data = await process.stdout.read(4096)
                        if not data:
                            break
                        yield data
                except Exception as e:
                    logger.error(f"读取 ffmpeg 输出失败: {e}")

            # 创建写入任务
            write_task = asyncio.create_task(write_to_ffmpeg())

            # 同时读取输出
            async for data in read_from_ffmpeg():
                yield data

            # 等待写入任务完成
            await write_task

            # 等待 ffmpeg 进程结束
            await process.wait()

        except Exception as e:
            error_msg = f"Opus 流式转码失败: {e}"
            logger.error(error_msg)
            raise Exception(error_msg)


# 全局 TTS 实例
_tts_provider = None


def get_tts_provider(voice="zh-CN-XiaoxiaoNeural"):
    """获取 TTS 服务单例"""
    global _tts_provider
    if _tts_provider is None:
        _tts_provider = TTSProvider({"voice": voice}, False)
    return _tts_provider


async def text_to_speech(text, voice="zh-CN-XiaoxiaoNeural"):
    """
    将文本转换为语音（返回 base64 编码的音频数据）

    Args:
        text: 要转换的文本
        voice: 语音模型

    Returns:
        base64 编码的音频数据 (mp3格式)
    """
    try:
        tts = get_tts_provider(voice)
        audio_bytes = await tts.text_to_speak(text)
        # 转换为 base64
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"TTS 转换失败: {e}")
        return None






if __name__ == '__main__':
    speaker_role = {
        "zh-CN-XiaoxiaoNeural"  : "（晓晓）"
    }
    # asyncio.run(test_tts_edge())