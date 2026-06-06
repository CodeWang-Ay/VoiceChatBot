"""
WebSocket 聊天服务器
使用 websockets 库实现
"""
import os
import json
import time
import asyncio
import webbrowser
import threading
import websockets

from loguru                     import logger
from websockets.exceptions      import ConnectionClosed
from http.server                import HTTPServer, SimpleHTTPRequestHandler
from dotenv                     import load_dotenv
from core.chat_service          import ChatService
from core.asr                   import preload_model,  preload_streaming_models
from core.storage_service       import get_storage_service

ROOT = os.path.dirname(__file__)
PORT = 8008
WS_PORT = 8765  # WebSocket 端口
HOST = "localhost"
# 加载环境变量
load_dotenv(".env")

logger.info(f'{os.getenv("OPENAI_API_KEY")}')


# 存储所有活跃的 WebSocket 连接
ws_connections = set()


class StaticHandler(SimpleHTTPRequestHandler):
    """静态文件处理器"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)


def run_http_server():
    """运行 HTTP 服务器"""
    http_server = HTTPServer((HOST, PORT), StaticHandler)
    logger.info(f"HTTP 服务器启动在 http://{HOST}:{PORT}")
    http_server.serve_forever()


async def handle_client(websocket):
    """
    处理 WebSocket 连接
    """
    logger.info("收到 WebSocket 连接请求")
    ws_connections.add(websocket)

    # 创建聊天服务实例
    chat_service = ChatService(websocket)
    await chat_service.initialize()

    logger.info("WebSocket 连接已建立")

    try:
        # 接收消息循环
        async for message in websocket:
            await chat_service.handle_message(message)

    except ConnectionClosed:
        logger.info("WebSocket 连接已断开")
    except Exception as e:
        logger.error(f"WebSocket 处理异常: {e}")
    finally:
        # 清理连接
        await chat_service.close()
        ws_connections.discard(websocket)
        logger.info("WebSocket 连接已关闭")


async def start_ws_server():
    """启动 WebSocket 服务器"""
    async with websockets.serve(handle_client, HOST, WS_PORT):
        logger.info(f"WebSocket 服务器启动在 ws://{HOST}:{WS_PORT}")
        # 保持服务运行
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("正在关闭 WebSocket 服务器...")
            raise


def open_browser():
    """延迟打开浏览器"""
    time.sleep(1)
    webbrowser.open(f'http://{HOST}:{PORT}')


def run():
    """启动服务器"""
    # 预加载 ASR 模型（根据配置自动选择）
    asr_mode = os.getenv("ASR_MODE")
    logger.info(f"当前服务使用的asr识别模式: asr_mode: {asr_mode}")
    if asr_mode in ["offline", "2pass"]:
        preload_model()
    if asr_mode in ["streaming", "2pass"]:
        preload_streaming_models()

    # 初始化存储服务
    get_storage_service()
    logger.info("存储服务已初始化")

    # 在后台线程中启动 HTTP 服务器
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # 在后台线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()

    # 启动 WebSocket 服务器
    try:
        asyncio.run(start_ws_server())
    except KeyboardInterrupt:
        logger.info("服务器已停止")


if __name__ == "__main__":
    run()