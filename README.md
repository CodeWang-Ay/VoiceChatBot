# VoiceChatBot

一个基于 WebSocket 的实时语音聊天机器人应用，支持语音识别（ASR）、语音合成（TTS）和 AI 对话功能。

## 功能特点

- 🎤 **实时语音识别**：支持离线、流式、两阶段识别[2-pass]三种模式
- 🔊 **流式语音合成**：使用 Edge TTS，实时转码为 Opus 格式，边收边播
- 💬 **AI 对话**：集成 OpenAI API，智能回复用户消息
- 👥 **多用户多会话**：支持用户管理、会话创建/切换/删除/重命名
- 🌐 **WebSocket 实时通信**：低延迟的双向通信

## 项目结构

```
VoiceChatBot/
├── server.py              # 主服务器入口（HTTP + WebSocket）
├── index.html             # 前端页面（聊天界面）
├── pyproject.toml         # 项目依赖配置
├── .env                   # 环境变量配置
├── .gitignore             # Git 忽略文件
├── data/
│   └── sessions.json      # 用户和会话数据存储
├── core/
│   ├── chat_service.py    # 聊天服务（OpenAI API 集成）
│   ├── storage_service.py # 会话存储服务
│   ├── asr/               # ASR 语音识别模块
│   │   ├── factory.py     # ASR 工厂
│   │   ├── asr_router.py  # ASR 路由服务
│   │   ├── interfaces/    # 接口定义
│   │   ├── strategies/    # 策略模式实现
│   │   │   ├── offline_strategy.py    # 离线识别
│   │   │   ├── streaming_strategy.py  # 流式识别
│   │   │   ├── two_pass_strategy.py   # 两阶段识别
│   │   │   └── base_strategy.py       # 策略基类
│   │   └── adapters/      # 后端适配器
│   │       ├── funasr_offline_adapter.py    # FunASR 离线
│   │       ├── funasr_streaming_adapter.py  # FunASR 流式
│   │       ├── qwen_offline_adapter.py      # Qwen 离线
│   │       ├── qwen_streaming_adapter.py    # Qwen 流式
│   │       └── base_adapter.py              # 适配器基类
│   └── tts/               # TTS 语音合成模块
│       ├── tts_base.py    # TTS 基类
│       └── edge.py        # Edge TTS 实现
```

## 技术架构

### ASR（语音识别）

采用 **策略模式 + 工厂模式** 架构，实现模式与后端解耦：

- **三种识别模式**：
  - `offline`：收完整音频 → 单次推理 → 返回结果
  - `streaming`：按 chunk 推流 → 持续返回 partial/final 结果
  - `2pass`：流式输出草稿（低延迟），音频结束后离线精校

- **两种后端支持**：
  - FunASR（阿里开源）
  - Qwen（阿里云 API）

### TTS（语音合成）

使用 Edge TTS + ffmpeg 实时转码：

- Edge TTS 生成 MP3 音频
- ffmpeg 实时转码为 Opus/WebM 格式
- 流式发送，前端使用 MSE（Media Source Extensions）边收边播

### 聊天服务

集成 OpenAI API，支持：

- 多轮对话历史管理
- 会话持久化存储
- 流式 TTS 回复

## 快速开始

### 1. 安装依赖

推荐使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境：

```bash
uv sync
```

或使用 pip：

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# OpenAI API 配置
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# ASR 模式配置
ASR_MODE=streaming  # 可选: offline, streaming, 2pass

# ffmpeg 路径（可选，默认使用系统 PATH）
FFMPEG_PATH=ffmpeg
```

### 3. 启动服务器

```bash
python server.py
```

服务器启动后：
- HTTP 服务：`http://localhost:8008`
- WebSocket 服务：`ws://localhost:8765`

浏览器会自动打开聊天界面。

### 4. 使用说明

- **文字聊天**：在输入框输入文字，点击发送
- **语音聊天**：点击「按住说话」按钮，或按空格键开始录音
- **新建会话**：点击左侧「+ 新建会话」
- **切换会话**：点击左侧会话列表
- **重命名会话**：双击会话标题或点击编辑按钮
- **删除会话**：点击会话右侧删除按钮

## WebSocket 消息协议

### 客户端 → 服务器

| 类型 | 说明 |
|------|------|
| `init` | 初始化连接，携带 `user_id` |
| `chat` | 发送聊天消息 |
| `audio_stream_start` | 开始流式录音 |
| `audio_stream_chunk` | 发送音频块（float32 格式） |
| `audio_stream_end` | 结束流式录音 |
| `create_session` | 创建新会话 |
| `switch_session` | 切换会话 |
| `delete_session` | 删除会话 |
| `rename_session` | 重命名会话 |

### 服务器 → 客户端

| 类型 | 说明 |
|------|------|
| `init_reply` | 初始化响应 |
| `chat_reply` | 聊天回复 |
| `asr_partial` | 流式 ASR 中间结果 |
| `asr_final` | ASR 最终结果 |
| `tts_stream_start` | TTS 流开始（携带 codec 信息） |
| `tts_stream_end` | TTS 流结束 |
| (binary) | Opus/WebM 音频数据 |

## 依赖说明

| 依赖 | 用途 |
|------|------|
| `websockets` | WebSocket 服务器 |
| `openai` | OpenAI API 客户端 |
| `edge-tts` | Edge TTS 语音合成 |
| `funasr` | FunASR 语音识别 |
| `torch` / `torchaudio` | FunASR 模型推理 |
| `aiohttp` | 异步 HTTP 客户端 |
| `loguru` | 日志记录 |

## 开发说明

### 添加新的 ASR 后端

1. 在 `core/asr/adapters/` 创建新适配器，继承 `ASRBackendAdapter`
2. 在 `core/asr/factory.py` 注册新后端
3. 配置环境变量指定新后端

### 添加新的 TTS 后端

1. 在 `core/tts/` 创建新实现，继承 `TTSBase`
2. 修改 `get_tts_provider()` 函数返回新实例

## 系统要求

- Python 3.13+
- ffmpeg（用于 Opus 转码）

## 许可证

MIT License