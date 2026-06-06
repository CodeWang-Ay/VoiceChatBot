这是个很典型的需要分层抽象的架构设计问题。核心思路是：用**策略模式 + 工厂模式**将模式（离线/流式/2-pass）与后端（FunASR/Qwen）解耦，通过统一接口对外屏蔽差异。先看整体架构图：架构分四层，核心是**策略模式解耦模式与后端**。以下是各层的设计要点：

---

**第一层：统一接口**

对调用方只暴露一个接口，屏蔽所有内部差异：

```python
class ASRService(ABC):
    @abstractmethod
    async def transcribe(self, audio: AudioInput, config: ASRConfig) -> ASRResult:
        ...
    
    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncIterator[bytes], config: ASRConfig) -> AsyncIterator[ASRPartialResult]:
        ...
```

`ASRConfig` 里携带 `mode="offline"|"streaming"|"2pass"` 和 `backend="funasr"|"qwen"` 两个维度。

---

**第二层：模式策略层**

三种模式实现同一个 `ASRModeStrategy` 接口，差异只在内部流程：

- `OfflineStrategy`：收完整音频 → 单次推理 → 返回最终结果
- `StreamingStrategy`：按 chunk 推流 → 持续返回 `partial` / `final` 结果
- `TwoPassStrategy`：先跑流式输出草稿（低延迟），音频结束后再调离线模型精校，合并两份结果

2-pass 里的两次调用可以指向**不同后端**（比如流式用 FunASR、精校用 Qwen），这就体现了策略与后端解耦的价值。

---

**第三层：后端适配器**

每个后端实现统一的 `ASRBackendAdapter`：

```python
class ASRBackendAdapter(ABC):
    @abstractmethod
    async def infer(self, audio: np.ndarray, params: dict) -> RawASROutput:
        ...
    
    @abstractmethod
    async def infer_streaming(self, chunk: bytes, state: dict) -> RawASROutput:
        ...
```

FunASR 和 Qwen 各自封装连接管理、协议转换、错误重试，策略层不感知任何后端细节。

---

**工厂组装**

```python
service = ASRServiceFactory.create(
    backend="funasr",
    mode="2pass",
    first_pass_backend="funasr",   # 流式 pass
    second_pass_backend="qwen",    # 精校 pass（可选）
)
```

---

**几个关键决策点**

| 问题 | 建议 |
|---|---|
| 2-pass 两次是否用同一后端？ | 配置化，可相同可不同 |
| 流式结果如何返回？ | 用 `AsyncGenerator` + `ASRPartialResult(is_final: bool)` |
| 后端连接池怎么管？ | 在 adapter 内部维护，工厂创建时按配置初始化 |
| 新增后端怎么扩展？ | 实现 `ASRBackendAdapter`，在工厂注册 `"whisper": WhisperAdapter` |

点击图中任意节点可以继续深入某个部分。你目前用的是什么语言栈（Python/Go/其他），以及 FunASR 是走本地推理还是 gRPC 服务？可以进一步细化某层的代码骨架。