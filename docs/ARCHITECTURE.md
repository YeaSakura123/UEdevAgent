# 架构设计说明

## 核心判断

本项目参考 `learn-claude-code-main` 的设计理念：Agent 的推理能力来自模型，工程侧要做的是 harness。也就是给模型提供稳定、可观察、可控的工具环境。

当前实现按 s01-s12 搭建完整 harness 闭环：

```text
用户任务 -> LLM -> JSON 动作 -> 工具分发 -> 执行结果 -> LLM -> ... -> final
```

## 模块分层

- `myagent.llm`：封装 OpenAI 兼容 Chat Completions 调用。
- `myagent.protocol`：解析模型输出的 JSON 动作，支持 `tool`、兼容旧版 `shell`、`final`。
- `myagent.loop`：agent loop、工具注册、chat slash commands、早停校验、运行时观察注入。
- `myagent.shell`：跨平台 shell 执行和人工确认。
- `myagent.workspace`：安全文件读写编辑工具。
- `myagent.tasks`：持久化 todo 和任务图，对齐 TodoWrite 与 Task System。
- `myagent.skills`：按需加载 `SKILL.md`。
- `myagent.context`：micro compact、transcript、手动/自动压缩。
- `myagent.background`：后台任务和完成通知。
- `myagent.team`：JSONL inbox、队友状态、协议握手、自主认领任务。
- `myagent.worktrees`：task-aware git worktree 隔离执行。
- `myagent.ue`：UE 项目发现、UE Python 脚本包装、dry-run 命令生成与显式执行。
- `myagent.cli`：命令行入口，对外暴露 `run`、`chat`、`tasks`、`ue`。

## 为什么先做这些

UE 游戏客户端开发的 agent 不应该只会聊天。它至少要能：

- 观察项目：读取文件、运行命令、发现 `.uproject` 和编辑器路径。
- 规划任务：把复杂需求拆成可以回溯的步骤。
- 操作编辑器：通过 UE Python API 查询、校验和批处理资源。
- 控制风险：默认不启动 UE，不做破坏性命令，关键执行必须显式授权。

这四点对应岗位要求中的“Agent 工具链”“游戏研发流程”“工程落地能力”。相比一次性实现多 agent 协作，当前版本更适合面试中讲清楚问题边界和迭代路线。

完整 s01-s12 对照见 `docs/STAGED_HARNESS_IMPLEMENTATION.md`。

## 工具协议

推荐模型输出：

```json
{"type":"tool","name":"todo_update","input":{"items":[{"id":"1","text":"检查项目","status":"in_progress"}]}}
```

UE 操作示例：

```json
{"type":"tool","name":"ue_run_python","input":{"kind":"list_assets","mode":"commandlet","script":"","execute":false}}
```

旧版协议仍兼容：

```json
{"type":"shell","command":"git status --short","reason":"检查工作区"}
```

## 安全边界

- Shell 命令默认需要人工确认，除非用户传 `--yes`。
- UE Python 默认 dry-run，只生成脚本和命令。
- Agent 想启动 UE 时，还必须由 CLI 会话传入 `--allow-ue-execute`。
- 独立 CLI 的 `myagent ue run-python` 也必须传 `--execute` 才会启动 UE。
- 本项目开发过程不启动 UE，UE 真实项目验证由用户自己完成。
