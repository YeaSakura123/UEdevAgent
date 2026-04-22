# 架构设计说明

## 核心判断

本项目参考 `learn-claude-code-main` 的设计理念：Agent 的推理能力来自模型，工程侧要做的是 harness。也就是给模型提供稳定、可观察、可控的工具环境。

当前实现按 s01-s12 搭建完整 harness 闭环：

```text
用户任务 -> LLM -> 原生 tool_calls -> 工具分发 -> tool 结果 -> LLM -> ... -> final
```

## 模块分层

- `uedev.llm`：封装 OpenAI 兼容 Chat Completions 调用，支持原生 tool/function calling。
- `uedev.tool_specs`：集中声明工具名称、描述和 JSON Schema 参数。
- `uedev.protocol`：兼容旧版模型手写 JSON 动作，支持 `tool`、`shell`、`final` 兜底解析。
- `uedev.loop`：agent loop、工具注册、chat slash commands、早停校验、运行时观察注入。
- `uedev.shell`：跨平台 shell 执行和人工确认。
- `uedev.workspace`：安全文件读写编辑工具。
- `uedev.tasks`：持久化 todo 和任务图，对齐 TodoWrite 与 Task System。
- `uedev.skills`：按需加载 `SKILL.md`。
- `uedev.context`：micro compact、transcript、手动/自动压缩。
- `uedev.background`：后台任务和完成通知。
- `uedev.team`：JSONL inbox、队友状态、协议握手、自主认领任务。
- `uedev.worktrees`：task-aware git worktree 隔离执行。
- `uedev.ue`：UE 项目发现、UE Python 脚本包装、dry-run 命令生成与显式执行。
- `uedev.cli`：命令行入口，对外暴露 `run`、`chat`、`tasks`、`ue`。

## 为什么先做这些

UE 游戏客户端开发的 agent 不应该只会聊天。它至少要能：

- 观察项目：读取文件、运行命令、发现 `.uproject` 和编辑器路径。
- 规划任务：把复杂需求拆成可以回溯的步骤。
- 操作编辑器：通过 UE Python API 查询、校验和批处理资源。
- 控制风险：默认不启动 UE，不做破坏性命令，关键执行必须显式授权。

这四点对应岗位要求中的“Agent 工具链”“游戏研发流程”“工程落地能力”。相比一次性实现多 agent 协作，当前版本更适合面试中讲清楚问题边界和迭代路线。

完整 s01-s12 对照见 `docs/STAGED_HARNESS_IMPLEMENTATION.md`。

## 工具协议

当前主路径是模型侧原生 tool/function calling：CLI 把 `uedev.tool_specs.get_tool_specs()` 传给模型，模型返回结构化 `tool_calls`，harness 执行工具后用 `tool_call_id` 回填观察结果。

旧版 JSON action 仍保留为兼容路径，例如：

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
- 独立 CLI 的 `uedev ue run-python` 也必须传 `--execute` 才会启动 UE。
- 本项目开发过程不启动 UE，UE 真实项目验证由用户自己完成。
