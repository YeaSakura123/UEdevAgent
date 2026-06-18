# 架构设计说明

## 核心判断

本项目参考 `learn-claude-code-main` 的 harness 思路：模型负责推理、规划和决定调用什么工具，工程侧负责提供稳定、可观察、可控的工具环境。

当前主循环是：

```text
用户任务 -> LLM -> 原生 tool_calls -> 工具分发 -> tool 结果 -> LLM -> ... -> final
```

模型输出的普通文本只会被当作最终回答或继续提示依据，不会再被当作工具动作解析。

## 模块分层

- `uedev.llm`：封装 OpenAI 兼容 Chat Completions 调用，支持原生 tool/function calling。
- `uedev.tool_specs`：集中声明工具名称、描述和 JSON Schema 参数。
- `uedev.loop`：agent loop、工具注册、chat slash commands、早停校验、运行时观察注入。
- `uedev.shell`：跨平台 shell 执行和人工确认。
- `uedev.workspace`：文件读写编辑工具。
- `uedev.tasks`：持久化 todo 和任务图，对齐 TodoWrite 与 Task System。
- `uedev.skills`：按需加载 `SKILL.md`。
- `uedev.context`：micro compact、transcript、手动/自动压缩。
- `uedev.background`：后台任务和完成通知。
- `uedev.team`：JSONL inbox、队友状态、协作握手、自主认领任务。
- `uedev.worktrees`：task-aware git worktree 隔离执行。
- `uedev.ue`：UE 项目发现、UE Python 脚本包装、命令生成与显式执行。
- `uedev.cli`：命令行入口，对外暴露 `run`、`chat`、`tasks`、`ue`。

## 工具协议

CLI 把 `uedev.tool_specs.get_tool_specs()` 传给模型，模型返回结构化 `tool_calls`。`AgentRuntime.run_turn_events()` 把每个 `tool_call` 转成内部 `ToolAction`，交给 `_execute_tool_with_status()` 分发到 `_build_tool_handlers()` 中注册的 handler。

工具执行后，loop 会追加 `role="tool"` 的 observation，并带上原始 `tool_call_id`。模型拿到 observation 后继续下一轮，直到返回没有 `tool_calls` 的最终回答。

UE 操作示例是模型调用原生工具，而不是把 JSON 写进正文：

```json
{
  "name": "ue_run_python",
  "arguments": {
    "script_path": "D:\\Path\\To\\script.py",
    "mode": "commandlet",
    "cwd": "."
  }
}
```

## 安全边界

- Shell 命令默认需要人工确认，除非用户传 `--yes`。
- Agent 想启动 UE 时，harness 会在执行前展示命令并要求用户 y/N 确认。
- 独立 CLI 的 `uedev ue run-python` 仍必须传 `--execute` 才会启动 UE。
- 真实 UE 项目验证由用户在自己的测试工程中执行，避免开发过程误启动或误改项目。

# API transport note

Profiles with `gpt_model=true` use the OpenAI Responses API. Other profiles keep
the existing OpenAI-compatible Chat Completions protocol for providers such as
DeepSeek-compatible endpoints. The runtime keeps the same internal `ToolCall`
and `ChatMessage` abstractions in both modes.
