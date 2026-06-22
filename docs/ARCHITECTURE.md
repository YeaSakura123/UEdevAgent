# 架构设计说明

## 核心判断

本项目参考 Claude Code 风格 harness：模型负责推理、规划和决定调用哪些工具；工程侧负责提供稳定、可观察、可授权的工具环境。

当前主循环是：

```text
用户任务 -> LLM -> 原生 tool/function calls -> 工具分发 -> tool 结果 -> LLM -> ... -> final
```

模型输出的普通文本只会作为最终回答或继续提示依据，不再被解析成工具动作。

## 模块分层

- `uedev.llm`：封装 OpenAI Responses API 与 OpenAI-compatible Chat Completions 调用。
- `uedev.tools.specs`：集中声明 canonical 工具名称、描述和 JSON Schema 参数。
- `uedev.runtime.agent`：AgentRuntime 编排、turn loop、工具注册、slash command 分发、早停校验和运行时观察注入。
- `uedev.runtime.context`：token 估算、micro compact、compact 请求构建和 transcript 保存。
- `uedev.runtime.history`：session、messages/display history、metadata 和 transcript 持久化。
- `uedev.runtime.subagents`：当前 session 内的子 agent 执行和回放。
- `uedev.runtime.skills`：按需加载 `SKILL.md`。
- `uedev.state.tasks`：持久化 todo 和任务图，对齐 TodoWrite 与 task graph。
- `uedev.tools.worktrees`：task-aware Git worktree 隔离执行。
- `uedev.tools.workspace` / `uedev.tools.shell` / `uedev.tools.background`：文件、shell、后台任务等基础工具实现。
- `uedev.ue`：UE 项目发现、UE Python 脚本包装、构建、Perforce 和执行器辅助。
- `uedev.cli`：命令行入口，对外暴露 `init`、`doctor`、`run`、`chat`、`tasks`、`worktrees`、`ue`。

当前协作能力以 `subagent`、task graph 和 worktree 为边界。

## 工具协议

CLI 将 `uedev.tools.specs.get_tool_specs()` 生成的 canonical function tool schema 传入 LLM 边界。Chat Completions 直接使用该 schema；Responses API 在 `uedev.llm` 边界转换为 Responses function tool 结构。

`AgentRuntime.run_turn_events()` 将模型返回的每个 tool call 转成内部 `ToolAction`，交给 `_execute_tool_with_status()` 分发到 `_build_tool_handlers()` 注册的 handler。工具执行后，loop 追加 `role="tool"` 的 observation，并带上原始 `tool_call_id` / `call_id`，直到模型返回无 tool calls 的最终回答。

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

- Shell、UE、写文件、Perforce 等高风险操作按权限模式要求确认。
- Agent 想启动 UE 时，harness 会在执行前展示命令并等待用户确认。
- 独立 CLI 的 `uedev ue run-python` 仍必须传 `--execute` 才会启动 UE。
- 真实 UE 项目验证由用户在自己的测试工程中执行，避免开发过程误启动或误改项目。

## API Transport Note

Profiles with `gpt_model=true` use the OpenAI Responses API. Other profiles keep the OpenAI-compatible Chat Completions protocol. The runtime keeps the same internal `ChatMessage` and `ToolCall` abstractions in both modes.
