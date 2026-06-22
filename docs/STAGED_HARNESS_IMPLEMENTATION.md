# 分阶段实现对照

本项目按 Claude Code 风格 harness 机制迁移到 UE agent CLI。当前实现同时支持 OpenAI-compatible Chat Completions 和 OpenAI Responses API；模型通过原生 tool/function calling 请求工具，正文不再被解析成工具动作。

## s01 Agent Loop

位置：`uedev.runtime.agent.AgentRuntime.run_turn_events`

- 用户消息进入 `messages`
- 模型返回原生 tool/function calls 或普通最终回答
- CLI 执行工具并追加 `tool` observation
- 直到模型返回无工具调用的最终回答

## s02 Tool Use

位置：`uedev.tools.specs` 与 `AgentRuntime._build_tool_handlers()`

- 所有本地工具共用一套 canonical tool schema
- Chat Completions 原样使用 schema；Responses API 在 LLM 边界转换工具结构
- 新增工具需要同时增加 schema 与 handler，不改主 loop 协议
- 当前基础工具包括 `shell`、`read_file`、`write_file`、`edit_file`、`list_files`、`grep`

## s03 TodoWrite

位置：`uedev.state.tasks.TodoManager`

- `todo_update`
- `todo_list`
- 最多一个 `in_progress`
- 状态保存到 `.agent/todos.json`
- 多轮不更新 todo 时，loop 注入 reminder

## s04 Subagent

位置：`uedev.runtime.subagents`

- `subagent` 工具创建独立 `messages`
- 子任务上下文不污染主会话
- explore 类型默认只允许读文件、列文件和受限 shell
- 子 agent 记录在当前 session 的 `subagents/` 目录下

## s05 Skill Loading

位置：`uedev.runtime.skills.SkillLoader`

- 扫描 `skills/**/SKILL.md`
- system prompt 只放技能名称和描述
- `load_skill` 按需注入完整技能正文

## s06 Context Compact

位置：`uedev.runtime.context` 与 `AgentRuntime._compact_messages()`

- `micro_compact` 压缩旧工具结果
- 接近上下文阈值时自动 compact
- `/compact` 支持手动压缩
- 完整 compact 源 transcript 写入当前 session 的 `transcript.jsonl`

## s07 Task System

位置：`uedev.state.tasks.TaskManager`

- `.agent/tasks/task_<id>.json`
- `task_create`
- `task_get`
- `task_update`
- `task_list`
- `claim_task`
- `blockedBy` 依赖完成后自动解除

## s08 Background Tasks

位置：`uedev.tools.background.BackgroundManager`

- `background_run` 使用线程执行慢命令
- `background_check` 查询状态
- 每轮 LLM 前 drain 完成通知并注入 `<background-results>`

## s09 Worktree Task Isolation

位置：`uedev.tools.worktrees.WorktreeManager`

- `.agent/worktrees/index.json`
- `.agent/worktrees/events.jsonl`
- `worktree_create`
- `worktree_list`
- `worktree_run`
- `worktree_keep`
- `worktree_remove`
- worktree 可与 task id 绑定，完成后更新任务状态

## UE Agent 适配

位置：`uedev.ue`

UE 工具作为同一 dispatch map 的领域工具存在：

- `ue_doctor`
- `ue_run_python`
- `ue_build`
- `ue_stop_executor`
- `p4_status`
- `p4_opened`
- `p4_diff`
- `p4_file_state`
- `p4_checkout`
- `p4_add`
- `p4_delete`
- `p4_reconcile`

安全边界：

- agent 会话在启动 UE 前必须展示命令并等待用户确认
- 独立 `uedev ue ...` 命令默认 dry-run，必须显式传 `--execute`
- 开发和测试阶段不自动启动 UE

## API Transport Note

Profiles with `gpt_model=true` use the OpenAI Responses API. Other profiles keep the OpenAI-compatible Chat Completions protocol. Both modes share the same internal `ChatMessage`, `ToolCall`, and tool dispatch flow.
