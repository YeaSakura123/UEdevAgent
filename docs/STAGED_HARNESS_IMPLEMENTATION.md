# s01-s12 分阶段实现对照

本项目按 `D:\Code\learn-claude-code-main` README 的节奏，把 Claude Code 风格 harness 机制迁移到 UE agent CLI。实现方式保留本项目的 OpenAI-compatible Chat Completions 与 JSON 单动作协议。

## s01 Agent Loop

位置：`myagent.loop.AgentRuntime.run_turn`

实现：

- 用户消息进入 `messages`
- 模型返回一个 JSON action
- CLI 执行动作并追加 observation
- 直到模型返回 `final`

## s02 Tool Use

位置：`AgentRuntime._build_tool_handlers`

实现：

- 所有工具进入 dispatch map
- 新增工具只加 handler，不改主 loop
- 当前基础工具：`shell`、`read_file`、`write_file`、`edit_file`、`list_files`

## s03 TodoWrite

位置：`myagent.tasks.TodoManager`

实现：

- `todo_update`
- `todo_list`
- 最多一个 `in_progress`
- 状态保存到 `.agent/todos.json`
- 多轮不更新 todo 时，loop 注入 reminder

## s04 Subagent

位置：`AgentRuntime._run_subagent`

实现：

- `subagent` 工具创建独立 `messages`
- 子任务上下文不会污染主会话
- explore 类型默认只允许读文件、列文件和 shell

## s05 Skill Loading

位置：`myagent.skills.SkillLoader`

实现：

- 扫描 `skills/**/SKILL.md`
- system prompt 只放技能名称和描述
- `load_skill` 按需注入完整技能正文
- 已添加 `skills/ue-editor/SKILL.md`

## s06 Context Compact

位置：`myagent.context`

实现：

- `micro_compact` 压缩旧工具结果
- 超过阈值后把完整 transcript 存入 `.agent/transcripts`
- `compact` 工具支持手动压缩

## s07 Task System

位置：`myagent.tasks.TaskManager`

实现：

- `.tasks/task_<id>.json`
- `task_create`
- `task_get`
- `task_update`
- `task_list`
- `claim_task`
- `blockedBy` 依赖完成后自动解除

## s08 Background Tasks

位置：`myagent.background.BackgroundManager`

实现：

- `background_run` 使用线程执行慢命令
- `background_check` 查询状态
- 每轮 LLM 前 drain 完成通知并注入 `<background-results>`

## s09 Agent Teams

位置：`myagent.team.MessageBus`、`TeamManager`

实现：

- `.team/config.json` 保存成员
- `.team/inbox/*.jsonl` 作为队友 inbox
- `spawn_teammate`
- `list_teammates`
- `send_message`
- `read_inbox`
- `broadcast`

## s10 Team Protocols

位置：`myagent.team.TeamManager`

实现：

- `shutdown_request`
- `shutdown_response`
- `plan_submit`
- `plan_review`
- `.team/requests.json` 保存 request_id 与状态

## s11 Autonomous Agents

位置：`TeamManager.claim_ready_task` 与 `claim_task`

实现：

- 队友可以从 `.tasks` 中认领 ready task
- ready 条件：`pending`、无 owner、无 `blockedBy`
- `idle` 工具更新成员状态

说明：当前版本实现自主认领的数据结构与工具入口，尚未让后台 LLM 队友持续轮询，以避免在用户未授权时产生额外模型调用。

## s12 Worktree Task Isolation

位置：`myagent.worktrees.WorktreeManager`

实现：

- `.worktrees/index.json`
- `.worktrees/events.jsonl`
- `worktree_create`
- `worktree_list`
- `worktree_run`
- `worktree_keep`
- `worktree_remove`
- worktree 与 task id 绑定，完成后可自动更新任务状态

## UE Agent 适配

位置：`myagent.ue`

UE 工具作为同一 dispatch map 的领域工具存在：

- `ue_doctor`
- `ue_run_python`

支持模板：

- `kind=custom`
- `kind=list_assets`
- `kind=validate_assets`

安全边界：

- 默认 dry-run
- agent 会话必须传 `--allow-ue-execute` 才能真正执行 UE
- 本项目开发和测试阶段不启动 UE
