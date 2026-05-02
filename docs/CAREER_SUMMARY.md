# 求职展示总结

## 项目一句话

这是一个面向 UE 游戏客户端开发的 Python Agent CLI 原型。它参考 Claude Code / OpenCode 一类 CLI agent 的 harness 思路：模型负责规划和决策，CLI 负责工具、权限、执行、观察和回填；同时加入 UE Python API 通道，让 agent 可以安全地生成、审阅并执行编辑器自动化脚本。

## 当前亮点

- **Agent loop 清晰**：LLM 通过原生 tool/function calling 请求工具，CLI 执行工具并把 observation 回填，能讲清楚 ReAct / tool-use 的闭环。
- **工具协议可扩展**：从早期 `shell/final` 升级到 `tool` 分发，后续加 MCP、RAG、后台任务不需要重写 loop。
- **安全边界明确**：shell 默认人工确认；UE 默认 dry-run；真正启动 UE 需要 `--execute` 或 `--allow-ue-execute`。
- **UE 场景聚焦**：支持 `.uproject`/编辑器路径发现、UE Python 脚本包装、资源列表、Data Validation 脚本模板。
- **可回溯开发**：使用 Git 分阶段提交，并维护 `docs/TASKS.md`，方便面试时展示迭代过程。
- **中文工程说明**：`docs/ARCHITECTURE.md` 和 `docs/UE_AGENT_DESIGN.md` 解释了取舍、风险和后续路线。

## 类似项目调研

### mcp-unreal

[mcp-unreal](https://github.com/remiphilippe/mcp-unreal) 是面向 Claude Code、Cursor、Codex CLI 等 agent 客户端的 UE MCP server。它提供编辑器状态、Actor、Blueprint、Material、PIE、日志、截图、文档查询等工具，并强调“先检查 status、写 UE 代码前查 docs、修改后 build/test”的工作流。

差异点：

- mcp-unreal 是成熟 MCP 工具层，目标是“让外部 agent 全面控制 UE”。
- 本项目是 agent CLI/harness 原型，目标是展示 loop、权限、UE Python 执行边界。
- 本项目暂不要求安装 UE 插件，更适合先做命令行与 commandlet 自动化验证。

### soft-ue-cli

[soft-ue-cli](https://pypi.org/project/soft-ue-cli/1.0.2/) 是 Python CLI + UE C++ bridge plugin，通过 HTTP/JSON-RPC 控制运行中的 UE，覆盖 Actor、Blueprint、Material、PIE、截图、性能分析等 50+ 操作。

差异点：

- soft-ue-cli 需要项目集成插件，优势是运行时交互能力强。
- 本项目不侵入 UE 工程，先通过 UE 官方 Python/commandlet 路径做轻量自动化。
- 本项目的面试表达重点是“为什么先 dry-run、为什么先做资源观察/校验，而不是直接改蓝图”。

### Autonomix

Reddit 上的 [Autonomix 介绍](https://www.reddit.com/r/UnrealEngine5/comments/1rpsqj1/opensourced_an_autonomous_ai_agent_plugin_for/) 提到它是 UE5 内部 AI agent 插件，暴露 60+ engine tools，能生成 Blueprint、编译资产、配置输入、修改项目设置，并加入安全层、重复调用检测、git checkpoints 和执行日志。

差异点：

- Autonomix 是大型 UE 插件工程，覆盖面广。
- 本项目是小而清晰的 CLI agent，展示求职时更容易讲透的子问题：工具调用可靠性、UE Python 包装、人工审批与回滚意识。

### AutoUE

[AutoUE 论文](https://arxiv.org/abs/2603.07106) 研究多 agent 自动生成 3D 游戏，覆盖模型检索、场景生成、玩法交互代码生成和自动化测试；论文特别强调用 RAG 给 agent 提供 UE 工具文档，以缓解工具幻觉。

差异点：

- AutoUE 是研究型 multi-agent 系统，目标是端到端生成游戏。
- 本项目是工程型 harness，目标是让单个 agent 可靠进入 UE 编辑器工作流。
- 后续可借鉴 AutoUE：给 UE Python API、项目规范、Data Validation 规则做检索增强。

## 企业面试关注点

从牛客和 Agent 岗位资料看，面试高频点不只是“会调模型”：

- [牛客 Agent 面试攻略](https://www.nowcoder.com/discuss/1628704) 强调 Agent 架构、ReAct、工具调用可靠性、Human-in-the-loop、参数校验和重试自修复。
- [游戏类 AI agent 应用面经](https://www.nowcoder.com/feed/main/detail/416c3118a9c84fe0a43dc00f9eb9bec2) 提到项目介绍、MCP 工具、上下文管理、ReAct、复杂内容生产流程设计、评估方法和“先发一版保留哪些节点”。
- [百度 Agent 实习面经](https://www.nowcoder.com/feed/main/detail/bb76c9109b8d4958ad46204f5da8609d) 提到 multi-agent、上下文压缩、上下文存储，也会追问后端基础、缓存、消息队列、索引等。
- [牛客大模型应用开发话题](https://www.nowcoder.com/creation/subject/a9f4e74eefa6427e97a56826ac5cff66) 里不少讨论都指向同一件事：大模型应用开发仍然很看重后端工程能力，Agent 只是业务落地的一层。
- [AgentGuide](https://github.com/adongwanai/AgentGuide) 总结的开发线能力包括系统稳定性、工具编排、上下文工程、记忆、任务规划和评估体系。

## 面试讲法

可以这样讲这个项目：

1. 我没有把重点放在“写很多 prompt”，而是先做 agent harness：工具协议、执行器、观察结果、权限边界。
2. UE 是高风险大型软件，所以我先选了最小可控的入口：UE Python + commandlet + dry-run，先观察资源和跑校验，再考虑写操作。
3. 我知道市面上有 mcp-unreal、soft-ue-cli、Autonomix 这类更完整项目，所以我的差异化不是工具数量，而是安全可解释的原型和清楚的演进路线。
4. 如果继续做，我会补三件事：UE API/项目规范 RAG、后台任务与日志、编辑器内 bridge/MCP，使它从“能生成 UE Python”进化成“能和已打开编辑器稳定协作”。

## 可主动承认的不足

- 没有启动 UE 做真实 API 兼容性测试，这符合当前开发约束，但需要后续在测试工程中验证。
- `validate_assets` 的 UE Python API 可能随 UE 版本变化，需要版本适配层。
- 还没有做完整上下文压缩、长期记忆、multi-agent、MCP server。
- 目前没有项目级写操作白名单和 git checkpoint，后续做批量修改前必须补上。
