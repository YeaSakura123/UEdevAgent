# 综合课程设计期末报告：面向 UE 游戏开发流程的 AI Agent CLI 设计与实现

> 当前代码状态说明：Markdown 报告保留课程归档叙述，但当前代码已移除旧协作通信机制；现有协作边界以 subagent、task graph 和 worktree 为主。对应 `.docx` 归档文件本轮不重新生成。

## 一、项目概述

本课程设计项目实现了一个名为 `uedev-cli` 的 AI Agent 命令行工具。项目目标不是构建一个普通聊天机器人，而是将大语言模型接入真实的软件开发环境，使其能够围绕开发任务完成观察、规划、工具调用、结果反馈和过程记录。项目重点面向 Unreal Engine 游戏客户端开发场景，探索 AI Agent 在代码维护、资源检查、编辑器自动化和多步骤研发流程中的应用方式。

从整体形态看，`uedev-cli` 是一个 Python CLI 工具。用户可以通过 `uedev run` 发起一次性任务，也可以通过 `uedev chat` 进入交互式会话。Agent 在会话中能够读取项目文件、执行受控命令、维护 todo、创建持久任务、调用子 agent、压缩上下文、处理 UE Python 脚本，并通过权限策略限制高风险操作。

本项目的核心思路是：大模型负责理解需求、拆解任务和决定下一步动作；工程系统负责提供稳定、可观察、可授权的工具环境；用户负责对关键风险操作进行确认。这种设计使 AI 工具从“回答问题”进一步转向“参与开发流程”。

## 二、选题背景与意义

随着大语言模型能力提升，AI 工具在软件开发中的作用正在从代码补全、问答解释，逐渐扩展到任务规划、自动化执行和工程协作。传统问答式 AI 工具通常只能根据用户提供的上下文生成文本，无法直接了解本地项目状态，也无法验证自己的判断是否正确。在真实开发中，许多问题需要先读取项目结构、检查配置、运行测试、分析报错，再继续修改和验证。因此，仅靠单轮问答很难满足复杂工程任务需求。

游戏开发场景对这类工具有更强需求。Unreal Engine 项目通常具有资源数量多、目录层级深、编辑器启动成本高、资源检查流程重复等特点。例如，开发者经常需要检查 `/Game` 目录下的资产类型、执行 Data Validation、整理命名和目录规范，或者生成批处理脚本。若这些工作完全依赖人工完成，效率较低；若完全交给自动化工具，又存在误改资源和难以回溯的风险。

因此，本项目选择“面向 UE 游戏开发流程的 AI Agent CLI”作为课程设计主题，尝试构建一个兼顾自动化能力和安全控制的开发助手。项目的意义主要体现在三个方面：

1. 探索 AI Agent 如何嵌入真实工程环境，而不是停留在文本问答层面。
2. 验证工具调用、权限控制、上下文管理等机制在开发流程中的作用。
3. 面向 UE 游戏开发场景，设计可逐步扩展的资源检查和编辑器自动化方案。

## 三、需求分析

根据项目目标，系统需要满足以下功能需求。

第一，系统需要具备基础 agent loop。用户输入任务后，模型可以根据当前上下文选择直接回答，也可以发起工具调用；工具执行结果需要作为 observation 回填给模型，模型再继续下一轮推理，直到给出最终答案。

第二，系统需要提供统一的工具协议。文件读取、文件写入、文本搜索、shell 执行、todo 管理、任务系统、UE 操作等能力都应通过结构化工具 schema 暴露给模型，避免用自然语言或非结构化文本解析工具动作。

第三，系统需要支持长任务管理。复杂开发任务往往包含多个步骤，因此需要 todo 和持久 task graph 来记录任务状态、依赖关系、负责人和工作树绑定关系。

第四，系统需要支持上下文管理。长期会话中消息历史会不断增长，系统应能估算上下文规模，并在接近阈值时进行压缩，同时保存完整 transcript 以便回溯。

第五，系统需要提供安全边界。AI Agent 不能无条件执行本地命令、修改文件或启动 UE 编辑器。系统必须根据权限模式区分只读、工作区写入、自动审查和完全访问等场景，并对高风险动作要求人工确认。

第六，系统需要适配 Unreal Engine 开发流程。至少应支持 UE 项目发现、引擎配置检查、UE Python 脚本准备和执行、资产列表统计、Data Validation 等基础能力，并默认采用 dry-run 机制降低误操作风险。

第七，系统需要具备可测试性。核心模块应通过自动化单元测试验证，开发者也应能根据文档进行 TUI、slash command、权限模式和 UE dry-run 等手工验证。

## 四、系统总体设计

项目采用分层结构组织，主要模块包括：

- `uedev.cli`：命令行入口，负责解析 `init`、`doctor`、`run`、`chat`、`tasks`、`worktrees`、`ue` 等命令。
- `uedev.runtime`：负责 agent 运行时，包括主循环、历史记录、上下文压缩、slash command、subagent 和 prompt 构建。
- `uedev.llm`：封装模型调用，支持 OpenAI Responses API 以及 OpenAI-compatible Chat Completions。
- `uedev.tools`：提供工具 schema、workspace 文件工具、shell、后台任务、UE 工具和 worktree 工具。
- `uedev.state`：管理本地配置、todo、持久任务和计划记录。
- `uedev.policy`：实现权限模式、命令风险分类和沙箱路径约束。
- `uedev.ui`：实现 Rich + Prompt Toolkit 的终端交互界面和事件渲染。
- `uedev.ue`：实现 UE 项目发现、引擎路径解析、UE Python 包装执行、构建结果渲染和 Perforce 状态检查。
- `test`：保存自动化测试用例，覆盖配置、运行时、历史记录、权限、TUI、UE、MCP、任务系统等模块。

系统核心流程如下：

```text
用户任务
  -> AgentRuntime.run_turn_events()
  -> 调用模型并传入 tool schema
  -> 模型返回普通回答或 tool_calls
  -> 权限策略检查工具调用
  -> 工具 handler 执行动作
  -> 结果作为 role="tool" observation 回填
  -> 模型继续推理
  -> 输出 final answer
```

这一流程保证模型的执行动作来自结构化 tool calling，而不是从正文中解析命令。这样可以降低歧义，也便于对每类工具进行权限控制、日志记录和测试。

## 五、核心功能实现

### 1. Agent 主循环

Agent 主循环主要位于 `uedev.runtime.agent.AgentRuntime`。`run_turn_events()` 是核心入口，它负责将用户消息加入上下文、检查是否需要压缩历史、调用模型、处理流式输出、执行工具调用，并以事件形式向 TUI 或 plain renderer 汇报过程。

模型返回 `tool_calls` 时，系统会将其转换为内部 `ToolAction`，再交给 `_execute_tool_with_status()` 分发执行。工具执行结果会以 `role="tool"` 消息追加到上下文，并保留原始 `tool_call_id`，确保模型能够将 observation 与具体工具调用对应起来。

这一设计形成了“任务 - 工具 - 反馈”的闭环，使 Agent 能够根据真实执行结果继续推理，而不是一次性猜测答案。

### 2. 统一工具系统

项目在 `uedev.tools.specs.get_tool_specs()` 中集中声明工具 schema，在 `AgentRuntime._build_tool_handlers()` 中注册对应 handler。当前工具能力包括：

- 基础开发工具：`read_file`、`write_file`、`edit_file`、`list_files`、`grep`、`shell`。
- 任务规划工具：`todo_update`、`todo_list`、`task_create`、`task_update`、`task_list`、`claim_task`。
- 上下文与技能工具：`compact`、`load_skill`。
- 协作工具：`subagent`、task graph、worktree 隔离执行。
- 后台与隔离执行：`background_run`、`background_check`、`worktree_create`、`worktree_run`、`worktree_remove`。
- UE 相关工具：`ue_doctor`、`ue_run_python`、`ue_build`、`ue_stop_executor`。
- Perforce 辅助工具：`p4_status`、`p4_file_state`、`p4_checkout`、`p4_add`、`p4_delete`、`p4_diff` 等。

工具 schema 与 handler 分离后，后续新增工具只需要补充声明和执行函数，不需要改动主循环逻辑，系统扩展性更好。

### 3. 权限控制与安全机制

项目在 `uedev.policy.permissions` 中实现了多种权限模式：

- `read-only`：只允许读操作和只读命令，高风险操作需要确认或被拒绝。
- `default`：允许工作区内常规读写，对网络、危险命令等要求确认。
- `auto-review`：允许本地常规操作，对网络访问和未知工具加强审查。
- `full-access`：用于用户明确授权后的完整访问。

系统会根据工具类型和命令风险分类决定允许、拒绝或询问用户。例如，`p4_delete` 始终需要显式确认；UE 启动和构建在只读模式下需要确认；shell 命令会通过命令分类器判断是否属于只读、变更、网络、危险或未知类别。

此外，工作区文件工具受到 sandbox 路径约束，避免 Agent 越界访问或修改不属于当前项目的文件。UE 操作默认 dry-run，只有在用户明确传入 `--execute` 或通过权限流程确认后才会真正启动编辑器。

### 4. 任务系统与协作边界

项目同时实现了轻量 todo 和持久 task graph。Todo 更适合当前会话内的多步骤任务，保存在 `.agent/todos.json`；持久任务系统保存在 `.agent/tasks/`，支持任务创建、状态更新、依赖关系、负责人、worktree 绑定和 ready task 认领。

协作边界方面，当前版本保留 subagent、持久 task graph 和 worktree 隔离执行。subagent 用于拆分一次会话内的子任务；task graph 用于保存长期任务状态；worktree 用于隔离复杂代码修改。

### 5. 上下文压缩与历史记录

长期会话中，模型上下文可能逐渐增大。项目通过 `estimate_tokens()` 估算消息体量，并在达到阈值时触发 compact。压缩过程中，系统会保留关键任务状态和最近消息，同时将完整 transcript 保存到会话目录，便于后续回溯。

会话历史保存于 `.agent/sessions/YYYY/MM/DD/<session_id>/`，主要包括：

- `messages.jsonl`：模型上下文消息。
- `display.jsonl`：可重放的 UI 显示记录。
- `metadata.json`：会话元数据。
- `transcript.jsonl`：压缩前完整 transcript。

这种设计兼顾了模型上下文效率和工程可追溯性。

### 6. Unreal Engine 适配

UE 适配是本项目的重要应用方向。系统提供 `uedev ue doctor`、`uedev ue run-python`、`uedev ue list-assets`、`uedev ue validate-assets`、`uedev ue build` 等命令。

`ue doctor` 会查找 `.uproject`，读取 `EngineAssociation`，并根据 `~/.uedev/config.json` 中的 `ue.engines` 配置匹配 UE 引擎路径。系统不会在版本不匹配时随意选择其他引擎，避免误启动错误版本编辑器。

`ue run-python` 会将普通 Python 脚本包装成 UE 可执行脚本，外层捕获异常并输出 JSON。默认情况下它只生成脚本路径和将要执行的 UE 命令，不真正启动编辑器。`list-assets` 和 `validate-assets` 则是基于 UE Python 的内置脚本模板，用于资产列表和数据校验。

这种设计让 Agent 可以先承担“生成脚本、展示命令、辅助检查”的角色，等用户确认后再执行，从而适应 UE 项目高风险、重资源的开发特点。

### 7. 终端交互界面

项目提供 plain 和 full-screen TUI 两种交互方式。plain 模式适合脚本和管道场景；full-screen TUI 使用 Rich 与 Prompt Toolkit，实现持续 transcript、底部固定输入、状态栏、进度提示、slash command 补全和权限确认弹窗。

常用 slash command 包括 `/help`、`/context`、`/diff`、`/todos`、`/tasks`、`/history`、`/subagents`、`/worktree`、`/model`、`/mcp`、`/plan`、`/permissions`、`/compact`、`/clear`、`/exit`、`/ue doctor` 等。它们使用户能够在会话中快速切换模型、查看上下文、调整权限和管理任务。

## 六、实验过程说明

本项目实验过程大致分为五个阶段。

第一阶段是项目定位和参考分析。实验开始时，先明确本项目不是训练模型，而是构建一个面向开发流程的 agent harness。通过分析 Claude Code 风格工具调用流程，将项目目标确定为“模型推理 + 工具执行 + 人工确认”的可控工作流。

第二阶段是搭建最小 Agent CLI。该阶段实现了 `run` 和 `chat` 基础命令，完成用户输入、模型调用、工具调用、observation 回填和最终回答输出。此时系统已经能围绕简单开发任务形成闭环。

第三阶段是扩展工具系统和任务机制。项目逐步加入文件工具、shell、grep、todo、持久任务、subagent、skill loading、background task、worktree 隔离等能力。这一阶段的重点是让 Agent 能够处理多步骤任务，并在任务变复杂时保持状态。

第四阶段是增强安全和可回溯能力。项目加入权限模式、命令风险分类、sandbox 路径限制、历史记录、上下文压缩和 transcript 保存。该阶段解决的问题是：Agent 不仅要“能做事”，还要“可控地做事”。

第五阶段是 UE 开发流程适配。项目实现 UE 项目发现、引擎配置匹配、UE Python 脚本包装、资产列表、Data Validation、构建辅助和 Perforce 状态检查。为了降低风险，UE 执行默认 dry-run，让用户先看到脚本和命令，再决定是否真正执行。

在实验过程中，项目始终采用“先建立可控框架，再逐步扩展能力”的策略。尤其是 UE 场景中，没有直接让 AI 大范围修改资源，而是先从检查、生成脚本和 dry-run 预览切入。这种方式更符合实际游戏开发中的风险控制要求。

## 七、实验验证

本次期末报告撰写前，对当前项目进行了自动化验证：

```powershell
python -m compileall -q uedev test
python -m unittest discover -s test
```

验证结果如下：

- Python 编译检查通过。
- 单元测试共运行 221 个用例。
- 测试结果为 `OK`。

测试覆盖范围包括配置加载、运行时逻辑、历史记录、slash command、TUI、权限策略、UE runner、Perforce、MCP、任务系统和计划模式等模块。

除自动化测试外，项目还在 `docs/VALIDATION.md` 中整理了手工验证清单，包括：

- 启动 `python -m uedev chat` 检查 full-screen TUI。
- 验证 slash command 补全与 `/help`、`/model`、`/plan`、`/permissions` 等命令。
- 验证权限确认弹窗。
- 验证 plain 模式。
- 验证 UE dry-run 命令输出。
- 在真实 UE 项目环境中按需执行 `ue list-assets --execute`。

自动化测试与手工验证清单结合，保证项目不仅在代码层面可运行，也在交互流程上具备可检查路径。

## 八、实验结果与项目亮点

通过本次课程设计，项目已经形成一个较完整的 AI Agent CLI 原型。它能够承接自然语言开发任务，结合本地项目上下文进行分析，并通过工具系统逐步执行和验证。

项目亮点主要包括：

1. 从问答式 AI 转向流程式 AI。Agent 能够读取文件、执行工具、观察结果并继续推理，更接近真实开发流程。
2. 使用结构化 tool calling。工具调用不依赖正文解析，降低了歧义，也便于权限审查和错误处理。
3. 安全机制较完整。系统提供多种权限模式、命令风险分类、sandbox 路径限制和 UE dry-run 机制。
4. 支持长任务协作。Todo、持久 task graph、subagent 和 worktree 机制为复杂任务提供了状态基础。
5. 面向 UE 场景有明确适配。项目不是泛化聊天助手，而是针对 UE 资源检查、脚本执行、构建和 Perforce 状态等实际需求设计。
6. 可验证性较好。项目拥有较多自动化测试，并提供手工验证文档，便于后续维护和扩展。

## 九、遇到的问题与解决方法

第一个问题是如何避免模型输出不可控命令。解决方法是使用原生 tool/function calling，让模型只能通过工具 schema 发起动作，再由权限系统判断是否允许执行。

第二个问题是长期会话上下文过大。解决方法是加入 token 估算、自动 compact 和 transcript 保存，在保留关键信息的同时减少模型输入负担。

第三个问题是 UE 操作风险较高。解决方法是默认 dry-run，先输出生成脚本和命令，不直接启动编辑器；真正执行前需要用户明确授权。

第四个问题是复杂任务状态容易丢失。解决方法是同时实现 lightweight todo 和 persistent task graph，并将任务、worktree 等状态保存在 `.agent/` 目录。

第五个问题是命令行交互信息密度较高。解决方法是提供 full-screen TUI，将 transcript、输入框、状态栏、进度和审批弹窗整合到统一界面，同时保留 plain 模式满足自动化需求。

## 十、存在不足

虽然项目已经具备较完整的原型能力，但仍存在一些不足。

首先，UE 深度集成仍需在更多真实项目中验证。不同 UE 版本的 Python API、Data Validation 接口和编辑器行为可能存在差异，需要继续适配。

其次，当前 UE 自动化主要通过命令行启动编辑器或 commandlet。对于大型项目，反复启动编辑器成本较高，后续应考虑通过 UE 插件、TCP 或 HTTP 与已打开编辑器通信。

再次，多 Agent 协作目前以 session 内 subagent 为主，尚未扩展为持续运行的后台协作者。后续如果要增强协作能力，需要进一步设计调用预算、权限边界和任务调度策略。

此外，项目规范知识主要依赖用户提供。若要更好支持团队项目，应接入项目文档检索、命名规则、资源规范和代码风格检查，使 Agent 的判断更贴合具体团队流程。

最后，当前项目主要是 CLI 形态。虽然 TUI 已经改善交互体验，但对于非技术用户或美术、策划等角色，未来可能需要提供更友好的图形界面或 UE 编辑器内面板。

## 十一、改进方向

后续可从以下方向继续完善：

1. 增加 UE 插件桥接能力，与已打开编辑器进行长连接通信，减少重复启动成本。
2. 增加资源操作白名单和脚本审计，执行前列出将调用的 `unreal.*` API 和目标资源路径。
3. 接入项目规范文档和检索增强能力，使 Agent 能按团队规则检查命名、目录和蓝图结构。
4. 完善 Perforce 工作流，例如 changelist 管理、锁冲突提示和二进制资源差异说明。
5. 强化测试体系，增加端到端 CLI 测试和更多真实 UE dry-run 样例。
6. 扩展 TUI 或编辑器内 UI，使任务状态、审批、日志和结果展示更加直观。
7. 在预算限制和显式授权机制完善后，扩展更可控的多 Agent 协作能力。

## 十二、课程设计总结

本次期末课程设计完成了一个面向软件开发和 UE 游戏开发流程的 AI Agent CLI 原型。项目围绕“模型推理、工具执行、人工确认、结果回溯”这一核心思想，构建了 agent loop、工具系统、权限策略、任务管理、上下文压缩、TUI 交互和 UE 适配等模块。

通过实验可以看出，AI 工具真正进入工程实践时，关键不只是模型是否能生成正确文本，更在于它能否安全地连接真实环境、能否根据执行结果持续调整、能否在风险操作前让用户确认、能否留下可追溯记录。本项目在这些方面进行了较系统的探索。

总体而言，`uedev-cli` 已经具备一个 AI 开发助手的基本形态：它能够理解自然语言任务，观察项目状态，调用受控工具，维护任务进度，并面向 UE 开发提供初步自动化能力。后续若继续完善 UE 插件通信、项目规范检索、多 Agent 协作和图形化界面，它可以进一步发展为更实用的智能研发工具。
