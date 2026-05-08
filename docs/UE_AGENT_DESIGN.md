# 面向 UE 游戏客户端开发的 Agent 设计

## 目标场景

本项目先聚焦“编辑器内容操作”这一条主线：让 agent 能通过 UE Python API 对资源进行观察和批处理。

典型任务包括：

- 列出 `/Game` 下资源，按类型统计资产。
- 执行 Data Validation，输出失败资源和错误摘要。
- 批量修改资源元数据、目录结构或命名。
- 生成脚本后让开发者审阅，再决定是否执行。

## 当前已实现能力

### `uedev ue doctor`

检查：

- `.uproject` 是否可发现。
- `UnrealEditor-Cmd.exe` 是否配置。
- `UnrealEditor.exe` 是否配置。

配置来源：

UE engine paths are configured in `~/.uedev/config.json` under `ue.engines`.
`uedev ue doctor` reads the project's `.uproject` `EngineAssociation` and
matches it to an engine key or alias before deriving editor executable paths.

### `uedev ue run-python`

将普通 Python 脚本包装成 UE 内执行脚本，外层捕获异常并打印 JSON。默认 dry-run，只输出：

- 生成的脚本路径。
- 将要运行的 UE 命令。
- 是否真的执行。

只有传 `--execute` 才会启动 UE。

### Agent 内置 UE 工具

Agent 可请求：

- `ue_doctor`
- `ue_run_python`

`ue_run_python` 支持：

- `script`：执行模型生成或用户给出的 inline UE Python 代码。
- `script_path`：执行指定 `.py` 文件。

执行前由 harness 展示将要启动的 UE 命令，并等待用户 y/N 确认。

## 可能出现的问题

- UE Python API 在不同 UE 版本中会有差异，尤其是 Data Validation 相关接口。
- `commandlet` 模式比 full editor 安全和轻量，但部分编辑器 API 可能只能在 full editor 下工作。
- 大项目资源扫描耗时长，后续需要后台任务和进度观察。
- 自动批量修改资源存在高风险，必须结合版本控制、dry-run diff、人工审批。

## 后续迭代路线

1. 增加资源操作白名单，例如只允许 `/Game/Test` 或指定目录。
2. 增加脚本审计：执行前输出将调用的 `unreal.*` API 和目标路径。
3. 增加 `.agent/ue_runs/*.jsonl` 执行日志，便于回溯。
4. 增加 UE 插件桥接，通过 TCP/HTTP 与已打开的编辑器通信，避免反复启动编辑器。
5. 接入项目规范文档，让 agent 根据团队命名、目录、蓝图规范做检查。
