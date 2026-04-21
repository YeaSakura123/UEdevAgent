# 项目任务清单

本清单用于约束开发范围，避免“看起来很像 agent，但无法讲清楚”的功能堆叠。

## 阶段 1：基线与参考理解

- [x] 阅读当前项目结构与 README
- [x] 阅读 `learn-claude-code-main` README 与 s01-s12 分节文档
- [x] 初始化本地 Git 仓库并提交基线
- [x] 明确本项目优先落地 harness，而不是训练模型本身

## 阶段 2：CLI Agent Harness

- [x] 保留最小 agent loop：模型输出动作，CLI 执行动作并回填观察结果
- [x] 将旧 shell/final 协议扩展为可注册工具协议
- [x] 增加持久化 todo 状态，支持多步任务规划
- [x] 增加 chat slash commands：`/help`、`/todos`、`/ue doctor`
- [ ] 后续可扩展：会话 resume、上下文压缩、后台任务、团队 inbox

## 阶段 3：UE Python 能力

- [x] 增加 `ue doctor`：发现 `.uproject` 与 UE 可执行文件配置
- [x] 增加 `ue run-python`：生成或执行 UE Python 脚本
- [x] UE 执行默认 dry-run，防止误启动大型编辑器
- [x] 内置 `list_assets` 与 `validate_assets` 脚本模板
- [ ] 后续可扩展：蓝图/资源批处理工具、Data Validation 规则包、编辑器插件通信

## 阶段 4：求职材料

- [ ] 搜索类似 UE/游戏开发 agent 项目
- [ ] 搜索 agent 开发相关面经和岗位关注点
- [ ] 输出项目亮点、差异点、可讲解风险与改进路线
