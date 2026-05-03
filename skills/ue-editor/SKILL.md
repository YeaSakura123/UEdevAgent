---
name: ue-editor
description: UE 编辑器自动化工作流，优先使用资源观察、Data Validation 和人工审批。
---

# UE 编辑器自动化技能

## 使用原则

1. 先调用 `ue_doctor`，确认 `.uproject` 与编辑器路径。
2. 对资源进行写操作前，先列出目标路径、预期改动和回滚方案。
3. 大批量资产操作前，要求用户确认 Git 状态或提交点。

## 常见工具选择

- 资源盘点：`ue_run_python` with `kind=list_assets`
- 资源校验：`ue_run_python` with `kind=validate_assets`
- 自定义脚本：`ue_run_python` with `kind=custom`
- 编辑器路径检查：`ue_doctor`

## 脚本执行方式

- 临时脚本：直接把完整 UE Python 内容传给 `ue_run_python.script`，由 harness 写入 `.agent/ue_runs/<run_id>/user_script.py`。
- 已存在的持久脚本：使用 `ue_run_python.script_path`，由 harness 读取原文件并在 run 目录中保存快照。
- 不要把 `runpy.run_path(...)` loader 当作 inline `script` 传入；这会让 run 目录只保存跳转路径，无法追溯真实脚本内容。

## 风险提示

- UE Python API 会随版本变化，失败时先保留脚本与命令，便于用户在 UE 输出日志中定位。
- 批量重命名、移动、保存资产会影响 `.uasset`，必须避免无审批执行。
- commandlet 模式并不支持所有 Editor API，必要时切换 full editor。
