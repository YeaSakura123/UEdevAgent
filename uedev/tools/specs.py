from __future__ import annotations

from typing import Any


ToolSpec = dict[str, Any]


# 内部函数：生成一个 OpenAI function tool schema，统一工具声明的外部格式。
def function_tool(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> ToolSpec:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


# 外部函数：返回模型侧原生 tool/function calling 使用的统一工具 schema 列表。
def get_tool_specs(extra_tools: list[ToolSpec] | None = None) -> list[ToolSpec]:
    integer = {"type": "integer"}
    string = {"type": "string"}
    boolean = {"type": "boolean"}
    string_list = {"type": "array", "items": string}

    tools = [
        function_tool(
            "shell",
            "Run a shell command in the current workspace after harness permission checks.",
            {"command": string, "reason": string},
            ["command", "reason"],
        ),
        function_tool("read_file", "Read a UTF-8 text file from the workspace.", {"path": string, "limit": integer}, ["path"]),
        function_tool("write_file", "Write UTF-8 text content to a workspace file.", {"path": string, "content": string}, ["path", "content"]),
        function_tool(
            "edit_file",
            "Replace the first occurrence of old_text with new_text in a workspace file.",
            {"path": string, "old_text": string, "new_text": string},
            ["path", "old_text", "new_text"],
        ),
        function_tool(
            "list_files",
            "List files under a workspace directory.",
            {"path": string, "pattern": string, "limit": integer},
        ),
        function_tool(
            "todo_update",
            "Replace the lightweight todo list for meaningful multi-step progress in the current agent turn. Do not use this tool to acknowledge instructions, confirm future behavior, or mark a single status check as completed.",
            {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"id": string, "text": string, "status": string},
                    },
                }
            },
            ["items"],
        ),
        function_tool("todo_list", "Show the current lightweight todo list.", {}),
        function_tool("subagent", "Run a bounded child agent with fresh context.", {"prompt": string, "agent_type": string}, ["prompt"]),
        function_tool("load_skill", "Load an on-demand local skill by name.", {"name": string}, ["name"]),
        function_tool("compact", "Compact the current conversation context.", {}),
        function_tool(
            "task_create",
            "Create a persistent task graph item.",
            {"subject": string, "description": string, "blocked_by": {"type": "array", "items": integer}, "owner": string},
            ["subject"],
        ),
        function_tool("task_get", "Get one persistent task by id.", {"task_id": integer}, ["task_id"]),
        function_tool(
            "task_update",
            "Update persistent task status, ownership, dependencies, or worktree.",
            {
                "task_id": integer,
                "status": string,
                "owner": string,
                "add_blocked_by": {"type": "array", "items": integer},
                "remove_blocked_by": {"type": "array", "items": integer},
                "worktree": string,
            },
            ["task_id"],
        ),
        function_tool("task_list", "List persistent tasks.", {}),
        function_tool("claim_task", "Claim a persistent task for an agent.", {"task_id": integer, "owner": string}, ["task_id"]),
        function_tool(
            "background_run",
            "Start a shell command in the background after permission checks.",
            {"command": string, "reason": string, "timeout_seconds": integer},
            ["command"],
        ),
        function_tool("background_check", "Check background command status.", {"task_id": string}),
        function_tool("spawn_teammate", "Create a persistent teammate record.", {"name": string, "role": string, "prompt": string}, ["name"]),
        function_tool("list_teammates", "List persistent teammate records.", {}),
        function_tool("send_message", "Send a mailbox message to a teammate.", {"to": string, "content": string, "msg_type": string}, ["to", "content"]),
        function_tool("read_inbox", "Read pending lead-agent mailbox messages.", {}),
        function_tool("broadcast", "Broadcast a mailbox message to all teammates.", {"content": string}, ["content"]),
        function_tool("shutdown_request", "Request shutdown approval for a teammate.", {"teammate": string}, ["teammate"]),
        function_tool("shutdown_response", "Respond to a teammate shutdown request.", {"request_id": string, "approve": boolean, "reason": string}, ["request_id", "approve"]),
        function_tool("plan_submit", "Submit a teammate plan for review.", {"teammate": string, "plan": string}, ["plan"]),
        function_tool("plan_review", "Approve or reject a submitted plan.", {"request_id": string, "approve": boolean, "feedback": string}, ["request_id", "approve"]),
        function_tool("idle", "Mark a teammate as idle.", {"teammate": string}),
        function_tool("worktree_create", "Create an isolated worktree.", {"name": string, "task_id": integer, "base_ref": string}, ["name"]),
        function_tool("worktree_list", "List managed worktrees.", {}),
        function_tool("worktree_run", "Run a command inside a managed worktree.", {"name": string, "command": string, "timeout_seconds": integer}, ["name", "command"]),
        function_tool("worktree_keep", "Keep a managed worktree and detach it from cleanup.", {"name": string}, ["name"]),
        function_tool("worktree_remove", "Remove a managed worktree.", {"name": string, "force": boolean, "complete_task": boolean}, ["name"]),
        function_tool(
            "ue_doctor",
            "Inspect Unreal Engine project state, including .uproject discovery, EngineAssociation, configured UE engine/editor paths, and Perforce read-only status. Use this as the sole default check for UE project/engine/Perforce questions; do not follow with shell p4 info unless the user explicitly requests raw p4 diagnostics or this tool reports unknown, timeout, or an error. Optional cwd points at a UE project or workspace directory.",
            {"cwd": string},
        ),
        function_tool(
            "ue_run_python",
            "Run Unreal Engine Python after harness permission checks. Use script for complete inline Python code or script_path for an existing .py file path; do not pass inline runpy.run_path loader scripts. Optional mode is commandlet or full_editor; optional cwd points at a UE project or workspace directory. To return data to the agent, set _uedev_result or call _uedev_emit(key, value); unreal.log output is captured as logs for diagnosis.",
            {"script": string, "script_path": string, "mode": string, "cwd": string},
        ),
        function_tool(
            "ue_stop_executor",
            "Queue a stop request for the full_editor UE executor in the given cwd. Use only when the user wants the editor-side executor to stop polling.",
            {"cwd": string},
        ),
        function_tool(
            "p4_status",
            "Inspect Perforce workspace status for the current UE workspace, including client, user, server, root, project tracking, and opened file summary. This is read-only.",
            {"cwd": string},
        ),
        function_tool(
            "p4_file_state",
            "Inspect Perforce state for workspace files before editing. Returns depot path, tracked/opened state, action, type, revision, and other-user lock/open conflict hints. This is read-only.",
            {"paths": string_list, "cwd": string},
            ["paths"],
        ),
        function_tool(
            "p4_opened",
            "List files currently opened in the Perforce workspace or a specific changelist. This is read-only.",
            {"cwd": string, "changelist": string},
        ),
        function_tool(
            "p4_checkout",
            "Run p4 edit for existing Perforce-controlled files before modifying them. Stops on UE binary asset lock conflicts instead of bypassing them.",
            {"paths": string_list, "cwd": string, "changelist": string},
            ["paths"],
        ),
        function_tool(
            "p4_add",
            "Run p4 add for new workspace files after creating them.",
            {"paths": string_list, "cwd": string, "changelist": string},
            ["paths"],
        ),
        function_tool(
            "p4_delete",
            "Run p4 delete for Perforce-controlled workspace files. This requires explicit approval because it schedules deletions.",
            {"paths": string_list, "cwd": string, "changelist": string},
            ["paths"],
        ),
        function_tool(
            "p4_reconcile",
            "Run p4 reconcile for the workspace or selected paths to detect missed adds, edits, and deletes after filesystem changes.",
            {"paths": string_list, "cwd": string, "changelist": string},
        ),
        function_tool(
            "p4_diff",
            "Show Perforce text diffs for opened files or selected paths. UE binary assets are reported as skipped instead of being read as text.",
            {"paths": string_list, "cwd": string},
        ),
    ]
    if extra_tools:
        tools.extend(extra_tools)
    return tools


# 外部函数：返回工具 schema 中声明的工具名称集合，用于测试和运行时校验。
def get_tool_names(extra_tools: list[ToolSpec] | None = None) -> set[str]:
    return {str(tool["function"]["name"]) for tool in get_tool_specs(extra_tools)}
