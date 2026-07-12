# uedev-cli

`uedev` is a Python CLI agent harness for coding, local automation, and
Unreal Engine editor workflows. It keeps a conversation, lets the model request
tools, enforces configurable permission modes, and feeds observations back into
the model until the task is complete.

Tool use goes through native tool/function calling. GPT model profiles can use
the OpenAI Responses API, while OpenAI-compatible model profiles continue to use
Chat Completions. Tool schemas live in `uedev.tools.specs`, while implementations
live in `AgentRuntime._build_tool_handlers()`. The loop executes returned tool
calls, adds tool results, and asks the model to continue until it gives a final
answer or reaches a configured runtime budget.

## Install

Python 3.11 or newer is required.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Create local configuration:

```bash
uedev init
```

Edit the system JSON config at `~/.uedev/config.json`:

```json
{
  "version": 2,
  "default_model": "openai-gpt",
  "models": {
    "openai-gpt": {
      "response": true,
      "effort": "medium",
      "model": "gpt-5",
      "base_url": "https://api.openai.com/v1",
      "api_key": "your_api_key",
      "context_window": 262144,
      "timeout_seconds": 120,
      "responses": {
        "store": false,
        "reasoning": {
          "summary": null
        },
        "text": {
          "format": {
            "type": "text"
          }
        },
        "tool_choice": "auto",
        "parallel_tool_calls": true,
        "strict_function_tools": false,
        "max_output_tokens": null,
        "truncation": "disabled",
        "include": [],
        "built_in_tools": {
          "web_search": {
            "enabled": false
          },
          "file_search": {
            "enabled": false,
            "vector_store_ids": []
          },
          "remote_mcp": []
        }
      }
    },
    "deepseek": {
      "response": false,
      "effort": "high",
      "model": "deepseek-v4-pro",
      "base_url": "https://your.api.com/v1",
      "api_key": "your_api_key",
      "context_window": 262144,
      "timeout_seconds": 120,
      "requires_reasoning_content": false
    }
  },
  "display": {
    "diff_output_max_chars": 20000
  },
  "runtime": {
    "default_max_steps": 8,
    "budgets": {
      "model_request_hard_limit": 8,
      "tool_call_soft_limit": 24,
      "tool_call_limits": {
        "compact": 2,
        "subagent": 4,
        "background_run": 4,
        "write_file": 16,
        "edit_file": 16,
        "shell": 12,
        "worktree_run": 12,
        "ue_run_python": 12,
        "ue_build": 12
      },
      "wall_clock_seconds": 900,
      "consecutive_tool_failures": 3,
      "permission_denials": 2,
      "no_progress_rounds": 3,
      "output_token_soft_ratio": 0.8,
      "context_compact_ratio": 0.9
    }
  },
  "workspace": {
    "excluded_dirs": [".agent", ".git", ".vs", "Binaries", "Intermediate", "Saved", "DerivedDataCache"]
  },
  "worktrees": {
    "default_root": ""
  },
  "subagents": {
    "model_profile": null
  },
  "mcp": {
    "servers": {}
  },
  "ue": {
    "engines": {
      "5.4": {
        "root": "D:/Program Files/Epic Games/UE_5.4"
      },
      "5.5-source": {
        "root": "D:/UE/UE_5.5_Source",
        "aliases": ["5.5"]
      }
    }
  }
}
```

The CLI uses the official `openai` Python package. Set `response` to `true` for
OpenAI GPT profiles that should use the Responses API; leave it unset or `false`
for OpenAI-compatible Chat Completions endpoints such as DeepSeek-compatible
profiles.
`effort` is the profile's default reasoning effort. Supported values are
`low`, `medium`, `high`, and `xhigh`. Responses profiles send it as
`reasoning.effort`; DeepSeek-compatible Chat Completions profiles send it as
the top-level `reasoning_effort` parameter. Interactive chat can override the
profile default for the current session with `/effort`.
Every successful model request records token usage. Responses profiles read
`usage` from the completed response. All Chat Completions profiles, regardless
of model name or provider URL, request the DeepSeek/OpenAI-compatible final
streaming usage chunk with `stream_options.include_usage`. If an API does not
return usage, the CLI records a clearly labelled local estimate.
`context_window` is optional for each model profile and defaults to 262144
estimated tokens. Auto compaction defaults to 90% of that window unless
`--context-threshold` is supplied.
`timeout_seconds` is optional for each model profile and defaults to 120.
`requires_reasoning_content` is optional and defaults to `false`. Enable it for
models such as DeepSeek thinking mode that require assistant
`reasoning_content` to be replayed with later requests. It only applies when
`response` is `false`. Responses profiles send the configured Responses API
options, including reasoning, tool choice, parallel tool calls, max output
tokens, truncation, include, text formatting, storage, built-in tools, and
function tool schemas adapted from the same canonical project tool definitions
used by Chat Completions profiles.
`display.diff_output_max_chars` controls per-section `/diff` output truncation
and defaults to 20000 characters.
`runtime.default_max_steps` is retained as the default model-request limit.
`runtime.budgets` configures model requests, tool-call soft and per-tool limits,
wall-clock time, repeated failures or denials, output size, and the automatic
compaction ratio. `--max-steps` overrides `model_request_hard_limit` for one run.
`workspace.excluded_dirs` controls directory names skipped by workspace file
tools and defaults to the listed agent, VCS, IDE, and UE generated folders.
`worktrees.default_root` overrides the managed worktree root when non-empty.
`subagents.model_profile` optionally selects a configured model for child agents.
`mcp.servers` configures optional stdio MCP servers; `/mcp` reports their status
and exposed tools.

## Usage

Run a single task:

```bash
uedev run "create a code folder and add helloworld.py"
```

Start an interactive session:

```bash
uedev chat
```

During development, the module entry point works too:

```bash
python -m uedev chat
```

Check configuration:

```bash
uedev doctor
```

Show the persisted agent todo board:

```bash
uedev tasks
uedev tasks --graph
uedev worktrees
```

Inspect UE configuration without launching Unreal Engine:

```bash
uedev ue doctor --cwd D:\Path\To\GameProject
```

Compile the UE Editor target through the configured engine Build.bat:

```bash
uedev ue build --cwd D:\Path\To\GameProject
```

Prepare a UE Python command without launching UE:

```bash
uedev ue run-python scripts\list_assets.py --cwd D:\Path\To\GameProject
uedev ue list-assets --cwd D:\Path\To\GameProject
uedev ue validate-assets --cwd D:\Path\To\GameProject
```

Useful options:

```bash
uedev chat --timeout 180
uedev chat --max-steps 16
uedev chat --plain
uedev run "inspect this project" --yes
uedev run "debug this folder" --verbose
```

Permission mode is controlled in chat with `/permissions`. The available modes
are `read-only`, `default`, `auto-review`, and `full-access`; the selected mode
applies to the current chat session only. `--yes` starts that run or chat session
in `full-access`. Standalone `ue run-python`, `ue list-assets`, and
`ue validate-assets` require `--execute`; omitting it produces a dry-run command
preview. `ue doctor` is read-only, while `ue build` runs the build immediately.

## Behavior

- `run` handles one task and exits.
- `chat` keeps the same message history across turns.
- `chat` shows the current version, model, and working directory when it starts.
- `chat` uses a full-screen Rich + Prompt Toolkit terminal UI when a terminal is
  available: Prompt Toolkit owns the persistent transcript, fixed input, status,
  selector, and approval surfaces, while Rich formats structured user, system,
  tool, approval, and assistant blocks.
- Full-screen chat keeps terminal-native copy behavior for visible text. It does
  not implement a custom mouse selection or clipboard layer; use `chat --plain`
  when a scrollback-first transcript is required.
- `chat` renders final model answers as Markdown, including common headings,
  lists, links, code blocks, and tables.
- `chat` shows loading, assistant streaming text, and tool events while running,
  then renders a collapsed process summary before the final answer.
- `chat` supports slash commands such as `/help`, `/context`, `/diff`, `/todos`,
  `/tasks`, `/history`, `/subagents`, `/worktree`, `/model`, `/effort`, `/usage`, `/mcp`,
  `/plan`, `/permissions`, `/compact`, `/clear`, `/exit`, and `/ue doctor`; type `/` to
  autocomplete commands with descriptions.
- `/context` shows the current estimated model-context usage, configured context
  window, auto compact threshold, and remaining capacity.
- `/usage` shows the current session total, the latest user-turn total, and the
  individual model requests in that turn. Turn summaries also show input,
  output, and total token counts.
- `/diff` shows a human-readable Git status/diff summary and Perforce workspace
  status plus opened files for the current workspace.
- `/model` opens an interactive profile selector in TUI chat. In plain mode,
  `/model <profile>` stores the active model for the current project in
  `.agent/config.json`; `/model reset` returns to the first configured model,
  unless `default_model` is explicitly set.
- `/plan` enters Plan Mode; `/plan off` exits it. In the TUI, Plan Mode shows a
  right-aligned `Plan mode (Shift+Tab to exit)` hint on the status line directly
  under the current input while the same line shows the active model and directory.
- `/permissions` opens an interactive permission-mode selector in the TUI;
  `/permissions <mode>` switches the current chat session between `read-only`,
  `default`, `auto-review`, and `full-access`.
- `/compact` summarizes and rewrites the model context while keeping the visible
  chat transcript intact. The full pre-compact transcript is written to the
  current session's `transcript.jsonl`; later compactions in the same session
  overwrite that file.
- `/history` lists previous conversations for the current project and lets an
  interactive TUI user choose one with the arrow keys; plain chat falls back to
  a numbered list. History is loaded from `.agent/sessions/YYYY/MM/DD/<session>/`.
  Loading a session also restores the last API model and reasoning effort used
  by that conversation. The API model identifier is used for matching, so this
  still works after the outer CLI display name is changed.
- `/subagents` lists child-agent conversations created by the current main
  session. Subagents are scoped to that session and are not shared across other
  restored conversations.
- `/worktree` prompts for a name, creates a Git branch with the same name using
  `git worktree add`, links `Content/` back to the source UE project, and copies
  only the current `.agent` chat session plus project config into the new project.
  By default it writes to
  `<project-parent>/.uedev-worktrees/<project-name>/<worktree-name>`; set
  `worktrees.default_root` in `~/.uedev/config.json` to override the root. The
  linked `Content/` is shared, so asset edits in the worktree modify the original
  project assets.
- `chat` keeps an in-session input history that can be browsed with the arrow keys.
- `chat --plain` keeps the script-friendly text renderer for pipes, non-TTY
  sessions, and automation.
- The harness implements the staged mechanisms from `learn-claude-code-main`:
  loop, tool dispatch, TodoWrite, subagent, skill loading, context compact,
  persistent task graph, background tasks, and task-aware git worktrees.
- Commands run in the selected working directory.
- Shell commands time out after 120 seconds by default.
- Internal iteration and tool diagnostics are hidden unless `--verbose` is enabled.
- If a filesystem task is answered without using a tool call, the CLI asks
  the model to call the appropriate tool instead.
- Conversations are stored in `.agent/sessions/YYYY/MM/DD/<session_id>/`.
  `messages.jsonl` is the model context, `display.jsonl` is the full replayable
  UI transcript, `metadata.json` stores session metadata, and `transcript.jsonl`
  stores the latest compact source transcript. The session's last model display
  name, API model identifier, and reasoning effort are stored in
  `metadata.json` and as a `session_state` record in `transcript.jsonl`. Each
  model request is stored as a `token_usage` record; session totals are mirrored
  in `metadata.json`, and compaction preserves these records.
- Session subagents are stored below
  `.agent/sessions/YYYY/MM/DD/<session_id>/subagents/`, with one `index.jsonl`
  plus one directory per subagent containing `messages.jsonl`, `display.jsonl`,
  and `metadata.json`.
- Todos are stored in `.agent/todos.json`.
- Persistent tasks are stored in `.agent/tasks/`.
- Managed worktree indexes and directories are stored in `.agent/worktrees/`.
- UE run artifacts are stored in `.agent/ue_runs/<run_id>/`; UE build artifacts are stored in `.agent/ue_builds/<run_id>/`. The authoritative executed script snapshot is `user_script.py` in the UE run directory. `.agent/ue_scripts/` is a legacy location kept only for old workspaces and is not read as the current execution entrypoint.
- Manual validation steps are documented in `docs/VALIDATION.md`.

## Unreal Engine Direction

The agent now uses first-class UE tools instead of ad hoc shell commands:

- `ue_doctor`: find `.uproject`, resolve `EngineAssociation`, configured editor paths, and Perforce status
- `ue_build`: compile `<ProjectName>Editor Win64 Development`, capture UBT/UHT/MSVC diagnostics, and store logs under `.agent/ue_builds/`
- `ue_run_python`: run inline code or an existing script in `commandlet` or `full_editor` mode and return structured results and logs
- `ue_stop_executor`: stop the persistent full-editor executor polling loop
- `p4_status`, `p4_file_state`, `p4_opened`, `p4_checkout`, `p4_add`, `p4_delete`, `p4_reconcile`, and `p4_diff`: inspect and modify Perforce state through permission-aware interfaces

The standalone CLI additionally provides `ue run-python`, `ue list-assets`, and
`ue validate-assets` commands, with dry-run behavior unless `--execute` is set.

UE engine selection is driven by the project's `.uproject` `EngineAssociation`.
The system config can store multiple UE versions under `ue.engines`; the doctor
matches the project version to an engine key or alias and does not fall back to a
different engine when the declared version is not configured.

See:

- `docs/ARCHITECTURE.md`
- `docs/STAGED_HARNESS_IMPLEMENTATION.md`
- `docs/UE_AGENT_DESIGN.md`
- `docs/TASKS.md`
