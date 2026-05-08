# uedev-cli

`uedev` is a Python CLI agent harness for coding, local automation, and
Unreal Engine editor workflows. It keeps a conversation, lets the model request
tools, enforces configurable permission modes, and feeds observations back into
the model until the task is complete.

Tool use goes through OpenAI-compatible native tool/function calling. Tool
schemas live in `uedev.tool_specs`, while implementations live in
`AgentRuntime._build_tool_handlers`. The loop executes returned `tool_calls`,
adds `tool` result messages, and asks the model to continue until it gives a
normal final answer.

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
  "version": 1,
  "models": {
    "work": {
      "model": "your_model",
      "base_url": "https://api.openai.com/v1",
      "api_key": "your_api_key"
    }
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

The CLI uses the official `openai` Python package and supports
OpenAI-compatible Chat Completions endpoints.

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
uedev team
uedev worktrees
```

Inspect UE configuration without launching Unreal Engine:

```bash
uedev ue doctor --cwd D:\Path\To\GameProject
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
uedev chat --plain
uedev run "inspect this project" --yes
uedev run "debug this folder" --verbose
```

Permission mode is controlled in chat with `/permissions`. The available modes
are `read-only`, `default`, `auto-review`, and `full-access`; the selected mode
applies to the current chat session only. `--yes` starts that run or chat session
in `full-access`. Standalone `uedev ue ...` commands still require `--execute`;
omitting it produces a dry-run command preview.

## Behavior

- `run` handles one task and exits.
- `chat` keeps the same message history across turns.
- `chat` shows the current version, model, and working directory when it starts.
- `chat` uses a Rich + Prompt Toolkit terminal UI when a terminal is available:
  Prompt Toolkit handles input/history/completion, while Rich renders structured
  user, system, tool, approval, and assistant blocks.
- `chat` renders final model answers as Markdown, including common headings,
  lists, links, code blocks, and tables.
- `chat` shows per-turn thinking/tool events while running, then renders a
  collapsed process summary before the final answer.
- `chat` supports slash commands such as `/help`, `/todos`, `/model`, `/plan`,
  `/permissions`, and `/ue doctor`;
  type `/` to autocomplete commands with descriptions.
- `/model` lists configured model profiles; `/model <profile>` stores the active
  model for the current project in `.agent/config.json`; `/model reset` returns
  to the first configured model, unless `default_model` is explicitly set.
- `/plan` enters Plan Mode; `/plan off` exits it. In the TUI, Plan Mode shows a
  right-aligned `Plan mode (Shift+Tab to exit)` hint on the status line directly
  under the current input while the same line shows the active model and directory.
- `/permissions` opens an interactive permission-mode selector in the TUI;
  `/permissions <mode>` switches the current chat session between `read-only`,
  `default`, `auto-review`, and `full-access`.
- `chat` keeps an in-session input history that can be browsed with the arrow keys.
- `chat --plain` keeps the script-friendly text renderer for pipes, non-TTY
  sessions, and automation.
- The harness implements the staged mechanisms from `learn-claude-code-main`:
  loop, tool dispatch, TodoWrite, subagent, skill loading, context compact,
  persistent task graph, background tasks, team inbox/protocols, autonomous task
  claiming, and task-aware git worktrees.
- Commands run in the selected working directory.
- Shell commands time out after 120 seconds by default.
- Internal iteration and tool diagnostics are hidden unless `--verbose` is enabled.
- If a filesystem task is answered without using a tool call, the CLI asks
  the model to call the appropriate tool instead.
- Todos are stored in `.agent/todos.json`.
- Persistent tasks are stored in `.agent/tasks/`.
- Team state is stored in `.agent/team/`.
- Managed worktree indexes and directories are stored in `.agent/worktrees/`.
- UE run artifacts are stored in `.agent/ue_runs/<run_id>/`; the authoritative executed script snapshot is `user_script.py` in that run directory. `.agent/ue_scripts/` is a legacy location and is not the current execution entrypoint.
- Manual validation steps are documented in `docs/VALIDATION.md`.

## Unreal Engine Direction

The project now includes first-class UE helper tools instead of relying on ad
hoc shell commands:

- `ue.doctor`: find `.uproject`, read its `EngineAssociation`, and match a configured UE engine
- `ue.run_python_commandlet`: call `UnrealEditor-Cmd.exe -run=pythonscript`
- `ue.run_python_full_editor`: call `UnrealEditor-Cmd.exe -ExecutePythonScript`
- `ue.list_assets`: wrap a UE Python script and return JSON
- `ue.validate_assets`: run UE Data Validation and summarize results

UE engine selection is driven by the project's `.uproject` `EngineAssociation`.
The system config can store multiple UE versions under `ue.engines`; the doctor
matches the project version to an engine key or alias and does not fall back to a
different engine when the declared version is not configured.

See:

- `docs/ARCHITECTURE.md`
- `docs/STAGED_HARNESS_IMPLEMENTATION.md`
- `docs/UE_AGENT_DESIGN.md`
- `docs/TASKS.md`
