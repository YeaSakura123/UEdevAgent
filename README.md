# uedev-cli

`uedev` is a Python CLI agent harness for coding, local automation, and
Unreal Engine editor workflows. It keeps a conversation, lets the model request
tools, asks before risky execution by default, and feeds observations back into
the model until the task is complete.

Tool use now goes through OpenAI-compatible native tool/function calling. Tool
schemas live in `uedev.tool_specs`, while implementations live in
`AgentRuntime._build_tool_handlers`. The loop executes returned `tool_calls`,
adds `tool` result messages, and asks the model to continue until it gives a
normal final answer.

The older hand-written JSON action protocol is still accepted as a fallback for
older models and transcripts:

```json
{"type":"tool","name":"shell","input":{"command":"...","reason":"..."}}
{"type":"shell","command":"...","reason":"..."}
{"type":"final","answer":"..."}
```

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

Edit `.env`:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=your_model
UE_PROJECT_PATH=D:\Path\To\Game.uproject
UE_ENGINE_ROOT=D:\Program Files\Epic Games\UE_5.4
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
uedev run "inspect this project" --yes
uedev run "debug this folder" --verbose
uedev chat --allow-ue-execute
```

By default, shell commands require confirmation before execution. UE execution is
even stricter: UE Python tools dry-run unless the command line explicitly passes
`--execute` or the agent session passes `--allow-ue-execute`.

## Behavior

- `run` handles one task and exits.
- `chat` keeps the same message history across turns.
- `chat` shows the current version, model, and working directory when it starts.
- `chat` supports slash commands such as `/help`, `/todos`, and `/ue doctor`;
  type `/` to autocomplete commands with descriptions.
- `chat` keeps an in-session input history that can be browsed with the arrow keys.
- The harness implements the staged mechanisms from `learn-claude-code-main`:
  loop, tool dispatch, TodoWrite, subagent, skill loading, context compact,
  persistent task graph, background tasks, team inbox/protocols, autonomous task
  claiming, and task-aware git worktrees.
- Commands run in the selected working directory.
- Shell commands time out after 120 seconds by default.
- Internal protocol diagnostics are hidden unless `--verbose` is enabled.
- If a filesystem task is answered without using a shell action, the CLI asks
  the model to choose an executable action instead.
- Todos are stored in `.agent/todos.json`.
- UE-generated scripts are stored in `.agent/ue_scripts/`.

## Unreal Engine Direction

The project now includes first-class UE helper tools instead of relying on ad
hoc shell commands:

- `ue.doctor`: find Unreal Engine and `.uproject`
- `ue.run_python_commandlet`: call `UnrealEditor-Cmd.exe -run=pythonscript`
- `ue.run_python_full_editor`: call `UnrealEditor-Cmd.exe -ExecutePythonScript`
- `ue.list_assets`: wrap a UE Python script and return JSON
- `ue.validate_assets`: run UE Data Validation and summarize results

See:

- `docs/ARCHITECTURE.md`
- `docs/STAGED_HARNESS_IMPLEMENTATION.md`
- `docs/UE_AGENT_DESIGN.md`
- `docs/TASKS.md`
