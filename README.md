# myAgentCli

`myagent` is a Python CLI agent harness for coding, local automation, and
Unreal Engine editor workflows. It keeps a conversation, lets the model request
tools, asks before risky execution by default, and feeds observations back into
the model until the task is complete.

The current action protocol is intentionally simple and stable. New code should
prefer the generic tool form:

```json
{"type":"tool","name":"shell","input":{"command":"...","reason":"..."}}
```

Other built-in tools:

```json
{"type":"tool","name":"todo_update","input":{"items":[{"id":"1","text":"...","status":"pending"}]}}
{"type":"tool","name":"subagent","input":{"prompt":"inspect tests and summarize"}}
{"type":"tool","name":"task_create","input":{"subject":"Validate UE assets","description":"Run dry-run first"}}
{"type":"tool","name":"background_run","input":{"command":"python -m unittest discover -s test","reason":"run tests"}}
{"type":"tool","name":"spawn_teammate","input":{"name":"alice","role":"tester","prompt":"claim ready tasks"}}
{"type":"tool","name":"worktree_create","input":{"name":"asset-validation","task_id":1}}
{"type":"tool","name":"ue_doctor","input":{}}
{"type":"tool","name":"ue_run_python","input":{"kind":"list_assets","mode":"commandlet","script":"","execute":false}}
```

The legacy shell form is still accepted:

```json
{"type":"shell","command":"...","reason":"..."}
```

or:

```json
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
python -m myagent init
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
python -m myagent run "create a code folder and add helloworld.py"
```

Start an interactive session:

```bash
python -m myagent chat
```

Check configuration:

```bash
python -m myagent doctor
```

Show the persisted agent todo board:

```bash
python -m myagent tasks
python -m myagent tasks --graph
python -m myagent team
python -m myagent worktrees
```

Inspect UE configuration without launching Unreal Engine:

```bash
python -m myagent ue doctor --cwd D:\Path\To\GameProject
```

Prepare a UE Python command without launching UE:

```bash
python -m myagent ue run-python scripts\list_assets.py --cwd D:\Path\To\GameProject
python -m myagent ue list-assets --cwd D:\Path\To\GameProject
python -m myagent ue validate-assets --cwd D:\Path\To\GameProject
```

Useful options:

```bash
python -m myagent chat --timeout 180
python -m myagent run "inspect this project" --yes
python -m myagent run "debug this folder" --verbose
python -m myagent chat --allow-ue-execute
```

By default, shell commands require confirmation before execution. UE execution is
even stricter: UE Python tools dry-run unless the command line explicitly passes
`--execute` or the agent session passes `--allow-ue-execute`.

## Behavior

- `run` handles one task and exits.
- `chat` keeps the same message history across turns.
- `chat` supports slash commands: `/help`, `/todos`, `/ue doctor`.
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
