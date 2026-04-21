# myAgentCli

`myagent` is a Python CLI agent for coding and local automation workflows. It
keeps a conversation, requests shell actions when work needs to happen on disk,
asks before executing commands by default, and feeds command results back into
the model until the task is complete.

The current action protocol is intentionally simple and stable:

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

Useful options:

```bash
python -m myagent chat --timeout 180
python -m myagent run "inspect this project" --yes
python -m myagent run "debug this folder" --verbose
```

By default, shell commands require confirmation before execution.

## Behavior

- `run` handles one task and exits.
- `chat` keeps the same message history across turns.
- Commands run in the selected working directory.
- Shell commands time out after 120 seconds by default.
- Internal protocol diagnostics are hidden unless `--verbose` is enabled.
- If a filesystem task is answered without using a shell action, the CLI asks
  the model to choose an executable action instead.

## Unreal Engine Direction

The next production step is to add first-class UE tools instead of relying on
ad hoc shell commands:

- `ue.doctor`: find Unreal Engine and `.uproject`
- `ue.run_python_commandlet`: call `UnrealEditor-Cmd.exe -run=pythonscript`
- `ue.run_python_full_editor`: call `UnrealEditor-Cmd.exe -ExecutePythonScript`
- `ue.list_assets`: wrap a UE Python script and return JSON
- `ue.validate_assets`: run UE Data Validation and summarize results
