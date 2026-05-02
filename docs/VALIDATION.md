# Validation Checklist

Use this checklist after changes to the agent loop, TUI, tool calling, or UE
integration.

## Automated checks

```powershell
python -m compileall -q uedev test
python -m unittest discover -s test
```

## Chat and TUI checks

```powershell
python -m uedev chat
```

- Confirm the full-screen TUI opens with status, transcript, and input areas.
- Type `test`; it should answer directly without tool activity.
- Type `/`; slash commands should complete with descriptions.
- Run `/help`, `/clear`, and `/ue doctor`.
- Ask the agent to read a small file; the turn should show tool activity while
  running and collapse to a summary after the final answer.
- Ask the agent to edit `test/main.py`; the turn should show edit activity and
  collapse after the final answer.

## UE dry-run checks

Set these variables when needed:

```powershell
$env:UE_PROJECT_PATH="D:\Path\Project.uproject"
$env:UE_EDITOR_CMD_PATH="D:\Path\UnrealEditor-Cmd.exe"
$env:UE_EDITOR_PATH="D:\Path\UnrealEditor.exe"
```

Then run:

```powershell
python -m uedev ue doctor --cwd .
python -m uedev ue list-assets --cwd .
```

The default UE run must be dry-run only. It should report the generated script,
the command line, and `executed: False`.

## UE execution check

Only run this when the local UE project and editor paths are valid:

```powershell
python -m uedev ue list-assets --cwd . --execute
```

Agent-driven UE execution must prompt for y/N before launching Unreal Editor.
