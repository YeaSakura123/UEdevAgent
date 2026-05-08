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

- Confirm the coding-agent terminal UI opens with structured user, system, tool,
  approval, and assistant blocks.
- Type `test`; it should answer directly without tool activity.
- Type `/`; slash commands should complete with descriptions.
- Run `/help`, `/model`, `/plan`, `/plan off`, `/permissions`, `/clear`, and `/ue doctor`.
- Confirm the TUI status bar below the input line shows the active model and
  directory.
- After `/plan`, confirm the same status bar shows `Plan mode (Shift+Tab to exit)`
  on the right, and that Shift+Tab exits Plan Mode.
- Run `/permissions`; the permission selector should open and support up/down
  selection plus Enter. Also verify `/permissions read-only`, `/permissions default`,
  `/permissions auto-review`, and `/permissions full-access`; the selected value
  should affect only the current chat session.
- Ask for a response with Markdown headings, a list, a table, and a code block;
  the final answer should render as rich terminal Markdown.
- Ask the agent to read a small file; the turn should show tool activity while
  running and collapse to a summary after the final answer.
- Ask the agent to edit `test/main.py`; the turn should show edit activity and
  collapse after the final answer.
- Run `python -m uedev chat --plain`; it should keep the script-friendly plain
  transcript renderer.

## UE dry-run checks

Configure UE engines in `~/.uedev/config.json`. The project's `.uproject`
`EngineAssociation` must match one engine key or alias:

```json
{
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

Agent-driven UE execution must follow the active `/permissions` mode before
launching Unreal Editor.
