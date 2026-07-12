# Repository Guidelines

## Project Structure & Module Organization

`uedev/` contains the installable package. Keep CLI parsing in `uedev/cli.py`, orchestration in `uedev/runtime/`, tools in `uedev/tools/`, persisted state in `uedev/state/`, rendering in `uedev/ui/`, policy checks in `uedev/policy/`, and Unreal integration in `uedev/ue/`. Tests live in `test/` and mirror package concerns (for example, `test/test_permissions.py`). Samples are under `examples/`, design notes under `docs/`, and the bundled UE skill under `skills/ue-editor/`.

## Build, Test, and Development Commands

- `python -m venv .venv` creates a local virtual environment.
- `python -m pip install -e .` installs the CLI in editable mode.
- `python -m unittest discover -s test` runs the complete test suite.
- `python -m compileall -q uedev test` catches syntax and import-time compilation errors quickly.
- `python -m uedev doctor` validates local configuration.
- `python -m uedev chat --plain` starts a script-friendly local development session.

Run commands from the repository root. Unreal build and execution commands require a project and configured engine; `ue doctor` can diagnose missing setup.

## Coding Style & Naming Conventions

Use Python 3.11+ features, four-space indentation, type hints, and `from __future__ import annotations` where modules use modern annotations. Follow PEP 8: `snake_case` for functions and modules, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep platform-specific behavior in `uedev/platforms/` and avoid mixing UI rendering with runtime logic. No formatter or linter is currently configured, so keep imports grouped, lines readable, and changes consistent with nearby code.

## Testing Guidelines

Tests use the standard-library `unittest` framework and `unittest.mock`. Name files `test_<area>.py` and methods `test_<behavior>`. Use temporary directories below `.tmp/tests/` or `tempfile`; do not rely on a developer's global configuration, network access, or installed Unreal Engine. Add regression tests for bug fixes and cover both success and failure paths for tools and permission checks.

## Commit & Pull Request Guidelines

Recent history uses short, verb-led summaries in English or Chinese, without Conventional Commit prefixes. Keep each commit focused and describe the observable change (for example, `Fix history replay rendering`). Pull requests should summarize behavior changes, list verification commands, link relevant issues, and call out configuration or compatibility impacts. Include terminal output or screenshots when TUI behavior changes.

## Security & Configuration

Never commit API keys or user configuration from `~/.uedev/config.json` or project state from `.agent/`. Preserve dry-run and permission checks around shell, filesystem, and Unreal execution paths; tests should use placeholder credentials only.
