from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


ApprovalProvider = Callable[[str, str], bool]


@dataclass(frozen=True)
class ShellResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str


def shell_name() -> str:
    return "PowerShell" if sys.platform == "win32" else "bash"


def confirm_command(command: str, reason: str) -> bool:
    answer = input(f"\nRun command?\nReason: {reason}\n> {command}\n[y/N] ")
    return answer.strip().lower() == "y"


def run_shell(command: str, cwd: Path, timeout_seconds: int) -> ShellResult:
    if shell_name() == "PowerShell":
        args = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command]
    else:
        args = ["bash", "-lc", command]

    process = subprocess.Popen(
        args,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        timeout_message = f"Error: command timed out after {timeout_seconds}s"
        stderr = f"{stderr.rstrip()}\n{timeout_message}\n" if stderr else f"{timeout_message}\n"
        return_code = 124
    else:
        return_code = process.returncode

    return ShellResult(
        command=command,
        exit_code=return_code,
        stdout=stdout,
        stderr=stderr,
    )
