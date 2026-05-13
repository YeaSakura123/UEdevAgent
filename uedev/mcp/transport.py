from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .types import McpError, McpServerConfig


class McpStdioTransport:
    def __init__(self, config: McpServerConfig):
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self._next_id = 1
        self._stdout: queue.Queue[str] = queue.Queue()
        self._stderr_lines: list[str] = []

    def start(self) -> None:
        if self.process is not None:
            return
        try:
            self.process = subprocess.Popen(
                [self.config.command, *self.config.args],
                cwd=str(self.config.cwd) if self.config.cwd else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError as error:
            raise McpError(f"cannot start MCP server {self.config.name}: {error}") from error
        if self.process.stdin is None or self.process.stdout is None:
            raise McpError(f"MCP server {self.config.name} did not expose stdio pipes")
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def request(self, method: str, params: dict[str, Any] | None = None, timeout_seconds: int | None = None) -> dict[str, Any]:
        self.start()
        request_id = self._next_id
        self._next_id += 1
        self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}})

        deadline = time.monotonic() + float(timeout_seconds or self.config.timeout_seconds)
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None and self._stdout.empty():
                raise McpError(f"MCP server {self.config.name} exited during {method}: {self.stderr_summary()}")
            try:
                line = self._stdout.get(timeout=max(0.01, min(0.1, deadline - time.monotonic())))
            except queue.Empty:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise McpError(f"MCP {method} failed: {json.dumps(message['error'], ensure_ascii=False)}")
            result = message.get("result")
            return result if isinstance(result, dict) else {}
        raise McpError(f"MCP {method} timed out after {timeout_seconds or self.config.timeout_seconds}s: {self.stderr_summary()}")

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self.start()
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def close(self) -> None:
        process = self.process
        self.process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()

    def stderr_summary(self, max_lines: int = 5) -> str:
        return "\n".join(self._stderr_lines[-max_lines:]).strip()

    def _write(self, payload: dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            raise McpError(f"MCP server {self.config.name} is not running")
        try:
            self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except OSError as error:
            raise McpError(f"cannot write to MCP server {self.config.name}: {error}") from error

    def _read_stdout(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            stripped = line.strip()
            if stripped:
                self._stdout.put(stripped)

    def _read_stderr(self) -> None:
        process = self.process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            stripped = line.strip()
            if stripped:
                self._stderr_lines.append(stripped)
                if len(self._stderr_lines) > 50:
                    self._stderr_lines = self._stderr_lines[-50:]
