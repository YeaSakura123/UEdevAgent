from __future__ import annotations

from .base import PlatformBackend


class WindowsBackend(PlatformBackend):
    def __init__(self) -> None:
        super().__init__(name="windows", shell="PowerShell")
