from __future__ import annotations

import sys

from .base import PlatformBackend
from .posix import PosixBackend
from .windows import WindowsBackend


def current_platform_backend() -> PlatformBackend:
    if sys.platform == "win32":
        return WindowsBackend()
    return PosixBackend()


__all__ = ["PlatformBackend", "WindowsBackend", "PosixBackend", "current_platform_backend"]
