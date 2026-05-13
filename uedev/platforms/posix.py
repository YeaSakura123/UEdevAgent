from __future__ import annotations

from .base import PlatformBackend


class PosixBackend(PlatformBackend):
    def __init__(self) -> None:
        super().__init__(name="posix", shell="bash")
