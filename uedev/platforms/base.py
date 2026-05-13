from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformBackend:
    name: str
    shell: str

    @property
    def supports_process_sandbox(self) -> bool:
        return False
