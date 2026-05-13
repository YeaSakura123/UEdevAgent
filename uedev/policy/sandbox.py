from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SandboxPolicy:
    primary_root: Path
    extra_roots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.primary_root = self.primary_root.expanduser().resolve()
        self.extra_roots = [root.expanduser().resolve() for root in self.extra_roots]

    def allowed_roots(self) -> tuple[Path, ...]:
        return (self.primary_root, *self.extra_roots)

    def is_allowed(self, path: Path) -> bool:
        resolved = path.expanduser().resolve()
        return any(resolved == root or root in resolved.parents for root in self.allowed_roots())

    def require_path(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not self.is_allowed(resolved):
            raise ValueError(f"Path escapes sandbox: {path}")
        return resolved

    def add_dir(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise ValueError(f"Sandbox directory does not exist: {path}")
        if resolved != self.primary_root and resolved not in self.extra_roots:
            self.extra_roots.append(resolved)
        return resolved

    def render(self) -> str:
        lines = ["Sandbox roots:"]
        lines.extend(f"- {root}" for root in self.allowed_roots())
        return "\n".join(lines)


__all__ = ["SandboxPolicy"]
