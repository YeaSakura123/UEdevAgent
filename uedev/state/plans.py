from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from . import config


PlanStatus = Literal["pending", "approved", "rejected", "missing"]


@dataclass(frozen=True)
class PlanRecord:
    id: str
    session_id: str
    turn_id: str
    title: str
    status: PlanStatus
    path: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlanManager:
    def __init__(self, plans_dir: Path | None = None):
        self._plans_dir = plans_dir.expanduser().resolve() if plans_dir is not None else None

    @property
    def plans_dir(self) -> Path:
        return self._plans_dir or default_plan_dir()

    def save_proposed_plan(self, session_id: str, turn_id: str, content: str) -> PlanRecord:
        now = _utc_now()
        plan_id = uuid.uuid4().hex[:8]
        title = extract_plan_title(content)
        safe_session = _safe_filename_part(session_id or "session")
        filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{safe_session}-{plan_id}.md"
        path = self.plans_dir / filename
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(content.strip() + "\n", encoding="utf-8")
        return PlanRecord(
            id=plan_id,
            session_id=session_id,
            turn_id=turn_id,
            title=title,
            status="pending",
            path=str(path),
            created_at=now,
            updated_at=now,
        )

    def with_status(self, record: PlanRecord, status: PlanStatus) -> PlanRecord:
        return replace(record, status=status, updated_at=_utc_now())

    def read_content(self, record: PlanRecord) -> tuple[str, PlanRecord]:
        path = Path(record.path).expanduser()
        if not path.exists():
            return "", self.with_status(record, "missing")
        return path.read_text(encoding="utf-8"), record


def default_plan_dir() -> Path:
    return config.default_system_config_path().expanduser().resolve().parent / "plan"


def extract_proposed_plan_content(answer: str) -> str:
    stripped = answer.strip()
    prefix = "<proposed_plan>"
    suffix = "</proposed_plan>"
    if not (stripped.startswith(prefix) and stripped.endswith(suffix)):
        return stripped
    return stripped[len(prefix) : -len(suffix)].strip()


def extract_plan_title(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or "Proposed plan"
        return stripped[:80]
    return "Proposed plan"


def plan_record_from_dict(raw: Any) -> PlanRecord | None:
    if not isinstance(raw, dict):
        return None
    try:
        status = str(raw.get("status") or "pending")
        if status not in {"pending", "approved", "rejected", "missing"}:
            status = "pending"
        return PlanRecord(
            id=str(raw.get("id") or ""),
            session_id=str(raw.get("session_id") or ""),
            turn_id=str(raw.get("turn_id") or ""),
            title=str(raw.get("title") or "Proposed plan"),
            status=status,  # type: ignore[arg-type]
            path=str(raw.get("path") or ""),
            created_at=str(raw.get("created_at") or ""),
            updated_at=str(raw.get("updated_at") or ""),
        )
    except (TypeError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip(".-")[:80] or "session"
