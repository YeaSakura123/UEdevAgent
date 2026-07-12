from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from ..llm.client import ChatMessage, ToolCall, call_model
from ..state.config import ModelProfile
from ..tools.specs import get_tool_specs
from ..ui.events import (
    final_event,
    stopped_event,
    thinking_event,
    tool_error_event,
    tool_result_event,
    tool_start_event,
    usage_event,
)
from .context import estimate_tokens, is_runtime_state_message, repair_tool_call_messages
from .history import (
    append_display_event,
    append_display_turn_start,
    append_history_message,
    append_transcript_token_usage,
    load_history_file,
    write_history_messages,
)


SubagentType = Literal["explorer", "worker", "default"]
SubagentStatus = Literal["running", "complete", "failed"]
SubagentToolExecutor = Callable[[str, dict[str, Any]], str]


@dataclass(frozen=True)
class SubagentSpec:
    agent_type: SubagentType
    task: str
    responsibility: str = ""
    paths: tuple[str, ...] = ()
    inherit_context: bool = False


@dataclass
class SubagentRecord:
    id: str
    agent_type: str
    task: str
    responsibility: str
    paths: list[str]
    inherit_context: bool
    status: SubagentStatus
    created_at: float
    completed_at: float | None
    history_path: str
    model_profile: str
    model: str
    display_history_path: str = ""
    metadata_path: str = ""
    parent_session_id: str = ""
    result: str = ""
    error: str = ""

    @property
    def label(self) -> str:
        task = " ".join(self.task.split())
        if len(task) > 72:
            task = task[:69] + "..."
        return f"{self.id} [{self.status}] {self.agent_type} - {task}"


@dataclass(frozen=True)
class SubagentResult:
    record: SubagentRecord

    @property
    def output(self) -> str:
        lines = [
            f"subagent_id: {self.record.id}",
            f"status: {self.record.status}",
            f"type: {self.record.agent_type}",
            f"model_profile: {self.record.model_profile}",
            f"model: {self.record.model or '(missing model)'}",
            f"history: {self.record.history_path}",
        ]
        if self.record.error:
            lines.extend(["error:", self.record.error])
        else:
            lines.extend(["result:", self.record.result or "(no result)"])
        return "\n".join(lines)


class SubagentManager:
    def __init__(
        self,
        agent_dir: Path,
        max_steps: int,
        execute_tool: SubagentToolExecutor,
        model_profile_provider: Callable[[], ModelProfile],
    ):
        self.agent_dir = agent_dir
        self.max_steps = max_steps
        self.execute_tool = execute_tool
        self.model_profile_provider = model_profile_provider
        self._counter = 0
        self._lock = threading.Lock()

    def validate_spec(self, spec: SubagentSpec) -> None:
        if spec.agent_type not in {"explorer", "worker", "default"}:
            raise ValueError("subagent agent_type must be explorer, worker, or default")
        if not spec.task.strip():
            raise ValueError("subagent requires task")
        if spec.agent_type == "worker":
            if not spec.responsibility.strip():
                raise ValueError("worker subagent requires responsibility")
            if not spec.paths:
                raise ValueError("worker subagent requires paths")

    def run_batch(
        self,
        specs: list[SubagentSpec],
        main_messages: list[ChatMessage],
        subagents_dir: Path,
        parent_turn_id: str = "",
    ) -> list[SubagentResult]:
        if not specs:
            return []
        for spec in specs:
            self.validate_spec(spec)
        with ThreadPoolExecutor(max_workers=len(specs), thread_name_prefix="uedev-subagent") as executor:
            futures = [executor.submit(self.run_one, spec, main_messages, subagents_dir, parent_turn_id) for spec in specs]
            return [future.result() for future in futures]

    def run_one(
        self,
        spec: SubagentSpec,
        main_messages: list[ChatMessage],
        subagents_dir: Path,
        parent_turn_id: str = "",
    ) -> SubagentResult:
        self.validate_spec(spec)
        profile = self.model_profile_provider()
        record = self._new_record(spec, profile, subagents_dir)
        self._append_index(record, subagents_dir)

        child_messages = self._build_initial_messages(spec, main_messages, profile)
        write_history_messages(Path(record.history_path), child_messages)
        display_path = Path(record.display_history_path)
        turn_id = record.id
        append_display_turn_start(display_path, turn_id, _subagent_task_message(spec))
        allowed_tools = _allowed_tools(spec.agent_type)
        subagent_tools = [tool for tool in get_tool_specs() if str(tool["function"]["name"]) in allowed_tools]

        started_at = time.perf_counter()
        total_steps = min(6, self.max_steps)
        try:
            for step in range(1, total_steps + 1):
                append_display_event(display_path, thinking_event(step, total_steps, turn_id))
                repair_tool_call_messages(child_messages, require_reasoning_content=profile.requires_reasoning_content)
                response = call_model(child_messages, profile, tools=subagent_tools)
                if response.usage is not None:
                    usage_payload: dict[str, object] = {
                        "request_id": f"req_{uuid4().hex}",
                        "created_at": time.time(),
                        "turn_id": parent_turn_id or turn_id,
                        "subagent_turn_id": turn_id,
                        "purpose": "subagent",
                        "subagent_id": record.id,
                        "profile_name": profile.name,
                        "model": profile.model,
                        "api_mode": "responses" if profile.response else "chat_completions",
                        "effort": profile.effort,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "cached_input_tokens": response.usage.cached_input_tokens,
                        "reasoning_tokens": response.usage.reasoning_tokens,
                        "source": response.usage.source,
                    }
                    append_transcript_token_usage(subagents_dir.parent / "transcript.jsonl", usage_payload)
                    append_display_event(display_path, usage_event(usage_payload, turn_id))
                if response.tool_calls:
                    assistant = ChatMessage(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                        reasoning_content=response.reasoning_content,
                    )
                    child_messages.append(assistant)
                    append_history_message(Path(record.history_path), assistant)
                    for tool_call in response.tool_calls:
                        append_display_event(display_path, tool_start_event(tool_call.name, tool_call.arguments, turn_id))
                        output, is_error = self._execute_subagent_tool(tool_call, allowed_tools)
                        if is_error:
                            append_display_event(display_path, tool_error_event(tool_call.name, output, turn_id))
                        else:
                            append_display_event(display_path, tool_result_event(tool_call.name, output, turn_id))
                        tool_message = ChatMessage(
                            role="tool",
                            content=f"Tool result for: {tool_call.name}\n{_truncate(output)}",
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                        child_messages.append(tool_message)
                        append_history_message(Path(record.history_path), tool_message)
                    continue

                final = ChatMessage(role="assistant", content=response.content, reasoning_content=response.reasoning_content)
                child_messages.append(final)
                append_history_message(Path(record.history_path), final)
                record.status = "complete"
                record.result = response.content.strip()
                record.completed_at = time.time()
                self._append_index(record, subagents_dir)
                append_display_event(display_path, final_event(response.content.strip(), turn_id, _duration_ms(started_at)))
                return SubagentResult(record)

            record.status = "failed"
            record.error = "Subagent stopped after bounded steps."
            record.completed_at = time.time()
            self._append_index(record, subagents_dir)
            append_display_event(display_path, stopped_event(record.error, turn_id, _duration_ms(started_at)))
            return SubagentResult(record)
        except Exception as error:
            record.status = "failed"
            record.error = str(error)
            record.completed_at = time.time()
            self._append_index(record, subagents_dir)
            append_display_event(display_path, stopped_event(record.error, turn_id, _duration_ms(started_at)))
            return SubagentResult(record)

    def list_records(self, subagents_dir: Path | None) -> list[SubagentRecord]:
        if subagents_dir is None:
            return []
        index_path = subagents_dir / "index.jsonl"
        if not index_path.exists():
            return []
        records: dict[str, SubagentRecord] = {}
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            record = _record_from_dict(raw)
            if record is not None:
                records[record.id] = record
        return sorted(records.values(), key=lambda item: item.created_at, reverse=True)

    def load_messages(self, record: SubagentRecord) -> list[ChatMessage]:
        return load_history_file(Path(record.history_path))

    def render_list(self, subagents_dir: Path | None) -> str:
        records = self.list_records(subagents_dir)
        if not records:
            return "Main conversation\nNo subagents."
        return "\n".join(["Main conversation", *[record.label for record in records]])

    def _new_record(self, spec: SubagentSpec, profile: ModelProfile, subagents_dir: Path) -> SubagentRecord:
        subagents_dir.mkdir(parents=True, exist_ok=True)
        while True:
            with self._lock:
                self._counter += 1
                subagent_id = f"sa_{time.time_ns()}_{self._counter}_{uuid4().hex[:8]}"
            subagent_dir = subagents_dir / subagent_id
            try:
                subagent_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            break
        metadata_path = subagent_dir / "metadata.json"
        history_path = subagent_dir / "messages.jsonl"
        display_history_path = subagent_dir / "display.jsonl"
        return SubagentRecord(
            id=subagent_id,
            agent_type=spec.agent_type,
            task=spec.task,
            responsibility=spec.responsibility,
            paths=list(spec.paths),
            inherit_context=spec.inherit_context,
            status="running",
            created_at=time.time(),
            completed_at=None,
            metadata_path=str(metadata_path),
            history_path=str(history_path),
            display_history_path=str(display_history_path),
            parent_session_id=subagents_dir.parent.name,
            model_profile=profile.name,
            model=profile.model,
        )

    def _build_initial_messages(
        self,
        spec: SubagentSpec,
        main_messages: list[ChatMessage],
        profile: ModelProfile,
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=_subagent_system_prompt(spec))]
        if spec.inherit_context:
            inherited = _render_inherited_context(main_messages, profile)
            if inherited:
                messages.append(ChatMessage(role="user", content=f"Main conversation context:\n{inherited}"))
        messages.append(ChatMessage(role="user", content=_subagent_task_message(spec)))
        repair_tool_call_messages(messages, require_reasoning_content=profile.requires_reasoning_content)
        return messages

    def _execute_subagent_tool(self, tool_call: ToolCall, allowed_tools: set[str]) -> tuple[str, bool]:
        if tool_call.name not in allowed_tools:
            return f"Subagent tool not allowed: {tool_call.name}", True
        output = self.execute_tool(tool_call.name, tool_call.arguments)
        return output, output.startswith(f"Tool {tool_call.name} failed:")

    def _append_index(self, record: SubagentRecord, subagents_dir: Path) -> None:
        index_path = subagents_dir / "index.jsonl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if record.metadata_path:
                metadata_path = Path(record.metadata_path)
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def parse_subagent_spec(tool_input: dict[str, object]) -> SubagentSpec:
    raw_type = str(tool_input.get("agent_type") or "default").strip().lower()
    if raw_type in {"explore", "readonly", "read_only"}:
        raw_type = "explorer"
    if raw_type not in {"explorer", "worker", "default"}:
        raise ValueError("subagent agent_type must be explorer, worker, or default")
    task = str(tool_input.get("task") or tool_input.get("prompt") or "").strip()
    responsibility = str(tool_input.get("responsibility") or "").strip()
    paths = tuple(_string_items(tool_input.get("paths")))
    inherit_context = bool(tool_input.get("inherit_context", False))
    return SubagentSpec(
        agent_type=raw_type,  # type: ignore[arg-type]
        task=task,
        responsibility=responsibility,
        paths=paths,
        inherit_context=inherit_context,
    )


def _allowed_tools(agent_type: str) -> set[str]:
    allowed = {"read_file", "list_files", "grep", "shell"}
    if agent_type == "worker":
        allowed.update({"write_file", "edit_file"})
    return allowed


def _subagent_system_prompt(spec: SubagentSpec) -> str:
    lines = [
        "You are a focused subagent spawned by the main agent.",
        "You do not answer the user directly; return results for the main agent to integrate.",
        "Stay inside your assigned responsibility and paths.",
        "Do not undo unrelated user, main-agent, or other subagent changes.",
        f"Subagent type: {spec.agent_type}",
    ]
    if spec.agent_type == "explorer":
        lines.extend(["Read-only. Find facts, relevant files, risks, and concise conclusions.", "Do not edit files."])
    elif spec.agent_type == "worker":
        lines.extend([
            "You may edit files only within the assigned paths.",
            "Return changed files, summary, tests run, and remaining risks.",
        ])
    else:
        lines.extend(["Analyze and synthesize. Prefer read-only unless the task explicitly requires edits."])
    return "\n".join(lines)


def _subagent_task_message(spec: SubagentSpec) -> str:
    lines = ["Task:", spec.task]
    if spec.responsibility:
        lines.extend(["", "Responsibility:", spec.responsibility])
    if spec.paths:
        lines.append("")
        lines.append("Assigned paths:")
        lines.extend(f"- {path}" for path in spec.paths)
    lines.extend(
        [
            "",
            "Output:",
            "- summary",
            "- findings or changed files",
            "- tests run when relevant",
            "- risks or next recommendation",
        ]
    )
    return "\n".join(lines)


def _render_inherited_context(main_messages: list[ChatMessage], profile: ModelProfile) -> str:
    budget = max(1, min(20000, int(profile.context_window * 0.5)))
    rendered: list[str] = []
    for message in reversed(main_messages):
        if message.role == "system" or is_runtime_state_message(message):
            continue
        if message.role == "tool":
            content = _truncate(message.content, 1200)
        else:
            content = _truncate(message.content, 2400)
        if not content.strip():
            continue
        candidate = f"{message.role}: {content}"
        trial = [ChatMessage(role="user", content="\n\n".join([candidate, *rendered]))]
        if estimate_tokens(trial) > budget:
            break
        rendered.insert(0, candidate)
    return "\n\n".join(rendered)


def _record_from_dict(raw: Any) -> SubagentRecord | None:
    if not isinstance(raw, dict):
        return None
    try:
        return SubagentRecord(
            id=str(raw.get("id") or ""),
            agent_type=str(raw.get("agent_type") or ""),
            task=str(raw.get("task") or ""),
            responsibility=str(raw.get("responsibility") or ""),
            paths=_string_items(raw.get("paths")),
            inherit_context=bool(raw.get("inherit_context", False)),
            status=str(raw.get("status") or "failed"),  # type: ignore[arg-type]
            created_at=float(raw.get("created_at") or 0.0),
            completed_at=float(raw["completed_at"]) if raw.get("completed_at") is not None else None,
            metadata_path=str(raw.get("metadata_path") or ""),
            history_path=str(raw.get("history_path") or ""),
            display_history_path=str(raw.get("display_history_path") or ""),
            parent_session_id=str(raw.get("parent_session_id") or ""),
            model_profile=str(raw.get("model_profile") or ""),
            model=str(raw.get("model") or ""),
            result=str(raw.get("result") or ""),
            error=str(raw.get("error") or ""),
        )
    except (TypeError, ValueError):
        return None


def _string_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _truncate(value: str, max_length: int = 12000) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}\n...[truncated {len(value) - max_length} chars]"


def _duration_ms(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))
