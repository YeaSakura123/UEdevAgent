from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from ..mcp.types import McpServerConfig
from ..policy.permissions import PermissionMode, normalize_permission_mode


CONFIG_VERSION = 2
SYSTEM_CONFIG_DIR = ".uedev"
CONFIG_FILE = "config.json"
DEFAULT_CONTEXT_WINDOW = 256 * 1024
DEFAULT_DIFF_OUTPUT_MAX_CHARS = 20000
DEFAULT_WORKTREE_ROOT = ""
DEFAULT_MAX_STEPS = 8
DEFAULT_MODEL_TIMEOUT_SECONDS = 120
DEFAULT_WORKSPACE_EXCLUDED_DIRS = (
    ".agent",
    ".git",
    ".vs",
    "Binaries",
    "Intermediate",
    "Saved",
    "DerivedDataCache",
)


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    base_url: str
    api_key: str
    context_window: int = DEFAULT_CONTEXT_WINDOW
    timeout_seconds: int = DEFAULT_MODEL_TIMEOUT_SECONDS
    gpt_model: bool = False
    requires_reasoning_content: bool = False
    responses: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UeEngineProfile:
    name: str
    root: Path
    aliases: tuple[str, ...] = ()

    @property
    def editor_cmd_path(self) -> Path:
        return self.root / "Engine" / "Binaries" / "Win64" / "UnrealEditor-Cmd.exe"

    @property
    def editor_gui_path(self) -> Path:
        return self.root / "Engine" / "Binaries" / "Win64" / "UnrealEditor.exe"


@dataclass(frozen=True)
class SystemConfig:
    path: Path
    default_model: str
    models: dict[str, ModelProfile]
    ue_engines: dict[str, UeEngineProfile]
    mcp_servers: dict[str, McpServerConfig]
    subagent_model_profile: str | None = None
    diff_output_max_chars: int = DEFAULT_DIFF_OUTPUT_MAX_CHARS
    worktree_default_root: Path | None = None
    runtime_default_max_steps: int = DEFAULT_MAX_STEPS
    workspace_excluded_dirs: tuple[str, ...] = DEFAULT_WORKSPACE_EXCLUDED_DIRS


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    active_model: str | None
    permission_mode: PermissionMode


def default_system_config_path() -> Path:
    return Path.home() / SYSTEM_CONFIG_DIR / CONFIG_FILE


def agent_dir(cwd: Path) -> Path:
    return cwd.resolve() / ".agent"


def project_config_path(cwd: Path) -> Path:
    return agent_dir(cwd) / CONFIG_FILE


def system_config_template() -> dict[str, Any]:
    return {
        "version": CONFIG_VERSION,
        "default_model": "openai-gpt",
        "models": {
            "openai-gpt": {
                "gpt_model": True,
                "model": "gpt-5",
                "base_url": "https://api.openai.com/v1",
                "api_key": "",
                "context_window": DEFAULT_CONTEXT_WINDOW,
                "timeout_seconds": DEFAULT_MODEL_TIMEOUT_SECONDS,
                "responses": default_responses_options(),
            },
            "compatible-chat": {
                "gpt_model": False,
                "model": "",
                "base_url": "https://your.api.com/v1",
                "api_key": "",
                "context_window": DEFAULT_CONTEXT_WINDOW,
                "timeout_seconds": DEFAULT_MODEL_TIMEOUT_SECONDS,
                "requires_reasoning_content": False,
            },
        },
        "ue": {
            "engines": {
                "5.4": {
                    "root": "D:/Program Files/Epic Games/UE_5.4",
                }
            }
        },
        "display": {
            "diff_output_max_chars": DEFAULT_DIFF_OUTPUT_MAX_CHARS,
        },
        "runtime": {
            "default_max_steps": DEFAULT_MAX_STEPS,
        },
        "workspace": {
            "excluded_dirs": list(DEFAULT_WORKSPACE_EXCLUDED_DIRS),
        },
        "worktrees": {
            "default_root": DEFAULT_WORKTREE_ROOT,
        },
        "subagents": {
            "model_profile": None,
        },
        "mcp": {
            "servers": {},
        },
    }


def default_responses_options() -> dict[str, Any]:
    return deepcopy(
        {
            "store": False,
            "reasoning": {
                "effort": None,
                "summary": None,
            },
            "text": {
                "format": {"type": "text"},
            },
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "strict_function_tools": False,
            "max_output_tokens": None,
            "truncation": "disabled",
            "include": [],
            "built_in_tools": {
                "web_search": {"enabled": False},
                "file_search": {"enabled": False, "vector_store_ids": []},
                "remote_mcp": [],
            },
        }
    )


def project_config_template(active_model: str | None = None, permission_mode: PermissionMode | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"version": CONFIG_VERSION}
    if active_model:
        payload["active_model"] = active_model
    if permission_mode:
        payload["permission_mode"] = permission_mode
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ConfigError(f"Invalid JSON config file: {path}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a JSON object: {path}")
    return data


def load_system_config(path: Path | None = None) -> SystemConfig:
    config_path = (path or default_system_config_path()).expanduser().resolve()
    data = read_json_object(config_path)

    raw_models = data.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ConfigError(f"System config models must be a non-empty object: {config_path}")

    models: dict[str, ModelProfile] = {}
    for name, raw in raw_models.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"Model profile names must be non-empty strings: {config_path}")
        if not isinstance(raw, dict):
            raise ConfigError(f"Model profile {name!r} must be an object: {config_path}")
        gpt_model = _optional_bool(raw.get("gpt_model"), False, config_path, f"models.{name}.gpt_model")
        requires_reasoning_content = _optional_bool(
            raw.get("requires_reasoning_content"),
            False,
            config_path,
            f"models.{name}.requires_reasoning_content",
        )
        profile = ModelProfile(
            name=name,
            model=str(raw.get("model") or "").strip(),
            base_url=str(raw.get("base_url") or "https://your.api.com/v1"),
            api_key=str(raw.get("api_key") or "").strip(),
            context_window=_optional_positive_int(raw.get("context_window"), DEFAULT_CONTEXT_WINDOW, config_path, f"models.{name}.context_window"),
            timeout_seconds=_optional_positive_int(
                raw.get("timeout_seconds"),
                DEFAULT_MODEL_TIMEOUT_SECONDS,
                config_path,
                f"models.{name}.timeout_seconds",
            ),
            gpt_model=gpt_model,
            requires_reasoning_content=requires_reasoning_content and not gpt_model,
            responses=_parse_responses_options(raw.get("responses"), config_path, f"models.{name}.responses") if gpt_model else {},
        )
        models[name] = profile

    default_model = str(data.get("default_model") or "").strip()
    if not default_model:
        default_model = next(iter(models))
    if default_model not in models:
        raise ConfigError(f"default_model {default_model!r} is not defined in models: {config_path}")

    raw_ue = data.get("ue", {})
    if raw_ue is None:
        raw_ue = {}
    if not isinstance(raw_ue, dict):
        raise ConfigError(f"System config ue must be an object: {config_path}")
    raw_engines = raw_ue.get("engines", {})
    if raw_engines is None:
        raw_engines = {}
    if not isinstance(raw_engines, dict):
        raise ConfigError(f"System config ue.engines must be an object: {config_path}")

    engines: dict[str, UeEngineProfile] = {}
    for name, raw in raw_engines.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigError(f"UE engine names must be non-empty strings: {config_path}")
        if not isinstance(raw, dict):
            raise ConfigError(f"UE engine {name!r} must be an object: {config_path}")
        root = _required_str(raw, "root", config_path, f"ue.engines.{name}")
        aliases = _optional_string_tuple(raw.get("aliases"), config_path, f"ue.engines.{name}.aliases")
        engines[name] = UeEngineProfile(name=name, root=Path(root).expanduser().resolve(), aliases=aliases)

    mcp_servers = _parse_mcp_servers(data.get("mcp", {}), config_path)
    diff_output_max_chars = _parse_display_config(data.get("display", {}), config_path)
    runtime_default_max_steps = _parse_runtime_config(data.get("runtime", {}), config_path)
    workspace_excluded_dirs = _parse_workspace_config(data.get("workspace", {}), config_path)
    worktree_default_root = _parse_worktree_config(data.get("worktrees", {}), config_path)
    subagent_model_profile = _parse_subagent_config(data.get("subagents", {}), models, config_path)

    return SystemConfig(
        path=config_path,
        default_model=default_model,
        models=models,
        ue_engines=engines,
        mcp_servers=mcp_servers,
        subagent_model_profile=subagent_model_profile,
        diff_output_max_chars=diff_output_max_chars,
        worktree_default_root=worktree_default_root,
        runtime_default_max_steps=runtime_default_max_steps,
        workspace_excluded_dirs=workspace_excluded_dirs,
    )


def load_project_config(cwd: Path) -> ProjectConfig:
    path = project_config_path(cwd)
    if not path.exists():
        return ProjectConfig(path=path, active_model=None, permission_mode="default")
    data = read_json_object(path)
    raw_active = data.get("active_model")
    active_model = str(raw_active).strip() if raw_active is not None else ""
    permission_mode = normalize_permission_mode(str(data.get("permission_mode") or "default")) or "default"
    return ProjectConfig(path=path, active_model=active_model or None, permission_mode=permission_mode)


def save_project_active_model(cwd: Path, model_name: str) -> None:
    data = _read_project_config_data(cwd)
    data["active_model"] = model_name
    data["version"] = CONFIG_VERSION
    write_json(project_config_path(cwd), data)


def save_project_permission_mode(cwd: Path, permission_mode: PermissionMode) -> None:
    data = _read_project_config_data(cwd)
    data["permission_mode"] = permission_mode
    data["version"] = CONFIG_VERSION
    write_json(project_config_path(cwd), data)


def reset_project_active_model(cwd: Path) -> None:
    path = project_config_path(cwd)
    if not path.exists():
        return
    data = read_json_object(path)
    data.pop("active_model", None)
    data["version"] = CONFIG_VERSION
    write_json(path, data)


def _read_project_config_data(cwd: Path) -> dict[str, Any]:
    path = project_config_path(cwd)
    if not path.exists():
        return {"version": CONFIG_VERSION}
    data = read_json_object(path)
    data["version"] = CONFIG_VERSION
    return data


def resolve_model_profile(cwd: Path, system_config: SystemConfig | None = None) -> ModelProfile:
    config = system_config or load_system_config()
    project = load_project_config(cwd)
    model_name = project.active_model or config.default_model
    profile = config.models.get(model_name)
    if profile is None:
        raise ConfigError(
            f"Active model {model_name!r} is not defined in {config.path}. "
            "Use /model to choose a configured profile."
        )
    return profile


def resolve_subagent_model_profile(
    cwd: Path,
    main_profile: ModelProfile | None = None,
    system_config: SystemConfig | None = None,
) -> ModelProfile:
    config = system_config or load_system_config()
    if config.subagent_model_profile:
        profile = config.models.get(config.subagent_model_profile)
        if profile is None:
            raise ConfigError(
                f"subagents.model_profile {config.subagent_model_profile!r} is not defined in models: {config.path}"
            )
        return profile
    return main_profile or resolve_model_profile(cwd, config)


def active_model_name(cwd: Path, system_config: SystemConfig | None = None) -> str:
    config = system_config or load_system_config()
    project = load_project_config(cwd)
    return project.active_model or config.default_model


def format_model_profiles(cwd: Path, system_config: SystemConfig | None = None) -> str:
    config = system_config or load_system_config()
    active = active_model_name(cwd, config)
    lines = ["Model profiles:"]
    for name, profile in sorted(config.models.items()):
        markers: list[str] = []
        if name == active:
            markers.append("active")
        if name == config.default_model:
            markers.append("default")
        suffix = f" ({', '.join(markers)})" if markers else ""
        key_state = "set" if profile.api_key else "missing"
        lines.append(
            f"- {name}{suffix}: {profile.model or '(missing model)'} "
            f"mode={'responses' if profile.gpt_model else 'chat_completions'} "
            f"context_window={profile.context_window} "
            f"timeout_seconds={profile.timeout_seconds} "
            f"requires_reasoning_content={profile.requires_reasoning_content} "
            f"api_key={key_state} base_url={profile.base_url}"
        )
    return "\n".join(lines)


def _required_str(data: dict[str, Any], key: str, path: Path, prefix: str = "") -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        label = f"{prefix}.{key}" if prefix else key
        raise ConfigError(f"Missing required string {label}: {path}")
    return value.strip()


def _optional_string_tuple(value: object, path: Path, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ConfigError(f"{label} must be an array of strings: {path}")
    return tuple(item.strip() for item in value if item.strip())


def _optional_positive_int(value: object, default: int, path: Path, label: str) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ConfigError(f"{label} must be a positive integer: {path}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{label} must be a positive integer: {path}") from error
    if parsed <= 0:
        raise ConfigError(f"{label} must be a positive integer: {path}")
    return parsed


def _optional_positive_int_or_none(value: object, path: Path, label: str) -> int | None:
    if value is None or value == "":
        return None
    return _optional_positive_int(value, 1, path, label)


def _optional_bool(value: object, default: bool, path: Path, label: str) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ConfigError(f"{label} must be a boolean: {path}")
    return value


def _optional_str_or_none(value: object, path: Path, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{label} must be a string or null: {path}")
    return value.strip() or None


def _parse_responses_options(raw: object, config_path: Path, label: str) -> dict[str, Any]:
    options = default_responses_options()
    if raw is None:
        return options
    if not isinstance(raw, dict):
        raise ConfigError(f"{label} must be an object: {config_path}")

    if "store" in raw:
        options["store"] = _optional_bool(raw.get("store"), False, config_path, f"{label}.store")
    if "tool_choice" in raw:
        tool_choice = raw.get("tool_choice")
        if not isinstance(tool_choice, (str, dict)):
            raise ConfigError(f"{label}.tool_choice must be a string or object: {config_path}")
        options["tool_choice"] = tool_choice
    if "parallel_tool_calls" in raw:
        options["parallel_tool_calls"] = _optional_bool(raw.get("parallel_tool_calls"), True, config_path, f"{label}.parallel_tool_calls")
    if "strict_function_tools" in raw:
        options["strict_function_tools"] = _optional_bool(raw.get("strict_function_tools"), False, config_path, f"{label}.strict_function_tools")
    if "max_output_tokens" in raw:
        options["max_output_tokens"] = _optional_positive_int_or_none(raw.get("max_output_tokens"), config_path, f"{label}.max_output_tokens")
    if "truncation" in raw:
        options["truncation"] = _optional_str_or_none(raw.get("truncation"), config_path, f"{label}.truncation")
    if "include" in raw:
        include = raw.get("include")
        if include is None:
            options["include"] = []
        elif not isinstance(include, list) or not all(isinstance(item, str) for item in include):
            raise ConfigError(f"{label}.include must be an array of strings: {config_path}")
        else:
            options["include"] = [item.strip() for item in include if item.strip()]

    if "reasoning" in raw:
        reasoning = raw.get("reasoning")
        if reasoning is None:
            options["reasoning"] = {"effort": None, "summary": None}
        elif not isinstance(reasoning, dict):
            raise ConfigError(f"{label}.reasoning must be an object or null: {config_path}")
        else:
            parsed_reasoning = dict(options["reasoning"])
            if "effort" in reasoning:
                parsed_reasoning["effort"] = _optional_str_or_none(reasoning.get("effort"), config_path, f"{label}.reasoning.effort")
            if "summary" in reasoning:
                parsed_reasoning["summary"] = _optional_str_or_none(reasoning.get("summary"), config_path, f"{label}.reasoning.summary")
            options["reasoning"] = parsed_reasoning

    if "text" in raw:
        text = raw.get("text")
        if not isinstance(text, dict):
            raise ConfigError(f"{label}.text must be an object: {config_path}")
        text_format = text.get("format", options["text"]["format"])
        if not isinstance(text_format, dict):
            raise ConfigError(f"{label}.text.format must be an object: {config_path}")
        options["text"] = {"format": text_format}

    if "built_in_tools" in raw:
        options["built_in_tools"] = _parse_responses_built_in_tools(raw.get("built_in_tools"), config_path, f"{label}.built_in_tools")

    return options


def _parse_responses_built_in_tools(raw: object, config_path: Path, label: str) -> dict[str, Any]:
    built_in = default_responses_options()["built_in_tools"]
    if raw is None:
        return built_in
    if not isinstance(raw, dict):
        raise ConfigError(f"{label} must be an object: {config_path}")

    if "web_search" in raw:
        web_search = raw.get("web_search")
        if not isinstance(web_search, dict):
            raise ConfigError(f"{label}.web_search must be an object: {config_path}")
        built_in["web_search"] = {
            "enabled": _optional_bool(web_search.get("enabled"), False, config_path, f"{label}.web_search.enabled")
        }

    if "file_search" in raw:
        file_search = raw.get("file_search")
        if not isinstance(file_search, dict):
            raise ConfigError(f"{label}.file_search must be an object: {config_path}")
        vector_store_ids = file_search.get("vector_store_ids", [])
        if not isinstance(vector_store_ids, list) or not all(isinstance(item, str) for item in vector_store_ids):
            raise ConfigError(f"{label}.file_search.vector_store_ids must be an array of strings: {config_path}")
        built_in["file_search"] = {
            "enabled": _optional_bool(file_search.get("enabled"), False, config_path, f"{label}.file_search.enabled"),
            "vector_store_ids": [item.strip() for item in vector_store_ids if item.strip()],
        }

    if "remote_mcp" in raw:
        remote_mcp = raw.get("remote_mcp")
        if not isinstance(remote_mcp, list) or not all(isinstance(item, dict) for item in remote_mcp):
            raise ConfigError(f"{label}.remote_mcp must be an array of objects: {config_path}")
        built_in["remote_mcp"] = [dict(item) for item in remote_mcp]

    return built_in


def _parse_mcp_servers(raw_mcp: object, config_path: Path) -> dict[str, McpServerConfig]:
    if raw_mcp is None:
        return {}
    if not isinstance(raw_mcp, dict):
        raise ConfigError(f"System config mcp must be an object: {config_path}")
    raw_servers = raw_mcp.get("servers", {})
    if raw_servers is None:
        return {}
    if not isinstance(raw_servers, dict):
        raise ConfigError(f"System config mcp.servers must be an object: {config_path}")

    servers: dict[str, McpServerConfig] = {}
    for name, raw in raw_servers.items():
        if not isinstance(name, str) or not _is_safe_mcp_name(name):
            raise ConfigError(f"MCP server names must use letters, numbers, underscore, or hyphen: {config_path}")
        if not isinstance(raw, dict):
            raise ConfigError(f"MCP server {name!r} must be an object: {config_path}")
        transport = str(raw.get("transport") or "stdio").strip()
        if transport != "stdio":
            raise ConfigError(f"MCP server {name!r} has unsupported transport {transport!r}: {config_path}")
        command = _required_str(raw, "command", config_path, f"mcp.servers.{name}")
        args = _optional_string_tuple(raw.get("args"), config_path, f"mcp.servers.{name}.args")
        cwd_raw = raw.get("cwd")
        cwd = Path(str(cwd_raw)).expanduser().resolve() if isinstance(cwd_raw, str) and cwd_raw.strip() else None
        timeout_raw = raw.get("timeout_seconds")
        timeout_seconds = 10
        if timeout_raw is not None and timeout_raw != "":
            try:
                timeout_seconds = int(timeout_raw)
            except (TypeError, ValueError) as error:
                raise ConfigError(f"mcp.servers.{name}.timeout_seconds must be an integer: {config_path}") from error
        enabled_raw = raw.get("enabled", True)
        if not isinstance(enabled_raw, bool):
            raise ConfigError(f"mcp.servers.{name}.enabled must be a boolean: {config_path}")
        servers[name] = McpServerConfig(
            name=name,
            transport=transport,
            command=command,
            args=args,
            cwd=cwd,
            enabled=enabled_raw,
            timeout_seconds=timeout_seconds,
        )
    return servers


def _parse_display_config(raw_display: object, config_path: Path) -> int:
    if raw_display is None:
        return DEFAULT_DIFF_OUTPUT_MAX_CHARS
    if not isinstance(raw_display, dict):
        raise ConfigError(f"System config display must be an object: {config_path}")
    return _optional_positive_int(
        raw_display.get("diff_output_max_chars"),
        DEFAULT_DIFF_OUTPUT_MAX_CHARS,
        config_path,
        "display.diff_output_max_chars",
    )


def _parse_runtime_config(raw_runtime: object, config_path: Path) -> int:
    if raw_runtime is None:
        return DEFAULT_MAX_STEPS
    if not isinstance(raw_runtime, dict):
        raise ConfigError(f"System config runtime must be an object: {config_path}")
    return _optional_positive_int(
        raw_runtime.get("default_max_steps"),
        DEFAULT_MAX_STEPS,
        config_path,
        "runtime.default_max_steps",
    )


def _parse_workspace_config(raw_workspace: object, config_path: Path) -> tuple[str, ...]:
    if raw_workspace is None:
        return DEFAULT_WORKSPACE_EXCLUDED_DIRS
    if not isinstance(raw_workspace, dict):
        raise ConfigError(f"System config workspace must be an object: {config_path}")
    raw_excluded = raw_workspace.get("excluded_dirs")
    if raw_excluded is None:
        return DEFAULT_WORKSPACE_EXCLUDED_DIRS
    if not isinstance(raw_excluded, list):
        raise ConfigError(f"workspace.excluded_dirs must be an array of directory names: {config_path}")
    names: list[str] = []
    for item in raw_excluded:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"workspace.excluded_dirs must be an array of directory names: {config_path}")
        value = item.strip()
        if _is_invalid_excluded_dir(value):
            raise ConfigError(f"workspace.excluded_dirs entries must be single directory names: {config_path}")
        names.append(value)
    return tuple(names)


def _is_invalid_excluded_dir(value: str) -> bool:
    if value in {".", ".."}:
        return True
    if "/" in value or "\\" in value:
        return True
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        return True
    return bool(PureWindowsPath(value).drive)


def _parse_worktree_config(raw_worktrees: object, config_path: Path) -> Path | None:
    if raw_worktrees is None:
        return None
    if not isinstance(raw_worktrees, dict):
        raise ConfigError(f"System config worktrees must be an object: {config_path}")
    raw_default = raw_worktrees.get("default_root", DEFAULT_WORKTREE_ROOT)
    if raw_default is None:
        return None
    if not isinstance(raw_default, str):
        raise ConfigError(f"worktrees.default_root must be a string: {config_path}")
    value = raw_default.strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _parse_subagent_config(raw_subagents: object, models: dict[str, ModelProfile], config_path: Path) -> str | None:
    if raw_subagents is None:
        return None
    if not isinstance(raw_subagents, dict):
        raise ConfigError(f"System config subagents must be an object: {config_path}")
    raw_profile = raw_subagents.get("model_profile")
    if raw_profile is None:
        return None
    profile = str(raw_profile).strip()
    if not profile:
        return None
    if profile not in models:
        raise ConfigError(f"subagents.model_profile {profile!r} is not defined in models: {config_path}")
    return profile


def _is_safe_mcp_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", value.strip()))
