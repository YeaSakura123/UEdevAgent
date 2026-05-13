from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..mcp.types import McpServerConfig
from ..permissions import PermissionMode, normalize_permission_mode


CONFIG_VERSION = 1
SYSTEM_CONFIG_DIR = ".uedev"
CONFIG_FILE = "config.json"


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    base_url: str
    api_key: str


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
        "models": {
            "my-model": {
                "model": "",
                "base_url": "https://your.api.com/v1",
                "api_key": "",
            }
        },
        "ue": {
            "engines": {
                "5.4": {
                    "root": "D:/Program Files/Epic Games/UE_5.4",
                }
            }
        },
        "mcp": {
            "servers": {},
        },
    }


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
        profile = ModelProfile(
            name=name,
            model=str(raw.get("model") or "").strip(),
            base_url=str(raw.get("base_url") or "https://your.api.com/v1"),
            api_key=str(raw.get("api_key") or "").strip(),
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

    return SystemConfig(path=config_path, default_model=default_model, models=models, ue_engines=engines, mcp_servers=mcp_servers)


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
        lines.append(f"- {name}{suffix}: {profile.model or '(missing model)'} api_key={key_state} base_url={profile.base_url}")
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


def _is_safe_mcp_name(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]+", value.strip()))
