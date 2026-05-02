"""
UE 5.7 live-editor smoke test for agent capability checks.

Run this only while the full Unreal Editor is open. It intentionally exercises
editor-only APIs, so it is not suitable for true commandlet mode.

Suggested run paths:
  1. UE editor: Window -> Developer Tools -> Python Console -> paste exec(open(...).read())
  2. CLI dry-run/execute path:
     uedev ue run-python examples/editor_live_agent_smoke.py --mode full_editor --execute

The script creates:
  - a saved Material under /Game/_UedevAgentSmoke
  - three transient cube actors in the current editor level

Check the UE Output Log for:
  UDEV_AGENT_SMOKE_BEGIN
  UDEV_AGENT_SMOKE_STEP ...
  UDEV_AGENT_SMOKE_RESULT {json}
  UDEV_AGENT_SMOKE_END status=PASS
"""

from __future__ import annotations

import json
import time
import traceback

import unreal


TEST_ROOT = "/Game/_UedevAgentSmoke"
ASSET_BASE_NAME = "M_AgentSmoke"
LOG_PREFIX = "UDEV_AGENT_SMOKE"


def log_step(message: str, **payload: object) -> None:
    if payload:
        rendered = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        unreal.log(f"{LOG_PREFIX}_STEP {message} {rendered}")
    else:
        unreal.log(f"{LOG_PREFIX}_STEP {message}")


def log_result(status: str, data: dict[str, object]) -> None:
    result = {"status": status, **data}
    unreal.log(f"{LOG_PREFIX}_RESULT {json.dumps(result, ensure_ascii=True, sort_keys=True, default=str)}")


def require_subsystem(cls):
    subsystem = unreal.get_editor_subsystem(cls)
    if subsystem is None:
        raise RuntimeError(f"Could not get editor subsystem: {cls}")
    return subsystem


def make_unique_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def create_material_asset(asset_subsystem, run_id: str) -> dict[str, object]:
    asset_tools = unreal.AssetToolsHelpers.get_asset_tools()
    asset_subsystem.make_directory(TEST_ROOT)

    base_package_name = f"{TEST_ROOT}/{ASSET_BASE_NAME}_{run_id}"
    package_name, asset_name = asset_tools.create_unique_asset_name(base_package_name, "")
    package_path = package_name.rsplit("/", 1)[0]

    factory = unreal.MaterialFactoryNew()
    try:
        factory.set_editor_property("edit_after_new", False)
    except Exception:
        unreal.log_warning(f"{LOG_PREFIX}_STEP factory_edit_after_new_unavailable")

    asset = asset_tools.create_asset(asset_name, package_path, unreal.Material, factory)
    if asset is None:
        raise RuntimeError(f"Failed to create Material at {package_path}/{asset_name}")

    asset_path = asset_subsystem.get_path_name_for_loaded_asset(asset)
    asset_subsystem.set_metadata_tag(asset, "uedev_agent_smoke", run_id)
    saved = asset_subsystem.save_loaded_asset(asset, only_if_is_dirty=False)
    exists = asset_subsystem.does_asset_exist(asset_path)
    reloaded = asset_subsystem.load_asset(asset_path)
    metadata_value = asset_subsystem.get_metadata_tag(reloaded, "uedev_agent_smoke") if reloaded else ""

    if not saved or not exists or str(metadata_value) != run_id:
        raise RuntimeError(
            "Material verification failed: "
            f"saved={saved}, exists={exists}, metadata={metadata_value!r}"
        )

    log_step("asset_created", asset_path=asset_path, metadata_value=str(metadata_value))
    return {
        "asset_path": asset_path,
        "asset_saved": saved,
        "asset_exists": exists,
        "metadata_value": str(metadata_value),
    }


def create_level_actors(actor_subsystem, run_id: str) -> dict[str, object]:
    cube_mesh = unreal.load_asset("/Engine/BasicShapes/Cube.Cube")
    if cube_mesh is None:
        raise RuntimeError("Could not load /Engine/BasicShapes/Cube.Cube")

    created_labels: list[str] = []
    created_paths: list[str] = []
    locations = [
        unreal.Vector(0.0, 0.0, 120.0),
        unreal.Vector(180.0, 0.0, 120.0),
        unreal.Vector(360.0, 0.0, 120.0),
    ]

    actor_subsystem.select_nothing()
    for index, location in enumerate(locations, start=1):
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor,
            location,
            unreal.Rotator(0.0, 0.0, 0.0),
            transient=False,
        )
        if actor is None:
            raise RuntimeError(f"Failed to spawn smoke actor {index}")

        label = f"UDEV_AGENT_SMOKE_{run_id}_{index}"
        actor.set_actor_label(label)
        actor.set_actor_scale3d(unreal.Vector(0.75, 0.75, 0.75))

        mesh_component = actor.get_component_by_class(unreal.StaticMeshComponent)
        if mesh_component is not None:
            mesh_component.set_static_mesh(cube_mesh)

        created_labels.append(label)
        created_paths.append(actor.get_path_name())
        log_step("actor_spawned", label=label, path=actor.get_path_name())

    all_actors = actor_subsystem.get_all_level_actors()
    found_count = sum(1 for actor in all_actors if actor.get_actor_label() in created_labels)
    if found_count != len(created_labels):
        raise RuntimeError(f"Actor verification failed: expected={len(created_labels)}, found={found_count}")

    actor_subsystem.set_selected_level_actors(
        [actor for actor in all_actors if actor.get_actor_label() in created_labels]
    )

    return {
        "actor_count": len(created_labels),
        "actor_found_count": found_count,
        "actor_labels": created_labels,
        "actor_paths": created_paths,
    }


def run_smoke_test() -> dict[str, object]:
    run_id = make_unique_run_id()
    unreal.log(f"{LOG_PREFIX}_BEGIN run_id={run_id}")

    editor_subsystem = require_subsystem(unreal.UnrealEditorSubsystem)
    actor_subsystem = require_subsystem(unreal.EditorActorSubsystem)
    asset_subsystem = require_subsystem(unreal.EditorAssetSubsystem)

    world = editor_subsystem.get_editor_world()
    if world is None:
        raise RuntimeError("No editor world is loaded. Open a level before running the smoke test.")

    log_step(
        "editor_context",
        engine_version=unreal.SystemLibrary.get_engine_version(),
        project_name=unreal.SystemLibrary.get_game_name(),
        world_name=world.get_name(),
    )

    asset_info = create_material_asset(asset_subsystem, run_id)
    actor_info = create_level_actors(actor_subsystem, run_id)

    result = {
        "run_id": run_id,
        "world_name": world.get_name(),
        **asset_info,
        **actor_info,
    }
    log_result("PASS", result)
    unreal.log(f"{LOG_PREFIX}_END status=PASS run_id={run_id}")
    return result


try:
    _uedev_result = run_smoke_test()
except Exception as exc:
    error = {"error": str(exc), "traceback": traceback.format_exc()}
    unreal.log_error(f"{LOG_PREFIX}_ERROR {json.dumps(error, ensure_ascii=True, sort_keys=True)}")
    log_result("FAIL", error)
    unreal.log(f"{LOG_PREFIX}_END status=FAIL")
    raise
