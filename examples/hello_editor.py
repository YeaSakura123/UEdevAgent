"""
UE Python API 入门示例：获取当前关卡信息

最简单的 UE Python API 上手脚本。不需要项目里有任何内容。

运行方式：
  1. UE 编辑器内：Window → Developer Tools → Python Console，粘贴运行
  2. 命令行：UnrealEditor-Cmd.exe "C:/YourProject.uproject" -ExecutePythonScript="path/to/this/file.py"
"""

import unreal


def hello_editor():
    """打印编辑器当前状态 —— 每个字段都是 UE Python API 的入口。"""

    # ---- 获取编辑器子系统 ----
    editor_subsystem = unreal.get_editor_subsystem(
        unreal.UnrealEditorSubsystem
    )

    # ---- 关卡信息 ----
    world = editor_subsystem.get_editor_world()
    if world:
        level_name = world.get_name()
        print(f"Current Level : {level_name}")

        # 获取关卡中的所有 Actor
        # UE 5.4+ 用 EditorActorSubsystem 替代废弃的 EditorLevelLibrary
        # 注意：Python 中是 EditorActorSubsystem，C++ 中才是 EditorActorUtilitiesSubsystem
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        all_actors = actor_subsystem.get_all_level_actors()
        actor_count = len(all_actors)
        print(f"Actor Count   : {actor_count}")

        # 列出前 5 个 Actor
        print("\nFirst 5 actors in the level:")
        for i, actor in enumerate(all_actors[:5]):
            label = actor.get_actor_label()
            location = actor.get_actor_location()
            actor_class = actor.get_class().get_name()
            print(f"  [{i+1}] {label:<30} class={actor_class:<25} pos=({location.x:.0f}, {location.y:.0f}, {location.z:.0f})")
    else:
        print("No world loaded. Open a level in the editor first.")

    # ---- 项目信息 ----
    print(f"\nProject Name  : {unreal.SystemLibrary.get_game_name()}")
    print(f"Engine Version : {unreal.SystemLibrary.get_engine_version()}")

    # ---- 资源注册表状态 ----
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    # is_loading_assets() 在 5.1+ 可用，旧版本可能没有
    try:
        loading = registry.is_loading_assets()
        print(f"AssetRegistry Loading : {loading}")
    except AttributeError:
        pass

    # ---- 选中的资源 ----
    selected_assets = unreal.EditorUtilityLibrary.get_selected_assets()
    if selected_assets:
        print(f"\nSelected Assets ({len(selected_assets)}):")
        for asset in selected_assets:
            print(f"  {asset.get_name()}  ({asset.get_class().get_name()})")
    else:
        print("\nNo assets selected in Content Browser.")


if __name__ == "__main__":
    hello_editor()
