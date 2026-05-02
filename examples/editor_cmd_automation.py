"""
UE 5.7 Editor-Cmd 自动化脚本 —— 纯 commandlet 模式，不启动编辑器。

本脚本使用真正的 commandlet 模式（-run=pythonscript），编辑器子系统不会初始化，
因此仅依赖 Asset Registry 和 SystemLibrary 等无需编辑器环境的 API。

================================================================================
运行方式（纯 commandlet，不启动编辑器）：
  UnrealEditor-Cmd.exe "C:/YourProject.uproject" ^
      -run=pythonscript ^
      -script="C:/path/to/editor_cmd_automation.py" ^
      -stdout -unattended -nullrhi -nosplash

子任务选择（通过 -cmdline 参数或修改脚本底部的 __main__ 逻辑）：
  UnrealEditor-Cmd.exe "..." -run=pythonscript -script="..." -cmdline="asset_inventory"
  UnrealEditor-Cmd.exe "..." -run=pythonscript -script="..." -cmdline="project_info"
  UnrealEditor-Cmd.exe "..." -run=pythonscript -script="..." -cmdline="all"

  不传则默认执行 all（全部子任务按顺序运行）。

================================================================================
API 可用性说明（-run=pythonscript 真 commandlet 模式）：

  ✅ 可用:
    - unreal.log / log_warning / log_error
    - AssetRegistryHelpers.get_asset_registry() → 遍历资源、读元数据
    - asset_data.asset_class_path / package_name / get_full_name()
    - SystemLibrary.get_engine_version / get_game_name / get_project_directory
    - unreal.load_asset()（基础 Python 便利函数，非 EditorAssetLibrary）

  ❌ 不可用（编辑器未初始化）:
    - EditorAssetLibrary（load/rename/delete/save/metadata 全部不可用）
    - EditorLevelLibrary / EditorActorSubsystem
    - get_editor_subsystem() 系列
    - EditorPythonScripting
    - 任何关卡/Actor 操作

================================================================================
包含的子任务（全部 commandlet 安全，纯只读）：

  1. asset_inventory    — 资源类型分布统计（Asset Registry 遍历）
  2. find_redirectors   — 查找 ObjectRedirector 残留
  3. project_info       — 项目/引擎/目录信息
  4. asset_size_report  — 按类型统计资源磁盘占用（Asset Registry + package_path）

每次运行默认执行 all，可通过 -cmdline=xxx 指定单个。
================================================================================
"""

import unreal
import json
import sys
import os
from collections import Counter, defaultdict


# ==============================================================================
#                           IMPORTANT NOTICE
# ==============================================================================
# This script is designed for TRUE COMMANDLET MODE (-run=pythonscript).
# It does NOT use EditorAssetLibrary, EditorLevelLibrary, or any
# get_editor_subsystem() calls because those require a fully booted editor.
#
# If you need to RENAME, DELETE, DUPLICATE, SAVE, or MODIFY assets, or
# interact with LEVELS/ACTORS, you must use -ExecutePythonScript instead:
#
#   UnrealEditor-Cmd.exe "Project.uproject" -ExecutePythonScript="script.py"
#
# That mode boots the full editor (slow startup) but exposes all editor APIs.
# ==============================================================================


# ==============================================================================
# 工具函数
# ==============================================================================

def log_section(title: str) -> None:
    """打印分隔标题。"""
    line = "=" * 70
    unreal.log(f"\n{line}\n  {title}\n{line}")


def log_json(data: object) -> None:
    """以 JSON 格式输出（同时到 stdout 和 UE Output Log）。"""
    rendered = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    print(rendered)
    unreal.log(rendered)


def get_asset_class_name(asset_data) -> str:
    """兼容 UE 5.1+ 的 asset_class_path.asset_name。"""
    return str(asset_data.asset_class_path.asset_name)


def get_package_name(asset_data) -> str:
    """去掉 .AssetName 后缀的纯路径名。"""
    package = str(asset_data.package_name)
    return package.split(".")[0] if "." in package else package


# ==============================================================================
# 子任务 1: 资源类型分布统计 (Asset Inventory)
# ==============================================================================

def asset_inventory(base_path: str = "/Game/", max_assets: int = 5000) -> dict:
    """
    扫描项目资源目录，输出完整的类型分布。

    纯 Asset Registry 遍历，不加载任何资源文件，速度快、内存安全。

    用途：快速了解项目资源构成，发现异常类型或数量异常。
    """
    log_section("Asset Inventory")

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    unreal.log("Waiting for asset registry scan to complete...")
    asset_registry.wait_for_completion()

    type_counter: Counter = Counter()
    path_by_type: dict[str, list[str]] = defaultdict(list)
    dir_counter: Counter = Counter()
    total_scanned = 0
    size_estimates: dict[str, int] = defaultdict(int)

    unreal.log(f"Scanning assets under {base_path} (max={max_assets})...")
    assets = asset_registry.get_assets_by_path(base_path, recursive=True)

    for asset_data in assets:
        if total_scanned >= max_assets:
            unreal.log_warning(f"Reached max_assets limit ({max_assets}), stopping scan.")
            break

        class_name = get_asset_class_name(asset_data)
        pkg_name = get_package_name(asset_data)

        type_counter[class_name] += 1

        # 每种类型存前 3 个示例
        if len(path_by_type[class_name]) < 3:
            path_by_type[class_name].append(pkg_name)

        # 统计目录分布（只取顶层，如 /Game/Characters）
        parts = pkg_name.split("/")
        if len(parts) >= 3:
            top_dir = "/".join(parts[:3])
            dir_counter[top_dir] += 1

        # 磁盘占用估算（仅 metadata，不加载 asset）
        try:
            disk_size = asset_data.get_tag_value("DiskSize")
            if disk_size:
                size_estimates[class_name] += int(disk_size)
        except Exception:
            pass

        total_scanned += 1
        if total_scanned % 1000 == 0:
            unreal.log(f"  ... scanned {total_scanned} assets, {len(type_counter)} unique types so far")

    # 按数量排序的类型分布
    top_types = []
    for class_name, count in type_counter.most_common(30):
        top_types.append({
            "type": class_name,
            "count": count,
            "pct": f"{count / total_scanned * 100:.1f}%",
            "examples": path_by_type.get(class_name, []),
        })

    # 按数量排序的目录分布
    top_dirs = [{"directory": d, "count": c} for d, c in dir_counter.most_common(15)]

    result = {
        "base_path": base_path,
        "scanned_count": total_scanned,
        "unique_types": len(type_counter),
        "unique_directories": len(dir_counter),
        "top_types": top_types,
        "top_directories": top_dirs,
    }

    log_json(result)
    return result


# ==============================================================================
# 子任务 2: 查找 ObjectRedirector（重定向器）
# ==============================================================================

def find_redirectors(base_path: str = "/Game/") -> dict:
    """
    查找项目中所有 ObjectRedirector。

    ObjectRedirector 是资源被移动/重命名后残留的"指针"。
    大量 redirector 会影响加载性能，应该定期清理。

    本脚本只查找和报告，不执行修复。
    修复需要用 EditorAssetLibrary（需要 -ExecutePythonScript 模式）。

    用途：定期审计项目健康度。
    """
    log_section("Find ObjectRedirectors")

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    asset_registry.wait_for_completion()

    redirectors = []
    assets = asset_registry.get_assets_by_path(base_path, recursive=True)

    for asset_data in assets:
        class_name = get_asset_class_name(asset_data)
        if class_name == "ObjectRedirector":
            pkg_name = get_package_name(asset_data)
            # 尝试获取 redirector 指向的目标
            destination = ""
            try:
                destination = str(asset_data.get_tag_value("DestinationObject"))
            except Exception:
                pass
            redirectors.append({
                "path": pkg_name,
                "destination": destination,
            })

    result = {
        "base_path": base_path,
        "total_redirectors": len(redirectors),
        "redirectors": redirectors[:200],
        "hint": (
            "To fix redirectors, use -ExecutePythonScript mode to access "
            "EditorAssetLibrary.rename_asset() or use the Content Browser "
            "Fix Up Redirectors command in the full editor."
        ),
    }

    log_json(result)
    return result


# ==============================================================================
# 子任务 3: 项目信息诊断
# ==============================================================================

def project_info() -> dict:
    """
    输出当前项目的基础信息。

    仅使用 SystemLibrary（引擎级 API），不依赖编辑器子系统。
    """
    log_section("Project Info")

    engine_version = unreal.SystemLibrary.get_engine_version()
    project_name = unreal.SystemLibrary.get_game_name()
    project_dir = unreal.SystemLibrary.get_project_directory()
    content_dir = unreal.SystemLibrary.get_project_content_directory()

    # Asset Registry 状态
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    registry_loading = False
    try:
        registry_loading = asset_registry.is_loading_assets()
    except AttributeError:
        pass

    # 尝试获取总体资源数量
    total_assets = 0
    try:
        all_assets = asset_registry.get_assets_by_path("/Game/", recursive=True)
        total_assets = len(all_assets)
    except Exception:
        pass

    result = {
        "engine_version": engine_version,
        "project_name": project_name,
        "project_directory": project_dir,
        "content_directory": content_dir,
        "asset_registry_loading": registry_loading,
        "total_assets_under_game": total_assets,
        "mode": "commandlet (-run=pythonscript)",
        "note": (
            "Running in true commandlet mode. EditorAssetLibrary, "
            "EditorLevelLibrary, and get_editor_subsystem() are NOT available. "
            "Only Asset Registry and SystemLibrary APIs are accessible."
        ),
    }

    log_json(result)
    return result


# ==============================================================================
# 子任务 4: 资源磁盘占用报告
# ==============================================================================

def asset_size_report(base_path: str = "/Game/", max_assets: int = 3000) -> dict:
    """
    按资源类型统计磁盘占用（估算，基于 Asset Registry metadata）。

    从 Asset Registry 的 tags 中读取 DiskSize 等元数据，不需要加载资源文件。
    注意：并非所有资源类型都注册了 DiskSize tag，结果可能不完整。

    用途：发现体积最大的资源类型，辅助优化决策。
    """
    log_section("Asset Size Report")

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    asset_registry.wait_for_completion()

    type_size: dict[str, int] = defaultdict(int)
    type_count: Counter = Counter()
    all_sizes: list[tuple[str, str, int]] = []  # (class, path, size)
    total_scanned = 0
    assets_with_size = 0

    unreal.log(f"Collecting size metadata under {base_path} (max={max_assets})...")
    assets = asset_registry.get_assets_by_path(base_path, recursive=True)

    for asset_data in assets:
        if total_scanned >= max_assets:
            break

        class_name = get_asset_class_name(asset_data)
        pkg_name = get_package_name(asset_data)
        type_count[class_name] += 1

        # DiskSize tag —— 不是所有资源都有
        try:
            raw = asset_data.get_tag_value("DiskSize")
            if raw:
                size_bytes = int(raw)
                type_size[class_name] += size_bytes
                assets_with_size += 1
                all_sizes.append((class_name, pkg_name, size_bytes))
        except Exception:
            pass

        total_scanned += 1

    # 按类型总大小排序
    sorted_types = sorted(type_size.items(), key=lambda x: x[1], reverse=True)
    top_types_by_size = []
    for class_name, total_size in sorted_types[:20]:
        top_types_by_size.append({
            "type": class_name,
            "count": type_count[class_name],
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "avg_size_mb": round(total_size / type_count[class_name] / (1024 * 1024), 2) if type_count[class_name] else 0,
        })

    # Top 20 最大的单个资源
    all_sizes.sort(key=lambda x: x[2], reverse=True)
    largest_assets = []
    for cls_name, path, size in all_sizes[:30]:
        largest_assets.append({
            "type": cls_name,
            "path": path,
            "size_mb": round(size / (1024 * 1024), 2),
        })

    total_disk_mb = round(sum(type_size.values()) / (1024 * 1024), 2)

    result = {
        "base_path": base_path,
        "scanned_count": total_scanned,
        "assets_with_size_tag": assets_with_size,
        "total_disk_mb_estimated": total_disk_mb,
        "top_types_by_size": top_types_by_size,
        "largest_individual_assets": largest_assets,
        "note": (
            "Sizes are estimates from Asset Registry DiskSize tags. "
            "Not all asset types expose this tag. For accurate sizes, "
            "use the full editor's Reference Viewer or audit tools."
        ),
    }

    log_json(result)
    return result


# ==============================================================================
# 子任务 5: 查找命名不规范/空目录
# ==============================================================================

def directory_audit(base_path: str = "/Game/") -> dict:
    """
    审计 Content Browser 目录结构：
      - 找出空目录（无任何资源）
      - 找出过于扁平的目录（资源都在根层级，没有子目录）
      - 列出所有顶层子目录

    全部通过 Asset Registry 完成，不涉及文件系统操作。
    """
    log_section("Directory Audit")

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    asset_registry.wait_for_completion()

    # 收集所有子目录路径
    # 注意：UE 5.7 中参数名是 recurse（不是 recursive）
    all_sub_paths = set()
    sub_paths = asset_registry.get_sub_paths(base_path, recurse=True)
    all_sub_paths = set(str(p) for p in sub_paths)

    # 收集每个目录下的资源数量
    dir_asset_count: Counter = Counter()
    assets = asset_registry.get_assets_by_path(base_path, recursive=True)

    for asset_data in assets:
        pkg_name = get_package_name(asset_data)
        parent_dir = "/".join(pkg_name.split("/")[:-1])
        dir_asset_count[parent_dir] += 1

    # 找出空目录（在子路径中存在但无资源）
    empty_dirs = sorted([d for d in all_sub_paths if dir_asset_count.get(d, 0) == 0])

    # 找出顶层子目录及其资源数
    top_dirs = asset_registry.get_sub_paths(base_path, recurse=False)
    top_dir_stats = []
    for td in top_dirs:
        td_str = str(td)
        # 递归计算该目录下所有资源
        count = 0
        for d, c in dir_asset_count.items():
            if d.startswith(td_str):
                count += c
        top_dir_stats.append({"directory": td_str, "total_assets": count})

    top_dir_stats.sort(key=lambda x: x["total_assets"], reverse=True)

    result = {
        "base_path": base_path,
        "total_directories": len(all_sub_paths),
        "directories_with_assets": len(dir_asset_count),
        "empty_directories": len(empty_dirs),
        "empty_directory_list": empty_dirs[:50],
        "top_level_directories": top_dir_stats,
    }

    log_json(result)
    return result


# ==============================================================================
# 子任务 6: 按 Tag 搜索资源
# ==============================================================================

def search_by_tag(tag_name: str = "", tag_value: str = "", base_path: str = "/Game/") -> dict:
    """
    通过 Asset Registry tag 搜索资源。

    如果同时指定 tag_name 和 tag_value，精确匹配。
    如果只指定 tag_name，列出所有有该 tag 的资源及其值。
    如果都不指定，列出 Metadata 中 tag 值的统计分布。

    用途：查找带有特定 Metadata 标签的资源（非 EditorAssetLibrary 的元数据，
          而是 Asset Registry 级别的 tags）。
    """
    if not tag_name:
        label = "tags overview"
    elif tag_name and not tag_value:
        label = f"assets with tag '{tag_name}'"
    else:
        label = f"tag '{tag_name}' = '{tag_value}'"

    log_section(f"Search by Tag: {label}")

    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()
    asset_registry.wait_for_completion()

    assets = asset_registry.get_assets_by_path(base_path, recursive=True)
    matches = []
    all_tags_distribution: Counter = Counter()
    total_scanned = 0

    for asset_data in assets:
        pkg_name = get_package_name(asset_data)
        class_name = get_asset_class_name(asset_data)

        if tag_name:
            try:
                value = str(asset_data.get_tag_value(tag_name))
                if not tag_value or value.lower() == tag_value.lower():
                    matches.append({
                        "path": pkg_name,
                        "type": class_name,
                        "tag": tag_name,
                        "value": value,
                    })
            except Exception:
                pass
        else:
            # 尝试获取所有已知 tags
            try:
                # AssetData 没有直接列出所有 tag 的方法，
                # 但我们可以尝试几个常见的 tag
                for known_tag in [
                    "DiskSize", "Packages", "ChunkID", "HasEditorData",
                    "SourceAsset", "NativeClass", "GeneratedClass",
                ]:
                    try:
                        val = asset_data.get_tag_value(known_tag)
                        if val:
                            all_tags_distribution[known_tag] += 1
                    except Exception:
                        pass
            except Exception:
                pass

        total_scanned += 1

    if tag_name:
        result = {
            "tag_name": tag_name,
            "tag_value_filter": tag_value or "(any)",
            "scanned": total_scanned,
            "matched": len(matches),
            "results": matches[:200],
        }
    else:
        result = {
            "scanned": total_scanned,
            "common_tags": [
                {"tag": t, "asset_count": c}
                for t, c in all_tags_distribution.most_common(20)
            ],
            "hint": (
                "Only a few Asset Registry tags are queried here. "
                "For full metadata inspection, use the editor GUI "
                "or EditorAssetLibrary.get_metadata_tag_values() "
                "in -ExecutePythonScript mode."
            ),
        }

    log_json(result)
    return result


# ==============================================================================
# 主入口
# ==============================================================================

# 所有 commandlet 安全的子任务
TASKS = {
    "asset_inventory":  ("Asset type distribution scan", asset_inventory),
    "find_redirectors": ("Find ObjectRedirectors in the project", find_redirectors),
    "project_info":     ("Project, engine, and directory info", project_info),
    "asset_size_report": ("Disk size estimates by asset type", asset_size_report),
    "directory_audit":  ("Directory structure audit (empty dirs, etc.)", directory_audit),
    "search_by_tag":    ("Search assets by Asset Registry tag", search_by_tag),
}


def print_usage() -> None:
    """打印命令行帮助。"""
    print("\n" + "=" * 70)
    print("  UE 5.7 Commandlet-Safe Automation Script")
    print("=" * 70)
    print()
    print("Run with:")
    print('  UnrealEditor-Cmd.exe "Project.uproject" -run=pythonscript \\')
    print('      -script="this_script.py" -stdout -unattended -nullrhi -nosplash')
    print()
    print("Select tasks via -cmdline=xxx:")
    for name, (desc, _) in TASKS.items():
        print(f"  {name:<22} {desc}")
    print(f"  {'all':<22} Run all tasks in sequence (default)")
    print()
    print("Compatibility:")
    print("  This script uses ONLY Asset Registry and SystemLibrary APIs.")
    print("  It works in TRUE commandlet mode (-run=pythonscript).")
    print("  No editor subsystems are required.")
    print("=" * 70)
    print()


def parse_cmdline_arg() -> str:
    """解析 -cmdline=xxx 参数。"""
    for arg in sys.argv:
        if arg.startswith("-cmdline="):
            return arg.split("=", 1)[1].strip()
    return "all"


def run_task(task_name: str) -> None:
    """执行单个命名子任务。"""
    if task_name not in TASKS:
        print(f"Unknown task: {task_name}")
        print_usage()
        return

    _, func = TASKS[task_name]
    func()


def run_all() -> None:
    """按顺序执行所有子任务。"""
    unreal.log("\n" + "█" * 70)
    unreal.log("  UE 5.7 Commandlet Automation — Running All Tasks")
    unreal.log("  Mode: -run=pythonscript (true commandlet, no editor subsystems)")
    unreal.log("█" * 70)

    for name, (desc, func) in TASKS.items():
        unreal.log(f"\n>>> Task: {name} — {desc}")
        try:
            func()
        except Exception as exc:
            unreal.log_error(f"Task '{name}' failed: {exc}")
            import traceback
            unreal.log_error(traceback.format_exc())

    unreal.log("\n" + "█" * 70)
    unreal.log("  All tasks completed.")
    unreal.log("█" * 70)


if __name__ == "__main__":
    task = parse_cmdline_arg()

    if task in ("-h", "--help", "help"):
        print_usage()
        sys.exit(0)

    print_usage()

    if task == "all":
        run_all()
    else:
        run_task(task)
