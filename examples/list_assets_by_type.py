"""
UE Python API 示例：统计项目资源类型分布

这个脚本展示以下 UE Python API 的用法：
- unreal.AssetRegistryHelpers.get_asset_registry()  — 获取资源注册表
- asset_registry.get_assets_by_path()                — 按路径遍历资源
- asset_data.asset_class_path.asset_name            — 获取资源类型名（UE 5.1+）
- asset_data.package_name                            — 获取资源包名

运行方式：
  1. UE 编辑器内：Window → Python Console，粘贴运行
  2. 命令行：UnrealEditor-Cmd.exe "C:/YourProject.uproject" -ExecutePythonScript="path/to/this/file.py"
  3. 通过 uedev：uedev ue run-python -s examples/list_assets_by_type.py --execute
"""

import unreal
import json
from collections import Counter


def list_assets_by_type(base_path="/Game/", max_assets=500, top_n=20):
    """
    遍历指定路径下的资源，按类型统计并输出 Top N。

    Args:
        base_path: 资源根路径，默认 /Game/
        max_assets: 最多扫描的资源数（防止大项目卡死）
        top_n:    输出前 N 种类型

    Returns:
        dict: {"total": 总数, "types": {类型名: 数量}, "top_10_largest": [...]}
    """
    # 1. 获取全局 AssetRegistry
    asset_registry = unreal.AssetRegistryHelpers.get_asset_registry()

    # 2. 等待资源扫描完成（重要！不等待可能拿到空结果）
    unreal.log("Waiting for asset registry to finish scanning...")
    asset_registry.wait_for_completion()
    unreal.log("Asset registry scan complete.")

    # 3. 遍历指定路径下的所有资源
    type_counter = Counter()
    sample_assets = {}  # 每种类型存一个示例路径，方便后续查看
    count = 0

    asset_list = asset_registry.get_assets_by_path(base_path, recursive=True)

    for asset_data in asset_list:
        if count >= max_assets:
            unreal.log_warning(f"Reached max_assets limit ({max_assets}), stopping scan.")
            break

        # UE 5.1+：asset_class 废弃，改用 asset_class_path.asset_name
        class_name = str(asset_data.asset_class_path.asset_name)
        package_name = str(asset_data.package_name)

        type_counter[class_name] += 1

        # 每种类型只存第一个示例
        if class_name not in sample_assets:
            sample_assets[class_name] = package_name

        count += 1

    # 4. 组装结果
    result = {
        "base_path": base_path,
        "scanned_count": count,
        "unique_types": len(type_counter),
        "types": dict(type_counter.most_common(top_n)),
        "sample_assets": sample_assets,
    }

    # 5. 输出（编辑器内会显示在 Output Log；命令行会打印到 stdout）
    print(json.dumps(result, indent=2, ensure_ascii=False))
    unreal.log(f"\n===== Asset Summary for {base_path} =====")
    unreal.log(f"Total scanned : {count}")
    unreal.log(f"Unique types  : {len(type_counter)}")
    unreal.log(f"\nTop {min(top_n, len(type_counter))} types:")
    for type_name, cnt in type_counter.most_common(top_n):
        example = sample_assets.get(type_name, "N/A")
        unreal.log(f"  {cnt:>6}  {type_name:<40}  e.g. {example}")

    return result


if __name__ == "__main__":
    list_assets_by_type(base_path="/Game/", max_assets=1000, top_n=15)
