from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from uedev.protocol import FinalAction, ToolAction, parse_agent_action
from uedev.tasks import TodoManager
from uedev.ue import build_python_script, discover_ue, run_ue_python


class ProtocolTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 协议、todo 和 UE helper 回归测试 中的 test_parse_tool_action 场景。
    def test_parse_tool_action(self) -> None:
        action = parse_agent_action(
            '{"type":"tool","name":"todo_list","input":{}}'
        )

        self.assertIsInstance(action, ToolAction)
        self.assertEqual(action.name, "todo_list")

    # 测试函数：由 unittest 执行，验证缺少 type 的 answer 简写会被兼容为 final action。
    def test_parse_answer_shorthand_as_final_action(self) -> None:
        action = parse_agent_action('{"answer":"test"}')

        self.assertIsInstance(action, FinalAction)
        self.assertEqual(action.answer, "test")

    # 测试函数：由 unittest 执行，验证多个 JSON action 粘连时会解析第一个可执行 action。
    def test_parse_first_action_from_concatenated_json_objects(self) -> None:
        action = parse_agent_action(
            '{"type":"tool","name":"list_files","input":{"path":"test"}}{"type":"final","answer":"done"}'
        )

        self.assertIsInstance(action, ToolAction)
        self.assertEqual(action.name, "list_files")


class TodoManagerTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 协议、todo 和 UE helper 回归测试 中的 test_update_rejects_multiple_in_progress_items 场景。
    def test_update_rejects_multiple_in_progress_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = TodoManager(Path(temp) / ".agent")

            with self.assertRaises(ValueError):
                manager.update(
                    [
                        {"id": "1", "text": "A", "status": "in_progress"},
                        {"id": "2", "text": "B", "status": "in_progress"},
                    ]
                )

    # 测试函数：由 unittest 执行，验证 协议、todo 和 UE helper 回归测试 中的 test_update_persists_items 场景。
    def test_update_persists_items(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = TodoManager(Path(temp) / ".agent")
            manager.update([{"id": "1", "text": "检查 UE 配置", "status": "completed"}])

            self.assertIn("检查 UE 配置", manager.render_current())


class UeTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 协议、todo 和 UE helper 回归测试 中的 test_discover_project_from_environment 场景。
    def test_discover_project_from_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "Demo.uproject"
            project.write_text("{}", encoding="utf-8")

            old = __import__("os").environ.get("UE_PROJECT_PATH")
            __import__("os").environ["UE_PROJECT_PATH"] = str(project)
            try:
                result = discover_ue(root)
                self.assertEqual(result.project_path, project.resolve())
            finally:
                if old is None:
                    __import__("os").environ.pop("UE_PROJECT_PATH", None)
                else:
                    __import__("os").environ["UE_PROJECT_PATH"] = old

    # 测试函数：由 unittest 执行，验证 协议、todo 和 UE helper 回归测试 中的 test_build_script_wraps_json_errors 场景。
    def test_build_script_wraps_json_errors(self) -> None:
        wrapped = build_python_script("custom", "print('ok')")

        self.assertIn("traceback.format_exc", wrapped)
        self.assertIn("print('ok')", wrapped)

    # 测试函数：由 unittest 执行，验证 协议、todo 和 UE helper 回归测试 中的 test_run_python_dry_run_requires_editor_path 场景。
    def test_run_python_dry_run_requires_editor_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project = root / "Demo.uproject"
            editor = root / "UnrealEditor-Cmd.exe"
            project.write_text("{}", encoding="utf-8")
            editor.write_text("", encoding="utf-8")

            os_module = __import__("os")
            old_project = os_module.environ.get("UE_PROJECT_PATH")
            old_editor = os_module.environ.get("UE_EDITOR_CMD_PATH")
            os_module.environ["UE_PROJECT_PATH"] = str(project)
            os_module.environ["UE_EDITOR_CMD_PATH"] = str(editor)
            try:
                result = run_ue_python(root, root / ".agent", "print('ok')", execute=False)
                self.assertFalse(result.executed)
                self.assertIn("UnrealEditor-Cmd.exe", result.command)
            finally:
                if old_project is None:
                    os_module.environ.pop("UE_PROJECT_PATH", None)
                else:
                    os_module.environ["UE_PROJECT_PATH"] = old_project
                if old_editor is None:
                    os_module.environ.pop("UE_EDITOR_CMD_PATH", None)
                else:
                    os_module.environ["UE_EDITOR_CMD_PATH"] = old_editor


if __name__ == "__main__":
    unittest.main()
