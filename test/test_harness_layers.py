from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from prompt_toolkit.document import Document

from uedev.background import BackgroundManager
from uedev.context import compact_locally, estimate_tokens, micro_compact
from uedev.llm import ChatMessage, ToolCall, _serialize_message
from uedev.loop import (
    SLASH_COMMANDS,
    AgentOptions,
    AgentRuntime,
    SlashCommandCompleter,
    render_chat_banner,
    render_slash_help,
    requires_tool_action,
)
from uedev.skills import SkillLoader
from uedev.tasks import TaskManager
from uedev.team import MessageBus, TeamManager
from uedev.tool_specs import get_tool_names
from uedev.workspace import edit_file, read_file, write_file
from uedev.worktrees import WorktreeManager


class WorkspaceToolTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_read_write_edit_are_sandboxed 场景。
    def test_read_write_edit_are_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_file(root, "a.txt", "hello")
            edit_file(root, "a.txt", "hello", "world")

            self.assertEqual(read_file(root, "a.txt"), "world")
            with self.assertRaises(ValueError):
                read_file(root, "../outside.txt")

    # 测试函数：由 unittest 执行，验证 edit_file 工具兼容 edits 列表和 camelCase 字段。
    def test_edit_file_tool_accepts_edits_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_file(root, "test/main.py", "print('hello')\n")
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=root,
                timeout_seconds=120,
                verbose=False,
            )
            runtime = AgentRuntime(options)

            result = runtime.tools["edit_file"](
                {
                    "path": "test/main.py",
                    "edits": [{"oldText": "print('hello')", "newText": "print('updated')"}],
                }
            )

            self.assertIn("Edited", result)
            self.assertEqual(read_file(root, "test/main.py"), "print('updated')")


class SkillLoaderTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_load_skill_from_frontmatter 场景。
    def test_load_skill_from_frontmatter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            skill_dir = Path(temp) / "skills" / "ue"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: ue\ndescription: UE workflow\n---\nUse UE Python.",
                encoding="utf-8",
            )

            loader = SkillLoader(Path(temp) / "skills")
            self.assertIn("ue", loader.descriptions())
            self.assertIn("Use UE Python", loader.load("ue"))


class ContextTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_micro_compact_old_tool_results 场景。
    def test_micro_compact_old_tool_results(self) -> None:
        messages = [ChatMessage(role="user", content=f"Tool result for: read_file\n{'x' * 5000}") for _ in range(10)]

        micro_compact(messages, keep_recent=2, max_content=100)

        self.assertIn("compacted", messages[0].content)
        self.assertGreater(estimate_tokens(messages), 0)

    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_compact_locally_saves_transcript 场景。
    def test_compact_locally_saves_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            messages = [ChatMessage(role="user", content="hello")]
            compacted = compact_locally(messages, Path(temp), "test")

            self.assertEqual(len(compacted), 1)
            self.assertTrue(list(Path(temp).glob("transcript_*.jsonl")))

    # 测试函数：由 unittest 执行，验证包含原生工具调用的消息也能估算上下文。
    def test_estimate_tokens_accepts_native_tool_calls(self) -> None:
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
            )
        ]

        self.assertGreater(estimate_tokens(messages), 0)


class LlmMessageTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 assistant tool_calls 会序列化成 OpenAI 原生工具调用格式。
    def test_serialize_assistant_tool_call_message(self) -> None:
        message = ChatMessage(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
        )

        serialized = _serialize_message(message)

        self.assertEqual(serialized["role"], "assistant")
        self.assertEqual(serialized["tool_calls"][0]["function"]["name"], "read_file")

    # 测试函数：由 unittest 执行，验证 tool 结果消息会携带 tool_call_id。
    def test_serialize_tool_result_message(self) -> None:
        message = ChatMessage(role="tool", content="ok", tool_call_id="call_1", name="read_file")

        serialized = _serialize_message(message)

        self.assertEqual(serialized["role"], "tool")
        self.assertEqual(serialized["tool_call_id"], "call_1")


class TaskAndTeamTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_task_dependencies_clear_on_completion 场景。
    def test_task_dependencies_clear_on_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = TaskManager(Path(temp) / ".tasks")
            manager.create("A")
            manager.create("B", blocked_by=[1])
            manager.update(1, status="completed")

            self.assertNotIn("blockedBy=[1]", manager.list_all())

    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_team_message_and_claim 场景。
    def test_team_message_and_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            task_manager = TaskManager(root / ".tasks")
            bus = MessageBus(root / ".team")
            team = TeamManager(root / ".team", task_manager, bus)
            task_manager.create("Ready task")
            team.spawn("alice", "coder")

            self.assertIn("alice", team.list_all())
            self.assertIn("claimed task #1", team.claim_ready_task("alice"))
            bus.send("lead", "alice", "hello")
            self.assertEqual(bus.read_inbox("alice")[0]["content"], "hello")


class BackgroundTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_background_check_empty 场景。
    def test_background_check_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = BackgroundManager(Path(temp))
            self.assertIn("No background", manager.check())


class SlashCommandTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_help_includes_command_descriptions 场景。
    def test_help_includes_command_descriptions(self) -> None:
        help_text = render_slash_help()

        self.assertIn("/help", help_text)
        self.assertIn("Show available chat slash commands.", help_text)
        self.assertIn("/ue doctor", help_text)
        self.assertIn("Inspect Unreal Engine project and editor configuration.", help_text)
        self.assertIn("/clear", help_text)
        self.assertIn("Reset the current chat conversation context.", help_text)

    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_chat_banner_includes_runtime_details 场景。
    def test_chat_banner_includes_runtime_details(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=False,
                cwd=Path(temp),
                timeout_seconds=120,
                verbose=False,
            )

            banner = render_chat_banner(options)

            self.assertIn("uedev", banner)
            self.assertIn("model:", banner)
            self.assertIn(str(Path(temp)), banner)
            self.assertIn('Type "/" for commands', banner)

    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_slash_completer_returns_all_commands_for_slash 场景。
    def test_slash_completer_returns_all_commands_for_slash(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/"), None))

        self.assertEqual([completion.text for completion in completions], [command for command, _ in SLASH_COMMANDS])
        self.assertIn("Show available chat slash commands.", str(completions[0].display_meta))

    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_slash_completer_filters_by_prefix 场景。
    def test_slash_completer_filters_by_prefix(self) -> None:
        completions = list(SlashCommandCompleter().get_completions(Document("/ue"), None))

        self.assertEqual([completion.text for completion in completions], ["/ue doctor"])


class ToolRequirementTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证低意图聊天不会被误判为必须使用工具。
    def test_low_intent_chat_does_not_require_tool_action(self) -> None:
        self.assertFalse(requires_tool_action("test"))
        self.assertFalse(requires_tool_action("hello"))
        self.assertFalse(requires_tool_action("测试"))

    # 测试函数：由 unittest 执行，验证明确的执行类测试请求仍然必须使用工具。
    def test_explicit_test_run_requires_tool_action(self) -> None:
        self.assertTrue(requires_tool_action("run tests"))
        self.assertTrue(requires_tool_action("pytest"))


class ToolSpecTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证原生工具 schema 和运行时工具处理器保持同名覆盖。
    def test_native_tool_specs_match_runtime_handlers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            options = AgentOptions(
                task="",
                max_steps=1,
                auto_approve=True,
                cwd=Path(temp),
                timeout_seconds=120,
                verbose=False,
            )
            runtime = AgentRuntime(options)

            self.assertEqual(get_tool_names(), set(runtime.tools))


class WorktreeTests(unittest.TestCase):
    # 测试函数：由 unittest 执行，验证 harness 各层行为回归测试 中的 test_worktree_index_starts_empty 场景。
    def test_worktree_index_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = TaskManager(root / ".tasks")
            manager = WorktreeManager(root, root / ".worktrees", tasks)

            self.assertIn("No managed", manager.list_all())


if __name__ == "__main__":
    unittest.main()
