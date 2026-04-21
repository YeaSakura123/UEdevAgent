from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from myagent.background import BackgroundManager
from myagent.context import compact_locally, estimate_tokens, micro_compact
from myagent.llm import ChatMessage
from myagent.skills import SkillLoader
from myagent.tasks import TaskManager
from myagent.team import MessageBus, TeamManager
from myagent.workspace import edit_file, read_file, write_file
from myagent.worktrees import WorktreeManager


class WorkspaceToolTests(unittest.TestCase):
    def test_read_write_edit_are_sandboxed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_file(root, "a.txt", "hello")
            edit_file(root, "a.txt", "hello", "world")

            self.assertEqual(read_file(root, "a.txt"), "world")
            with self.assertRaises(ValueError):
                read_file(root, "../outside.txt")


class SkillLoaderTests(unittest.TestCase):
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
    def test_micro_compact_old_tool_results(self) -> None:
        messages = [ChatMessage(role="user", content=f"Tool result for: read_file\n{'x' * 5000}") for _ in range(10)]

        micro_compact(messages, keep_recent=2, max_content=100)

        self.assertIn("compacted", messages[0].content)
        self.assertGreater(estimate_tokens(messages), 0)

    def test_compact_locally_saves_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            messages = [ChatMessage(role="user", content="hello")]
            compacted = compact_locally(messages, Path(temp), "test")

            self.assertEqual(len(compacted), 1)
            self.assertTrue(list(Path(temp).glob("transcript_*.jsonl")))


class TaskAndTeamTests(unittest.TestCase):
    def test_task_dependencies_clear_on_completion(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = TaskManager(Path(temp) / ".tasks")
            manager.create("A")
            manager.create("B", blocked_by=[1])
            manager.update(1, status="completed")

            self.assertNotIn("blockedBy=[1]", manager.list_all())

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
    def test_background_check_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manager = BackgroundManager(Path(temp))
            self.assertIn("No background", manager.check())


class WorktreeTests(unittest.TestCase):
    def test_worktree_index_starts_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tasks = TaskManager(root / ".tasks")
            manager = WorktreeManager(root, root / ".worktrees", tasks)

            self.assertIn("No managed", manager.list_all())


if __name__ == "__main__":
    unittest.main()
