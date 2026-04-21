from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str
    path: Path


class SkillLoader:
    """按需加载 SKILL.md，避免把全部领域知识塞进 system prompt。"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = skills_dir
        self.skills = self._scan()

    def _scan(self) -> dict[str, Skill]:
        skills: dict[str, Skill] = {}
        if not self.skills_dir.exists():
            return skills

        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            text = path.read_text(encoding="utf-8", errors="replace")
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name") or path.parent.name
            description = meta.get("description") or ""
            skills[name] = Skill(name=name, description=description, body=body, path=path)
        return skills

    def descriptions(self) -> str:
        if not self.skills:
            return "(no skills found)"
        return "\n".join(f"- {skill.name}: {skill.description}" for skill in self.skills.values())

    def load(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "(none)"
            return f"Error: unknown skill '{name}'. Available: {available}"
        return f"<skill name=\"{skill.name}\" path=\"{skill.path}\">\n{skill.body}\n</skill>"

    def _parse_frontmatter(self, text: str) -> tuple[dict[str, str], str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
        if not match:
            return {}, text

        meta: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                meta[key.strip()] = value.strip().strip('"').strip("'")
        return meta, match.group(2).strip()
