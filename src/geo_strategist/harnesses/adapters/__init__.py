"""Harness adapter metadata for agentic Skills launcher/execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PromptMode = Literal["none", "stdin", "argument"]


@dataclass(frozen=True)
class HarnessAdapter:
    harness: str
    command: str
    skill_source: str
    automation_supported: bool = False
    command_template: tuple[str, ...] = ()
    prompt_mode: PromptMode = "none"
    timeout_seconds: int = 1800

    def build_command(
        self,
        *,
        repo_root: Path,
        run_dir: Path,
        launcher_prompt_path: Path,
        manual_result_path: Path,
    ) -> list[str]:
        """Build a non-interactive command from the adapter template.

        Real subscription CLIs default to no template and therefore manual
        fallback. Tests and future stable adapters can provide templates with
        ``{repo_root}``, ``{run_dir}``, ``{launcher_prompt}``, and
        ``{manual_result}`` placeholders.
        """

        if not self.automation_supported or not self.command_template:
            return []
        values = {
            "repo_root": str(repo_root),
            "run_dir": str(run_dir),
            "launcher_prompt": str(launcher_prompt_path),
            "manual_result": str(manual_result_path),
        }
        return [part.format(**values) for part in self.command_template]


def adapter_for(harness: str) -> HarnessAdapter:
    if harness == "antigravity":
        from geo_strategist.harnesses.adapters.antigravity import ADAPTER
    elif harness == "codex":
        from geo_strategist.harnesses.adapters.codex import ADAPTER
    elif harness == "claude_code":
        from geo_strategist.harnesses.adapters.claude_code import ADAPTER
    elif harness == "opencode":
        from geo_strategist.harnesses.adapters.opencode import ADAPTER
    else:
        raise ValueError(f"unsupported agentic Skills harness: {harness}")
    return ADAPTER
