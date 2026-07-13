"""Codex adapter metadata for Skills launcher mode."""

from __future__ import annotations

from geo_strategist.harnesses.adapters import HarnessAdapter

ADAPTER = HarnessAdapter(
    harness="codex",
    command="codex --ask-for-approval never exec --cd <repo> --sandbox workspace-write -",
    skill_source=".agents/skills",
    automation_supported=True,
    command_template=(
        "codex",
        "--ask-for-approval", "never",
        "exec",
        "--cd", "{repo_root}",
        "--sandbox", "workspace-write",
        "-",
    ),
    prompt_mode="stdin",
    timeout_seconds=7200,
)
