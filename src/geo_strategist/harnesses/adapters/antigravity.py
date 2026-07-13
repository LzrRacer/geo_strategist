"""Antigravity adapter metadata for Skills launcher mode."""

from __future__ import annotations

from geo_strategist.harnesses.adapters import HarnessAdapter

ADAPTER = HarnessAdapter(
    harness="antigravity",
    command="agy --print <launcher prompt>",
    skill_source=".agents/skills",
    automation_supported=True,
    command_template=("agy", "--print"),
    prompt_mode="argument",
    timeout_seconds=7200,
)
