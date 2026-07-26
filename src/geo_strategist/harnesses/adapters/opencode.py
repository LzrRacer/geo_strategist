"""OpenCode adapter metadata for Skills launcher mode."""

from __future__ import annotations

from geo_strategist.harnesses.adapters import HarnessAdapter

ADAPTER = HarnessAdapter(
    harness="opencode",
    command="opencode run --model opencode_go/deepseek-v4-flash <launcher prompt>",
    skill_source=".claude/skills",
    automation_supported=True,
    command_template=(
        "opencode", "run",
        "--model", "opencode_go/deepseek-v4-flash",
    ),
    prompt_mode="argument",
    timeout_seconds=7200,
)
