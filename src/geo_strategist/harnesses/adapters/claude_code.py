"""Claude Code adapter metadata for Skills launcher mode."""

from __future__ import annotations

from geo_strategist.harnesses.adapters import HarnessAdapter

ADAPTER = HarnessAdapter(
    harness="claude_code",
    command="claude -p --model sonnet --permission-mode bypassPermissions <launcher prompt>",
    skill_source=".claude/skills",
    automation_supported=True,
    command_template=(
        "claude", "-p",
        "--model", "sonnet",
        "--permission-mode", "bypassPermissions",
    ),
    prompt_mode="argument",
    timeout_seconds=7200,
)
