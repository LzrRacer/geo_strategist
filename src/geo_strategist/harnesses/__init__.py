"""Coding-agent harness wrappers (Codex, Claude Code, OpenCode).

These wrappers check availability/login state and produce manual handoff
prompt files when a harness cannot be driven non-interactively. They never
fall back to deterministic rankings: an unrunnable harness condition is
reported as ``waiting_for_manual_harness``.
"""

from geo_strategist.harnesses.status import (
    HarnessStatus,
    check_claude_code,
    check_codex,
    check_opencode,
)

__all__ = ["HarnessStatus", "check_claude_code", "check_codex", "check_opencode"]
