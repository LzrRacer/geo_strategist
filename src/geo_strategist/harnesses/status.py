"""Availability / login checks for the coding-agent harness CLIs."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class HarnessStatus:
    harness: str
    binary: str | None
    available: bool
    version: str | None = None
    logged_in: bool | None = None
    model: str | None = None
    non_interactive_supported: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 1, f"{type(exc).__name__}"
    return completed.returncode, (completed.stdout + "\n" + completed.stderr).strip()


def check_antigravity() -> HarnessStatus:
    binary = shutil.which("agy")
    if not binary:
        return HarnessStatus(harness="antigravity", binary=None, available=False,
                             notes=["agy (Antigravity) CLI not found on PATH"])
    _rc, version = _run(["agy", "--version"])
    return HarnessStatus(
        harness="antigravity", binary=binary, available=True,
        version=version.splitlines()[0] if version else None,
        logged_in=None,
        model=os.environ.get("C5_MODEL", "gemini-3.5-pro"),
        non_interactive_supported=True,
        notes=[
            "agy --print is available for opt-in automation via "
            "--auto-agentic-harness; manual launcher fallback remains under "
            "outputs/condition_proposals/live/manual_harness/ for C5.",
        ],
    )


def check_codex() -> HarnessStatus:
    binary = shutil.which("codex")
    if not binary:
        return HarnessStatus(harness="codex", binary=None, available=False,
                             notes=["codex CLI not found on PATH"])
    _rc, version = _run(["codex", "--version"])
    rc, login = _run(["codex", "login", "status"])
    logged_in = rc == 0 and "logged in" in login.lower()
    status = HarnessStatus(
        harness="codex", binary=binary, available=True,
        version=version.splitlines()[0] if version else None,
        logged_in=logged_in,
        model=os.environ.get("CODEX_MODEL", "default"),
        non_interactive_supported=True,
        notes=[
            "codex exec is available for opt-in automation via "
            "--auto-agentic-harness; manual prompt fallback remains under "
            "outputs/condition_proposals/live/manual_harness/.",
        ],
    )
    if not logged_in:
        status.notes.append("codex is not logged in; run `codex login` manually")
    return status


def check_claude_code() -> HarnessStatus:
    binary = shutil.which("claude")
    if not binary:
        return HarnessStatus(harness="claude_code", binary=None, available=False,
                             notes=["claude CLI not found on PATH"])
    _rc, version = _run(["claude", "--version"])
    return HarnessStatus(
        harness="claude_code", binary=binary, available=True,
        version=version.splitlines()[0] if version else None,
        logged_in=None,
        model=os.environ.get("CLAUDE_CODE_MODEL") or os.environ.get("ANTHROPIC_MODEL"),
        non_interactive_supported=True,
        notes=[
            "claude -p is available for opt-in automation via "
            "--auto-agentic-harness; manual prompt fallback remains under "
            "outputs/condition_proposals/live/manual_harness/.",
        ],
    )


def check_opencode() -> HarnessStatus:
    binary = shutil.which("opencode")
    if not binary:
        return HarnessStatus(harness="opencode", binary=None, available=False,
                             notes=["opencode CLI not found on PATH"])
    _rc, version = _run(["opencode", "--version"])
    rc, auth = _run(["opencode", "auth", "list"])
    has_opencode_credential = rc == 0 and "opencode" in auth.lower() and "0 credentials" not in auth
    has_opencode_env = bool(os.environ.get("OPENCODE_API_KEY"))
    status = HarnessStatus(
        harness="opencode", binary=binary, available=True,
        version=version.splitlines()[0] if version else None,
        logged_in=has_opencode_credential or has_opencode_env,
        model=f"{os.environ.get('OPENCODE_PROVIDER', 'opencode_go')}/"
              f"{os.environ.get('OPENCODE_GO_FAST_MODEL', 'deepseek-v4-flash')}",
        non_interactive_supported=True,
        notes=[],
    )
    if not (has_opencode_credential or has_opencode_env):
        status.notes.append(
            "opencode CLI has no stored opencode_go credential and "
            "OPENCODE_API_KEY is not present; C8 automation may fail until "
            "one of those authentication paths is configured."
        )
    else:
        status.notes.append(
            "opencode run is available for opt-in automation via "
            "--auto-agentic-harness; manual launcher fallback remains under "
            "outputs/condition_proposals/live/manual_harness/."
        )
    return status


def all_harness_statuses() -> list[HarnessStatus]:
    return [check_antigravity(), check_codex(), check_claude_code(), check_opencode()]
