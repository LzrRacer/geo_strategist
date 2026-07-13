"""Static safety guard for generated experiment code.

Complements the process isolation in interpreter.py: before any generated
code is executed, it is scanned for prohibited operations (destructive shell
commands, privilege escalation, subprocess/daemon spawning, credential reads,
and — when live search is disabled — network access). Absolute paths outside
the repository or the agent run directory are rejected so reads stay on
repo data/config/reference files and writes stay inside the run directory.
"""

from __future__ import annotations

import re
from pathlib import Path

# (pattern, violation code). Patterns are matched on the raw code text.
_DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"rm\s+-rf", "destructive_shell_command"),
    (r"\bsudo\b", "privilege_escalation"),
    (r"chmod\s+-R", "recursive_permission_change"),
    (r"\bshutil\.rmtree\b", "recursive_delete"),
    (r"\bos\.system\b", "shell_execution"),
    (r"\bsubprocess\b", "subprocess_spawn"),
    (r"\bos\.fork\b|\bos\.setsid\b|\bnohup\b|\bdaemon\b", "background_daemon"),
    (r"\bos\.remove\b|\bos\.unlink\b|\bos\.rmdir\b", "file_deletion"),
    (r"\bos\.chdir\b", "working_directory_escape"),
    (r"\beval\s*\(|\bexec\s*\(", "dynamic_code_execution"),
    (r"os\.environ|getenv", "environment_read"),
)

_NETWORK_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bimport\s+(socket|requests|urllib|http\.client|httpx|ftplib|smtplib)\b", "network_access_disabled"),
    (r"\bfrom\s+(socket|requests|urllib|http|httpx|ftplib|smtplib)\b", "network_access_disabled"),
)

_ABS_PATH_RE = re.compile(r"[\"'](/[^\"']+)[\"']")


def scan_generated_code(
    code: str,
    *,
    repo_root: str | Path,
    run_dir: str | Path,
    allow_live_search: bool = False,
) -> list[dict[str, str]]:
    """Return a list of violations; empty means the code may be executed."""

    violations: list[dict[str, str]] = []
    patterns = list(_DENY_PATTERNS)
    if not allow_live_search:
        patterns.extend(_NETWORK_PATTERNS)
    for pattern, code_name in patterns:
        match = re.search(pattern, code)
        if match:
            violations.append({"violation": code_name, "match": match.group(0)})

    repo = Path(repo_root).resolve()
    run = Path(run_dir).resolve()
    for raw in _ABS_PATH_RE.findall(code):
        path = Path(raw)
        try:
            path.relative_to(repo)
            continue
        except ValueError:
            pass
        try:
            path.relative_to(run)
            continue
        except ValueError:
            violations.append({"violation": "path_outside_allowed_roots", "match": raw})
    return violations


def audit_artifact_locations(run_dir: str | Path, artifact_paths: list[Path]) -> list[dict[str, str]]:
    """Post-run check that every produced artifact stayed inside the run dir."""

    run = Path(run_dir).resolve()
    problems: list[dict[str, str]] = []
    for path in artifact_paths:
        try:
            path.resolve().relative_to(run)
        except ValueError:
            problems.append({"violation": "artifact_outside_run_dir", "match": str(path)})
    return problems
