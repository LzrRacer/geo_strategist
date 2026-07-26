"""Sandboxed executor for LLM/agent-generated experiment code.

Adapted from AI Scientist-v2 ``treesearch/interpreter.py``: code is written to
``agent_file_name`` inside an isolated working directory and executed in a
separate process with a hard timeout, capturing stdout/stderr and the
exception type. This adaptation uses ``subprocess`` instead of a long-lived
``multiprocessing`` REPL session (each node run is independent), scrubs
credential-bearing environment variables, and never runs shell commands.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_CREDENTIAL_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "APP_ID", "CLIENT_ID")


@dataclass(frozen=True)
class ExecutionResult:
    """Result of executing a generated code file (cf. reference ExecutionResult)."""

    term_out: list[str] = field(default_factory=list)
    exec_time: float = 0.0
    exc_type: str | None = None
    returncode: int | None = None

    @property
    def stderr_tail(self) -> str:
        return "\n".join(self.term_out[-30:])


def _scrubbed_env() -> dict[str, str]:
    env = {}
    for name, value in os.environ.items():
        if any(marker in name.upper() for marker in _CREDENTIAL_MARKERS):
            continue
        env[name] = value
    return env


def _exc_type_from_output(lines: list[str]) -> str | None:
    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            continue
        head = stripped.split(":")[0].split(" ")[0]
        if head.endswith("Error") or head.endswith("Exception") or head == "KeyboardInterrupt":
            return head
    return "ExecutionError"


class Interpreter:
    """Run one generated Python file per call inside ``working_dir``."""

    def __init__(
        self,
        working_dir: str | Path,
        *,
        timeout: int = 120,
        agent_file_name: str = "runfile.py",
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        self.timeout = timeout
        self.agent_file_name = agent_file_name

    def run(self, code: str) -> ExecutionResult:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        runfile = self.working_dir / self.agent_file_name
        runfile.write_text(code, encoding="utf-8")
        start = time.time()
        try:
            completed = subprocess.run(
                [sys.executable, self.agent_file_name],
                cwd=str(self.working_dir),
                env=_scrubbed_env(),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed = time.time() - start
            out = ((exc.stdout or "") + "\n" + (exc.stderr or "")).splitlines()
            out.append(f"TimeoutError: execution exceeded the time limit of {self.timeout}s")
            return ExecutionResult(term_out=out, exec_time=elapsed, exc_type="TimeoutError", returncode=None)
        elapsed = time.time() - start
        out = (completed.stdout + "\n" + completed.stderr).splitlines()
        if completed.returncode == 0:
            exc_type = None
        else:
            exc_type = _exc_type_from_output(completed.stderr.splitlines())
        out.append(f"Execution time: {elapsed:.2f}s (limit {self.timeout}s)")
        return ExecutionResult(
            term_out=out,
            exec_time=elapsed,
            exc_type=exc_type,
            returncode=completed.returncode,
        )
