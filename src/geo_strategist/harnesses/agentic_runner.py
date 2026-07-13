"""Agentic Skills harness launcher support for C5-C8.

Phase 1 keeps execution manual because the supported CLIs differ in
authentication, TUI, and approval behavior. This module validates the shared
AGENTS.md + Skill packages contract and writes concise launcher prompts.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from geo_strategist.agent.skill_registry import build_default_skill_registry
from geo_strategist.experiments.branch_objectives import BRANCH_OBJECTIVES
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.harnesses.adapters import adapter_for
from geo_strategist.harnesses.skills_installer import validate_installed_skill_packages
from geo_strategist.providers.base import redact_secrets


@dataclass(frozen=True)
class AgenticHarnessLaunch:
    run_dir: Path
    launcher_prompt_path: Path
    validation_issues: list[str]


AgenticExecutionStatus = Literal[
    "not_supported", "succeeded", "agent_command_failed", "agent_timeout",
    "missing_agent_output",
]


@dataclass(frozen=True)
class AgenticHarnessExecution:
    status: AgenticExecutionStatus
    manual_result_path: Path
    stdout_path: Path
    stderr_path: Path
    metadata_path: Path
    command: list[str]
    returncode: int | None = None
    detail: str = ""


_RATE_LIMIT_MARKERS = (
    "usage limit",
    "rate limit",
    "rate-limit",
    "rate_limited",
    "too many requests",
    "http_429",
    "429",
)
_RETRY_AFTER_RE = re.compile(
    r"(?:retry(?:\s|-)?after|try again in)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)?",
    re.IGNORECASE,
)
_TRY_AGAIN_AT_RE = re.compile(
    r"try again at\s+"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*"
    r"(?P<ampm>am|pm)?",
    re.IGNORECASE,
)
_DEFAULT_MAX_RATE_LIMIT_SLEEP_SECONDS = 6 * 60 * 60
_RATE_LIMIT_RESTORE_BUFFER_SECONDS = 5.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _bounded_rate_limit_delay(delay_seconds: float) -> float | None:
    delay = max(0.0, delay_seconds) + _RATE_LIMIT_RESTORE_BUFFER_SECONDS
    max_sleep = _env_float(
        "GEO_STRATEGIST_MAX_HARNESS_RATE_LIMIT_SLEEP_SECONDS",
        _DEFAULT_MAX_RATE_LIMIT_SLEEP_SECONDS,
    )
    if max_sleep <= 0 or delay > max_sleep:
        return None
    return delay


def _parse_harness_rate_limit_delay(
    text: str,
    *,
    now: datetime | None = None,
) -> float | None:
    """Return seconds to wait for a CLI rate limit, or None if not actionable."""

    lowered = text.lower()
    if not any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
        return None

    retry_after = _RETRY_AFTER_RE.search(text)
    if retry_after is not None:
        value = float(retry_after.group("value"))
        unit = (retry_after.group("unit") or "seconds").lower()
        if unit.startswith(("h", "hr", "hour")):
            value *= 60 * 60
        elif unit.startswith(("m", "min")):
            value *= 60
        return _bounded_rate_limit_delay(value)

    restore_time = _TRY_AGAIN_AT_RE.search(text)
    if restore_time is None:
        return None
    current = now or datetime.now().astimezone()
    hour = int(restore_time.group("hour"))
    minute = int(restore_time.group("minute") or 0)
    ampm = (restore_time.group("ampm") or "").lower()
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    target = current.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= current:
        target += timedelta(days=1)
    return _bounded_rate_limit_delay((target - current).total_seconds())


def _attempt_log(chunks: list[str]) -> str:
    if len(chunks) == 1:
        return chunks[0]
    return "\n".join(
        f"===== attempt {index} =====\n{chunk}"
        for index, chunk in enumerate(chunks, start=1)
    )


def launcher_filename(spec: ConditionSpec) -> str:
    return f"{spec.report_slug.removesuffix('_manual')}_launcher.md"


def build_agentic_launcher_prompt(spec: ConditionSpec, repo_root: Path, live_dir: Path) -> str:
    adapter = adapter_for(spec.harness)
    skills = list(build_default_skill_registry().values())
    skill_lines = [
        f"{index}. {skill.skill_package_name} (`{skill.codex_skill_path}` / `{skill.claude_skill_path}`)"
        for index, skill in enumerate(skills, start=1)
    ]
    objectives = [
        f"- `{objective.key}` ({objective.label}): {objective.description}"
        for objective in BRANCH_OBJECTIVES
    ]
    padded = spec.padded_id
    schema = {
        "condition_group": spec.group,
        "ranked_candidates": [{
            "candidate_id": "<exact id from candidate_actions.jsonl>",
            "rationale": "<why this candidate, <= 60 words>",
            "qualitative_discussion": {
                "regional": "...",
                "population": "...",
                "demand_supply": "...",
                "access": "...",
                "cost_financial": "...",
                "preferred_action": "...",
                "review_comments": "...",
            },
        }],
        "review_comments": ["<slate-level review findings>"],
        "skill_trace": [{
            "skill_id": "inspect_available_data",
            "status": "succeeded",
            "produced_outputs": ["data_bundle", "data_inventory"],
            "output_refs": ["runs/Cxx/..."],
            "payload": {},
        }],
        "model_call_summary": {"total_requests": 0},
    }
    return f"""# {spec.group} - {spec.label} launcher

Repository: `{repo_root}`

Run from the repository root with:

```bash
{adapter.command}
```

You are running condition `{spec.group}`.

Condition:
- provider: `{spec.provider}`
- model: `{spec.model}`
- harness: `{spec.harness}`
- algorithm: `{spec.algorithm}`
- instruction_mode: `{spec.instruction_mode}`
- skills_unified: `true`
- branch_search: `true`

Use the repository `AGENTS.md` and the filesystem Skill packages. For this
harness, the expected Skill source is `{adapter.skill_source}`. Do not use the
C2/C3 vanilla no-Skills baseline behavior.

Execute the Skills-unified contract in this exact order:

{chr(10).join(skill_lines)}

Use exactly these branch objectives:

{chr(10).join(objectives)}

Save the final result to:

`{live_dir}/runs/{padded}/manual_result.json`

Required return shape:

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

Rules:
- `candidate_id` values must come verbatim from `.data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl`.
- Provide at least 5 ranked candidates when enough valid candidates are available.
- Every candidate must include the full seven-part `qualitative_discussion`.
- Keep generated code under `{live_dir}/runs/{padded}/generated_code/`.
- Record a complete `skill_trace`; invalid C5-C8 traces are excluded from the comparison.
- Do not fabricate missing facts or substitute deterministic fallback rankings.

After saving `manual_result.json`, ingest it with:

```bash
.venv/bin/python -m geo_strategist.cli run-condition-proposals \\
  --conditions {spec.group} --output-dir {live_dir} \\
  --manual-result {live_dir}/runs/{padded}/manual_result.json --skip-judge
```
"""


def prepare_agentic_skills_harness(
    repo_root: Path,
    spec: ConditionSpec,
    out_dir: Path,
) -> AgenticHarnessLaunch:
    run_dir = out_dir / "runs" / spec.padded_id
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = out_dir / "manual_harness"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    issues: list[str] = []
    if not (repo_root / "AGENTS.md").exists():
        issues.append("missing_agents_md")
    issues.extend(validate_installed_skill_packages(repo_root))

    launcher_path = prompts_dir / launcher_filename(spec)
    launcher_path.write_text(
        build_agentic_launcher_prompt(spec, repo_root, out_dir),
        encoding="utf-8",
    )
    return AgenticHarnessLaunch(
        run_dir=run_dir,
        launcher_prompt_path=launcher_path,
        validation_issues=list(dict.fromkeys(issues)),
    )


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def execute_agentic_skills_harness(
    repo_root: Path,
    spec: ConditionSpec,
    launch: AgenticHarnessLaunch,
) -> AgenticHarnessExecution:
    """Run a supported non-interactive harness adapter and capture logs."""

    adapter = adapter_for(spec.harness)
    manual_result_path = launch.run_dir / "manual_result.json"
    stdout_path = launch.run_dir / "agent_stdout.log"
    stderr_path = launch.run_dir / "agent_stderr.log"
    metadata_path = launch.run_dir / "agent_execution.json"
    command = adapter.build_command(
        repo_root=repo_root,
        run_dir=launch.run_dir,
        launcher_prompt_path=launch.launcher_prompt_path,
        manual_result_path=manual_result_path,
    )
    prompt_text = launch.launcher_prompt_path.read_text(encoding="utf-8")
    command_input = prompt_text if adapter.prompt_mode == "stdin" else None
    if command and adapter.prompt_mode == "argument":
        command = [*command, prompt_text]

    metadata_command = [
        f"<launcher_prompt:{launch.launcher_prompt_path}>"
        if part == prompt_text else part
        for part in command
    ]

    def write_metadata(
        status: AgenticExecutionStatus,
        *,
        returncode: int | None = None,
        detail: str = "",
        attempts: int = 1,
        rate_limit_detected: bool = False,
        retry_after_seconds: float | None = None,
        rate_limit_retry_attempted: bool = False,
    ) -> AgenticHarnessExecution:
        payload = {
            "condition_group": spec.group,
            "harness": spec.harness,
            "status": status,
            "automation_supported": adapter.automation_supported,
            "command": [redact_secrets(part) for part in metadata_command],
            "returncode": returncode,
            "detail": redact_secrets(detail),
            "attempts": attempts,
            "launcher_prompt": str(launch.launcher_prompt_path),
            "manual_result": str(manual_result_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        if rate_limit_detected:
            payload.update({
                "rate_limit_detected": True,
                "retry_after_seconds": retry_after_seconds,
                "rate_limit_retry_attempted": rate_limit_retry_attempted,
            })
        metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        return AgenticHarnessExecution(
            status=status,
            manual_result_path=manual_result_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            metadata_path=metadata_path,
            command=command,
            returncode=returncode,
            detail=detail,
        )

    if not command:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return write_metadata("not_supported", detail="adapter has no automation command")

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    def run_once() -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=os.environ.copy(),
            input=command_input,
            capture_output=True,
            text=True,
            timeout=adapter.timeout_seconds,
            check=False,
        )
        stdout_chunks.append(completed.stdout or "")
        stderr_chunks.append(completed.stderr or "")
        return completed

    try:
        completed = run_once()
    except subprocess.TimeoutExpired as exc:
        stdout_chunks.append(_text(exc.stdout))
        stderr_chunks.append(_text(exc.stderr))
        stdout_path.write_text(redact_secrets(_attempt_log(stdout_chunks)), encoding="utf-8")
        stderr_path.write_text(redact_secrets(_attempt_log(stderr_chunks)), encoding="utf-8")
        return write_metadata("agent_timeout", detail=str(exc))

    rate_limit_detected = False
    retry_after_seconds: float | None = None
    rate_limit_retry_attempted = False
    if completed.returncode != 0:
        retry_after_seconds = _parse_harness_rate_limit_delay(
            "\n".join([stdout_chunks[-1], stderr_chunks[-1]])
        )
        if retry_after_seconds is not None:
            rate_limit_detected = True
            rate_limit_retry_attempted = True
            time.sleep(retry_after_seconds)
            try:
                completed = run_once()
            except subprocess.TimeoutExpired as exc:
                stdout_chunks.append(_text(exc.stdout))
                stderr_chunks.append(_text(exc.stderr))
                stdout_path.write_text(
                    redact_secrets(_attempt_log(stdout_chunks)), encoding="utf-8")
                stderr_path.write_text(
                    redact_secrets(_attempt_log(stderr_chunks)), encoding="utf-8")
                return write_metadata(
                    "agent_timeout",
                    detail=str(exc),
                    attempts=len(stdout_chunks),
                    rate_limit_detected=True,
                    retry_after_seconds=retry_after_seconds,
                    rate_limit_retry_attempted=True,
                )

    stdout_path.write_text(redact_secrets(_attempt_log(stdout_chunks)), encoding="utf-8")
    stderr_path.write_text(redact_secrets(_attempt_log(stderr_chunks)), encoding="utf-8")
    if completed.returncode != 0:
        return write_metadata(
            "agent_command_failed",
            returncode=completed.returncode,
            detail=f"adapter command exited {completed.returncode}",
            attempts=len(stdout_chunks),
            rate_limit_detected=rate_limit_detected,
            retry_after_seconds=retry_after_seconds,
            rate_limit_retry_attempted=rate_limit_retry_attempted,
        )
    if not manual_result_path.exists():
        return write_metadata(
            "missing_agent_output",
            returncode=completed.returncode,
            detail="adapter command completed but manual_result.json was not written",
            attempts=len(stdout_chunks),
            rate_limit_detected=rate_limit_detected,
            retry_after_seconds=retry_after_seconds,
            rate_limit_retry_attempted=rate_limit_retry_attempted,
        )
    return write_metadata(
        "succeeded",
        returncode=completed.returncode,
        attempts=len(stdout_chunks),
        rate_limit_detected=rate_limit_detected,
        retry_after_seconds=retry_after_seconds,
        rate_limit_retry_attempted=rate_limit_retry_attempted,
    )
