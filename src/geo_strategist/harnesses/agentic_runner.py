"""Agentic Skills harness launcher support for C9-C12.

Phase 1 keeps execution manual because the supported CLIs differ in
authentication, TUI, and approval behavior. This module validates the shared
AGENTS.md + Skill packages contract and writes concise launcher prompts.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from geo_strategist.agent.skill_registry import build_default_skill_registry
from geo_strategist.experiments.branch_objectives import BRANCH_OBJECTIVES
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.decision_reporting_contract import (
    reporting_prompt_fragment,
    reporting_schema_example,
)
from geo_strategist.harnesses.adapters import adapter_for
from geo_strategist.harnesses.antigravity_support import (
    ensure_workspace_trusted,
    resolve_agy_model_slug,
)
from geo_strategist.harnesses.prompts import (
    _C0_SUBSTITUTION_RULES,
    _EVIDENCE_RULES,
    _QUALITATIVE_REQUIREMENT,
)
from geo_strategist.harnesses.skills_installer import validate_installed_skill_packages
from geo_strategist.providers.base import redact_secrets


@dataclass(frozen=True)
class AgenticHarnessLaunch:
    run_dir: Path
    launcher_prompt_path: Path
    validation_issues: list[str]


AgenticExecutionStatus = Literal[
    "not_supported", "succeeded", "succeeded_with_nonzero_exit",
    "agent_command_failed", "agent_timeout", "missing_agent_output",
]

FailureClass = Literal["terminal", "retryable"]

# Markers strong enough to call a failure terminal regardless of what
# artifacts exist — retrying would just repeat the same outcome and burn
# another full attempt budget for nothing.
_TERMINAL_FAILURE_MARKERS = (
    "not logged in", "not authenticated", "authentication failed",
    "unauthorized", "no credentials", "permission denied",
    "unsupported flag", "unrecognized model", "not recognized",
    "invalid model",
)

# Markers worth a bounded retry: transient provider/network conditions, or a
# timeout/crash that still left real intermediate work behind.
_RETRYABLE_FAILURE_MARKERS = (
    "rate limit", "usage limit", "429", "temporarily unavailable",
    "connection reset", "connection refused", "network error",
    "internal server error", "http_500", "timeout waiting for response",
    "deadline exceeded",
)


def classify_agentic_failure(
    status: AgenticExecutionStatus,
    *,
    stdout: str,
    stderr: str,
    manual_result_exists: bool,
    generated_code_exists: bool,
) -> FailureClass:
    """Terminal vs retryable classification for a failed agentic-harness run.

    Never called for a status that already represents success. A run with
    no failure-text markers at all but real intermediate artifacts is still
    treated as retryable (a resume prompt asking the agent to inspect and
    continue is cheap and strictly more informative than giving up), but a
    bare failure with nothing on disk is terminal — nothing to resume from.
    """

    if status == "not_supported":
        return "terminal"
    combined = f"{stdout}\n{stderr}".lower()
    if any(marker in combined for marker in _TERMINAL_FAILURE_MARKERS):
        return "terminal"
    if any(marker in combined for marker in _RETRYABLE_FAILURE_MARKERS):
        return "retryable"
    if manual_result_exists or generated_code_exists:
        return "retryable"
    return "terminal"


def build_resume_prompt(original_prompt: str, run_dir: Path, *, attempt: int) -> str:
    """A follow-up prompt for a retried attempt: inspect what already exists
    before repeating any step, rather than blindly restarting from scratch."""

    existing_files = sorted(
        str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()
    )
    manual_result = run_dir / "manual_result.json"
    status_line = (
        "`manual_result.json` already exists in this run directory — inspect "
        "it, validate it, and repair it rather than regenerating it from "
        "scratch." if manual_result.exists() else
        "`manual_result.json` does not exist yet in this run directory."
    )
    listing = "\n".join(f"- `{name}`" for name in existing_files) or "(no files yet)"
    return (
        f"# Resume — attempt {attempt}\n\n"
        "The previous attempt at this exact task did not finish cleanly, but "
        "it may have left real, reusable work in this run directory. Inspect "
        "what is already there before repeating any step.\n\n"
        f"Run directory: `{run_dir}`\n\n"
        f"Files already present:\n{listing}\n\n"
        f"{status_line}\n\n"
        "Do not blindly restart the whole analysis. Read any existing "
        "`generated_code/` scripts and their prior output first; reuse what "
        "is still valid, fix what failed, and continue from there. Record "
        "what changed since the previous attempt rather than silently "
        "repeating it unmodified.\n\n"
        "---\n\n"
        "## Original task\n\n" + original_prompt
    )


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
    from geo_strategist.agent.skills_budget import skills_budget_for

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
    budget = skills_budget_for(spec.group)
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
        "skill_trace": [
            {
                "skill_id": "inspect_available_data",
                "status": "succeeded",
                "produced_outputs": ["data_bundle", "data_inventory"],
                "output_refs": ["runs/Cxx/..."],
                "payload": {},
            },
            {
                "skill_id": "generate_research_hypotheses",
                "status": "succeeded",
                "produced_outputs": ["hypotheses"],
                "output_refs": ["runs/Cxx/..."],
                "payload": {"hypotheses": [{
                    "hypothesis_id": "h1", "mechanism": "...",
                    "required_data": ["..."], "implementation_plan": "...",
                    "expected_contribution": "...", "evaluation_method": "...",
                    "failure_modes": ["..."], "acceptance_evidence": "...",
                }]},
            },
            {
                "skill_id": "run_branch_search",
                "status": "succeeded",
                "produced_outputs": ["branch_results"],
                "output_refs": ["runs/Cxx/..."],
                "payload": {
                    "branch_objectives": [o.key for o in BRANCH_OBJECTIVES],
                    "branches": [{
                        "branch_id": "b1", "parent_id": None,
                        "objective": "elderly-demand", "hypothesis_id": "h1",
                        "status": "winner",
                    }],
                },
            },
            {
                "skill_id": "review_proposal",
                "status": "succeeded",
                "produced_outputs": ["reviews", "revision_requests"],
                "output_refs": ["runs/Cxx/..."],
                "payload": {"reviews": ["..."]},
            },
            {
                "skill_id": "write_final_condition_proposal",
                "status": "succeeded",
                "produced_outputs": ["condition_reports"],
                "output_refs": ["runs/Cxx/..."],
                "payload": {"source_branch_ids": ["b1"]},
            },
        ],
        "model_call_summary": {"total_requests": 0},
    }
    schema.update(reporting_schema_example(spec.group))
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

Select Skills dynamically from this operator library. Reuse an operator when
execution, critique, ablation, or a newly discovered regime requires it; the
list order is registry order, not an execution sequence:

{chr(10).join(skill_lines)}

Cover at least these strategy anchors, and add evidence-supported materially
different regimes when useful:

{chr(10).join(objectives)}

## Search budget (configured for this run — record actual usage, do not exceed silently)

- num_drafts / initial hypotheses: {budget.num_drafts}
- agent_steps: {budget.agent_steps}
- variants_per_branch: {budget.variants_per_branch}
- max_debug_depth: {budget.max_debug_depth}
- concurrency: {budget.concurrency} (where the harness supports it)
- max_revision_rounds: {budget.max_revision_rounds}

If this harness cannot enforce one of these limits exactly (e.g. an
interactive CLI session cannot be stepped like a direct-API loop), use the
closest technically valid equivalent and record the deviation explicitly in
`review_comments` rather than silently ignoring the budget.

{reporting_prompt_fragment(spec.group)}
Save the final result to:

`{live_dir}/runs/{padded}/manual_result.json`

Required return shape (`skill_trace` entries are illustrative; record every
invocation actually made, including repeated calls, with resolvable output
references; see each selected Skill's SKILL.md for its payload fields):

```json
{json.dumps(schema, ensure_ascii=False, indent=2)}
```

Rules:
- `candidate_id` values must come verbatim from `.data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl`.
- Provide at least 5 ranked candidates when enough valid candidates are available.
- Keep generated code under `{live_dir}/runs/{padded}/generated_code/`.
- Do not fabricate missing facts or substitute deterministic fallback rankings.

{_QUALITATIVE_REQUIREMENT}
{_EVIDENCE_RULES}
{_C0_SUBSTITUTION_RULES}
## Skills-unified-only rules (this is what a no-Skills coding-agent control
## condition on the same provider/model/harness does NOT have to satisfy)

- Record the complete dynamic `skill_trace`. A step that claims executed analysis with a broken or missing output reference behind it is an unsupported factual claim and excludes the run from the comparison; other lineage or trace-shape differences are recorded as deviations, not excluded.
- `generate_research_hypotheses` needs at least 2 materially distinct hypotheses, each with mechanism/required_data/implementation_plan/expected_contribution/evaluation_method/failure_modes/acceptance_evidence.
- `run_branch_search` needs a `branches` list with branch_id/parent_id/objective/hypothesis_id/status lineage for every branch explored.
- `debug_failed_code` retries must record what changed since the previous attempt; an unmodified repeat does not count as new search depth.
- `write_final_condition_proposal` needs `source_branch_ids` naming the actual branch(es) this slate derives from — a narrative-only final report is not comparable.

## Validate and repair before you stop (mandatory)

After you write `manual_result.json`, run the validator and repair any
reported issue before finishing — do not stop at "wrote the file":

```
.venv/bin/python -m geo_strategist.cli validate-skills-result {live_dir}/runs/{padded}/manual_result.json --expected-condition-group {spec.group}
```

1. Write `manual_result.json`.
2. Run the exact validator command above.
3. Read every reported error and every C0-substitution flag (a
   C0-substitution flag is never acceptable and must be fixed, not
   explained away). Skill_trace deviations are reported separately and are
   not by themselves blocking, but fix them too when practical.
4. Repair the output, the recorded `skill_trace`, or the supporting
   `generated_code/` that produced it.
5. Re-run the validator.
6. Repeat steps 3-5 for up to 3 repair attempts.
7. Stop only once the validator reports PASS, or — if it still fails after
   3 repair attempts — write an explicit failure note explaining what still
   fails and why, to
   `{live_dir}/runs/{padded}/unresolved_validation_failure.md`, rather than
   leaving a silently invalid `manual_result.json` as your only output.

Stop after the validator passes (or you write the failure note above). Do
NOT run `run-condition-proposals` or `run-condition-comparison-judge`
yourself — an orchestrator ingests the result and refreshes the comparison
automatically once every condition in the batch has finished. Running
these yourself from inside this session can race with, or block, that
orchestration.
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

    resolved_model: str | None = None
    trust_note: str | None = None
    if spec.harness == "antigravity" and command:
        # Every antigravity-specific flag is appended AFTER the prompt text,
        # never before it: `--print` consumes only the single token
        # immediately following it, so any flag placed between `--print`
        # and the prompt gets swallowed as the prompt itself (see
        # adapters/antigravity.py's module docstring for how this was found).
        from geo_strategist.harnesses.adapters.antigravity import POST_PROMPT_FLAGS

        command = [*command, *POST_PROMPT_FLAGS]
        resolved_model = resolve_agy_model_slug(spec.model)
        if resolved_model:
            command = [*command, "--model", resolved_model]
        registration = ensure_workspace_trusted(repo_root)
        trust_note = (
            f"trustedWorkspaces: {registration.workspace} "
            f"(already_trusted={registration.was_already_trusted}, "
            f"trusted_now={registration.trusted_now}"
            + (f", note={registration.note}" if registration.note else "") + ")"
        )

    metadata_command = [
        f"<launcher_prompt:{launch.launcher_prompt_path}>"
        if part == prompt_text else part
        for part in command
    ]
    prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    start_time = datetime.now(timezone.utc).isoformat()

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
            "exited_nonzero": bool(returncode),
            "detail": redact_secrets(detail),
            "attempts": attempts,
            "launcher_prompt": str(launch.launcher_prompt_path),
            "manual_result": str(manual_result_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "provider": spec.provider,
            "configured_model": spec.model,
            "resolved_model": resolved_model,
            "working_directory": str(repo_root),
            "prompt_sha256": prompt_sha256,
            "start_time_utc": start_time,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_trust": trust_note,
            "failure_classification": (
                None if status in ("succeeded", "succeeded_with_nonzero_exit") else
                classify_agentic_failure(
                    status, stdout=_safe_read(stdout_path), stderr=_safe_read(stderr_path),
                    manual_result_exists=manual_result_path.exists(),
                    generated_code_exists=(launch.run_dir / "generated_code").is_dir(),
                )
            ),
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
    # Check for a valid-looking artifact before trusting the process exit
    # code: a real C5 run produced a genuine manual_result.json well before
    # a later, unrelated step (see diagnostics/) ran long and eventually hit
    # the CLI's own response-wait timeout, discarding real completed work
    # for a reason unrelated to whether the analysis itself succeeded.
    result_present = manual_result_path.exists() and manual_result_path.stat().st_size > 0
    if completed.returncode != 0:
        if result_present:
            return write_metadata(
                "succeeded_with_nonzero_exit",
                returncode=completed.returncode,
                detail=(
                    f"adapter command exited {completed.returncode}, but "
                    f"manual_result.json already exists and is non-empty; "
                    "the abnormal exit is retained as diagnostic metadata "
                    "(see stderr/stdout logs), not treated as a failed run."
                ),
                attempts=len(stdout_chunks),
                rate_limit_detected=rate_limit_detected,
                retry_after_seconds=retry_after_seconds,
                rate_limit_retry_attempted=rate_limit_retry_attempted,
            )
        return write_metadata(
            "agent_command_failed",
            returncode=completed.returncode,
            detail=f"adapter command exited {completed.returncode}",
            attempts=len(stdout_chunks),
            rate_limit_detected=rate_limit_detected,
            retry_after_seconds=retry_after_seconds,
            rate_limit_retry_attempted=rate_limit_retry_attempted,
        )
    if not result_present:
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


RECOVERABLE_MIRROR_STATUSES: tuple[AgenticExecutionStatus, ...] = (
    "missing_agent_output", "agent_command_failed",
)


def recover_from_sanitized_workspace_mirror(
    execution: AgenticHarnessExecution,
    *,
    harness_cwd: Path,
    repo_root: Path,
) -> tuple[AgenticHarnessExecution, bool]:
    """Recover ``manual_result.json`` from a sanitized workspace's own
    mirror of the real output path, when the direct path is missing.

    Some sandboxes (observed with Codex) materialize a real directory over
    a symlinked ancestor instead of writing through it, stranding the
    agent's output inside the sanitized workspace's own mirror of the real
    path. Applies to a nonzero exit too (``agent_command_failed``): a run
    can finish real work and only fail later for an unrelated reason (e.g.
    the CLI's own response-wait timeout on a long-running final step) — the
    mirror may hold that real, already-computed result even when the direct
    path does not.

    Returns ``(execution, recovered)`` — the original ``execution`` is
    returned unchanged when recovery does not apply or the mirror has
    nothing to offer.
    """

    if execution.status not in RECOVERABLE_MIRROR_STATUSES:
        return execution, False
    if harness_cwd == repo_root or execution.manual_result_path.exists():
        return execution, False
    mirror_result = harness_cwd / execution.manual_result_path.relative_to(repo_root)
    if not (mirror_result.is_file() and mirror_result.stat().st_size > 0):
        return execution, False

    execution.manual_result_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mirror_result, execution.manual_result_path)
    mirror_generated_code = mirror_result.parent / "generated_code"
    if mirror_generated_code.is_dir():
        shutil.copytree(
            mirror_generated_code,
            execution.manual_result_path.parent / "generated_code",
            dirs_exist_ok=True,
        )
    recovered_status: AgenticExecutionStatus = (
        "succeeded_with_nonzero_exit" if execution.returncode else "succeeded")
    recovered_execution = AgenticHarnessExecution(
        status=recovered_status,
        manual_result_path=execution.manual_result_path,
        stdout_path=execution.stdout_path,
        stderr_path=execution.stderr_path,
        metadata_path=execution.metadata_path,
        command=execution.command,
        returncode=execution.returncode,
        detail=execution.detail,
    )
    return recovered_execution, True


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _prepend_previous_attempt_log(path: Path, previous_text: str, previous_attempt_number: int) -> None:
    if not previous_text:
        return
    current = _safe_read(path)
    combined = (
        f"===== attempt {previous_attempt_number} =====\n{previous_text}\n"
        f"===== attempt {previous_attempt_number + 1} =====\n{current}"
    )
    path.write_text(combined, encoding="utf-8")


def execute_agentic_skills_harness_with_retry(
    repo_root: Path,
    spec: ConditionSpec,
    launch: AgenticHarnessLaunch,
    *,
    max_attempts: int = 2,
) -> AgenticHarnessExecution:
    """Run the adapter, and on a retryable failure, resume with a prompt that
    tells the agent to inspect existing artifacts rather than restart blindly.

    Terminal failures (auth, unsupported mode, permission denial, or a bare
    failure with no reusable artifacts) are returned immediately without
    consuming the retry budget. The sanitized workspace, `generated_code/`,
    and any partial `manual_result.json` are never cleared between attempts.
    stdout/stderr from every attempt are preserved with `===== attempt N
    =====` separators, never overwritten.
    """

    original_prompt = launch.launcher_prompt_path.read_text(encoding="utf-8")
    execution = execute_agentic_skills_harness(repo_root, spec, launch)
    attempt = 1
    while attempt < max_attempts:
        if execution.status in ("succeeded", "succeeded_with_nonzero_exit", "not_supported"):
            break
        classification = classify_agentic_failure(
            execution.status,
            stdout=_safe_read(execution.stdout_path),
            stderr=_safe_read(execution.stderr_path),
            manual_result_exists=execution.manual_result_path.exists(),
            generated_code_exists=(launch.run_dir / "generated_code").is_dir(),
        )
        if classification != "retryable":
            break
        attempt += 1
        resume_prompt_path = launch.run_dir / f"resume_prompt_attempt{attempt}.md"
        resume_prompt_path.write_text(
            build_resume_prompt(original_prompt, launch.run_dir, attempt=attempt),
            encoding="utf-8",
        )
        previous_stdout = _safe_read(execution.stdout_path)
        previous_stderr = _safe_read(execution.stderr_path)
        resumed_launch = AgenticHarnessLaunch(
            run_dir=launch.run_dir,
            launcher_prompt_path=resume_prompt_path,
            validation_issues=[],
        )
        execution = execute_agentic_skills_harness(repo_root, spec, resumed_launch)
        _prepend_previous_attempt_log(execution.stdout_path, previous_stdout, attempt - 1)
        _prepend_previous_attempt_log(execution.stderr_path, previous_stderr, attempt - 1)

    if attempt > 1:
        metadata = json.loads(execution.metadata_path.read_text(encoding="utf-8"))
        metadata["resume_attempts"] = attempt
        metadata["max_attempts"] = max_attempts
        execution.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return execution
