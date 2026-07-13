"""Durable supervisor for rerunning the C0-C13 live condition track.

Runs each condition in a separate CLI process, records state under ``.runs``,
and retries rate-limited conditions after a bounded wait. It is intentionally
restartable: rerun with the same ``--run-dir`` to continue after a dropped
terminal/session without repeating completed conditions unless ``--force`` is
passed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

CONDITIONS = (
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13",
)
RETRYABLE_MODES = {"live_rate_limited", "output_truncated", "live_error"}
RATE_LIMIT_NEEDLES = (
    "live_rate_limited",
    "http_429",
    "429",
    "too many requests",
    "rate limit",
    "rate_limited",
    "usage limit",
)
DEFAULT_BRANCH_OBJECTIVES = (
    "elderly-demand,emergency-access,reorganization-feasibility,"
    "financial-risk,evidence-completeness"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def sleep_with_heartbeat(state_path: Path, state: dict[str, Any], seconds: float) -> None:
    """Sleep in short chunks so dropped sessions can resume from state."""
    deadline = datetime.now(timezone.utc) + timedelta(seconds=max(0.0, seconds))
    while True:
        remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return
        state["last_seen_at"] = now_iso()
        write_json(state_path, state)
        time.sleep(min(60.0, remaining))


def read_records(records_path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not records_path.exists():
        return records
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records[str(row.get("condition_group"))] = row
    return records


def retry_hint_from_trace(run_dir: Path) -> float | None:
    trace = run_dir / "model_call_trace.jsonl"
    if not trace.exists():
        return None
    hints: list[float] = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        retry_after = row.get("retry_after")
        if isinstance(retry_after, (int, float)) and retry_after > 0:
            hints.append(float(retry_after))
    return max(hints) if hints else None


def retryable_record(record: dict[str, Any], run_dir: Path) -> tuple[bool, str]:
    mode = str(record.get("execution_mode") or "")
    text = json.dumps({
        "execution_mode": mode,
        "exclusion_reason": record.get("exclusion_reason"),
        "failure_notes": record.get("failure_notes") or [],
        "model_call_summary": record.get("model_call_summary") or {},
    }, ensure_ascii=False).lower()
    if any(needle in text for needle in RATE_LIMIT_NEEDLES):
        return True, "rate_limit"
    if mode in RETRYABLE_MODES and retry_hint_from_trace(run_dir) is not None:
        return True, "retry_after_hint"
    return False, ""


def condition_complete(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    if record.get("comparable_for_e13") is True:
        return True
    # Auth failures and manual-harness waits are true terminal outcomes for
    # the current credentials/session; rate limits remain retryable.
    mode = str(record.get("execution_mode") or "")
    reason = str(record.get("exclusion_reason") or "").lower()
    return mode in {"live_auth_failed", "waiting_for_manual_harness"} and "429" not in reason


def default_wait_seconds(record: dict[str, Any], run_dir: Path, attempt: int) -> float:
    trace_hint = retry_hint_from_trace(run_dir)
    if trace_hint is not None:
        return min(trace_hint + 5.0, 6 * 60 * 60)
    # Provider APIs sometimes omit Retry-After on 429. Back off enough to let
    # per-minute quotas recover, but keep it bounded for unattended runs.
    return min(300.0 * (2 ** max(0, attempt - 1)), 60 * 60)


def run_command(command: list[str], *, log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== {now_iso()} command =====\n")
        log.write(" ".join(command) + "\n")
        log.flush()
        timeout = float(env.get("GEO_STRATEGIST_CONDITION_TIMEOUT_SECONDS", "7200"))
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout,
            )
            log.write(f"===== exit {completed.returncode} at {now_iso()} =====\n")
            return int(completed.returncode)
        except subprocess.TimeoutExpired:
            log.write(f"===== timeout after {timeout:.0f}s at {now_iso()} =====\n")
            return 124


def condition_command(condition: str, output_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable, "-m", "geo_strategist.cli", "run-condition-proposals",
        "--conditions", condition,
        "--output-dir", str(output_dir),
        "--top-k-sites", str(args.top_k_sites),
        "--max-review-rounds", str(args.max_review_rounds),
        "--require-live-agents",
        "--disable-deterministic-fallback-for-comparison",
        "--branch-objectives", args.branch_objectives,
        "--auto-agentic-harness",
        "--skip-judge",
    ]


def supervisor_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("GEMINI_RETRY_ATTEMPTS", "1")
    env.setdefault("GEMINI_REQUESTS_PER_MINUTE", "9")
    env.setdefault("GEMINI_TOKENS_PER_MINUTE", "220000")
    env.setdefault("GEMINI_REQUESTS_PER_DAY", "18")
    env.setdefault("C9_NUM_DRAFTS", "3")
    env.setdefault("C9_AGENT_STEPS", "8")
    env.setdefault("C9_VARIANTS_PER_BRANCH", "1")
    env.setdefault("C9_MAX_DEBUG_DEPTH", "0")
    env.setdefault("C9_CONCURRENCY", "1")
    env.setdefault("CANDIDATE_REVIEW_PROVIDER", "same_provider")
    env.setdefault("CANDIDATE_REVIEW_REVIEWERS", "healthcare_strategy")
    env.setdefault("CANDIDATE_REVIEW_MAX_CANDIDATES", "1")
    env.setdefault("CANDIDATE_REVIEW_MAX_WORKERS", "1")
    env.setdefault("CANDIDATE_REVIEW_MAX_DATA_REQUESTS", "0")
    env.setdefault("CANDIDATE_REVIEW_RETRY_ATTEMPTS", "1")
    env.setdefault("CANDIDATE_REVIEW_RETRY_BASE_DELAY", "10")
    env.setdefault("CANDIDATE_REVIEW_RETRY_MAX_DELAY", "65")
    env.setdefault("GEO_STRATEGIST_MAX_HARNESS_RATE_LIMIT_SLEEP_SECONDS", str(6 * 60 * 60))
    env.setdefault("GEO_STRATEGIST_CONDITION_TIMEOUT_SECONDS", str(2 * 60 * 60))
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/condition_proposals/live")
    parser.add_argument("--run-dir", default=".runs/live_condition_supervisor/latest")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--top-k-sites", type=int, default=5)
    parser.add_argument("--max-review-rounds", type=int, default=2)
    parser.add_argument("--branch-objectives", default=DEFAULT_BRANCH_OBJECTIVES)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    run_dir = (REPO_ROOT / args.run_dir).resolve()
    state_path = run_dir / "state.json"
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    state = read_json(state_path, {
        "started_at": now_iso(),
        "output_dir": str(output_dir),
        "conditions": {},
    })
    state["last_seen_at"] = now_iso()
    write_json(state_path, state)

    env = supervisor_env()
    for condition in conditions:
        state = read_json(state_path, state)
        condition_state = state.setdefault("conditions", {}).setdefault(condition, {
            "attempts": 0,
            "status": "pending",
        })
        records = read_records(output_dir / "condition_records.jsonl")
        record = records.get(condition)
        if not args.force and condition_complete(record):
            condition_state.update({
                "status": "complete",
                "completed_at": condition_state.get("completed_at") or now_iso(),
                "execution_mode": record.get("execution_mode") if record else None,
                "comparable_for_e13": record.get("comparable_for_e13") if record else None,
            })
            state["last_seen_at"] = now_iso()
            write_json(state_path, state)
            continue

        if (
            condition_state.get("status") == "waiting_rate_limit"
            and int(condition_state.get("attempts") or 0) < args.max_attempts
        ):
            wait_until = parse_iso(condition_state.get("wait_until"))
            if wait_until is None:
                started = parse_iso(condition_state.get("wait_started_at"))
                wait_seconds = float(condition_state.get("wait_seconds") or 0.0)
                if started is not None and wait_seconds > 0:
                    wait_until = started + timedelta(seconds=wait_seconds)
                    condition_state["wait_until"] = wait_until.isoformat()
                    state["last_seen_at"] = now_iso()
                    write_json(state_path, state)
            if wait_until is not None:
                remaining = (wait_until - datetime.now(timezone.utc)).total_seconds()
                if remaining > 0:
                    sleep_with_heartbeat(state_path, state, remaining)

        while int(condition_state.get("attempts") or 0) < args.max_attempts:
            condition_state["attempts"] = int(condition_state.get("attempts") or 0) + 1
            condition_state["status"] = "running"
            condition_state["started_at"] = now_iso()
            state["last_seen_at"] = now_iso()
            write_json(state_path, state)

            log_path = log_dir / f"{condition}_attempt{condition_state['attempts']}.log"
            returncode = run_command(
                condition_command(condition, output_dir, args),
                log_path=log_path,
                env=env,
            )
            records = read_records(output_dir / "condition_records.jsonl")
            record = records.get(condition)
            condition_state.update({
                "last_returncode": returncode,
                "last_log": str(log_path),
                "execution_mode": record.get("execution_mode") if record else None,
                "comparable_for_e13": record.get("comparable_for_e13") if record else None,
                "exclusion_reason": record.get("exclusion_reason") if record else None,
                "last_finished_at": now_iso(),
            })
            if returncode == 0 and condition_complete(record):
                condition_state["status"] = "complete"
                condition_state["completed_at"] = now_iso()
                state["last_seen_at"] = now_iso()
                write_json(state_path, state)
                break

            retryable, retry_reason = retryable_record(
                record or {}, output_dir / "runs" / f"C{int(condition[1:]):02d}"
            )
            if not retryable:
                condition_state["status"] = "failed_terminal"
                condition_state["retry_reason"] = retry_reason
                state["last_seen_at"] = now_iso()
                write_json(state_path, state)
                break

            if int(condition_state["attempts"]) >= args.max_attempts:
                condition_state["status"] = "attempts_exhausted"
                condition_state["retry_reason"] = retry_reason
                state["last_seen_at"] = now_iso()
                write_json(state_path, state)
                break

            wait_seconds = default_wait_seconds(
                record or {}, output_dir / "runs" / f"C{int(condition[1:]):02d}",
                int(condition_state["attempts"]),
            )
            condition_state.update({
                "status": "waiting_rate_limit",
                "retry_reason": retry_reason,
                "wait_seconds": wait_seconds,
                "wait_started_at": now_iso(),
                "wait_until": (
                    datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
                ).isoformat(),
            })
            state["last_seen_at"] = now_iso()
            write_json(state_path, state)
            sleep_with_heartbeat(state_path, state, wait_seconds)
        else:
            condition_state["status"] = "attempts_exhausted"
            state["last_seen_at"] = now_iso()
            write_json(state_path, state)

    rebuild_log = log_dir / "rebuild_live_artifacts.log"
    rebuild_rc = run_command(
        [sys.executable, "scripts/rebuild_live_artifacts.py", "--target", str(output_dir)],
        log_path=rebuild_log,
        env=env,
    )
    state["rebuild"] = {
        "status": "complete" if rebuild_rc == 0 else "failed",
        "returncode": rebuild_rc,
        "log": str(rebuild_log),
        "finished_at": now_iso(),
    }
    state["finished_at"] = now_iso()
    state["last_seen_at"] = now_iso()
    write_json(state_path, state)
    return rebuild_rc


if __name__ == "__main__":
    raise SystemExit(main())
