"""Aggregate provider and harness usage from condition-run artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProviderUsageReport:
    summary_json_path: str
    calls_csv_path: str
    markdown_path: str
    total_requests: int
    total_errors: int


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def _condition_from_run_dir(path: Path) -> str:
    name = path.name
    if name.startswith("C") and name[1:].isdigit():
        return f"C{int(name[1:]):02d}"
    return name


def _condition_group_from_run_dir(path: Path) -> str:
    name = _condition_from_run_dir(path)
    if name.startswith("C"):
        return f"C{int(name[1:])}"
    return name


def _trace_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace_path in sorted(runs_dir.glob("C*/model_call_trace.jsonl")):
        run_dir = trace_path.parent
        group = _condition_group_from_run_dir(run_dir)
        for row in _read_jsonl(trace_path):
            rows.append({
                "condition_group": row.get("condition_group") or group,
                "condition_run_dir": run_dir.name,
                "call_index": row.get("call_index"),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "purpose": row.get("purpose"),
                "status": row.get("status"),
                "error_detail": row.get("error_detail"),
                "finish_reason": row.get("finish_reason"),
                "request_count": int(row.get("request_count") or 1),
                "prompt_tokens": int(row.get("prompt_tokens") or 0),
                "completion_tokens": int(row.get("completion_tokens") or 0),
                "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
                "latency_seconds": float(row.get("latency_seconds") or 0.0),
                "source_artifact": str(trace_path),
            })
    return rows


def _summary_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(runs_dir.glob("C*/model_call_summary.json")):
        run_dir = summary_path.parent
        group = _condition_group_from_run_dir(run_dir)
        payload = _read_json(summary_path)
        rows.append({
            "condition_group": group,
            "condition_run_dir": run_dir.name,
            "total_requests": int(payload.get("total_requests") or 0),
            "total_errors": int(payload.get("total_errors") or 0),
            "total_prompt_tokens": int(payload.get("total_prompt_tokens") or 0),
            "total_completion_tokens": int(payload.get("total_completion_tokens") or 0),
            "total_reasoning_tokens": int(payload.get("total_reasoning_tokens") or 0),
            "source_artifact": str(summary_path),
        })
    return rows


def _harness_rows(runs_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for execution_path in sorted(runs_dir.glob("C*/agent_execution.json")):
        run_dir = execution_path.parent
        group = _condition_group_from_run_dir(run_dir)
        payload = _read_json(execution_path)
        rows.append({
            "condition_group": payload.get("condition_group") or group,
            "condition_run_dir": run_dir.name,
            "harness": payload.get("harness"),
            "status": payload.get("status"),
            "returncode": payload.get("returncode"),
            "attempts": payload.get("attempts"),
            "rate_limit_detected": payload.get("rate_limit_detected"),
            "retry_after_seconds": payload.get("retry_after_seconds"),
            "rate_limit_retry_attempted": payload.get("rate_limit_retry_attempted"),
            "detail": payload.get("detail"),
            "source_artifact": str(execution_path),
        })
    return rows


def _rollup_trace_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollup: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("condition_group") or ""),
            str(row.get("provider") or ""),
            str(row.get("model") or ""),
            str(row.get("purpose") or ""),
            str(row.get("status") or ""),
        )
        target = rollup.setdefault(key, {
            "condition_group": key[0],
            "provider": key[1],
            "model": key[2],
            "purpose": key[3],
            "status": key[4],
            "call_rows": 0,
            "request_count": 0,
            "error_count": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "latency_seconds": 0.0,
        })
        target["call_rows"] += 1
        target["request_count"] += int(row.get("request_count") or 0)
        target["error_count"] += 0 if row.get("status") == "ok" else 1
        target["prompt_tokens"] += int(row.get("prompt_tokens") or 0)
        target["completion_tokens"] += int(row.get("completion_tokens") or 0)
        target["reasoning_tokens"] += int(row.get("reasoning_tokens") or 0)
        target["latency_seconds"] = round(
            float(target["latency_seconds"]) + float(row.get("latency_seconds") or 0.0),
            3,
        )
    return sorted(
        rollup.values(),
        key=lambda r: (r["condition_group"], r["provider"], r["model"], r["purpose"], r["status"]),
    )


def _diagnostics(trace_rows: list[dict[str, Any]],
                 harness_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        grouped[str(row.get("condition_group") or "")].append(row)
    harness_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in harness_rows:
        harness_grouped[str(row.get("condition_group") or "")].append(row)

    diagnostics: list[dict[str, str]] = []
    for group, rows in sorted(grouped.items()):
        statuses = {str(r.get("status")) for r in rows}
        details = {str(r.get("error_detail")) for r in rows if r.get("error_detail")}
        provider = ",".join(sorted({str(r.get("provider")) for r in rows if r.get("provider")}))
        if "ok" in statuses and len(statuses) > 1:
            diagnosis = "mixed_success_and_provider_errors"
        elif "live_auth_failed" in statuses:
            diagnosis = "auth_or_key_configuration"
        elif "live_rate_limited" in statuses:
            diagnosis = "provider_rate_limit"
        elif "live_error" in statuses:
            diagnosis = "provider_server_or_response_error"
        elif statuses == {"ok"}:
            diagnosis = "provider_calls_succeeded"
        else:
            diagnosis = "mixed_provider_status"
        diagnostics.append({
            "condition_group": group,
            "provider": provider,
            "diagnosis": diagnosis,
            "evidence": "; ".join(sorted(details))[:500],
        })
    for group, rows in sorted(harness_grouped.items()):
        statuses = {str(r.get("status")) for r in rows}
        details = {str(r.get("detail")) for r in rows if r.get("detail")}
        if "agent_command_failed" in statuses:
            diagnosis = "harness_command_failed_before_manual_result"
        elif "succeeded" in statuses:
            diagnosis = "harness_completed"
        else:
            diagnosis = "harness_status_recorded"
        diagnostics.append({
            "condition_group": group,
            "provider": ",".join(sorted({str(r.get("harness")) for r in rows if r.get("harness")})),
            "diagnosis": diagnosis,
            "evidence": "; ".join(sorted(details))[:500],
        })
    return diagnostics


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition_group", "provider", "model", "purpose", "status",
        "call_rows", "request_count", "error_count", "prompt_tokens",
        "completion_tokens", "reasoning_tokens", "latency_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Provider Usage Report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This report is rebuilt from redacted raw trace rows and harness execution metadata.",
        "",
        "## Totals",
        "",
        f"- Trace call rows: {payload['totals']['trace_call_rows']}",
        f"- Provider request count: {payload['totals']['request_count']}",
        f"- Provider error count: {payload['totals']['error_count']}",
        f"- Prompt tokens: {payload['totals']['prompt_tokens']}",
        f"- Completion tokens: {payload['totals']['completion_tokens']}",
        f"- Reasoning tokens: {payload['totals']['reasoning_tokens']}",
        f"- Summary-reported requests: {payload['totals']['summary_reported_request_count']}",
        f"- Summary-reported errors: {payload['totals']['summary_reported_error_count']}",
        f"- Summary-reported prompt tokens: {payload['totals']['summary_reported_prompt_tokens']}",
        f"- Summary-reported completion tokens: {payload['totals']['summary_reported_completion_tokens']}",
        f"- Summary-reported reasoning tokens: {payload['totals']['summary_reported_reasoning_tokens']}",
        "",
        "## By Condition / Provider",
        "",
        "| condition | provider | model | purpose | status | requests | errors | prompt | completion | reasoning |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["trace_rollup"]:
        lines.append(
            f"| {row['condition_group']} | {row['provider']} | {row['model']} "
            f"| {row['purpose']} | {row['status']} | {row['request_count']} "
            f"| {row['error_count']} | {row['prompt_tokens']} "
            f"| {row['completion_tokens']} | {row['reasoning_tokens']} |"
        )
    lines.extend([
        "",
        "## Diagnostics",
        "",
        "| condition | provider/harness | diagnosis | evidence |",
        "| --- | --- | --- | --- |",
    ])
    for row in payload["diagnostics"]:
        evidence = str(row.get("evidence") or "").replace("|", "\\|")
        lines.append(
            f"| {row['condition_group']} | {row['provider']} | {row['diagnosis']} | {evidence} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_provider_usage_report(
    proposals_dir: str | Path = "outputs/condition_proposals/live",
) -> ProviderUsageReport:
    target = Path(proposals_dir)
    runs_dir = target / "runs"
    trace_rows = _trace_rows(runs_dir)
    summary_rows = _summary_rows(runs_dir)
    harness_rows = _harness_rows(runs_dir)
    trace_rollup = _rollup_trace_rows(trace_rows)
    totals = {
        "trace_call_rows": len(trace_rows),
        "request_count": sum(int(row.get("request_count") or 0) for row in trace_rows),
        "error_count": sum(0 if row.get("status") == "ok" else 1 for row in trace_rows),
        "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in trace_rows),
        "completion_tokens": sum(int(row.get("completion_tokens") or 0) for row in trace_rows),
        "reasoning_tokens": sum(int(row.get("reasoning_tokens") or 0) for row in trace_rows),
        "summary_reported_request_count": sum(
            int(row.get("total_requests") or 0) for row in summary_rows),
        "summary_reported_error_count": sum(
            int(row.get("total_errors") or 0) for row in summary_rows),
        "summary_reported_prompt_tokens": sum(
            int(row.get("total_prompt_tokens") or 0) for row in summary_rows),
        "summary_reported_completion_tokens": sum(
            int(row.get("total_completion_tokens") or 0) for row in summary_rows),
        "summary_reported_reasoning_tokens": sum(
            int(row.get("total_reasoning_tokens") or 0) for row in summary_rows),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proposals_dir": str(target),
        "totals": totals,
        "trace_rollup": trace_rollup,
        "summary_rows": summary_rows,
        "harness_rows": harness_rows,
        "diagnostics": _diagnostics(trace_rows, harness_rows),
        "notes": [
            "Use trace_rollup for paper-level provider usage; it is rebuilt from raw call traces.",
            "model_call_summary.json rows are included for audit but may be stale after partial retries.",
            "For older traces without request_count, trace totals count one request per trace row; summary_reported_* totals preserve provider-ledger values when available.",
            "Prompt/response excerpts in raw traces are redacted and truncated; this report stores no API keys.",
        ],
    }
    reports_dir = target / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = reports_dir / "provider_usage_summary.json"
    csv_path = reports_dir / "provider_usage_by_condition.csv"
    md_path = reports_dir / "provider_usage_report.md"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, trace_rollup)
    _write_markdown(md_path, payload)
    return ProviderUsageReport(
        summary_json_path=str(summary_path),
        calls_csv_path=str(csv_path),
        markdown_path=str(md_path),
        total_requests=totals["request_count"],
        total_errors=totals["error_count"],
    )
