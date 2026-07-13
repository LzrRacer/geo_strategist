#!/usr/bin/env python3
"""Phase 10 — e-Stat getStatsData value retrieval runner.

Reads E3 retrieval requests, builds a value retrieval plan from cached metadata,
and optionally executes getStatsData calls for selected unambiguous table IDs.

Usage:
    # Cache-only planning (no network, no credentials required):
    .venv/bin/python scripts/run_estat_value_retrieval.py --latest-e3

    # Live e-Stat value retrieval (source .env first):
    set -a; . ./.env; set +a
    .venv/bin/python scripts/run_estat_value_retrieval.py --latest-e3 --allow-network

    # Explicit E3 run directory:
    .venv/bin/python scripts/run_estat_value_retrieval.py \\
        --e3-run-dir .runs/experiments/e3_estat_retrieval_llm/<run_id>

Security:
    - ESTAT_APP_ID is read from environment only; never printed.
    - Credential presence is reported as boolean only.
    - No .env file is opened in this script.
    - No fake table IDs or fabricated values are generated.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

E3_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "e3_estat_retrieval_llm"
VALUE_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_value_retrieval"
ESTAT_CACHE_ROOT = REPO_ROOT / ".data" / "api_raw" / "estat"


def _find_latest_e3() -> Path | None:
    if not E3_RUN_ROOT.exists():
        return None
    runs = [p for p in E3_RUN_ROOT.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if runs else None


def _load_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, records: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )


def run_value_retrieval(
    e3_run_dir: Path,
    allow_network: bool = False,
) -> Path:
    from geo_strategist.data.estat_retrieval import fetch_estat_stats_data, EStatRetrievalIssue
    from geo_strategist.data.estat_value_planning import (
        build_value_retrieval_plan, plan_to_dict, plan_to_md,
    )

    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = VALUE_RUN_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Credential check (boolean only)
    app_id_present = bool(os.environ.get("ESTAT_APP_ID", ""))
    print(f"ESTAT_APP_ID present: {str(app_id_present).lower()}")

    # Load E3 artifacts
    manifest = _load_json(e3_run_dir / "manifest.json") or {}
    e3_run_id = manifest.get("run_id", str(e3_run_dir.name))
    rr_data = _load_json(e3_run_dir / "retrieval_requests.json") or {}
    e3_requests = rr_data.get("valid_requests", [])

    print(f"E3 run: {e3_run_id}")
    print(f"E3 retrieval requests: {len(e3_requests)}")

    # Write input summary
    _write_json(out_dir / "input_e3_run.json", {
        "e3_run_id": e3_run_id,
        "e3_run_dir": str(e3_run_dir),
        "e3_request_count": len(e3_requests),
        "requests": e3_requests,
    })

    # Build plan
    plan = build_value_retrieval_plan(
        e3_requests=e3_requests,
        cache_root=ESTAT_CACHE_ROOT,
    )
    plan_dict = plan_to_dict(plan)
    plan_md = plan_to_md(plan)

    _write_json(out_dir / "estat_value_retrieval_plan.json", plan_dict)
    (out_dir / "estat_value_retrieval_plan.md").write_text(plan_md, encoding="utf-8")

    planning_issues = [
        {
            "issue_id": i.issue_id,
            "severity": i.severity,
            "issue_code": i.issue_code,
            "message": i.message,
            "context": i.context,
        }
        for i in plan.issues
    ]
    _write_jsonl(out_dir / "estat_value_planning_issues.jsonl", planning_issues)

    print(f"Plan: {plan.candidate_count} candidate(s), {plan.unambiguous_count} unambiguous")
    if plan.no_unambiguous_note:
        print(f"Note: {plan.no_unambiguous_note}")

    # Execute getStatsData for selected candidates
    selected = [c for c in plan.candidates if c.selected]
    all_records: list[dict] = []
    all_issues: list[dict] = []
    all_results: list[dict] = []
    cache_hits = 0
    live_fetches = 0

    for candidate in selected:
        rid = str(uuid.uuid4())
        result = fetch_estat_stats_data(
            stats_data_id=candidate.stats_data_id,
            retrieval_id=rid,
            cache_only=not allow_network,
            allow_network=allow_network,
            cache_root=ESTAT_CACHE_ROOT,
        )

        if result.cache_hit:
            cache_hits += 1
        if result.live_fetch:
            live_fetches += 1

        result_dict: dict = {
            "retrieval_id": result.retrieval_id,
            "stats_data_id": result.stats_data_id,
            "status": result.status,
            "cache_hit": result.cache_hit,
            "live_fetch": result.live_fetch,
            "record_count": result.record_count,
            "retrieved_at": result.retrieved_at,
            "cache_path": result.cache_path,
            "error_message": result.error_message,
        }
        all_results.append(result_dict)

        for rec in result.records:
            all_records.append(rec.model_dump())

        if result.issue is not None:
            iss = result.issue
            all_issues.append({
                "issue_id": iss.issue_id,
                "severity": iss.severity,
                "issue_code": iss.issue_code,
                "message": iss.detail,
                "context": iss.query_params,
            })

        status_label = result.status
        if result.issue:
            status_label += f" ({result.issue.issue_code})"
        print(f"  {candidate.stats_data_id}: {status_label}, {result.record_count} record(s)")

    _write_jsonl(out_dir / "estat_value_results.jsonl", all_results)
    _write_jsonl(out_dir / "estat_value_records.jsonl", all_records)
    _write_jsonl(out_dir / "estat_value_issues.jsonl", all_issues + planning_issues)

    actual_values = any(r.get("status") == "ok" and r.get("record_count", 0) > 0 for r in all_results)
    suitable_for_e3_rerun = actual_values and len(all_records) > 0

    manifest_out = {
        "run_id": run_id,
        "generated_at": generated_at,
        "e3_run_id": e3_run_id,
        "e3_run_dir": str(e3_run_dir),
        "plan_id": plan.plan_id,
        "allow_network": allow_network,
        "estat_app_id_present": app_id_present,
        "plan_candidate_count": plan.candidate_count,
        "plan_unambiguous_count": plan.unambiguous_count,
        "plan_selected_count": plan.selected_count,
        "retrieval_attempts": len(selected),
        "cache_hits": cache_hits,
        "live_fetches": live_fetches,
        "records_normalized": len(all_records),
        "issue_count": len(all_issues) + len(planning_issues),
        "actual_values_retrieved": actual_values,
        "suitable_for_e3_rerun": suitable_for_e3_rerun,
        "disclaimers": [
            "Experimental diagnostic output only.",
            "No LLM proposals generated.",
            "No cash-flow projections, parcel selection, or final recommendations.",
            "Values from getStatsData are official e-Stat data; interpret with provenance.",
            "No values are fabricated. No data available means no data available.",
        ],
    }
    _write_json(out_dir / "estat_value_manifest.json", manifest_out)

    # Report
    _write_report(out_dir, manifest_out, plan_dict, all_results, all_records, all_issues + planning_issues)

    print(f"\nOutput: {out_dir}/")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")

    return out_dir


def _write_report(
    out_dir: Path,
    manifest: dict,
    plan: dict,
    results: list[dict],
    records: list[dict],
    issues: list[dict],
) -> None:
    md = [
        "# e-Stat Value Retrieval Report",
        "",
        f"**Run ID:** `{manifest['run_id']}`",
        f"**Generated:** {manifest['generated_at']}",
        f"**E3 Run:** `{manifest['e3_run_id']}`",
        "",
        "## Planning Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| E3 requests analyzed | {manifest['plan_candidate_count']} |" ,
        f"| Cached metadata tables scanned | {plan.get('cached_metadata_tables_scanned', 0)} |",
        f"| Candidate tables found | {manifest['plan_candidate_count']} |",
        f"| Unambiguous table IDs | {manifest['plan_unambiguous_count']} |",
        f"| Selected for getStatsData | {manifest['plan_selected_count']} |",
        "",
    ]
    if plan.get("no_unambiguous_note"):
        md += [f"> **{plan['no_unambiguous_note']}**", ""]
    md += [f"**Recommendation:** {plan.get('recommendation', '')}", ""]

    md += [
        "## Retrieval Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| ESTAT_APP_ID present | {manifest['estat_app_id_present']} |",
        f"| Allow network | {manifest['allow_network']} |",
        f"| Retrieval attempts | {manifest['retrieval_attempts']} |",
        f"| Cache hits | {manifest['cache_hits']} |",
        f"| Live fetches | {manifest['live_fetches']} |",
        f"| Records normalized | {manifest['records_normalized']} |",
        f"| Actual values retrieved | {manifest['actual_values_retrieved']} |",
        f"| Suitable for E3 rerun | {manifest['suitable_for_e3_rerun']} |",
        "",
    ]

    if results:
        md += ["## Retrieval Results", "", "| statsDataId | Status | Records | Cache Hit |", "|-------------|--------|---------|-----------|"]
        for r in results:
            md.append(f"| `{r['stats_data_id']}` | {r['status']} | {r['record_count']} | {r['cache_hit']} |")
        md.append("")

    if issues:
        md += ["## Issues", ""]
        for iss in issues:
            md.append(f"- **[{iss.get('severity', 'info')}]** `{iss.get('issue_code', '')}`: {iss.get('message', '')}")
        md.append("")

    md += [
        "## Disclaimers",
        "",
    ]
    for d in manifest.get("disclaimers", []):
        md.append(f"- {d}")
    md.append("")

    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "report.json").write_text(
        json.dumps({
            "run_id": manifest["run_id"],
            "actual_values_retrieved": manifest["actual_values_retrieved"],
            "suitable_for_e3_rerun": manifest["suitable_for_e3_rerun"],
            "records_normalized": manifest["records_normalized"],
            "issue_count": manifest["issue_count"],
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="e-Stat getStatsData value retrieval.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest-e3", action="store_true", help="Use latest E3 run")
    group.add_argument("--e3-run-dir", help="Explicit E3 run directory")
    parser.add_argument("--allow-network", action="store_true", default=False,
                        help="Enable live e-Stat network calls (requires ESTAT_APP_ID in env)")
    args = parser.parse_args()

    if args.latest_e3:
        e3_dir = _find_latest_e3()
        if e3_dir is None:
            print("ERROR: No E3 run directory found.", file=sys.stderr)
            sys.exit(1)
    else:
        e3_dir = Path(args.e3_run_dir)
        if not e3_dir.exists():
            print(f"ERROR: E3 run directory not found: {e3_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"E3 run dir: {e3_dir}")
    out_dir = run_value_retrieval(e3_run_dir=e3_dir, allow_network=args.allow_network)
    print(f"\nValue retrieval complete: {out_dir}")


if __name__ == "__main__":
    main()
