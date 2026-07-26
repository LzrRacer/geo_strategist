#!/usr/bin/env python3
"""Phase 12 — cdArea-filtered e-Stat value retrieval for verified municipalities.

Reads a dimension inspection run and executes getStatsData only for tables whose
CLASS_INF area dimension explicitly contains the target municipality code.
Network access is disabled by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DIMENSION_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_dimension_inspection"
MUNICIPALITY_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_municipality_value_retrieval"
ESTAT_CACHE_ROOT = REPO_ROOT / ".data" / "api_raw" / "estat"


def _find_latest_dimension_inspection() -> Path | None:
    if not DIMENSION_RUN_ROOT.exists():
        return None
    runs = [p for p in DIMENSION_RUN_ROOT.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if runs else None


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _issue_counts(issues: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    by_severity = dict(sorted(Counter(i.get("severity", "info") for i in issues).items()))
    by_code = dict(sorted(Counter(i.get("issue_code", "") for i in issues).items()))
    return by_severity, by_code


def _load_dimension_report(dimension_run_dir: Path):
    from geo_strategist.data.estat_dimension_inspection import EStatDimensionInspectionReport, EStatTableInspection

    report_json = _load_json(dimension_run_dir / "dimension_inspection_report.json")
    table_rows = _load_jsonl(dimension_run_dir / "dimension_inspection_tables.jsonl")
    issue_rows = _load_jsonl(dimension_run_dir / "dimension_inspection_issues.jsonl")
    tables = tuple(EStatTableInspection(
        stats_data_id=str(row.get("stats_data_id", "")),
        table_title=row.get("table_title"),
        stat_name=row.get("stat_name"),
        survey_date=row.get("survey_date"),
        cache_path=row.get("cache_path"),
        status=str(row.get("status", "")),
        dimension_count=int(row.get("dimension_count", 0) or 0),
        class_count=int(row.get("class_count", 0) or 0),
        has_area_dimension=bool(row.get("has_area_dimension", False)),
        area_dimension_id=row.get("area_dimension_id"),
        area_dimension_name=row.get("area_dimension_name"),
        area_class_count=int(row.get("area_class_count", 0) or 0),
        target_municipality_codes_present=tuple(row.get("target_municipality_codes_present", [])),
        target_municipality_codes_missing=tuple(row.get("target_municipality_codes_missing", [])),
        hospital_facility_category_matches=tuple(row.get("hospital_facility_category_matches", [])),
        hospital_facility_category_match_count=int(row.get("hospital_facility_category_match_count", 0) or 0),
        readiness_classification=str(row.get("readiness_classification", "")),
        issue_codes=tuple(row.get("issue_codes", [])),
    ) for row in table_rows)
    report = EStatDimensionInspectionReport(
        generated_at=str(report_json.get("generated_at", "")),
        target_municipality_codes=tuple(report_json.get("target_municipality_codes", [])),
        table_count=int(report_json.get("table_count", len(tables)) or len(tables)),
        dimension_count=int(report_json.get("dimension_count", 0) or 0),
        class_count=int(report_json.get("class_count", 0) or 0),
        municipality_filter_ready_count=int(report_json.get("municipality_filter_ready_count", 0) or 0),
        not_municipality_grain_count=int(report_json.get("not_municipality_grain_count", 0) or 0),
        needs_manual_dimension_review_count=int(report_json.get("needs_manual_dimension_review_count", 0) or 0),
        unusable_for_e3_municipality_validation_count=int(report_json.get("unusable_for_e3_municipality_validation_count", 0) or 0),
        suitable_for_e3_rerun_with_municipality_values=bool(report_json.get("suitable_for_e3_rerun_with_municipality_values", False)),
        issue_counts_by_severity=dict(report_json.get("issue_counts_by_severity", {})),
        issue_counts_by_code=dict(report_json.get("issue_counts_by_code", {})),
        recommended_followup_queries=tuple(report_json.get("recommended_followup_queries", [])),
        tables=tables,
        issues=(),
    )
    return report, issue_rows


def run_municipality_value_retrieval(
    *,
    dimension_run_dir: Path,
    allow_network: bool = False,
    cache_root: Path = ESTAT_CACHE_ROOT,
) -> Path:
    from geo_strategist.data.estat_dimension_inspection import (
        dataclass_to_dict,
        municipality_plan_to_markdown,
        plan_municipality_filtered_retrievals,
        write_json,
        write_jsonl,
    )
    from geo_strategist.data.estat_retrieval import fetch_estat_stats_data

    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = MUNICIPALITY_RUN_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    app_id_present = bool(os.environ.get("ESTAT_APP_ID", ""))
    print(f"ESTAT_APP_ID present: {str(app_id_present).lower()}")
    print(f"Allow network:        {allow_network}")

    inspection_report, source_issues = _load_dimension_report(dimension_run_dir)
    plan = plan_municipality_filtered_retrievals(inspection_report)
    plan_payload = dataclass_to_dict(plan)
    write_json(out_dir / "municipality_value_plan.json", plan_payload)

    results: list[dict] = []
    records: list[dict] = []
    issues: list[dict] = []
    for issue in plan.issues:
        issues.append(dataclass_to_dict(issue))

    cache_hits = 0
    live_fetches = 0
    for item in plan.items:
        result = fetch_estat_stats_data(
            stats_data_id=item.stats_data_id,
            cd_area=item.target_municipality_code,
            retrieval_id=str(uuid.uuid4()),
            cache_only=not allow_network,
            allow_network=allow_network,
            cache_root=cache_root,
        )
        if result.cache_hit:
            cache_hits += 1
        if result.live_fetch:
            live_fetches += 1
        result_row = {
            "stats_data_id": item.stats_data_id,
            "table_title": item.table_title,
            "target_municipality_code": item.target_municipality_code,
            "area_dimension_id": item.area_dimension_id,
            "status": result.status,
            "cache_hit": result.cache_hit,
            "live_fetch": result.live_fetch,
            "record_count": result.record_count,
            "retrieved_at": result.retrieved_at,
            "cache_path": result.cache_path,
            "error_message": result.error_message,
        }
        results.append(result_row)
        for rec in result.records:
            row = rec.model_dump()
            row["target_municipality_code"] = item.target_municipality_code
            records.append(row)
        if result.issue is not None:
            issues.append({
                "issue_id": result.issue.issue_id,
                "severity": result.issue.severity,
                "issue_code": result.issue.issue_code,
                "message": result.issue.detail,
                "context": result.issue.query_params,
            })
        print(
            f"  {item.stats_data_id} cdArea={item.target_municipality_code}: "
            f"{result.status}, {result.record_count} record(s)"
        )

    if not allow_network and plan.retrieval_count > 0 and not records:
        issues.append({
            "issue_id": "cache_only_no_live_fetch",
            "severity": "info",
            "issue_code": "cache_only_mode_no_live_fetch",
            "message": "Cache-only mode performed no live e-Stat requests; use --allow-network only after credentials are provided in the environment.",
            "context": {"retrievals_planned": plan.retrieval_count},
        })

    write_jsonl(out_dir / "municipality_value_results.jsonl", results)
    write_jsonl(out_dir / "municipality_value_records.jsonl", records)
    write_jsonl(out_dir / "municipality_value_issues.jsonl", issues)

    actual_values = any(r.get("status") == "ok" and r.get("record_count", 0) > 0 for r in results)
    suitable = actual_values and bool(records)
    issue_counts_by_severity, issue_counts_by_code = _issue_counts(issues)
    source_issue_counts_by_severity, source_issue_counts_by_code = _issue_counts(source_issues)
    manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "dimension_run_dir": str(dimension_run_dir),
        "allow_network": allow_network,
        "estat_app_id_present": app_id_present,
        "inspection_table_count": inspection_report.table_count,
        "municipality_filter_ready_table_count": inspection_report.municipality_filter_ready_count,
        "retrievals_planned": plan.retrieval_count,
        "retrieval_attempts": len(results),
        "cache_hits": cache_hits,
        "live_fetches": live_fetches,
        "records_normalized": len(records),
        "actual_values_retrieved": actual_values,
        "suitable_for_e3_rerun_with_municipality_values": suitable,
        "issue_counts_by_severity": issue_counts_by_severity,
        "issue_counts_by_code": issue_counts_by_code,
        "source_dimension_issue_counts_by_severity": source_issue_counts_by_severity,
        "source_dimension_issue_counts_by_code": source_issue_counts_by_code,
        "output_paths": {
            "municipality_value_manifest": str(out_dir / "municipality_value_manifest.json"),
            "municipality_value_plan": str(out_dir / "municipality_value_plan.json"),
            "municipality_value_results": str(out_dir / "municipality_value_results.jsonl"),
            "municipality_value_records": str(out_dir / "municipality_value_records.jsonl"),
            "municipality_value_issues": str(out_dir / "municipality_value_issues.jsonl"),
            "municipality_value_report_json": str(out_dir / "municipality_value_report.json"),
            "municipality_value_report_md": str(out_dir / "municipality_value_report.md"),
        },
        "disclaimers": [
            "cdArea retrieval is attempted only for verified target municipality codes.",
            "No LLM proposal generation, reviewer agents, tree search, cash-flow modeling, site selection, or final recommendations are implemented.",
        ],
    }
    write_json(out_dir / "municipality_value_manifest.json", manifest)
    write_json(out_dir / "municipality_value_report.json", {
        "run_id": run_id,
        "generated_at": generated_at,
        "retrievals_planned": plan.retrieval_count,
        "retrieval_attempts": len(results),
        "records_normalized": len(records),
        "actual_values_retrieved": actual_values,
        "suitable_for_e3_rerun_with_municipality_values": suitable,
        "issue_counts_by_severity": issue_counts_by_severity,
        "issue_counts_by_code": issue_counts_by_code,
        "source_dimension_issue_counts_by_severity": source_issue_counts_by_severity,
        "source_dimension_issue_counts_by_code": source_issue_counts_by_code,
    })
    (out_dir / "municipality_value_report.md").write_text(
        municipality_plan_to_markdown(manifest, plan, results, issues),
        encoding="utf-8",
    )

    print(f"Retrievals planned: {plan.retrieval_count}")
    print(f"Retrievals attempted: {len(results)}")
    print(f"Records normalized: {len(records)}")
    print(f"Suitable for E3 rerun with municipality values: {str(suitable).lower()}")
    print(f"Output: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cdArea-filtered e-Stat retrieval from a dimension inspection run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest-dimension-inspection", action="store_true", help="Use latest dimension inspection run.")
    group.add_argument("--dimension-run-dir", help="Explicit dimension inspection run directory.")
    parser.add_argument("--allow-network", action="store_true", default=False, help="Enable live e-Stat network calls.")
    args = parser.parse_args()

    if args.latest_dimension_inspection:
        run_dir = _find_latest_dimension_inspection()
        if run_dir is None:
            print("ERROR: No dimension inspection run found.", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = Path(args.dimension_run_dir)
        if not run_dir.exists():
            print(f"ERROR: Dimension inspection run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

    run_municipality_value_retrieval(
        dimension_run_dir=run_dir,
        allow_network=args.allow_network,
    )


if __name__ == "__main__":
    main()
