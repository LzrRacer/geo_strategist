#!/usr/bin/env python3
"""Phase 12 — inspect cached e-Stat getStatsData dimensions.

This script reads cached getStatsData responses and writes CLASS_INF inspection
artifacts under .runs/experiments/estat_dimension_inspection/<run_id>/.

No network calls are made here. Credentials are not required or read.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

VALUE_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_value_retrieval"
DIMENSION_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_dimension_inspection"
ESTAT_CACHE_ROOT = REPO_ROOT / ".data" / "api_raw" / "estat"


def _find_latest_value_run() -> Path | None:
    if not VALUE_RUN_ROOT.exists():
        return None
    runs = [p for p in VALUE_RUN_ROOT.iterdir() if p.is_dir()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if runs else None


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


def _latest_value_run_cache_paths(value_run_dir: Path) -> list[tuple[str, Path]]:
    rows = _load_jsonl(value_run_dir / "estat_value_results.jsonl")
    paths: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for row in rows:
        cache_path = row.get("cache_path")
        stats_data_id = row.get("stats_data_id", "")
        if not cache_path:
            continue
        path = Path(cache_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path in seen:
            continue
        seen.add(path)
        paths.append((str(stats_data_id), path))
    return paths


def run_dimension_inspection(
    *,
    latest_value_run: bool = False,
    stats_data_ids: list[str] | None = None,
    target_municipalities: list[str] | None = None,
    cache_root: Path = ESTAT_CACHE_ROOT,
) -> Path:
    from geo_strategist.data.estat_dimension_inspection import (
        DEFAULT_TARGET_MUNICIPALITIES,
        LoadedGetStatsDataResponse,
        build_estat_dimension_inspection_report,
        load_cached_getstatsdata_responses,
        report_to_markdown,
        write_json,
        write_jsonl,
    )

    run_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    out_dir = DIMENSION_RUN_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = tuple(target_municipalities or DEFAULT_TARGET_MUNICIPALITIES)
    source_value_run_dir: Path | None = None

    if latest_value_run:
        source_value_run_dir = _find_latest_value_run()
        if source_value_run_dir is None:
            print("ERROR: No e-Stat value retrieval run found.", file=sys.stderr)
            sys.exit(1)
        loaded: list[LoadedGetStatsDataResponse] = []
        for stats_data_id, cache_path in _latest_value_run_cache_paths(source_value_run_dir):
            if not cache_path.exists():
                loaded.extend(load_cached_getstatsdata_responses(cache_root, [stats_data_id]))
                continue
            try:
                data = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                loaded.extend(load_cached_getstatsdata_responses(cache_path.parent, [stats_data_id]))
                continue
            loaded.append(LoadedGetStatsDataResponse(
                stats_data_id=stats_data_id,
                cache_path=cache_path,
                data=data,
            ))
    else:
        loaded = load_cached_getstatsdata_responses(cache_root, stats_data_ids)

    report, dimensions, classes = build_estat_dimension_inspection_report(
        loaded,
        target_municipality_codes=targets,
        generated_at=generated_at,
    )

    issue_rows = list(report.issues)
    table_rows = list(report.tables)
    write_jsonl(out_dir / "dimension_inspection_tables.jsonl", table_rows)
    write_jsonl(out_dir / "dimension_inspection_dimensions.jsonl", dimensions)
    write_jsonl(out_dir / "dimension_inspection_classes.jsonl", classes)
    write_jsonl(out_dir / "dimension_inspection_issues.jsonl", issue_rows)

    report_payload = {
        "generated_at": report.generated_at,
        "target_municipality_codes": list(report.target_municipality_codes),
        "table_count": report.table_count,
        "dimension_count": report.dimension_count,
        "class_count": report.class_count,
        "municipality_filter_ready_count": report.municipality_filter_ready_count,
        "not_municipality_grain_count": report.not_municipality_grain_count,
        "needs_manual_dimension_review_count": report.needs_manual_dimension_review_count,
        "unusable_for_e3_municipality_validation_count": report.unusable_for_e3_municipality_validation_count,
        "suitable_for_e3_rerun_with_municipality_values": report.suitable_for_e3_rerun_with_municipality_values,
        "issue_counts_by_severity": report.issue_counts_by_severity,
        "issue_counts_by_code": report.issue_counts_by_code,
        "recommended_followup_queries": list(report.recommended_followup_queries),
    }
    write_json(out_dir / "dimension_inspection_report.json", report_payload)
    (out_dir / "dimension_inspection_report.md").write_text(report_to_markdown(report), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "latest_value_run": latest_value_run,
        "source_value_run_dir": str(source_value_run_dir) if source_value_run_dir else None,
        "stats_data_ids": stats_data_ids or [t.stats_data_id for t in report.tables],
        "target_municipality_codes": list(targets),
        "cache_root": str(cache_root),
        "output_dir": str(out_dir),
        "output_paths": {
            "dimension_inspection_manifest": str(out_dir / "dimension_inspection_manifest.json"),
            "dimension_inspection_tables": str(out_dir / "dimension_inspection_tables.jsonl"),
            "dimension_inspection_dimensions": str(out_dir / "dimension_inspection_dimensions.jsonl"),
            "dimension_inspection_classes": str(out_dir / "dimension_inspection_classes.jsonl"),
            "dimension_inspection_issues": str(out_dir / "dimension_inspection_issues.jsonl"),
            "dimension_inspection_report_json": str(out_dir / "dimension_inspection_report.json"),
            "dimension_inspection_report_md": str(out_dir / "dimension_inspection_report.md"),
        },
        "table_count": report.table_count,
        "dimension_count": report.dimension_count,
        "class_count": report.class_count,
        "municipality_filter_ready_count": report.municipality_filter_ready_count,
        "not_municipality_grain_count": report.not_municipality_grain_count,
        "needs_manual_dimension_review_count": report.needs_manual_dimension_review_count,
        "unusable_for_e3_municipality_validation_count": report.unusable_for_e3_municipality_validation_count,
        "issue_counts_by_severity": report.issue_counts_by_severity,
        "issue_counts_by_code": report.issue_counts_by_code,
        "disclaimers": [
            "Dimension inspection only; no e-Stat network call is made.",
            "No LLM proposal generation, reviewer agents, tree search, cash-flow modeling, site selection, or final recommendations are implemented.",
        ],
    }
    write_json(out_dir / "dimension_inspection_manifest.json", manifest)

    print(f"Tables inspected: {report.table_count}")
    print(f"Municipality filter ready: {report.municipality_filter_ready_count}")
    print(f"Issues: {sum(report.issue_counts_by_severity.values())}")
    print(f"Output: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect cached e-Stat getStatsData CLASS_INF dimensions.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest-value-run", action="store_true", help="Inspect cache paths from the latest value retrieval run.")
    group.add_argument("--stats-data-ids", nargs="+", help="Inspect cached unfiltered getStatsData responses for these statsDataIds.")
    parser.add_argument("--target-municipalities", nargs="+", help="Target municipality JIS codes.")
    args = parser.parse_args()

    run_dimension_inspection(
        latest_value_run=args.latest_value_run,
        stats_data_ids=args.stats_data_ids,
        target_municipalities=args.target_municipalities,
    )


if __name__ == "__main__":
    main()
