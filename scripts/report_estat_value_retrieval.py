#!/usr/bin/env python3
"""Phase 10 — e-Stat value retrieval report generator.

Reads the latest (or specified) value retrieval run and produces a summary report.

Usage:
    .venv/bin/python scripts/report_estat_value_retrieval.py --latest
    .venv/bin/python scripts/report_estat_value_retrieval.py --run-dir .runs/experiments/estat_value_retrieval/<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
VALUE_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_value_retrieval"


def _find_latest(run_root: Path) -> Path | None:
    if not run_root.exists():
        return None
    runs = [p for p in run_root.iterdir() if p.is_dir()]
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


def report_run(run_dir: Path) -> None:
    manifest = _load_json(run_dir / "estat_value_manifest.json")
    plan = _load_json(run_dir / "estat_value_retrieval_plan.json")
    results = _load_jsonl(run_dir / "estat_value_results.jsonl")
    records = _load_jsonl(run_dir / "estat_value_records.jsonl")
    issues = _load_jsonl(run_dir / "estat_value_issues.jsonl")

    if manifest is None:
        print(f"ERROR: manifest not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("e-Stat Value Retrieval Report")
    print("=" * 70)
    print(f"Run ID:              {manifest.get('run_id', 'N/A')}")
    print(f"Generated:           {manifest.get('generated_at', 'N/A')}")
    print(f"E3 Run:              {manifest.get('e3_run_id', 'N/A')}")
    print(f"ESTAT_APP_ID present:{manifest.get('estat_app_id_present', 'N/A')}")
    print(f"Allow network:       {manifest.get('allow_network', False)}")
    print()
    print("Planning:")
    print(f"  E3 requests analyzed:      {plan.get('e3_request_count', 'N/A') if plan else 'N/A'}")
    print(f"  Metadata tables scanned:   {plan.get('cached_metadata_tables_scanned', 'N/A') if plan else 'N/A'}")
    print(f"  Candidate tables found:    {manifest.get('plan_candidate_count', 0)}")
    print(f"  Unambiguous table IDs:     {manifest.get('plan_unambiguous_count', 0)}")
    print(f"  Selected for getStatsData: {manifest.get('plan_selected_count', 0)}")
    print()
    print("Retrieval:")
    print(f"  Retrieval attempts:        {manifest.get('retrieval_attempts', 0)}")
    print(f"  Cache hits:                {manifest.get('cache_hits', 0)}")
    print(f"  Live fetches:              {manifest.get('live_fetches', 0)}")
    print(f"  Records normalized:        {manifest.get('records_normalized', 0)}")
    print(f"  Actual values retrieved:   {manifest.get('actual_values_retrieved', False)}")
    print(f"  Suitable for E3 rerun:     {manifest.get('suitable_for_e3_rerun', False)}")
    print()
    print(f"Issues: {manifest.get('issue_count', 0)}")

    if issues:
        for iss in issues:
            sev = iss.get("severity", "info")
            code = iss.get("issue_code", "")
            msg = iss.get("message", "")
            print(f"  [{sev}] {code}: {msg}")

    print()
    if plan and plan.get("no_unambiguous_note"):
        print(f"NOTE: {plan['no_unambiguous_note']}")
    if plan and plan.get("recommendation"):
        print(f"RECOMMENDATION: {plan['recommendation']}")

    if not manifest.get("actual_values_retrieved"):
        print()
        print(
            "No actual e-Stat statistical values were retrieved in this run. "
            "This is expected in a cache-only or no-credential environment. "
            "To retrieve values, set ESTAT_APP_ID in the environment and run "
            "with --allow-network."
        )

    for d in manifest.get("disclaimers", []):
        print(f"DISCLAIMER: {d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report on e-Stat value retrieval run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true", help="Use latest run")
    group.add_argument("--run-dir", help="Explicit run directory")
    args = parser.parse_args()

    if args.latest:
        run_dir = _find_latest(VALUE_RUN_ROOT)
        if run_dir is None:
            print("No value retrieval run found. Run run_estat_value_retrieval.py first.", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

    report_run(run_dir)


if __name__ == "__main__":
    main()
