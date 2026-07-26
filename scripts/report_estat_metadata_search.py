#!/usr/bin/env python3
"""Phase 11 — e-Stat metadata search report generator.

Reads the latest (or specified) metadata search run and produces a summary.

Usage:
    .venv/bin/python scripts/report_estat_metadata_search.py --latest
    .venv/bin/python scripts/report_estat_metadata_search.py \\
        --run-dir .runs/experiments/estat_metadata_search/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SEARCH_RUN_ROOT = REPO_ROOT / ".runs" / "experiments" / "estat_metadata_search"


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
    manifest = _load_json(run_dir / "metadata_search_manifest.json")
    results = _load_jsonl(run_dir / "metadata_search_results.jsonl")
    issues = _load_jsonl(run_dir / "metadata_search_issues.jsonl")

    if manifest is None:
        print(f"ERROR: manifest not found in {run_dir}", file=sys.stderr)
        sys.exit(1)

    print("=" * 70)
    print("e-Stat Metadata Search Report")
    print("=" * 70)
    print(f"Run ID:             {manifest.get('run_id', 'N/A')}")
    print(f"Generated:          {manifest.get('generated_at', 'N/A')}")
    print(f"ESTAT_APP_ID present:{manifest.get('estat_app_id_present', 'N/A')}")
    print(f"Allow network:      {manifest.get('allow_network', False)}")
    print()
    print(f"Query count:        {manifest.get('query_count', 0)}")
    print(f"Live fetches:       {manifest.get('live_fetch_count', 0)}")
    print(f"Cache hits:         {manifest.get('cache_hit_count', 0)}")
    print(f"Total tables found: {manifest.get('total_tables_found', 0)}")
    print(f"Unique table IDs:   {manifest.get('unique_candidate_count', 0)}")
    print(f"Issues:             {manifest.get('issue_count', 0)}")
    print()

    # Per-query breakdown
    print("Tables by query:")
    for r in results:
        print(f"  [{r.get('status', ''):20s}] '{r['query']}': {r['table_count']} table(s)")

    print()

    # Candidate IDs
    unique_ids = manifest.get("unique_candidate_ids", [])
    if unique_ids:
        print(f"Candidate 医療施設調査 table IDs ({len(unique_ids)}):")
        for sid in unique_ids:
            # Find title from results
            title = ""
            stat_name = ""
            for r in results:
                for t in r.get("tables", []):
                    if t.get("stats_data_id") == sid:
                        title = t.get("title") or t.get("statistics_name", "")
                        stat_name = t.get("stat_name", "")
                        break
            print(f"  {sid}: {stat_name} — {title[:50]}")
    else:
        print("No candidate table IDs found.")
        if not manifest.get("allow_network"):
            print()
            print("TIP: Run with --allow-network to fetch live metadata:")
            print("  set -a; . ./.env; set +a")
            print("  .venv/bin/python scripts/run_estat_metadata_search.py \\")
            print("    --allow-network --queries 医療施設調査 医療施設 病院 一般診療所")

    print()

    if issues:
        print("Issues:")
        for iss in issues:
            print(f"  [{iss.get('severity')}] {iss.get('issue_code')}: {iss.get('message', '')}")
        print()

    # Next step recommendation
    if unique_ids:
        print("Recommended next command:")
        print("  .venv/bin/python scripts/run_estat_value_retrieval.py --latest-e3")
    else:
        print("Recommended next command:")
        print("  Run with --allow-network to search for 医療施設調査 metadata.")

    print()
    for d in manifest.get("disclaimers", []):
        print(f"DISCLAIMER: {d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Report on metadata search run.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--latest", action="store_true")
    group.add_argument("--run-dir")
    args = parser.parse_args()

    if args.latest:
        run_dir = _find_latest(SEARCH_RUN_ROOT)
        if run_dir is None:
            print("No metadata search run found. Run run_estat_metadata_search.py first.", file=sys.stderr)
            sys.exit(1)
    else:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            print(f"Run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)

    report_run(run_dir)


if __name__ == "__main__":
    main()
