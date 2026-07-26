"""Ingest real healthcare supply data from Yahoo Local Search API (or report unavailable).

Usage examples:
  # Credentials-absent or cache-only run (safe, no network):
  .venv/bin/python scripts/ingest_healthcare_supply.py

  # Smoke run — 1 prefecture, network enabled:
  .venv/bin/python scripts/ingest_healthcare_supply.py --allow-network --prefecture-limit 1

  # Full run — all 3 prefectures:
  .venv/bin/python scripts/ingest_healthcare_supply.py --allow-network

  # Cache-only replay:
  .venv/bin/python scripts/ingest_healthcare_supply.py --cache-only
"""
import argparse
from pathlib import Path
from geo_strategist.data.healthcare_supply_ingestion import run_healthcare_supply_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Yahoo Local Search healthcare supply data")
    parser.add_argument(
        "--allow-network",
        action="store_true",
        default=False,
        help="Enable live API calls. Without this flag, only cached responses are used.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        default=False,
        help="Use only cached responses; do not make new API calls even if --allow-network is set.",
    )
    parser.add_argument(
        "--prefecture-limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit to first N target prefectures (useful for smoke runs).",
    )
    args = parser.parse_args()

    result = run_healthcare_supply_ingestion(
        repo_root=Path("."),
        allow_network=args.allow_network,
        cache_only=args.cache_only,
        prefecture_limit=args.prefecture_limit,
    )
    print(f"Study area: {result.study_area_id}")
    print(f"Source available: {result.source_available}")
    print(f"Healthcare supply records ingested: {result.records_ingested}")
    print(f"Municipality healthcare supply features written: {result.municipality_features_written}")
    print(f"Issues: {result.issue_count}")
    print(f"Blocking errors: {result.blocking_error_count}")
    if not result.source_available:
        print(
            "Note: Set YAHOO_CLIENT_ID and pass --allow-network to enable real healthcare supply data."
        )
    for label, path in result.output_paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
