"""Ingest real land-price data from MLIT Reinfolib API (or report unavailable).

Usage examples:
  # Credentials-absent or cache-only run (safe, no network):
  .venv/bin/python scripts/ingest_land_prices.py

  # Smoke run — 1 prefecture, 1 year, network enabled:
  .venv/bin/python scripts/ingest_land_prices.py --allow-network --prefecture-limit 1 --years 2023

  # Full run — all 3 prefectures, default years:
  .venv/bin/python scripts/ingest_land_prices.py --allow-network

  # Cache-only replay (use only pre-fetched cache, no new calls):
  .venv/bin/python scripts/ingest_land_prices.py --cache-only
"""
import argparse
from pathlib import Path
from geo_strategist.data.land_price_ingestion import run_land_price_ingestion


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MLIT Reinfolib land-price data")
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
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        metavar="YEAR",
        help="Survey years to fetch (e.g. --years 2023 2022). Defaults to [2023, 2022].",
    )
    args = parser.parse_args()

    result = run_land_price_ingestion(
        repo_root=Path("."),
        allow_network=args.allow_network,
        cache_only=args.cache_only,
        prefecture_limit=args.prefecture_limit,
        target_years=args.years,
    )
    print(f"Study area: {result.study_area_id}")
    print(f"Source available: {result.source_available}")
    print(f"Land-price records ingested: {result.records_ingested}")
    print(f"Municipality land features written: {result.municipality_features_written}")
    print(f"Issues: {result.issue_count}")
    print(f"Blocking errors: {result.blocking_error_count}")
    if not result.source_available:
        print(
            "Note: Set REINFOLIB_API_KEY and pass --allow-network to enable real land-price data."
        )
    for label, path in result.output_paths.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
