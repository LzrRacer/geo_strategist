#!/usr/bin/env python
"""Build pre-demand population-base views from geography-grain rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.population_base import build_population_base


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--config",
        default="configs/study_area_tokyo_aichi_osaka.yaml",
        help="Study-area config path.",
    )
    args = parser.parse_args()

    result = build_population_base(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
    )
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Municipality records: {result.municipality_records_written}")
    console.print(f"Prefecture-total records: {result.prefecture_total_records_written}")
    console.print(f"Excluded records: {result.excluded_records}")
    console.print(f"Issues: {result.issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
