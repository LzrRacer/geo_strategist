#!/usr/bin/env python
"""Filter population views to the configured Tokyo/Aichi/Osaka study area."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.study_area_filter import filter_study_area_population


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

    result = filter_study_area_population(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
    )
    console.print(f"Input found: {result.input_found}")
    console.print(f"Count rows read: {result.long_records_read}")
    console.print(f"Rate rows read: {result.rate_records_read}")
    console.print(f"Count rows written: {result.long_records_written}")
    console.print(f"Rate rows written: {result.rate_records_written}")
    console.print(f"Outside-scope rows: {result.outside_scope_rows}")
    console.print(f"Scope-unknown rows: {result.scope_unknown_rows}")
    console.print(f"Scope issues: {result.scope_issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
