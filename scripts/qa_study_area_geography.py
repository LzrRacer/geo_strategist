#!/usr/bin/env python
"""Run deterministic target-scoped geography QA for Tokyo/Aichi/Osaka."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.study_area_geography_qa import run_study_area_geography_qa


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

    result = run_study_area_geography_qa(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
    )
    console.print(f"Input found: {result.input_found}")
    console.print(f"Count rows read: {result.count_rows_read}")
    console.print(f"Rate rows read: {result.rate_rows_read}")
    console.print(f"Geography keys written: {result.geography_keys_written}")
    console.print(f"Duplicate target geography keys: {result.duplicate_key_count}")
    console.print(f"Issues: {result.issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
