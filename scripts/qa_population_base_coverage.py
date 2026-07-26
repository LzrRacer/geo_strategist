#!/usr/bin/env python
"""Run pre-demand coverage QA for population-base rows."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.population_base_coverage_qa import run_population_base_coverage_qa


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

    result = run_population_base_coverage_qa(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
    )
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Matrix rows written: {result.matrix_rows_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Model-blocking errors: {result.model_blocking_error_count}")
    console.print(f"Duplicate keys: {result.duplicate_key_count}")
    console.print(f"Conflicting values: {result.conflicting_value_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
