#!/usr/bin/env python
"""Report summary counts for pre-demand population-base views."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.population_base_report import build_population_base_report


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

    result = build_population_base_report(
        repo_root=Path(args.repo_root),
        config_path=Path(args.config),
    )
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Candidate model-input rows: {result.candidate_model_input_rows}")
    console.print(f"Context prefecture-total rows: {result.context_prefecture_total_rows}")
    console.print(f"Unknown geography-grain rows: {result.unknown_geography_grain_rows}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
