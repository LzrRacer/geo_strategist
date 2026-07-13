#!/usr/bin/env python
"""Build the population rates long view from normalized records."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.views.population_rates import build_population_rates_long


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    args = parser.parse_args()

    result = build_population_rates_long(repo_root=Path(args.repo_root))
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Quality issues: {result.quality_issue_count}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
