#!/usr/bin/env python
"""Run deterministic QA over population geography keys and units."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.population_geography_qa import build_population_geography_qa


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    args = parser.parse_args()

    result = build_population_geography_qa(repo_root=Path(args.repo_root))
    console.print(f"Input found: {result.input_found}")
    console.print(f"Long records read: {result.records_read}")
    console.print(f"Rate records read: {result.rate_records_read}")
    console.print(f"Geography keys written: {result.keys_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
