#!/usr/bin/env python
"""Inspect population workbook structures."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.population import inspect_population_data


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Return nonzero if population workbooks are missing.",
    )
    args = parser.parse_args()

    profile = inspect_population_data(repo_root=Path(args.repo_root))

    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Population directory: {profile.source_directory or 'none'}")
    console.print(f"Workbook files: {len(profile.files)}")
    console.print(f"Profile JSON: {profile.output_json}")
    console.print(f"Profile Markdown: {profile.output_markdown}")
    return 1 if args.require_data and not profile.found else 0


if __name__ == "__main__":
    raise SystemExit(main())
