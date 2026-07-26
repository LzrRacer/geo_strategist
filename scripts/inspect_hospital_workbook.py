#!/usr/bin/env python
"""Inspect the hospital cash-flow workbook structure."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.workbook import inspect_hospital_workbook


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Return nonzero if the workbook is missing.",
    )
    args = parser.parse_args()

    profile = inspect_hospital_workbook(repo_root=Path(args.repo_root))

    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    if profile.found:
        console.print(f"Workbook: {profile.path}")
        console.print(f"Sheets: {len(profile.sheet_names)}")
    else:
        console.print("Workbook: not found")
    console.print(f"Profile JSON: {profile.output_json}")
    console.print(f"Profile Markdown: {profile.output_markdown}")
    return 1 if args.require_data and not profile.found else 0


if __name__ == "__main__":
    raise SystemExit(main())
