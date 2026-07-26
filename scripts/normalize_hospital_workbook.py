#!/usr/bin/env python
"""Normalize the hospital workbook into auditable intermediate records."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.normalizers.hospital_workbook import normalize_hospital_workbook


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Return nonzero if the hospital workbook is missing.",
    )
    args = parser.parse_args()

    result = normalize_hospital_workbook(repo_root=Path(args.repo_root))
    console.print(f"Input found: {result.found}")
    console.print(f"Source tables: {len(result.source_tables)}")
    console.print(f"Normalized records: {len(result.normalized_records)}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 1 if args.require_data and not result.found else 0


if __name__ == "__main__":
    raise SystemExit(main())
