#!/usr/bin/env python
"""Inventory local real-data files by metadata only."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.inventory import inventory_local_data


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--no-provenance-jsonl",
        action="store_true",
        help="Skip writing JSONL provenance records.",
    )
    args = parser.parse_args()

    result = inventory_local_data(
        repo_root=Path(args.repo_root),
        provenance_path=None
        if args.no_provenance_jsonl
        else ".cache/inventory/local_data_provenance.jsonl",
    )

    for warning in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Scanned root: {result.scanned_root or 'none'}")
    console.print(f"Files inventoried: {len(result.files)}")
    console.print(f"Inventory JSON: {result.output_path}")
    if result.provenance_path:
        console.print(f"Provenance JSONL: {result.provenance_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
