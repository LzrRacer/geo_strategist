#!/usr/bin/env python
"""Review extraction mappings and prepare manual mapping candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.mapping_review import review_extraction_mappings


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    args = parser.parse_args()

    result = review_extraction_mappings(repo_root=Path(args.repo_root))
    console.print(f"Source tables: {result.source_table_count}")
    console.print(f"Normalized records: {result.normalized_record_count}")
    console.print(f"Inferred mappings: {result.inferred_mapping_count}")
    console.print(f"Unresolved mappings: {result.unresolved_mapping_count}")
    console.print(f"Warnings: {result.warning_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
