#!/usr/bin/env python
"""Diagnose population mapping evidence without editing configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.population_mapping_diagnostics import diagnose_population_mappings


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    args = parser.parse_args()

    result = diagnose_population_mappings(repo_root=Path(args.repo_root))
    console.print(f"Source tables: {result.source_table_count}")
    console.print(f"Diagnostics: {result.diagnostics_count}")
    console.print(f"Unresolved mappings: {result.unresolved_mapping_count}")
    console.print(f"Quarantined issues: {result.quarantined_issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
