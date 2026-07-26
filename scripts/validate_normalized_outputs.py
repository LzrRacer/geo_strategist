#!/usr/bin/env python
"""Validate normalized output artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.validate_normalized import validate_normalized_outputs


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Return nonzero if normalized outputs are absent.",
    )
    args = parser.parse_args()

    summary = validate_normalized_outputs(
        repo_root=Path(args.repo_root),
        require_outputs=args.require_outputs,
    )
    console.print(f"Checked outputs: {len(summary.checked_outputs)}")
    console.print(f"Missing outputs: {len(summary.missing_outputs)}")
    console.print(f"Source tables: {summary.source_table_count}")
    console.print(f"Normalized records: {summary.record_count}")
    console.print(f"Unresolved mappings: {summary.unresolved_mapping_count}")
    console.print(f"Warnings: {len(summary.warnings)}")
    if summary.errors:
        console.print("[red]Validation failed:[/red]")
        for error in summary.errors:
            console.print(f"- {error}")
        return 1
    console.print("[green]Normalized outputs validated.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
