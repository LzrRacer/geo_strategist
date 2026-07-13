#!/usr/bin/env python
"""Dry-run release cleanup for local-only generated artifacts."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rich.console import Console

try:
    from validate_no_mock_data import find_violations
except ModuleNotFoundError:
    from scripts.validate_no_mock_data import find_violations


console = Console()

DEFAULT_CLEAN_PATHS = [
    ".cache",
    ".runs",
    ".scratch",
    ".junk",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
]

LOCAL_DATA_PATHS = [
    ".data",
    "references/local",
]


def candidate_paths(root: Path, include_local_data: bool = False) -> list[Path]:
    """Return local-only paths that may be cleaned for a release."""

    names = [*DEFAULT_CLEAN_PATHS]
    if include_local_data:
        names.extend(LOCAL_DATA_PATHS)
    return [root / name for name in names if (root / name).exists()]


def remove_path(path: Path) -> None:
    """Remove a file or directory selected by an explicit cleanup request."""

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Repository root.")
    parser.add_argument("--apply", action="store_true", help="Actually remove cleanup paths.")
    parser.add_argument(
        "--include-local-data",
        action="store_true",
        help="Also include .data and references/local in cleanup candidates.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    violations = find_violations(root)
    if violations:
        console.print("[red]Release validation failed before cleanup.[/red]")
        for violation in violations:
            console.print(f"- {violation.path}: {violation.reason}")
        return 1

    paths = candidate_paths(root, include_local_data=args.include_local_data)
    if not paths:
        console.print("No cleanup candidates found.")
        return 0

    mode = "Applying cleanup" if args.apply else "Dry run cleanup candidates"
    console.print(mode)
    for path in paths:
        console.print(f"- {path.relative_to(root)}")
        if args.apply:
            remove_path(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
