#!/usr/bin/env python
"""Validate that release files do not contain mock data artifacts."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console


console = Console()

LOCAL_ONLY_PREFIXES = {
    ("references", "local"),
    ("data", "manual"),
    ("data", "api_raw"),
}

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".scratch",
    ".junk",
    ".cache",
    ".data",
    ".runs",
    # Git-ignored live-run artifacts: raw model traces legitimately discuss
    # (refusing) invented facts and must not gate the release scan.
    "outputs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}

DATA_SUFFIXES = {
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".ndjson",
    ".parquet",
    ".feather",
    ".xlsx",
    ".xls",
}

DATA_CONTEXT_PARTS = {
    "data",
    "datasets",
    "fixtures",
    "api_raw",
    "responses",
    "outputs",
    "tests",
}

SUSPICIOUS_NAME_TOKENS = {
    "mock",
    "fake",
    "dummy",
    "synthetic",
    "sample_api_response",
    "placeholder_api",
    "invented",
    "placeholder_data",
}

SUSPICIOUS_CONTENT_PATTERNS = {
    "mock data",
    "fake api",
    "dummy dataset",
    "synthetic dataset",
    "sample api response",
    "invented numbers",
    # Domain-specific mock patterns prohibited by the real-data-only policy
    "mock hospital",
    "fake hospital",
    "synthetic population",
    "dummy demand",
    "dummy land price",
    "placeholder revenue",
    "sample municipality",
    "test hospital",
    "invented hospital",
    "fake population",
    "mock population",
    "invented demand",
    "dummy population",
    "fake demand",
    "invented revenue",
}


@dataclass(frozen=True)
class Violation:
    """A release-safety validation failure."""

    path: Path
    reason: str


def _is_local_only_prefix(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return any(rel.parts[: len(prefix)] == prefix for prefix in LOCAL_ONLY_PREFIXES)


def _has_suspicious_name(path: Path) -> bool:
    lowered_parts = [part.lower() for part in path.parts]
    return any(token in part for part in lowered_parts for token in SUSPICIOUS_NAME_TOKENS)


def _is_data_context(path: Path) -> bool:
    lowered_parts = {part.lower() for part in path.parts}
    return bool(lowered_parts & DATA_CONTEXT_PARTS)


def _should_skip_dir(path: Path, root: Path) -> bool:
    return path.name in SKIP_DIR_NAMES or _is_local_only_prefix(path, root)


def _should_skip_file(path: Path, root: Path) -> bool:
    return path.name == ".env" or _is_local_only_prefix(path, root)


def _inspect_file(path: Path, root: Path) -> list[Violation]:
    rel = path.relative_to(root)
    violations: list[Violation] = []
    is_data_like = path.suffix.lower() in DATA_SUFFIXES
    suspicious_name = _has_suspicious_name(rel)

    if is_data_like and suspicious_name:
        violations.append(Violation(rel, "suspicious data-like file name"))

    if is_data_like and (suspicious_name or _is_data_context(rel)):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError as exc:
            violations.append(Violation(rel, f"could not inspect file: {exc}"))
            return violations

        for pattern in sorted(SUSPICIOUS_CONTENT_PATTERNS):
            if pattern in content:
                violations.append(Violation(rel, f"suspicious content pattern: {pattern}"))

    return violations


def find_violations(root: str | Path) -> list[Violation]:
    """Find tracked-release mock data and fake API response risks."""

    root_path = Path(root).resolve()
    violations: list[Violation] = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        current = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _should_skip_dir(current / dirname, root_path)
        ]

        for filename in filenames:
            path = current / filename
            if _should_skip_file(path, root_path):
                continue
            violations.extend(_inspect_file(path, root_path))

    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Repository root to validate.")
    args = parser.parse_args()

    violations = find_violations(args.root)
    if violations:
        console.print("[red]Mock data validation failed:[/red]")
        for violation in violations:
            console.print(f"- {violation.path}: {violation.reason}")
        return 1

    console.print("[green]No mock numeric datasets or fake API outputs found.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
