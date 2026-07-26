"""Migrate completed condition artifacts and re-render reports without model calls.

The source tree is read-only from this script's perspective. Normalized
``DecisionAnalysisBundle`` files, copied condition records, and regenerated
business reports are written under a separate destination.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def migrate(source: Path, destination: Path, groups: set[str] | None = None) -> list[str]:
    from geo_strategist.experiments.condition_registry import build_condition_registry
    from geo_strategist.experiments.condition_utils import _read_jsonl
    from geo_strategist.experiments.deterministic_evaluation_engine import load_data_bundle
    from scripts.rewrite_condition_reports import rewrite_reports

    records = _read_jsonl(source / "condition_records.jsonl")
    selected = [row for row in records
                if groups is None or str(row.get("condition_group")) in groups]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "condition_records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    return rewrite_reports(destination, build_condition_registry(), load_data_bundle(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--conditions", nargs="*")
    args = parser.parse_args()
    source = args.source if args.source.is_absolute() else REPO_ROOT / args.source
    destination = args.destination if args.destination.is_absolute() else REPO_ROOT / args.destination
    migrated = migrate(source, destination, set(args.conditions) if args.conditions else None)
    print(f"{destination}: migrated and re-rendered {len(migrated)} condition(s): {', '.join(migrated)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
