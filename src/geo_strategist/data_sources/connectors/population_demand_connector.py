"""Connector for the real municipality-level age-structure population source.

Loads `.data/interim/study_area/tokyo_aichi_osaka/population_base_age_normalized.jsonl`,
which already carries full cell-level provenance from the project's
population-normalization pipeline, and groups rows into one demand summary
record per municipality. Ratios (e.g. elderly share) are deterministic
calculations over those verified counts, not model guesses.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/population_base_age_normalized.jsonl")

AGE_GROUP_FIELDS = ("total", "age_0_14", "age_15_64", "age_65_plus", "age_75_plus")


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 6)


def load_records(repo_root: str | Path = ".", *, path: str | Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load one demand-summary record per municipality. Returns (records, issues)."""

    repo_root = Path(repo_root).resolve()
    resolved = Path(path) if path else repo_root / DEFAULT_PATH
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    if not resolved.exists():
        return [], [{
            "issue_code": "population_demand_source_missing",
            "severity": "error",
            "message": f"Expected source file not found: {resolved}",
        }]

    by_municipality: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    source_record_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    issues: list[dict[str, Any]] = []

    with resolved.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.append({
                    "issue_code": "population_demand_record_unparseable",
                    "severity": "warning",
                    "line_number": line_number,
                })
                continue
            municipality = row.get("municipality")
            prefecture = row.get("matched_target_prefecture")
            age_group = row.get("canonical_age_group_id")
            value = row.get("population_value")
            if not municipality or not prefecture or age_group not in AGE_GROUP_FIELDS or value is None:
                continue
            key = (prefecture, municipality)
            by_municipality[key][f"population_{age_group}"] = value
            by_municipality[key].setdefault("year", row.get("year"))
            if len(source_record_ids[key]) < 5:
                source_record_ids[key].append(row.get("record_id"))

    records: list[dict[str, Any]] = []
    for (prefecture, municipality), values in by_municipality.items():
        total = values.get("population_total")
        elderly_65 = values.get("population_age_65_plus")
        elderly_75 = values.get("population_age_75_plus")
        record = {
            "prefecture": prefecture,
            "municipality": municipality,
            "year": values.get("year"),
            "population_total": total,
            "population_age_0_14": values.get("population_age_0_14"),
            "population_age_15_64": values.get("population_age_15_64"),
            "population_age_65_plus": elderly_65,
            "population_age_75_plus": elderly_75,
            "elderly_ratio_65_plus": _safe_ratio(elderly_65, total),
            "elderly_ratio_75_plus": _safe_ratio(elderly_75, total),
            "source_artifact": str(DEFAULT_PATH),
            "source_record_ids": source_record_ids[(prefecture, municipality)],
            "evidence_grade_population": "verified_source",
            "evidence_grade_ratio": "derived_from_verified_source",
        }
        records.append(record)
    return records, issues
