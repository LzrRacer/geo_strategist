"""Connector for the real geocoded existing-facility source.

Loads `.data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_records.jsonl`,
a Yahoo Local Search API pull already normalized by the existing data
pipeline. Every record keeps its original address/coordinates/source_url;
this connector adds no invented fields.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_records.jsonl")

# The raw pull includes many non-hospital categories (veterinary clinics,
# pharmacies, prefectural offices). Candidate generation should default to
# this allowlist unless a caller explicitly wants the full raw set.
HOSPITAL_LIKE_CATEGORY_MARKERS = ("病院", "診療所")
NON_HOSPITAL_EXCLUSION_MARKERS = ("動物病院", "獣医")


def is_hospital_like_category(facility_category: str | None) -> bool:
    if not facility_category:
        return False
    if any(marker in facility_category for marker in NON_HOSPITAL_EXCLUSION_MARKERS):
        return False
    return any(marker in facility_category for marker in HOSPITAL_LIKE_CATEGORY_MARKERS)


def load_records(repo_root: str | Path = ".", *, path: str | Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load existing-facility records. Returns (records, issues); never fabricates data."""

    repo_root = Path(repo_root).resolve()
    resolved = Path(path) if path else repo_root / DEFAULT_PATH
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    if not resolved.exists():
        return [], [{
            "issue_code": "healthcare_facility_source_missing",
            "severity": "error",
            "message": f"Expected source file not found: {resolved}",
        }]

    records: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.append({
                    "issue_code": "healthcare_facility_record_unparseable",
                    "severity": "warning",
                    "line_number": line_number,
                })
                continue
            row = dict(row)
            row["source_artifact"] = str(DEFAULT_PATH)
            row["evidence_grade"] = "verified_source"
            row["is_hospital_like"] = is_hospital_like_category(row.get("facility_category"))
            records.append(row)
    return records, issues
