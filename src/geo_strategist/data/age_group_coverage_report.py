"""Coverage report for age-group-normalized population base."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.age_groups import (
    AgeGroupNormalizedRecord,
    AgeGroupQAIssue,
    AgeGroupQAManifest,
)
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json


class AgeGroupCoverageReportResult(BaseModel):
    """Result of age-group coverage report generation."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    issue_count: int = 0
    unknown_age_group_count: int = 0
    missing_age_group_count: int = 0
    duplicate_normalized_key_count: int = 0
    conflicting_value_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def build_age_group_coverage_report(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> AgeGroupCoverageReportResult:
    """Write summary report for age-group-normalized rows."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    records_path = root / outputs["population_base_age_normalized"]
    issues_path = root / outputs["age_group_qa_issues"]
    manifest_path = root / outputs["age_group_qa_manifest"]
    output_paths = {
        "json": outputs["age_group_coverage_report_json"],
        "markdown": outputs["age_group_coverage_report_markdown"],
    }
    if not records_path.exists() or not manifest_path.exists():
        return AgeGroupCoverageReportResult(input_found=False, output_paths=output_paths)

    records = [
        AgeGroupNormalizedRecord.model_validate(payload)
        for payload in _iter_jsonl(records_path)
    ]
    issues = [
        AgeGroupQAIssue.model_validate(payload)
        for payload in _iter_jsonl(issues_path)
    ]
    manifest = AgeGroupQAManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    issue_counts = Counter(issue.issue_type for issue in issues)
    rows_by_canonical = Counter(
        record.canonical_age_group_label or "unmatched" for record in records
    )
    rows_by_raw = Counter(record.raw_age_group or "missing" for record in records)
    rows_by_match = Counter(record.age_group_match_status.value for record in records)
    rows_by_kind = Counter(record.age_group_kind.value for record in records)
    rows_by_value_kind = Counter(record.value_kind.value for record in records)
    rows_by_grain = Counter(record.geography_grain.value for record in records)
    report = {
        "study_area_id": study_area.study_area_id,
        "raw_age_labels_observed": sorted(rows_by_raw),
        "canonical_age_groups_observed": sorted(
            {
                record.canonical_age_group_label
                for record in records
                if record.canonical_age_group_label
            }
        ),
        "rows_by_canonical_age_group": dict(rows_by_canonical),
        "rows_by_raw_age_label": dict(rows_by_raw),
        "rows_by_match_status": dict(rows_by_match),
        "rows_by_age_group_kind": dict(rows_by_kind),
        "rows_by_value_kind": dict(rows_by_value_kind),
        "rows_by_geography_grain": dict(rows_by_grain),
        "unknown_age_group_count": issue_counts.get("unknown_age_group", 0),
        "missing_age_group_count": issue_counts.get("missing_age_group", 0),
        "duplicate_normalized_key_count": issue_counts.get("age_group_duplicate_normalized_key", 0),
        "conflicting_value_count": issue_counts.get("age_group_conflicting_values", 0),
        "issue_counts": dict(issue_counts),
        "manifest_record_counts": manifest.record_counts,
    }
    write_json(root / outputs["age_group_coverage_report_json"], report)
    markdown_path = root / outputs["age_group_coverage_report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# Age Group Coverage Report",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Records read: {len(records)}",
                f"- Raw age labels observed: {len(rows_by_raw)}",
                f"- Canonical age groups observed: {len(report['canonical_age_groups_observed'])}",
                f"- Unknown age groups: {report['unknown_age_group_count']}",
                f"- Missing age groups: {report['missing_age_group_count']}",
                f"- Duplicate normalized keys: {report['duplicate_normalized_key_count']}",
                f"- Conflicting values: {report['conflicting_value_count']}",
                "- This report does not calculate demand.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return AgeGroupCoverageReportResult(
        input_found=True,
        records_read=len(records),
        issue_count=len(issues),
        unknown_age_group_count=report["unknown_age_group_count"],
        missing_age_group_count=report["missing_age_group_count"],
        duplicate_normalized_key_count=report["duplicate_normalized_key_count"],
        conflicting_value_count=report["conflicting_value_count"],
        output_paths=output_paths,
    )
