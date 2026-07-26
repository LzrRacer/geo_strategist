"""Report summaries for population-base coverage QA outputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.population_base_coverage import (
    CoverageSeverity,
    PopulationBaseCoverageIssue,
    PopulationBaseCoverageManifest,
    PopulationBaseCoverageMatrix,
)
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json


class PopulationBaseCoverageReportResult(BaseModel):
    """Result of coverage report generation."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    matrix_rows_read: int = 0
    issue_count: int = 0
    model_blocking_error_count: int = 0
    duplicate_key_count: int = 0
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


def build_population_base_coverage_report(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> PopulationBaseCoverageReportResult:
    """Write summary report for population-base coverage QA."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    matrix_path = root / outputs["population_base_coverage_matrix_json"]
    issues_path = root / outputs["population_base_coverage_issues"]
    manifest_path = root / outputs["population_base_coverage_manifest"]
    output_paths = {
        "json": outputs["population_base_coverage_report_json"],
        "markdown": outputs["population_base_coverage_report_markdown"],
    }
    if not matrix_path.exists() or not manifest_path.exists():
        return PopulationBaseCoverageReportResult(input_found=False, output_paths=output_paths)

    matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix_rows = [
        PopulationBaseCoverageMatrix.model_validate(row)
        for row in matrix_payload.get("matrix_rows", [])
    ]
    issues = [
        PopulationBaseCoverageIssue.model_validate(payload)
        for payload in _iter_jsonl(issues_path)
    ]
    manifest = PopulationBaseCoverageManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    issue_counts = Counter(issue.issue_type.value for issue in issues)
    severity_counts = Counter(issue.severity.value for issue in issues)
    matrix_axis_counts = Counter(row.coverage_axis.value for row in matrix_rows)
    target_prefectures = sorted(
        {
            row.target_prefecture
            for row in matrix_rows
            if row.target_prefecture
        }
    )
    years = sorted({str(row.year) for row in matrix_rows if row.year not in (None, "")})
    age_groups = sorted({row.age_group for row in matrix_rows if row.age_group})
    report = {
        "study_area_id": study_area.study_area_id,
        "target_prefectures_covered": target_prefectures,
        "years_covered": years,
        "age_groups_covered": age_groups,
        "expected_years": matrix_payload.get("expected_years", []),
        "expected_age_groups": matrix_payload.get("expected_age_groups", []),
        "matrix_rows": len(matrix_rows),
        "matrix_rows_by_axis": dict(matrix_axis_counts),
        "coverage_issue_counts": dict(issue_counts),
        "coverage_severity_counts": dict(severity_counts),
        "duplicate_key_count": issue_counts.get("duplicate_population_base_key", 0),
        "conflicting_value_count": issue_counts.get("conflicting_values_for_same_key", 0),
        "model_blocking_error_count": severity_counts.get(CoverageSeverity.ERROR.value, 0),
        "manifest_record_counts": manifest.record_counts,
        "municipality_candidate_coverage_rows": sum(
            1
            for row in matrix_rows
            if row.population_base_role == "model_input_candidate"
            or row.axis_values.get("population_base_role") == "model_input_candidate"
        ),
        "prefecture_total_context_coverage_rows": sum(
            1
            for row in matrix_rows
            if row.population_base_role == "context_prefecture_total"
            or row.axis_values.get("population_base_role") == "context_prefecture_total"
        ),
        "count_coverage_rows": sum(
            1 for row in matrix_rows if row.value_kind == "count" or row.axis_values.get("value_kind") == "count"
        ),
        "rate_coverage_rows": sum(
            1 for row in matrix_rows if row.value_kind == "rate" or row.axis_values.get("value_kind") == "rate"
        ),
    }
    write_json(root / outputs["population_base_coverage_report_json"], report)
    markdown_path = root / outputs["population_base_coverage_report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# Population Base Coverage Report",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Target prefectures covered: {', '.join(target_prefectures)}",
                f"- Years covered: {', '.join(years)}",
                f"- Age groups covered: {', '.join(age_groups)}",
                f"- Coverage issues: {len(issues)}",
                f"- Duplicate keys: {report['duplicate_key_count']}",
                f"- Conflicting values: {report['conflicting_value_count']}",
                f"- Model-blocking errors: {report['model_blocking_error_count']}",
                "- This report does not calculate demand.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return PopulationBaseCoverageReportResult(
        input_found=True,
        matrix_rows_read=len(matrix_rows),
        issue_count=len(issues),
        model_blocking_error_count=severity_counts.get(CoverageSeverity.ERROR.value, 0),
        duplicate_key_count=issue_counts.get("duplicate_population_base_key", 0),
        conflicting_value_count=issue_counts.get("conflicting_values_for_same_key", 0),
        output_paths=output_paths,
    )
