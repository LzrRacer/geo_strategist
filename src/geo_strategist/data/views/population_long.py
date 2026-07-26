"""Build conservative long-form population view from normalized records."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import (
    AnalysisViewManifest,
    IssueSeverity,
    PopulationLongRecord,
    SourceQualityIssue,
)
from geo_strategist.data.normalization import NormalizedRecord, ValueType, now_utc
from geo_strategist.data.population_views import PopulationValueKind
from geo_strategist.data.views.common import first_label, read_normalized_jsonl, write_json, write_jsonl


class PopulationLongResult(BaseModel):
    """Result of building population long view."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    records_written: int = 0
    quality_issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_config() -> dict:
    with Path("configs/analysis_views.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _quality_issue(
    record: NormalizedRecord,
    issue_type: str,
    message: str,
    recommended_action: str,
    severity: IssueSeverity = IssueSeverity.WARNING,
) -> SourceQualityIssue:
    return SourceQualityIssue(
        issue_id=f"population_quality:{issue_type}:{record.record_id}",
        severity=severity,
        source_file=record.source_file_path,
        sheet=record.source_sheet,
        issue_type=issue_type,
        message=message,
        recommended_action=recommended_action,
    )


def _record_to_population_long(record: NormalizedRecord) -> tuple[PopulationLongRecord | None, list[SourceQualityIssue]]:
    issues: list[SourceQualityIssue] = []
    if record.value_role == "population_rate":
        issues.append(
            _quality_issue(
                record,
                "not_population_count",
                "Population rate record is not a population count.",
                "Keep rate records separate from count-based population long views.",
            )
        )
        return None, issues
    if record.value_role != "population_count":
        issues.append(
            _quality_issue(
                record,
                "ambiguous_value_role",
                "Population record lacks explicit population_count semantics.",
                "Add a verified mapping override before promoting this record.",
            )
        )
        return None, issues

    if record.value_type is not ValueType.NUMBER:
        issues.append(
            _quality_issue(
                record,
                "non_numeric_population",
                "Population value record is not numeric.",
                "Use only numeric normalized population source records.",
                IssueSeverity.ERROR,
            )
        )
        return None, issues

    if not record.source_file_hash:
        issues.append(
            _quality_issue(
                record,
                "missing_source_hash",
                "Normalized population record lacks source hash.",
                "Regenerate normalized population records with source file hashes.",
                IssueSeverity.ERROR,
            )
        )
    if not record.provenance:
        issues.append(
            _quality_issue(
                record,
                "missing_provenance",
                "Normalized population record lacks provenance.",
                "Regenerate normalized population records with provenance.",
                IssueSeverity.ERROR,
            )
        )

    prefecture = first_label(record.geography_labels, ("prefecture", "都道府県"))
    municipality = first_label(record.geography_labels, ("municipality", "市区町村", "市町村"))
    year = first_label(record.time_labels, ("year", "年度", "年次", "調査年", "基準年"))
    age_group = first_label(record.age_labels, ("age", "年齢", "歳", "年齢階級"))

    if not (prefecture or municipality):
        issues.append(
            _quality_issue(
                record,
                "missing_geography",
                "Population record lacks geography labels.",
                "Confirm geography labels before using this row for analysis.",
            )
        )
    if not year:
        issues.append(
            _quality_issue(
                record,
                "missing_time_label",
                "Population record lacks time labels.",
                "Confirm time labels before using this row for analysis.",
            )
        )
    if not age_group:
        issues.append(
            _quality_issue(
                record,
                "missing_age_label",
                "Population record lacks age labels.",
                "Confirm age labels before using this row for age-structured analysis.",
            )
        )

    if issues:
        return None, issues

    year_value: int | str | None = year
    if isinstance(year, str) and year.isdigit():
        year_value = int(year)

    return (
        PopulationLongRecord(
            record_id=f"population_long:{record.record_id}",
            source_record_ids=[record.record_id],
            source_file_path=record.source_file_path,
            source_sheet=record.source_sheet,
            prefecture=prefecture,
            municipality=municipality,
            age_group=age_group,
            year=year_value,
            population_value=float(record.normalized_value),
            unit=record.unit,
            source_file_hash=record.source_file_hash,
            geography_labels=record.geography_labels,
            time_labels=record.time_labels,
            age_labels=record.age_labels,
            value_kind=PopulationValueKind.COUNT,
            provenance=record.provenance,
        ),
        issues,
    )


def _summary(result: PopulationLongResult) -> str:
    return (
        "# Population Long View Summary\n\n"
        f"- Input found: {result.input_found}\n"
        f"- Records read: {result.records_read}\n"
        f"- Population long records written: {result.records_written}\n"
        f"- Quality issues: {result.quality_issue_count}\n"
        f"- Warnings: {len(result.warnings)}\n"
    )


def build_population_long(repo_root: str | Path = ".") -> PopulationLongResult:
    """Build population long view without demand scoring or imputation."""

    root = Path(repo_root).resolve()
    config = _load_config()
    inputs = config["inputs"]
    outputs = config["outputs"]
    input_path = root / inputs["population_normalized_records"]
    normalized_records = read_normalized_jsonl(input_path)
    warnings: list[str] = []
    rows: list[PopulationLongRecord] = []
    issues: list[SourceQualityIssue] = []

    for record in normalized_records:
        row, record_issues = _record_to_population_long(record)
        issues.extend(record_issues)
        if row is not None:
            rows.append(row)

    output_files = [
        Path(outputs["population_long"]),
        Path(outputs["population_manifest"]),
        Path(outputs["population_summary"]),
        Path(outputs["population_quality_issues"]),
    ]
    manifest = AnalysisViewManifest(
        run_id=f"population_long:{now_utc().isoformat()}",
        view_name="population_long",
        generated_at=now_utc(),
        input_files=[Path(inputs["population_normalized_records"])],
        output_files=output_files,
        record_counts={
            "normalized_records_read": len(normalized_records),
            "population_records_written": len(rows),
            "quality_issues": len(issues),
        },
        warnings=warnings,
        unresolved_mapping_count=0,
    )
    output_paths = {
        "population_long": outputs["population_long"],
        "manifest": outputs["population_manifest"],
        "summary": outputs["population_summary"],
        "quality_issues": outputs["population_quality_issues"],
    }
    result = PopulationLongResult(
        input_found=input_path.exists(),
        records_read=len(normalized_records),
        records_written=len(rows),
        quality_issue_count=len(issues),
        warnings=warnings,
        output_paths=output_paths,
    )
    write_jsonl(root / outputs["population_long"], rows)
    write_jsonl(root / outputs["population_quality_issues"], issues)
    write_json(root / outputs["population_manifest"], manifest)
    summary_path = root / outputs["population_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary(result), encoding="utf-8")
    return result
