"""Build auditable pre-demand population base views from geography-grain rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import IssueSeverity
from geo_strategist.data.geography_grain import (
    GeographyGrain,
    GeographyGrainIssue,
    GeographyGrainRecord,
    PopulationBaseRole,
    StudyAreaPopulationBaseManifest,
    StudyAreaPopulationBaseRecord,
)
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class PopulationBaseBuildResult(BaseModel):
    """Result of pre-demand population-base view construction."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    records_written: int = 0
    municipality_records_written: int = 0
    prefecture_total_records_written: int = 0
    excluded_records: int = 0
    issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
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


def role_for_grain(grain: GeographyGrain) -> PopulationBaseRole:
    """Map deterministic geography grain to the pre-demand population-base role."""

    if grain is GeographyGrain.MUNICIPALITY:
        return PopulationBaseRole.MODEL_INPUT_CANDIDATE
    if grain is GeographyGrain.PREFECTURE_TOTAL:
        return PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
    return PopulationBaseRole.EXCLUDED_REQUIRES_REVIEW


def _base_record(record: GeographyGrainRecord) -> StudyAreaPopulationBaseRecord:
    return StudyAreaPopulationBaseRecord(
        base_record_id=f"population_base:{record.record_id}",
        grain_record_id=record.record_id,
        source_record_ids=record.source_record_ids,
        source_file_path=record.source_file_path,
        source_sheet=record.source_sheet,
        source_file_hash=record.source_file_hash,
        provenance=record.provenance,
        study_area_id=record.study_area_id,
        matched_target_prefecture=record.matched_target_prefecture,
        raw_municipality=record.raw_municipality,
        municipality=record.municipality,
        geography_grain=record.geography_grain,
        population_base_role=role_for_grain(record.geography_grain),
        year=record.year,
        age_group=record.age_group,
        population_value=record.population_value,
        rate_value=record.rate_value,
        unit=record.unit,
        value_kind=record.value_kind,
    )


def _issue_for_record(record: StudyAreaPopulationBaseRecord) -> GeographyGrainIssue | None:
    issue_type: str | None = None
    message: str | None = None
    severity = IssueSeverity.ERROR
    if (
        record.population_base_role is PopulationBaseRole.MODEL_INPUT_CANDIDATE
        and record.geography_grain is not GeographyGrain.MUNICIPALITY
    ):
        issue_type = "model_input_candidate_wrong_grain"
        message = "Model-input candidate row is not municipality grain."
    elif (
        record.population_base_role is PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
        and record.geography_grain is not GeographyGrain.PREFECTURE_TOTAL
    ):
        issue_type = "context_prefecture_total_wrong_grain"
        message = "Context prefecture-total row is not prefecture_total grain."
    elif record.geography_grain is GeographyGrain.MUNICIPALITY and not record.municipality:
        issue_type = "municipality_grain_missing_municipality"
        message = "Municipality grain row lacks municipality."
    elif record.geography_grain is GeographyGrain.PREFECTURE_TOTAL and record.municipality:
        issue_type = "prefecture_total_has_municipality"
        message = "Prefecture-total row has a municipality label."
    elif not record.source_record_ids or not record.source_file_hash or not record.provenance:
        issue_type = "population_base_missing_source_traceability"
        message = "Population-base row lacks source traceability."
    elif record.population_base_role is PopulationBaseRole.EXCLUDED_REQUIRES_REVIEW:
        issue_type = "geography_grain_unknown"
        message = "Unknown geography-grain row is excluded from model inputs pending review."
        severity = IssueSeverity.WARNING

    if issue_type is None or message is None:
        return None
    return GeographyGrainIssue(
        issue_id=f"population_base:{issue_type}:{record.base_record_id}",
        severity=severity,
        issue_type=issue_type,
        message=message,
        study_area_id=record.study_area_id,
        matched_target_prefecture=record.matched_target_prefecture,
        raw_municipality=record.raw_municipality,
        municipality=record.municipality,
        geography_grain=record.geography_grain,
        population_base_role=record.population_base_role,
        year=record.year,
        age_group=record.age_group,
        value_kind=record.value_kind,
        source_record_ids=record.source_record_ids,
        source_file_hash=record.source_file_hash,
        recommended_action="Review geography grain before using this row for demand modeling.",
    )


def build_population_base(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> PopulationBaseBuildResult:
    """Build pre-demand population-base views without scoring or aggregation."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    input_path = root / outputs["geography_grain_records"]
    payloads = _iter_jsonl(input_path)
    grain_records = [GeographyGrainRecord.model_validate(payload) for payload in payloads]
    base_records = [_base_record(record) for record in grain_records]
    municipality_records = [
        record
        for record in base_records
        if record.population_base_role is PopulationBaseRole.MODEL_INPUT_CANDIDATE
    ]
    prefecture_total_records = [
        record
        for record in base_records
        if record.population_base_role is PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
    ]
    issues = [
        issue
        for record in base_records
        if (issue := _issue_for_record(record)) is not None
    ]
    source_hashes = sorted({record.source_file_hash for record in base_records if record.source_file_hash})
    output_paths = {
        "population_base": outputs["population_base"],
        "population_base_municipality": outputs["population_base_municipality"],
        "population_base_prefecture_total": outputs["population_base_prefecture_total"],
        "manifest": outputs["population_base_manifest"],
        "summary": outputs["population_base_summary"],
        "issues": outputs["population_base_issues"],
    }
    if not grain_records:
        return PopulationBaseBuildResult(input_found=False, output_paths=output_paths)

    role_counts = Counter(record.population_base_role.value for record in base_records)
    grain_counts = Counter(record.geography_grain.value for record in base_records)
    value_kind_counts = Counter(record.value_kind.value for record in base_records)
    issue_counts = Counter(issue.issue_type for issue in issues)
    manifest = StudyAreaPopulationBaseManifest(
        run_id=f"population_base:{study_area.study_area_id}:{now_utc().isoformat()}",
        generated_at=now_utc(),
        study_area_id=study_area.study_area_id,
        target_prefectures=study_area.target_prefectures,
        input_files=[Path(outputs["geography_grain_records"])],
        output_files=[
            Path(outputs["population_base"]),
            Path(outputs["population_base_municipality"]),
            Path(outputs["population_base_prefecture_total"]),
            Path(outputs["population_base_manifest"]),
            Path(outputs["population_base_summary"]),
            Path(outputs["population_base_issues"]),
        ],
        source_file_hashes=source_hashes,
        record_counts={
            "geography_grain_records_read": len(grain_records),
            "population_base_records_written": len(base_records),
            "municipality_records_written": len(municipality_records),
            "prefecture_total_records_written": len(prefecture_total_records),
            "excluded_requires_review_records": role_counts[
                PopulationBaseRole.EXCLUDED_REQUIRES_REVIEW.value
            ],
            **{f"grain_{key}": value for key, value in grain_counts.items()},
            **{f"role_{key}": value for key, value in role_counts.items()},
            **{f"value_kind_{key}": value for key, value in value_kind_counts.items()},
        },
        issue_counts=dict(issue_counts),
        warnings=[],
    )
    write_jsonl(root / outputs["population_base"], base_records)
    write_jsonl(root / outputs["population_base_municipality"], municipality_records)
    write_jsonl(root / outputs["population_base_prefecture_total"], prefecture_total_records)
    write_jsonl(root / outputs["population_base_issues"], issues)
    write_json(root / outputs["population_base_manifest"], manifest)
    summary_path = root / outputs["population_base_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Population Base Summary",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Population base rows: {len(base_records)}",
                f"- Municipality model-input candidate rows: {len(municipality_records)}",
                f"- Prefecture-total context rows: {len(prefecture_total_records)}",
                f"- Excluded review rows: {role_counts[PopulationBaseRole.EXCLUDED_REQUIRES_REVIEW.value]}",
                "- This is a pre-demand base table and does not calculate demand.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return PopulationBaseBuildResult(
        input_found=input_path.exists(),
        records_read=len(grain_records),
        records_written=len(base_records),
        municipality_records_written=len(municipality_records),
        prefecture_total_records_written=len(prefecture_total_records),
        excluded_records=role_counts[PopulationBaseRole.EXCLUDED_REQUIRES_REVIEW.value],
        issue_count=len(issues),
        warnings=[],
        output_paths=output_paths,
    )
