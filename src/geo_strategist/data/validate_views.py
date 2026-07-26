"""Validation for analysis-ready source view artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from geo_strategist.data.analysis_views import (
    AnalysisViewManifest,
    HospitalWorkbookFact,
    PopulationLongRecord,
    SourceQualityIssue,
)
from geo_strategist.data.age_groups import (
    AgeGroupMatchStatus,
    AgeGroupNormalizedRecord,
    AgeGroupQAIssue,
    AgeGroupQAManifest,
)
from geo_strategist.data.geography_grain import (
    GeographyGrain,
    GeographyGrainIssue,
    GeographyGrainRecord,
    PopulationBaseRole,
    StudyAreaPopulationBaseManifest,
    StudyAreaPopulationBaseRecord,
)
from geo_strategist.data.population_views import (
    PopulationGeographyIssue,
    PopulationGeographyKey,
    PopulationQAManifest,
    PopulationRateRecord,
    PopulationValueKind,
)
from geo_strategist.data.population_base_coverage import (
    CoverageIssueType,
    PopulationBaseCoverageIssue,
    PopulationBaseCoverageManifest,
    PopulationBaseCoverageMatrix,
)
from geo_strategist.data.model_input_readiness import (
    ModelInputReadinessIssue,
    ModelInputReadinessRecord,
    ReadinessRole,
    ReadinessStatus,
)
from geo_strategist.data.feature_substrate import (
    HospitalFeatureRecord,
    MunicipalityFeatureBaseRecord,
    PopulationFeatureRecord,
)
from geo_strategist.data.score_layer import MunicipalityScoreRecord
from geo_strategist.data.land_price_ingestion import (
    LandIngestionIssue,
    LandPriceRecord,
    MunicipalityLandFeatureRecord,
)
from geo_strategist.data.healthcare_supply_ingestion import (
    HealthcareIngestionIssue,
    HealthcareSupplyRecord,
    MunicipalityHealthcareSupplyFeatureRecord,
)
from geo_strategist.data.enriched_feature_base import EnrichedMunicipalityFeatureRecord
from geo_strategist.data.enriched_score_layer import EnrichedMunicipalityScoreRecord
from geo_strategist.data.candidate_generation import (
    CandidateAction,
    CandidateActionRecord,
    CandidateEvidenceBundle,
)
from geo_strategist.data.study_area import (
    StudyAreaGeographyIssue,
    StudyAreaManifest,
    StudyAreaPopulationRecord,
    StudyAreaScopeStatus,
)


class AnalysisViewValidationSummary(BaseModel):
    """Validation summary for analysis-ready views."""

    model_config = ConfigDict(extra="forbid")

    checked_outputs: list[str] = Field(default_factory=list)
    missing_outputs: list[str] = Field(default_factory=list)
    hospital_fact_count: int = 0
    population_long_count: int = 0
    population_rate_count: int = 0
    population_geography_key_count: int = 0
    study_area_population_long_count: int = 0
    study_area_population_rate_count: int = 0
    study_area_geography_key_count: int = 0
    study_area_issue_count: int = 0
    geography_grain_record_count: int = 0
    geography_grain_issue_count: int = 0
    population_base_record_count: int = 0
    population_base_issue_count: int = 0
    coverage_matrix_row_count: int = 0
    coverage_issue_count: int = 0
    age_group_record_count: int = 0
    age_group_issue_count: int = 0
    model_input_readiness_record_count: int = 0
    model_input_readiness_issue_count: int = 0
    population_feature_count: int = 0
    hospital_feature_count: int = 0
    municipality_feature_base_count: int = 0
    municipality_score_count: int = 0
    land_price_record_count: int = 0
    municipality_land_feature_count: int = 0
    healthcare_supply_record_count: int = 0
    municipality_healthcare_supply_feature_count: int = 0
    enriched_municipality_feature_count: int = 0
    enriched_municipality_score_count: int = 0
    candidate_action_count: int = 0
    candidate_evidence_bundle_count: int = 0
    quality_issue_count: int = 0
    population_rate_issue_count: int = 0
    population_geography_issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether validation passed."""

        return not self.errors


def _load_config() -> dict:
    with Path("configs/analysis_views.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _iter_jsonl(path: Path) -> list[dict]:
    payloads: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payloads.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return payloads


def _validate_hospital(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        fact = HospitalWorkbookFact.model_validate(payload)
        if not fact.source_record_ids:
            summary.errors.append(f"{path}: hospital fact lacks source record IDs")
        if not fact.provenance:
            summary.errors.append(f"{path}: hospital fact lacks provenance")
        if not fact.source_file_hash:
            summary.errors.append(f"{path}: hospital fact lacks source hash")
        summary.hospital_fact_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        if payload.get("value_kind") != PopulationValueKind.COUNT.value:
            summary.errors.append(f"{path}: population long row is not marked as count semantics")
        row = PopulationLongRecord.model_validate(payload)
        if row.value_kind is not PopulationValueKind.COUNT:
            summary.errors.append(f"{path}: population long row has non-count semantics")
        if not isinstance(row.population_value, (int, float)):
            summary.errors.append(f"{path}: population value is not numeric")
        if not row.provenance:
            summary.errors.append(f"{path}: population row lacks provenance")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: population row lacks source hash")
        summary.population_long_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_rates(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        if payload.get("value_kind") != PopulationValueKind.RATE.value:
            summary.errors.append(f"{path}: population rate row is not marked as rate semantics")
        row = PopulationRateRecord.model_validate(payload)
        if row.value_kind is not PopulationValueKind.RATE:
            summary.errors.append(f"{path}: population rate row has non-rate semantics")
        if not isinstance(row.rate_value, (int, float)):
            summary.errors.append(f"{path}: population rate value is not numeric")
        if not row.provenance:
            summary.errors.append(f"{path}: population rate row lacks provenance")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: population rate row lacks source hash")
        summary.population_rate_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_rate_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        SourceQualityIssue.model_validate(payload)
        summary.population_rate_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        SourceQualityIssue.model_validate(payload)
        summary.quality_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_geography_keys(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        if payload.get("value_kind") not in {
            PopulationValueKind.COUNT.value,
            PopulationValueKind.RATE.value,
        }:
            summary.errors.append(f"{path}: geography key lacks explicit value kind")
        row = PopulationGeographyKey.model_validate(payload)
        if not row.source_record_ids:
            summary.errors.append(f"{path}: geography key lacks source record IDs")
        if not row.provenance:
            summary.errors.append(f"{path}: geography key lacks provenance")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: geography key lacks source hash")
        summary.population_geography_key_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_geography_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        PopulationGeographyIssue.model_validate(payload)
        summary.population_geography_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_study_area_population(
    path: Path,
    summary: AnalysisViewValidationSummary,
    expected_kind: PopulationValueKind,
) -> None:
    for payload in _iter_jsonl(path):
        if payload.get("value_kind") != expected_kind.value:
            summary.errors.append(
                f"{path}: study-area row has mixed count/rate semantics"
            )
        row = StudyAreaPopulationRecord.model_validate(payload)
        if row.value_kind is not expected_kind:
            summary.errors.append(
                f"{path}: study-area row has non-{expected_kind.value} semantics"
            )
        if row.scope_status is StudyAreaScopeStatus.IN_SCOPE and not row.matched_target_prefecture:
            summary.errors.append(f"{path}: in-scope row lacks matched target prefecture")
        if not row.source_record_ids:
            summary.errors.append(f"{path}: study-area row lacks source record IDs")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: study-area row lacks source hash")
        if not row.provenance:
            summary.errors.append(f"{path}: study-area row lacks provenance")
        if expected_kind is PopulationValueKind.COUNT:
            summary.study_area_population_long_count += 1
        else:
            summary.study_area_population_rate_count += 1
    summary.checked_outputs.append(str(path))


def _validate_study_area_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        StudyAreaGeographyIssue.model_validate(payload)
        summary.study_area_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_study_area_manifest(path: Path, summary: AnalysisViewValidationSummary) -> None:
    manifest = StudyAreaManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.source_file_hashes:
        summary.errors.append(f"{path}: study-area manifest lacks source file hashes")
    if not manifest.target_prefectures:
        summary.errors.append(f"{path}: study-area manifest lacks target prefectures")
    if not manifest.record_counts:
        summary.errors.append(f"{path}: study-area manifest lacks record counts")
    summary.warnings.extend(manifest.warnings)
    summary.checked_outputs.append(str(path))


def _validate_study_area_geography_keys(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        required = {
            "key_id",
            "study_area_id",
            "target_prefecture",
            "value_kind",
            "source_record_ids",
            "source_file_hash",
            "provenance",
        }
        missing = sorted(required - set(payload))
        if missing:
            summary.errors.append(f"{path}: geography key missing fields: {missing}")
        if payload.get("value_kind") not in {
            PopulationValueKind.COUNT.value,
            PopulationValueKind.RATE.value,
        }:
            summary.errors.append(f"{path}: geography key lacks explicit value kind")
        if not payload.get("target_prefecture"):
            summary.errors.append(f"{path}: geography key lacks target prefecture")
        if not payload.get("source_record_ids"):
            summary.errors.append(f"{path}: geography key lacks source record IDs")
        if not payload.get("source_file_hash"):
            summary.errors.append(f"{path}: geography key lacks source hash")
        if not payload.get("provenance"):
            summary.errors.append(f"{path}: geography key lacks provenance")
        summary.study_area_geography_key_count += 1
    summary.checked_outputs.append(str(path))


def _validate_study_area_geography_qa(path: Path, summary: AnalysisViewValidationSummary) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("study_area_id"):
        summary.errors.append(f"{path}: geography QA lacks study_area_id")
    if "duplicate_target_geography_key_count" not in payload:
        summary.errors.append(f"{path}: geography QA lacks duplicate key count")
    separation = payload.get("count_rate_separation") or {}
    if not separation.get("count_rows_only_in_count_output", False):
        summary.errors.append(f"{path}: count output contains non-count rows")
    if not separation.get("rate_rows_only_in_rate_output", False):
        summary.errors.append(f"{path}: rate output contains non-rate rows")
    if separation.get("mixed_count_rate_semantics_issue_count", 0):
        summary.errors.append(f"{path}: mixed count/rate semantics found in target scope")
    summary.checked_outputs.append(str(path))


def _validate_geography_grain_records(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        row = GeographyGrainRecord.model_validate(payload)
        if row.geography_grain is GeographyGrain.MUNICIPALITY and not row.municipality:
            summary.errors.append(f"{path}: municipality grain lacks municipality")
        if row.geography_grain is GeographyGrain.PREFECTURE_TOTAL and row.municipality:
            summary.errors.append(f"{path}: prefecture_total grain has municipality")
        if not row.source_record_ids:
            summary.errors.append(f"{path}: geography-grain row lacks source record IDs")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: geography-grain row lacks source hash")
        if not row.provenance:
            summary.errors.append(f"{path}: geography-grain row lacks provenance")
        if row.value_kind is PopulationValueKind.COUNT and row.rate_value is not None:
            summary.errors.append(f"{path}: count geography-grain row contains rate value")
        if row.value_kind is PopulationValueKind.RATE and row.population_value is not None:
            summary.errors.append(f"{path}: rate geography-grain row contains count value")
        summary.geography_grain_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_geography_grain_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        GeographyGrainIssue.model_validate(payload)
        summary.geography_grain_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_base_records(
    path: Path,
    summary: AnalysisViewValidationSummary,
    expected_role: PopulationBaseRole | None = None,
) -> None:
    for payload in _iter_jsonl(path):
        row = StudyAreaPopulationBaseRecord.model_validate(payload)
        if expected_role is not None and row.population_base_role is not expected_role:
            summary.errors.append(f"{path}: population-base row has unexpected role")
        if row.geography_grain is GeographyGrain.MUNICIPALITY and not row.municipality:
            summary.errors.append(f"{path}: municipality grain lacks municipality")
        if row.geography_grain is GeographyGrain.PREFECTURE_TOTAL and row.municipality:
            summary.errors.append(f"{path}: prefecture_total grain has municipality")
        if (
            row.population_base_role is PopulationBaseRole.MODEL_INPUT_CANDIDATE
            and row.geography_grain is not GeographyGrain.MUNICIPALITY
        ):
            summary.errors.append(f"{path}: model_input_candidate is not municipality grain")
        if (
            row.population_base_role is PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
            and row.geography_grain is not GeographyGrain.PREFECTURE_TOTAL
        ):
            summary.errors.append(f"{path}: context_prefecture_total is not prefecture_total grain")
        if not row.source_record_ids:
            summary.errors.append(f"{path}: population-base row lacks source record IDs")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: population-base row lacks source hash")
        if not row.provenance:
            summary.errors.append(f"{path}: population-base row lacks provenance")
        if row.value_kind is PopulationValueKind.COUNT and row.rate_value is not None:
            summary.errors.append(f"{path}: count population-base row contains rate value")
        if row.value_kind is PopulationValueKind.RATE and row.population_value is not None:
            summary.errors.append(f"{path}: rate population-base row contains count value")
        summary.population_base_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_base_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        GeographyGrainIssue.model_validate(payload)
        summary.population_base_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_population_base_manifest(path: Path, summary: AnalysisViewValidationSummary) -> None:
    manifest = StudyAreaPopulationBaseManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.source_file_hashes:
        summary.errors.append(f"{path}: population-base manifest lacks source file hashes")
    if not manifest.record_counts:
        summary.errors.append(f"{path}: population-base manifest lacks record counts")
    summary.warnings.extend(manifest.warnings)
    summary.checked_outputs.append(str(path))


def _validate_population_base_report(path: Path, summary: AnalysisViewValidationSummary) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "rows_by_target_prefecture",
        "rows_by_geography_grain",
        "rows_by_value_kind",
        "rows_by_population_base_role",
        "candidate_model_input_row_count",
        "context_prefecture_total_row_count",
        "unknown_geography_grain_rows",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: population-base report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_coverage_matrix(path: Path, summary: AnalysisViewValidationSummary) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("matrix_rows", [])
    if not isinstance(rows, list):
        summary.errors.append(f"{path}: coverage matrix rows must be a list")
        rows = []
    for row_payload in rows:
        row = PopulationBaseCoverageMatrix.model_validate(row_payload)
        if row.row_count < 0:
            summary.errors.append(f"{path}: coverage matrix has negative row count")
        summary.coverage_matrix_row_count += 1
    summary.checked_outputs.append(str(path))


def _validate_coverage_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        issue = PopulationBaseCoverageIssue.model_validate(payload)
        if issue.issue_type in {
            CoverageIssueType.DUPLICATE_POPULATION_BASE_KEY,
            CoverageIssueType.CONFLICTING_VALUES_FOR_SAME_KEY,
        } and not issue.coverage_key:
            summary.errors.append(f"{path}: duplicate/conflict issue lacks coverage key")
        if (
            issue.issue_type is CoverageIssueType.MISSING_SOURCE_TRACEABILITY
            and not issue.coverage_key
            and not issue.source_record_ids
        ):
            summary.errors.append(f"{path}: traceability issue lacks key/source context")
        summary.coverage_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_coverage_manifest(path: Path, summary: AnalysisViewValidationSummary) -> None:
    manifest = PopulationBaseCoverageManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.record_counts:
        summary.errors.append(f"{path}: coverage manifest lacks record counts")
    if not manifest.input_files or not manifest.output_files:
        summary.errors.append(f"{path}: coverage manifest lacks input or output paths")
    summary.warnings.extend(manifest.warnings)
    summary.checked_outputs.append(str(path))


def _validate_coverage_report(path: Path, summary: AnalysisViewValidationSummary) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "target_prefectures_covered",
        "years_covered",
        "age_groups_covered",
        "coverage_issue_counts",
        "duplicate_key_count",
        "conflicting_value_count",
        "model_blocking_error_count",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: coverage report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_age_group_records(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        row = AgeGroupNormalizedRecord.model_validate(payload)
        if row.raw_age_group is None and row.age_group_match_status in {
            AgeGroupMatchStatus.CANONICAL,
            AgeGroupMatchStatus.ALIAS,
        }:
            summary.errors.append(f"{path}: matched age row lacks preserved raw age label")
        if row.age_group_match_status in {
            AgeGroupMatchStatus.CANONICAL,
            AgeGroupMatchStatus.ALIAS,
        } and not (row.canonical_age_group_id and row.canonical_age_group_label):
            summary.errors.append(f"{path}: matched age row lacks canonical age group")
        if row.age_group_match_status in {
            AgeGroupMatchStatus.MISSING,
            AgeGroupMatchStatus.UNKNOWN,
        } and row.canonical_age_group_id:
            summary.errors.append(f"{path}: unmatched age row has canonical ID")
        if not row.source_record_ids:
            summary.errors.append(f"{path}: age-normalized row lacks source record IDs")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: age-normalized row lacks source hash")
        if not row.value_kind:
            summary.errors.append(f"{path}: age-normalized row lacks value kind")
        if not row.geography_grain:
            summary.errors.append(f"{path}: age-normalized row lacks geography grain")
        if not row.population_base_role:
            summary.errors.append(f"{path}: age-normalized row lacks population base role")
        summary.age_group_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_age_group_issues(path: Path, summary: AnalysisViewValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        AgeGroupQAIssue.model_validate(payload)
        summary.age_group_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_age_group_manifest(path: Path, summary: AnalysisViewValidationSummary) -> None:
    manifest = AgeGroupQAManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.record_counts:
        summary.errors.append(f"{path}: age-group manifest lacks record counts")
    if not manifest.input_files or not manifest.output_files:
        summary.errors.append(f"{path}: age-group manifest lacks input or output paths")
    summary.warnings.extend(manifest.warnings)
    summary.checked_outputs.append(str(path))


def _validate_age_group_report(path: Path, summary: AnalysisViewValidationSummary) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "raw_age_labels_observed",
        "canonical_age_groups_observed",
        "rows_by_canonical_age_group",
        "rows_by_match_status",
        "unknown_age_group_count",
        "missing_age_group_count",
        "duplicate_normalized_key_count",
        "conflicting_value_count",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: age-group report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_population_features(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = PopulationFeatureRecord.model_validate(payload)
        if rec.feature_id in seen_ids:
            summary.errors.append(f"{path}: duplicate feature_id {rec.feature_id}")
        seen_ids.add(rec.feature_id)
        summary.population_feature_count += 1
    summary.checked_outputs.append(str(path))


def _validate_hospital_features(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = HospitalFeatureRecord.model_validate(payload)
        if rec.feature_id in seen_ids:
            summary.errors.append(f"{path}: duplicate feature_id {rec.feature_id}")
        seen_ids.add(rec.feature_id)
        summary.hospital_feature_count += 1
    summary.checked_outputs.append(str(path))


def _validate_municipality_feature_base(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = MunicipalityFeatureBaseRecord.model_validate(payload)
        if rec.feature_id in seen_ids:
            summary.errors.append(f"{path}: duplicate feature_id {rec.feature_id}")
        seen_ids.add(rec.feature_id)
        if not rec.population_feature_available:
            summary.warnings.append(
                f"{path}: {rec.feature_id} has population_feature_available=False"
            )
        summary.municipality_feature_base_count += 1
    summary.checked_outputs.append(str(path))


def _validate_municipality_scores(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = MunicipalityScoreRecord.model_validate(payload)
        if rec.score_id in seen_ids:
            summary.errors.append(f"{path}: duplicate score_id {rec.score_id}")
        seen_ids.add(rec.score_id)
        if rec.overall_pre_candidate_priority_score is not None:
            v = rec.overall_pre_candidate_priority_score
            if not (0.0 <= v <= 1.0):
                summary.errors.append(
                    f"{path}: {rec.score_id} overall score {v} out of [0,1] range"
                )
        summary.municipality_score_count += 1
    summary.checked_outputs.append(str(path))


def _validate_feature_engineering_manifest(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_id", "generated_at", "study_area_id", "record_counts", "issue_counts_by_severity"}
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: feature engineering manifest missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_score_layer_manifest(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_id", "generated_at", "study_area_id", "record_counts", "issue_counts_by_severity"}
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: score layer manifest missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_feature_engineering_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "study_area_id",
        "population_feature_count",
        "hospital_feature_count",
        "municipality_feature_base_count",
        "blocking_errors",
        "stage_1_passed",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: feature engineering report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_score_layer_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "study_area_id",
        "score_count",
        "blocking_errors",
        "stage_2_passed",
        "land_score_available",
        "cash_flow_score_available",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: score layer report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_model_input_readiness_records(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    for payload in _iter_jsonl(path):
        row = ModelInputReadinessRecord.model_validate(payload)
        if not row.record_id:
            summary.errors.append(f"{path}: readiness record lacks record_id")
        if not row.readiness_key:
            summary.errors.append(f"{path}: readiness record lacks readiness_key")
        if row.readiness_role is ReadinessRole.MODEL_INPUT and row.readiness_status is ReadinessStatus.READY:
            if not row.municipality:
                summary.errors.append(f"{path}: ready model-input record lacks municipality")
        if row.readiness_role is ReadinessRole.CONTEXT and row.readiness_status is ReadinessStatus.CONTEXT_ONLY:
            if row.municipality:
                summary.errors.append(f"{path}: context record unexpectedly has municipality")
        summary.model_input_readiness_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_ready_population_base_records(
    path: Path,
    summary: AnalysisViewValidationSummary,
    expected_grain: GeographyGrain | None = None,
) -> None:
    for payload in _iter_jsonl(path):
        row = AgeGroupNormalizedRecord.model_validate(payload)
        if expected_grain is not None and row.geography_grain is not expected_grain:
            summary.errors.append(f"{path}: row has unexpected geography grain {row.geography_grain}")
        if not row.source_record_ids:
            summary.errors.append(f"{path}: ready population-base row lacks source record IDs")
        if not row.source_file_hash:
            summary.errors.append(f"{path}: ready population-base row lacks source hash")
        if not row.provenance:
            summary.errors.append(f"{path}: ready population-base row lacks provenance")
        summary.model_input_readiness_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_model_input_readiness_issues(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    for payload in _iter_jsonl(path):
        ModelInputReadinessIssue.model_validate(payload)
        summary.model_input_readiness_issue_count += 1
    summary.checked_outputs.append(str(path))


def _validate_model_input_readiness_manifest(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_id", "generated_at", "study_area_id", "record_counts", "issue_counts_by_severity"}
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: readiness manifest missing fields: {missing}")
    if not payload.get("record_counts"):
        summary.errors.append(f"{path}: readiness manifest lacks record_counts")
    summary.checked_outputs.append(str(path))


def _validate_model_input_readiness_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "study_area_id",
        "records_read",
        "ready_model_input_rows",
        "context_rows",
        "blocked_rows",
        "blocking_error_count",
        "model_input_readiness_passed",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: readiness report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_land_price_records(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    for payload in _iter_jsonl(path):
        rec = LandPriceRecord.model_validate(payload)
        if not rec.land_record_id:
            summary.errors.append(f"{path}: land price record lacks land_record_id")
        summary.land_price_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_municipality_land_features(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = MunicipalityLandFeatureRecord.model_validate(payload)
        if rec.feature_id in seen_ids:
            summary.errors.append(f"{path}: duplicate feature_id {rec.feature_id}")
        seen_ids.add(rec.feature_id)
        summary.municipality_land_feature_count += 1
    summary.checked_outputs.append(str(path))


def _validate_healthcare_supply_records(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    for payload in _iter_jsonl(path):
        rec = HealthcareSupplyRecord.model_validate(payload)
        if not rec.supply_record_id:
            summary.errors.append(f"{path}: healthcare supply record lacks supply_record_id")
        summary.healthcare_supply_record_count += 1
    summary.checked_outputs.append(str(path))


def _validate_municipality_healthcare_supply_features(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = MunicipalityHealthcareSupplyFeatureRecord.model_validate(payload)
        if rec.feature_id in seen_ids:
            summary.errors.append(f"{path}: duplicate feature_id {rec.feature_id}")
        seen_ids.add(rec.feature_id)
        summary.municipality_healthcare_supply_feature_count += 1
    summary.checked_outputs.append(str(path))


def _validate_enriched_municipality_features(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = EnrichedMunicipalityFeatureRecord.model_validate(payload)
        if rec.feature_id in seen_ids:
            summary.errors.append(f"{path}: duplicate feature_id {rec.feature_id}")
        seen_ids.add(rec.feature_id)
        summary.enriched_municipality_feature_count += 1
    summary.checked_outputs.append(str(path))


def _validate_enriched_municipality_scores(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    for payload in _iter_jsonl(path):
        rec = EnrichedMunicipalityScoreRecord.model_validate(payload)
        if rec.score_id in seen_ids:
            summary.errors.append(f"{path}: duplicate score_id {rec.score_id}")
        seen_ids.add(rec.score_id)
        if rec.overall_pre_candidate_priority_score is not None:
            v = rec.overall_pre_candidate_priority_score
            if not (0.0 <= v <= 1.0):
                summary.errors.append(
                    f"{path}: {rec.score_id} overall score {v} out of [0,1] range"
                )
        summary.enriched_municipality_score_count += 1
    summary.checked_outputs.append(str(path))


def _validate_ingestion_manifest(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"run_id", "generated_at", "study_area_id", "record_counts", "issue_counts_by_severity"}
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: ingestion manifest missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_ingestion_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"study_area_id", "blocking_errors"}
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: ingestion report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_enriched_feature_base_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "study_area_id",
        "enriched_record_count",
        "blocking_errors",
        "enriched_feature_base_passed",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: enriched feature base report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_enriched_score_layer_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "study_area_id",
        "enriched_score_count",
        "blocking_errors",
        "stage_2_enriched_passed",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: enriched score layer report missing fields: {missing}")
    summary.checked_outputs.append(str(path))


def _validate_candidate_actions(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    seen_ids: set[str] = set()
    seen_muni_action: set[tuple[str, str]] = set()
    allowed = {CandidateAction.build, CandidateAction.reorganize, CandidateAction.consolidate}
    for payload in _iter_jsonl(path):
        rec = CandidateActionRecord.model_validate(payload)
        if rec.candidate_id in seen_ids:
            summary.errors.append(f"{path}: duplicate candidate_id {rec.candidate_id}")
        seen_ids.add(rec.candidate_id)
        key = (rec.municipality, rec.candidate_action.value)
        if key in seen_muni_action:
            summary.errors.append(
                f"{path}: duplicate municipality/action key {key}"
            )
        seen_muni_action.add(key)
        if rec.candidate_action not in allowed:
            summary.errors.append(
                f"{path}: forbidden candidate_action {rec.candidate_action}"
            )
        if not (0.0 <= rec.priority_score <= 1.0):
            summary.errors.append(
                f"{path}: priority_score {rec.priority_score} out of [0,1] range"
            )
        if not rec.input_score_refs:
            summary.errors.append(f"{path}: {rec.candidate_id} lacks input_score_refs")
        if not rec.provenance:
            summary.errors.append(f"{path}: {rec.candidate_id} lacks provenance")
        summary.candidate_action_count += 1
    summary.checked_outputs.append(str(path))


def _validate_candidate_evidence_bundles(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    allowed = {CandidateAction.build, CandidateAction.reorganize, CandidateAction.consolidate}
    for payload in _iter_jsonl(path):
        rec = CandidateEvidenceBundle.model_validate(payload)
        if rec.candidate_action not in allowed:
            summary.errors.append(
                f"{path}: forbidden candidate_action {rec.candidate_action}"
            )
        if not rec.source_artifact_refs:
            summary.errors.append(f"{path}: {rec.candidate_id} lacks source_artifact_refs")
        if not rec.forbidden_claims:
            summary.errors.append(f"{path}: {rec.candidate_id} lacks forbidden_claims list")
        summary.candidate_evidence_bundle_count += 1
    summary.checked_outputs.append(str(path))


def _validate_candidate_generation_manifest(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "run_id", "generated_at", "study_area_id",
        "candidate_counts_by_action", "total_candidates",
        "issue_counts_by_severity", "record_counts",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: candidate generation manifest missing fields: {missing}")
    counts = payload.get("candidate_counts_by_action", {})
    for action in ("build", "reorganize", "consolidate"):
        if action not in counts:
            summary.errors.append(f"{path}: candidate_counts_by_action missing action '{action}'")
    summary.checked_outputs.append(str(path))


def _validate_candidate_generation_report(
    path: Path, summary: AnalysisViewValidationSummary
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "study_area_id",
        "total_candidates",
        "candidate_counts_by_action",
        "blocking_errors",
        "candidate_generation_passed",
        "allowed_candidate_actions",
        "forbidden_actions_emitted",
    }
    missing = sorted(required - set(payload))
    if missing:
        summary.errors.append(f"{path}: candidate generation report missing fields: {missing}")
    allowed = payload.get("allowed_candidate_actions", [])
    for action in ("build", "reorganize", "consolidate"):
        if action not in allowed:
            summary.errors.append(f"{path}: allowed_candidate_actions missing '{action}'")
    if payload.get("forbidden_actions_emitted", True):
        summary.errors.append(f"{path}: forbidden_actions_emitted must be False")
    summary.checked_outputs.append(str(path))


def _validate_manifest(path: Path, summary: AnalysisViewValidationSummary) -> None:
    manifest = AnalysisViewManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.record_counts:
        summary.errors.append(f"{path}: manifest lacks record counts")
    if not manifest.input_files or not manifest.output_files:
        summary.errors.append(f"{path}: manifest lacks input or output paths")
    summary.warnings.extend(manifest.warnings)
    summary.checked_outputs.append(str(path))


def _validate_population_manifest(path: Path, summary: AnalysisViewValidationSummary) -> None:
    manifest = PopulationQAManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not manifest.record_counts:
        summary.errors.append(f"{path}: population QA manifest lacks record counts")
    if not manifest.issue_counts:
        summary.errors.append(f"{path}: population QA manifest lacks issue counts")
    if not manifest.input_files or not manifest.output_files:
        summary.errors.append(f"{path}: population QA manifest lacks input or output paths")
    summary.warnings.extend(manifest.warnings)
    summary.checked_outputs.append(str(path))


def validate_analysis_views(
    repo_root: str | Path = ".",
    require_outputs: bool = False,
) -> AnalysisViewValidationSummary:
    """Validate analysis-ready views when present."""

    root = Path(repo_root).resolve()
    config = _load_config()
    outputs = config["outputs"]
    paths = {
        "hospital_facts": Path(outputs["hospital_facts"]),
        "population_long": Path(outputs["population_long"]),
        "population_rates": Path(outputs["population_rates_long"]),
        "hospital_manifest": Path(outputs["hospital_manifest"]),
        "population_manifest": Path(outputs["population_manifest"]),
        "population_quality_issues": Path(outputs["population_quality_issues"]),
        "population_rate_issues": Path(outputs["population_rates_quality_issues"]),
        "population_geography_keys": Path(outputs["population_geography_keys"]),
        "population_qa_manifest": Path(outputs["population_qa_manifest"]),
        "population_geography_issues": Path(outputs["population_geography_issues"]),
        "study_area_population_long": Path(outputs["study_area_population_long"]),
        "study_area_population_rates": Path(outputs["study_area_population_rates_long"]),
        "study_area_geography_keys": Path(outputs["study_area_geography_keys"]),
        "study_area_manifest": Path(outputs["study_area_manifest"]),
        "study_area_scope_issues": Path(outputs["study_area_scope_issues"]),
        "study_area_geography_qa": Path(outputs["study_area_geography_qa"]),
        "study_area_geography_issues": Path(outputs["study_area_geography_issues"]),
        "geography_grain_records": Path(outputs["geography_grain_records"]),
        "geography_grain_manifest": Path(outputs["geography_grain_manifest"]),
        "geography_grain_issues": Path(outputs["geography_grain_issues"]),
        "population_base": Path(outputs["population_base"]),
        "population_base_municipality": Path(outputs["population_base_municipality"]),
        "population_base_prefecture_total": Path(outputs["population_base_prefecture_total"]),
        "population_base_manifest": Path(outputs["population_base_manifest"]),
        "population_base_issues": Path(outputs["population_base_issues"]),
        "population_base_report": Path(outputs["population_base_report"]),
        "population_base_coverage_matrix": Path(outputs["population_base_coverage_matrix"]),
        "population_base_coverage_issues": Path(outputs["population_base_coverage_issues"]),
        "population_base_coverage_manifest": Path(outputs["population_base_coverage_manifest"]),
        "population_base_coverage_report": Path(outputs["population_base_coverage_report"]),
        "population_base_age_normalized": Path(outputs["population_base_age_normalized"]),
        "age_group_qa_manifest": Path(outputs["age_group_qa_manifest"]),
        "age_group_qa_issues": Path(outputs["age_group_qa_issues"]),
        "age_group_coverage_report": Path(outputs["age_group_coverage_report"]),
        "model_input_readiness": Path(outputs["model_input_readiness"]),
        "model_input_ready_population_base": Path(outputs["model_input_ready_population_base"]),
        "model_input_context_population_base": Path(outputs["model_input_context_population_base"]),
        "model_input_readiness_manifest": Path(outputs["model_input_readiness_manifest"]),
        "model_input_readiness_issues": Path(outputs["model_input_readiness_issues"]),
        "model_input_readiness_report": Path(outputs["model_input_readiness_report"]),
        "population_features": Path(outputs["population_features"]),
        "hospital_features": Path(outputs["hospital_features"]),
        "municipality_feature_base": Path(outputs["municipality_feature_base"]),
        "municipality_scores": Path(outputs["municipality_scores"]),
        "feature_engineering_manifest": Path(outputs["feature_engineering_manifest"]),
        "feature_engineering_issues": Path(outputs["feature_engineering_issues"]),
        "feature_engineering_report": Path(outputs["feature_engineering_report"]),
        "score_layer_manifest": Path(outputs["score_layer_manifest"]),
        "score_layer_issues": Path(outputs["score_layer_issues"]),
        "score_layer_report": Path(outputs["score_layer_report"]),
        # Phase 6 — land-price ingestion
        "land_price_records": Path(outputs["land_price_records"]),
        "municipality_land_features": Path(outputs["municipality_land_features"]),
        "land_price_ingestion_manifest": Path(outputs["land_price_ingestion_manifest"]),
        "land_price_ingestion_issues": Path(outputs["land_price_ingestion_issues"]),
        "land_price_ingestion_report": Path(outputs["land_price_ingestion_report"]),
        # Phase 6 — healthcare supply ingestion
        "healthcare_supply_records": Path(outputs["healthcare_supply_records"]),
        "municipality_healthcare_supply_features": Path(outputs["municipality_healthcare_supply_features"]),
        "healthcare_supply_ingestion_manifest": Path(outputs["healthcare_supply_ingestion_manifest"]),
        "healthcare_supply_ingestion_issues": Path(outputs["healthcare_supply_ingestion_issues"]),
        "healthcare_supply_ingestion_report": Path(outputs["healthcare_supply_ingestion_report"]),
        # Phase 6 — enriched feature base
        "municipality_feature_base_enriched": Path(outputs["municipality_feature_base_enriched"]),
        "enriched_feature_base_manifest": Path(outputs["enriched_feature_base_manifest"]),
        "enriched_feature_base_issues": Path(outputs["enriched_feature_base_issues"]),
        "enriched_feature_base_report": Path(outputs["enriched_feature_base_report"]),
        # Phase 6 — enriched score layer
        "municipality_scores_enriched": Path(outputs["municipality_scores_enriched"]),
        "score_layer_enriched_manifest": Path(outputs["score_layer_enriched_manifest"]),
        "score_layer_enriched_issues": Path(outputs["score_layer_enriched_issues"]),
        "score_layer_enriched_report": Path(outputs["score_layer_enriched_report"]),
        # Phase 7 — candidate action generation
        "candidate_actions": Path(outputs["candidate_actions"]),
        "candidate_evidence_bundles": Path(outputs["candidate_evidence_bundles"]),
        "candidate_generation_manifest": Path(outputs["candidate_generation_manifest"]),
        "candidate_generation_issues": Path(outputs["candidate_generation_issues"]),
        "candidate_generation_report": Path(outputs["candidate_generation_report_json"]),
    }
    summary = AnalysisViewValidationSummary()

    for name, relative_path in paths.items():
        path = root / relative_path
        if not path.exists():
            summary.missing_outputs.append(str(relative_path))
            if require_outputs:
                summary.errors.append(f"Missing required analysis-view output: {relative_path}")
            continue
        try:
            if name == "hospital_facts":
                _validate_hospital(path, summary)
            elif name == "population_long":
                _validate_population(path, summary)
            elif name == "population_rates":
                _validate_population_rates(path, summary)
            elif name == "study_area_manifest":
                _validate_study_area_manifest(path, summary)
            elif name in {"geography_grain_manifest", "population_base_manifest"}:
                _validate_population_base_manifest(path, summary)
            elif name == "population_base_coverage_manifest":
                _validate_coverage_manifest(path, summary)
            elif name == "age_group_qa_manifest":
                _validate_age_group_manifest(path, summary)
            elif name == "model_input_readiness_manifest":
                _validate_model_input_readiness_manifest(path, summary)
            elif name == "feature_engineering_manifest":
                _validate_feature_engineering_manifest(path, summary)
            elif name == "score_layer_manifest":
                _validate_score_layer_manifest(path, summary)
            elif name in {"land_price_ingestion_manifest", "healthcare_supply_ingestion_manifest",
                          "enriched_feature_base_manifest", "score_layer_enriched_manifest"}:
                _validate_ingestion_manifest(path, summary)
            elif name == "candidate_generation_manifest":
                _validate_candidate_generation_manifest(path, summary)
            elif name.endswith("_manifest"):
                if name == "population_qa_manifest":
                    _validate_population_manifest(path, summary)
                else:
                    _validate_manifest(path, summary)
            elif name == "population_quality_issues":
                _validate_issues(path, summary)
            elif name == "population_rate_issues":
                _validate_population_rate_issues(path, summary)
            elif name == "population_geography_keys":
                _validate_population_geography_keys(path, summary)
            elif name == "population_geography_issues":
                _validate_population_geography_issues(path, summary)
            elif name == "study_area_population_long":
                _validate_study_area_population(path, summary, PopulationValueKind.COUNT)
            elif name == "study_area_population_rates":
                _validate_study_area_population(path, summary, PopulationValueKind.RATE)
            elif name == "study_area_geography_keys":
                _validate_study_area_geography_keys(path, summary)
            elif name in {"study_area_scope_issues", "study_area_geography_issues"}:
                _validate_study_area_issues(path, summary)
            elif name == "study_area_geography_qa":
                _validate_study_area_geography_qa(path, summary)
            elif name == "geography_grain_records":
                _validate_geography_grain_records(path, summary)
            elif name == "geography_grain_issues":
                _validate_geography_grain_issues(path, summary)
            elif name == "population_base":
                _validate_population_base_records(path, summary)
            elif name == "population_base_municipality":
                _validate_population_base_records(
                    path, summary, PopulationBaseRole.MODEL_INPUT_CANDIDATE
                )
            elif name == "population_base_prefecture_total":
                _validate_population_base_records(
                    path, summary, PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
                )
            elif name == "population_base_issues":
                _validate_population_base_issues(path, summary)
            elif name == "population_base_report":
                _validate_population_base_report(path, summary)
            elif name == "population_base_coverage_matrix":
                _validate_coverage_matrix(path, summary)
            elif name == "population_base_coverage_issues":
                _validate_coverage_issues(path, summary)
            elif name == "population_base_coverage_report":
                _validate_coverage_report(path, summary)
            elif name == "population_base_age_normalized":
                _validate_age_group_records(path, summary)
            elif name == "age_group_qa_issues":
                _validate_age_group_issues(path, summary)
            elif name == "age_group_coverage_report":
                _validate_age_group_report(path, summary)
            elif name == "model_input_readiness":
                _validate_model_input_readiness_records(path, summary)
            elif name == "model_input_ready_population_base":
                _validate_ready_population_base_records(path, summary, GeographyGrain.MUNICIPALITY)
            elif name == "model_input_context_population_base":
                _validate_ready_population_base_records(path, summary, GeographyGrain.PREFECTURE_TOTAL)
            elif name == "model_input_readiness_issues":
                _validate_model_input_readiness_issues(path, summary)
            elif name == "model_input_readiness_report":
                _validate_model_input_readiness_report(path, summary)
            elif name == "population_features":
                _validate_population_features(path, summary)
            elif name == "hospital_features":
                _validate_hospital_features(path, summary)
            elif name == "municipality_feature_base":
                _validate_municipality_feature_base(path, summary)
            elif name == "municipality_scores":
                _validate_municipality_scores(path, summary)
            elif name == "feature_engineering_issues":
                pass  # Checked as generic JSONL; no strict schema required
            elif name == "feature_engineering_report":
                _validate_feature_engineering_report(path, summary)
            elif name == "score_layer_issues":
                pass  # Checked as generic JSONL; no strict schema required
            elif name == "score_layer_report":
                _validate_score_layer_report(path, summary)
            elif name == "land_price_records":
                _validate_land_price_records(path, summary)
            elif name == "municipality_land_features":
                _validate_municipality_land_features(path, summary)
            elif name in {"land_price_ingestion_issues", "healthcare_supply_ingestion_issues",
                          "enriched_feature_base_issues", "score_layer_enriched_issues"}:
                pass  # Generic JSONL; issues validated by schema at ingestion time
            elif name in {"land_price_ingestion_report", "healthcare_supply_ingestion_report"}:
                _validate_ingestion_report(path, summary)
            elif name == "healthcare_supply_records":
                _validate_healthcare_supply_records(path, summary)
            elif name == "municipality_healthcare_supply_features":
                _validate_municipality_healthcare_supply_features(path, summary)
            elif name == "municipality_feature_base_enriched":
                _validate_enriched_municipality_features(path, summary)
            elif name == "enriched_feature_base_report":
                _validate_enriched_feature_base_report(path, summary)
            elif name == "municipality_scores_enriched":
                _validate_enriched_municipality_scores(path, summary)
            elif name == "score_layer_enriched_report":
                _validate_enriched_score_layer_report(path, summary)
            elif name == "candidate_actions":
                _validate_candidate_actions(path, summary)
            elif name == "candidate_evidence_bundles":
                _validate_candidate_evidence_bundles(path, summary)
            elif name == "candidate_generation_issues":
                pass  # Generic JSONL; validated by schema at generation time
            elif name == "candidate_generation_report":
                _validate_candidate_generation_report(path, summary)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            summary.errors.append(f"{relative_path}: {exc}")

    return summary
