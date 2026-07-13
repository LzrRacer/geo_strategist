"""Geography-grain and pre-demand population-base contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.analysis_views import IssueSeverity
from geo_strategist.data.population_views import PopulationValueKind, normalize_label_text
from geo_strategist.data.provenance import ProvenanceRecord


class GeographyGrain(str, Enum):
    """Deterministic geography grain for target-scope population rows."""

    PREFECTURE_TOTAL = "prefecture_total"
    MUNICIPALITY = "municipality"
    UNKNOWN = "unknown"


class PopulationBaseRole(str, Enum):
    """Pre-demand role for population-base rows."""

    MODEL_INPUT_CANDIDATE = "model_input_candidate"
    CONTEXT_PREFECTURE_TOTAL = "context_prefecture_total"
    EXCLUDED_REQUIRES_REVIEW = "excluded_requires_review"


class _TraceablePopulationRow(BaseModel):
    """Shared validation for target-scope derived rows."""

    model_config = ConfigDict(extra="forbid")

    source_record_ids: list[str] = Field(min_length=1)
    source_file_path: Path | None = None
    source_sheet: str | None = None
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    matched_target_prefecture: str = Field(min_length=1)
    raw_municipality: str | None = None
    municipality: str | None = None
    year: int | str | None = None
    age_group: str | None = None
    population_value: float | None = None
    rate_value: float | None = None
    unit: str | None = None
    value_kind: PopulationValueKind

    @field_validator("source_record_ids")
    @classmethod
    def _source_record_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source record IDs must be non-empty")
        return cleaned

    @field_validator("municipality", "raw_municipality")
    @classmethod
    def _normalize_optional_labels(cls, value: str | None) -> str | None:
        return normalize_label_text(value)

    @model_validator(mode="after")
    def _traceability_and_value_semantics(self) -> "_TraceablePopulationRow":
        if not self.source_file_hash:
            raise ValueError("source_file_hash must be present")
        if not self.provenance:
            raise ValueError("provenance must be present")
        if not self.matched_target_prefecture:
            raise ValueError("matched target prefecture must be present")
        if self.value_kind is PopulationValueKind.COUNT:
            if self.population_value is None:
                raise ValueError("count rows require population_value")
            if self.rate_value is not None:
                raise ValueError("count rows must not include rate_value")
        elif self.value_kind is PopulationValueKind.RATE:
            if self.rate_value is None:
                raise ValueError("rate rows require rate_value")
            if self.population_value is not None:
                raise ValueError("rate rows must not include population_value")
        return self


class GeographyGrainRecord(_TraceablePopulationRow):
    """Population row classified by deterministic geography grain."""

    record_id: str = Field(min_length=1)
    raw_prefecture: str | None = None
    geography_labels: dict[str, str] = Field(default_factory=dict)
    geography_grain: GeographyGrain
    classification_evidence: list[str] = Field(default_factory=list)

    @field_validator("geography_labels")
    @classmethod
    def _labels_must_not_be_blank(cls, labels: dict[str, str]) -> dict[str, str]:
        cleaned = {
            normalize_label_text(key) or "": normalize_label_text(value) or ""
            for key, value in labels.items()
        }
        if any(not key for key in cleaned):
            raise ValueError("label keys must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_grain(self) -> "GeographyGrainRecord":
        if self.geography_grain is GeographyGrain.MUNICIPALITY and not self.municipality:
            raise ValueError("municipality grain requires municipality")
        if self.geography_grain is GeographyGrain.PREFECTURE_TOTAL and self.municipality:
            raise ValueError("prefecture_total grain must not include municipality")
        return self


class StudyAreaPopulationBaseRecord(_TraceablePopulationRow):
    """Auditable pre-demand population-base row."""

    base_record_id: str = Field(min_length=1)
    grain_record_id: str = Field(min_length=1)
    geography_grain: GeographyGrain
    population_base_role: PopulationBaseRole

    @model_validator(mode="after")
    def _validate_role_grain_consistency(self) -> "StudyAreaPopulationBaseRecord":
        if self.geography_grain is GeographyGrain.MUNICIPALITY and not self.municipality:
            raise ValueError("municipality grain requires municipality")
        if self.geography_grain is GeographyGrain.PREFECTURE_TOTAL and self.municipality:
            raise ValueError("prefecture_total grain must not include municipality")
        if (
            self.population_base_role is PopulationBaseRole.MODEL_INPUT_CANDIDATE
            and self.geography_grain is not GeographyGrain.MUNICIPALITY
        ):
            raise ValueError("model_input_candidate rows must be municipality grain")
        if (
            self.population_base_role is PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
            and self.geography_grain is not GeographyGrain.PREFECTURE_TOTAL
        ):
            raise ValueError("context_prefecture_total rows must be prefecture_total grain")
        if (
            self.population_base_role is PopulationBaseRole.EXCLUDED_REQUIRES_REVIEW
            and self.geography_grain is not GeographyGrain.UNKNOWN
        ):
            raise ValueError("excluded_requires_review rows must be unknown grain")
        return self


class GeographyGrainIssue(BaseModel):
    """Issue discovered during grain classification or population-base building."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: IssueSeverity
    issue_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    matched_target_prefecture: str = Field(min_length=1)
    raw_municipality: str | None = None
    municipality: str | None = None
    geography_grain: GeographyGrain
    population_base_role: PopulationBaseRole | None = None
    year: int | str | None = None
    age_group: str | None = None
    value_kind: PopulationValueKind
    source_record_ids: list[str] = Field(default_factory=list)
    source_file_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    recommended_action: str = Field(min_length=1)

    @field_validator("source_record_ids")
    @classmethod
    def _source_record_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source record IDs must be non-empty")
        return cleaned


class StudyAreaPopulationBaseManifest(BaseModel):
    """Manifest for geography-grain or pre-demand population-base builds."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    generated_at: datetime
    study_area_id: str = Field(min_length=1)
    target_prefectures: list[str] = Field(min_length=1)
    input_files: list[Path] = Field(default_factory=list)
    output_files: list[Path] = Field(default_factory=list)
    source_file_hashes: list[str] = Field(min_length=1)
    record_counts: dict[str, int] = Field(default_factory=dict)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("target_prefectures")
    @classmethod
    def _target_prefectures_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [normalize_label_text(value) or "" for value in values]
        if any(not value for value in cleaned):
            raise ValueError("target prefectures must be non-empty")
        return cleaned

    @field_validator("source_file_hashes")
    @classmethod
    def _source_hashes_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source file hashes must be present")
        return cleaned

    @field_validator("record_counts", "issue_counts")
    @classmethod
    def _counts_must_be_nonnegative(cls, counts: dict[str, int]) -> dict[str, int]:
        if any(value < 0 for value in counts.values()):
            raise ValueError("count fields must be nonnegative")
        return counts
