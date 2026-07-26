"""Contracts for pre-demand population-base coverage QA."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.geography_grain import GeographyGrain, PopulationBaseRole
from geo_strategist.data.population_views import PopulationValueKind, normalize_label_text


class CoverageAxis(str, Enum):
    """Coverage matrix axes."""

    TARGET_PREFECTURE = "target_prefecture"
    YEAR = "year"
    AGE_GROUP = "age_group"
    GEOGRAPHY_GRAIN = "geography_grain"
    VALUE_KIND = "value_kind"
    POPULATION_BASE_ROLE = "population_base_role"


class CoverageIssueType(str, Enum):
    """Coverage QA issue categories."""

    MISSING_EXPECTED_YEAR = "missing_expected_year"
    MISSING_EXPECTED_AGE_GROUP = "missing_expected_age_group"
    MISSING_PREFECTURE_YEAR_AGE_COMBINATION = "missing_prefecture_year_age_combination"
    MISSING_COUNT_FOR_RATE_COMBINATION = "missing_count_for_rate_combination"
    MISSING_RATE_FOR_COUNT_COMBINATION = "missing_rate_for_count_combination"
    DUPLICATE_POPULATION_BASE_KEY = "duplicate_population_base_key"
    CONFLICTING_VALUES_FOR_SAME_KEY = "conflicting_values_for_same_key"
    UNEXPECTED_GEOGRAPHY_GRAIN = "unexpected_geography_grain"
    UNEXPECTED_VALUE_KIND = "unexpected_value_kind"
    MISSING_SOURCE_TRACEABILITY = "missing_source_traceability"


class CoverageSeverity(str, Enum):
    """Coverage QA severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class PopulationBaseCoverageMatrix(BaseModel):
    """One coverage matrix row."""

    model_config = ConfigDict(extra="forbid")

    matrix_id: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    coverage_axis: CoverageAxis
    axis_values: dict[str, str | int | None] = Field(default_factory=dict)
    target_prefecture: str | None = None
    year: int | str | None = None
    age_group: str | None = None
    value_kind: PopulationValueKind | None = None
    geography_grain: GeographyGrain | None = None
    population_base_role: PopulationBaseRole | None = None
    row_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _study_area_and_axis_values_required(self) -> "PopulationBaseCoverageMatrix":
        if not self.study_area_id:
            raise ValueError("study_area_id must be present")
        if not self.axis_values:
            raise ValueError("axis_values must be present")
        return self


class PopulationBaseCoverageIssue(BaseModel):
    """One issue found during population-base coverage QA."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: CoverageSeverity
    issue_type: CoverageIssueType
    message: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    coverage_key: str | None = None
    target_prefecture: str | None = None
    year: int | str | None = None
    age_group: str | None = None
    value_kind: PopulationValueKind | None = None
    geography_grain: GeographyGrain | None = None
    population_base_role: PopulationBaseRole | None = None
    source_record_ids: list[str] = Field(default_factory=list)
    source_file_hashes: list[str] = Field(default_factory=list)
    recommended_action: str = Field(min_length=1)

    @field_validator("source_record_ids", "source_file_hashes")
    @classmethod
    def _ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source IDs and hashes must be non-empty when present")
        return cleaned

    @model_validator(mode="after")
    def _required_issue_context(self) -> "PopulationBaseCoverageIssue":
        if self.issue_type in {
            CoverageIssueType.DUPLICATE_POPULATION_BASE_KEY,
            CoverageIssueType.CONFLICTING_VALUES_FOR_SAME_KEY,
        } and not self.coverage_key:
            raise ValueError("duplicate/conflicting-value issues require coverage_key")
        if (
            self.issue_type is CoverageIssueType.MISSING_SOURCE_TRACEABILITY
            and not self.coverage_key
            and not self.source_record_ids
        ):
            raise ValueError("traceability issues require coverage_key or source_record_ids")
        return self


class PopulationBaseCoverageManifest(BaseModel):
    """Manifest for one population-base coverage QA run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    generated_at: datetime
    study_area_id: str = Field(min_length=1)
    target_prefectures: list[str] = Field(min_length=1)
    input_files: list[Path] = Field(default_factory=list)
    output_files: list[Path] = Field(default_factory=list)
    record_counts: dict[str, int] = Field(default_factory=dict)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    expectation_sources: list[dict[str, object]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("target_prefectures")
    @classmethod
    def _target_prefectures_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [normalize_label_text(value) or "" for value in values]
        if any(not value for value in cleaned):
            raise ValueError("target prefectures must be non-empty")
        return cleaned

    @field_validator("record_counts", "issue_counts")
    @classmethod
    def _counts_must_be_nonnegative(cls, counts: dict[str, int]) -> dict[str, int]:
        if any(value < 0 for value in counts.values()):
            raise ValueError("count fields must be nonnegative")
        return counts
