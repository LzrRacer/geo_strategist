"""Study-area contracts for deterministic target scoping.

These models only describe study-area membership and geography QA. They do not
score demand, land, candidate sites, cash flow, or proposals.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.analysis_views import IssueSeverity
from geo_strategist.data.population_views import PopulationValueKind, normalize_label_text
from geo_strategist.data.provenance import ProvenanceRecord


class StudyAreaScopeStatus(str, Enum):
    """Explicit scope status for source rows."""

    IN_SCOPE = "in_scope"
    OUTSIDE_SCOPE = "outside_scope"
    SCOPE_UNKNOWN = "scope_unknown"


class StudyArea(BaseModel):
    """Configured study area and deterministic prefecture aliases."""

    model_config = ConfigDict(extra="forbid")

    study_area_id: str = Field(min_length=1)
    target_prefectures: list[str] = Field(min_length=1)
    aliases: dict[str, list[str]] = Field(min_length=1)
    scope_policy: dict[str, object] = Field(default_factory=dict)

    @field_validator("target_prefectures")
    @classmethod
    def _target_prefectures_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [normalize_label_text(value) or "" for value in values]
        if any(not value for value in cleaned):
            raise ValueError("target prefectures must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("target prefectures must be unique")
        return cleaned

    @field_validator("aliases")
    @classmethod
    def _aliases_must_not_be_blank(cls, aliases: dict[str, list[str]]) -> dict[str, list[str]]:
        cleaned: dict[str, list[str]] = {}
        for target, values in aliases.items():
            clean_target = normalize_label_text(target) or ""
            clean_values = [normalize_label_text(value) or "" for value in values]
            if not clean_target or not clean_values or any(not value for value in clean_values):
                raise ValueError("aliases must use non-empty target and alias labels")
            cleaned[clean_target] = clean_values
        return cleaned

    @model_validator(mode="after")
    def _aliases_must_reference_targets(self) -> "StudyArea":
        missing = set(self.target_prefectures) - set(self.aliases)
        if missing:
            raise ValueError(f"target prefectures lack aliases: {sorted(missing)}")
        unknown = set(self.aliases) - set(self.target_prefectures)
        if unknown:
            raise ValueError(f"aliases reference non-target prefectures: {sorted(unknown)}")
        return self


class StudyAreaPopulationRecord(BaseModel):
    """Population row copied into a target-scoped study-area view."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    source_record_ids: list[str] = Field(min_length=1)
    source_file_path: Path | None = None
    source_sheet: str | None = None
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    scope_status: StudyAreaScopeStatus
    raw_prefecture: str | None = None
    matched_target_prefecture: str | None = None
    raw_municipality: str | None = None
    prefecture: str | None = None
    municipality: str | None = None
    year: int | str | None = None
    age_group: str | None = None
    geography_labels: dict[str, str] = Field(default_factory=dict)
    time_labels: dict[str, str] = Field(default_factory=dict)
    age_labels: dict[str, str] = Field(default_factory=dict)
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

    @field_validator("geography_labels", "time_labels", "age_labels")
    @classmethod
    def _labels_must_not_be_blank(cls, labels: dict[str, str]) -> dict[str, str]:
        cleaned = {
            normalize_label_text(key) or "": normalize_label_text(value) or ""
            for key, value in labels.items()
        }
        if any(not key or not value for key, value in cleaned.items()):
            raise ValueError("label keys and values must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_scope_and_value_semantics(self) -> "StudyAreaPopulationRecord":
        if not self.source_file_hash:
            raise ValueError("study-area rows require source_file_hash")
        if not self.provenance:
            raise ValueError("study-area rows require provenance")
        if self.scope_status is StudyAreaScopeStatus.IN_SCOPE and not self.matched_target_prefecture:
            raise ValueError("in-scope rows require matched_target_prefecture")
        if self.scope_status is not StudyAreaScopeStatus.IN_SCOPE and self.matched_target_prefecture:
            raise ValueError("only in-scope rows may have matched_target_prefecture")
        if self.value_kind is PopulationValueKind.COUNT:
            if self.population_value is None:
                raise ValueError("count study-area rows require population_value")
            if self.rate_value is not None:
                raise ValueError("count study-area rows must not include rate_value")
        elif self.value_kind is PopulationValueKind.RATE:
            if self.rate_value is None:
                raise ValueError("rate study-area rows require rate_value")
            if self.population_value is not None:
                raise ValueError("rate study-area rows must not include population_value")
        return self


class StudyAreaGeographyIssue(BaseModel):
    """Issue discovered during study-area scoping or targeted geography QA."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: IssueSeverity
    issue_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    scope_status: StudyAreaScopeStatus
    target_prefecture: str | None = None
    raw_prefecture: str | None = None
    raw_municipality: str | None = None
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

    @model_validator(mode="after")
    def _in_scope_issue_target_consistency(self) -> "StudyAreaGeographyIssue":
        if self.scope_status is StudyAreaScopeStatus.IN_SCOPE and not self.target_prefecture:
            raise ValueError("in-scope study-area issues require target_prefecture")
        return self


class StudyAreaManifest(BaseModel):
    """Manifest for deterministic study-area filtering or target QA."""

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
    def _hashes_must_be_present(cls, values: list[str]) -> list[str]:
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
