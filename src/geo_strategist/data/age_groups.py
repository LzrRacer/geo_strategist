"""Age-group alias and normalization contracts."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.geography_grain import GeographyGrain, PopulationBaseRole
from geo_strategist.data.population_base_coverage import CoverageSeverity
from geo_strategist.data.population_views import PopulationValueKind
from geo_strategist.data.provenance import ProvenanceRecord


class AgeGroupMatchStatus(str, Enum):
    """Status from deterministic age-group alias matching."""

    CANONICAL = "canonical"
    ALIAS = "alias"
    MISSING = "missing"
    UNKNOWN = "unknown"


class AgeGroupKind(str, Enum):
    """Semantic kind of age group label."""

    TOTAL = "total"
    AGE_BAND = "age_band"
    THRESHOLD_PLUS = "threshold_plus"
    UNKNOWN = "unknown"


class AgeGroupAlias(BaseModel):
    """Configured canonical age group and aliases."""

    model_config = ConfigDict(extra="forbid")

    age_group_id: str = Field(min_length=1)
    canonical_label: str = Field(min_length=1)
    kind: AgeGroupKind
    sort_key: int
    aliases: list[str] = Field(min_length=1)

    @field_validator("aliases")
    @classmethod
    def _aliases_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("age group aliases must be non-empty")
        return cleaned


def normalize_age_label(value: str | None) -> str | None:
    """Normalize labels for deterministic alias lookup while preserving semantics."""

    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    for dash in ("〜", "～", "~", "－", "ー", "–", "—"):
        text = text.replace(dash, "-")
    text = " ".join(text.split())
    return text or None


class AgeGroupNormalizedRecord(BaseModel):
    """Population-base row with raw and canonical age-group labels."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    base_record_id: str = Field(min_length=1)
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
    raw_age_group: str | None = None
    canonical_age_group_id: str | None = None
    canonical_age_group_label: str | None = None
    age_group_match_status: AgeGroupMatchStatus
    age_group_kind: AgeGroupKind
    age_sort_key: int | None = None
    population_value: float | None = None
    rate_value: float | None = None
    unit: str | None = None
    value_kind: PopulationValueKind
    geography_grain: GeographyGrain
    population_base_role: PopulationBaseRole

    @field_validator("source_record_ids")
    @classmethod
    def _source_record_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source record IDs must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_age_group_and_traceability(self) -> "AgeGroupNormalizedRecord":
        if not self.source_file_hash:
            raise ValueError("source_file_hash must be present")
        if not self.provenance:
            raise ValueError("provenance must be present")
        if self.age_group_match_status in {
            AgeGroupMatchStatus.CANONICAL,
            AgeGroupMatchStatus.ALIAS,
        }:
            if not self.canonical_age_group_id or not self.canonical_age_group_label:
                raise ValueError("matched age-group rows require canonical ID and label")
            if self.raw_age_group is None:
                raise ValueError("matched age-group rows must preserve raw age label")
        if self.age_group_match_status in {
            AgeGroupMatchStatus.MISSING,
            AgeGroupMatchStatus.UNKNOWN,
        } and self.canonical_age_group_id:
            raise ValueError("unmatched age-group rows must not have canonical ID")
        if self.canonical_age_group_id == "total" and self.age_group_kind is not AgeGroupKind.TOTAL:
            raise ValueError("total population rows must use total age group kind")
        if self.value_kind is PopulationValueKind.COUNT:
            if self.population_value is None or self.rate_value is not None:
                raise ValueError("count rows require only population_value")
        if self.value_kind is PopulationValueKind.RATE:
            if self.rate_value is None or self.population_value is not None:
                raise ValueError("rate rows require only rate_value")
        return self


class AgeGroupQAIssue(BaseModel):
    """Issue discovered during age-group normalization."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: CoverageSeverity
    issue_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    normalized_key: str | None = None
    raw_age_group: str | None = None
    canonical_age_group_id: str | None = None
    canonical_age_group_label: str | None = None
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


class AgeGroupQAManifest(BaseModel):
    """Manifest for deterministic age-group normalization."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    generated_at: datetime
    study_area_id: str = Field(min_length=1)
    input_files: list[Path] = Field(default_factory=list)
    output_files: list[Path] = Field(default_factory=list)
    record_counts: dict[str, int] = Field(default_factory=dict)
    issue_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("record_counts", "issue_counts")
    @classmethod
    def _counts_must_be_nonnegative(cls, counts: dict[str, int]) -> dict[str, int]:
        if any(value < 0 for value in counts.values()):
            raise ValueError("count fields must be nonnegative")
        return counts
