"""Population-specific analysis-ready and QA view contracts."""

from __future__ import annotations

import unicodedata
from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.provenance import ProvenanceRecord


class PopulationValueKind(str, Enum):
    """Explicit semantic kind for population values."""

    COUNT = "count"
    RATE = "rate"


class PopulationQASeverity(str, Enum):
    """Severity levels for population geography QA issues."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


def normalize_label_text(value: str | None) -> str | None:
    """Apply deterministic cleanup while preserving Japanese labels."""

    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\u3000", " ")
    text = " ".join(text.split())
    return text or None


class _PopulationProvenancedRow(BaseModel):
    """Common provenance validation for population-derived rows."""

    model_config = ConfigDict(extra="forbid")

    source_record_ids: list[str] = Field(min_length=1)
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    source_file_path: Path | None = None
    source_sheet: str | None = None
    geography_labels: dict[str, str] = Field(default_factory=dict)
    time_labels: dict[str, str] = Field(default_factory=dict)
    age_labels: dict[str, str] = Field(default_factory=dict)

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
    def _require_provenance_and_hash(self) -> "_PopulationProvenancedRow":
        if not self.provenance:
            raise ValueError("population rows require provenance")
        if not self.source_file_hash:
            raise ValueError("population rows require source_file_hash")
        return self


class PopulationLongRecord(_PopulationProvenancedRow):
    """Population count row ready for downstream feature building."""

    record_id: str = Field(min_length=1)
    prefecture: str | None = None
    municipality: str | None = None
    age_group: str | None = None
    year: int | str | None = None
    population_value: float
    unit: str | None = None
    value_kind: PopulationValueKind = PopulationValueKind.COUNT

    @model_validator(mode="after")
    def _validate_population_count_semantics(self) -> "PopulationLongRecord":
        if self.value_kind is not PopulationValueKind.COUNT:
            raise ValueError("population_long records must use count semantics")
        if not (self.prefecture or self.municipality):
            raise ValueError("population records require at least one geography label")
        return self


class PopulationRateRecord(_PopulationProvenancedRow):
    """Population rate row kept separate from count-based views."""

    record_id: str = Field(min_length=1)
    prefecture: str | None = None
    municipality: str | None = None
    age_group: str | None = None
    year: int | str | None = None
    rate_value: float
    unit: str | None = None
    value_kind: PopulationValueKind = PopulationValueKind.RATE

    @model_validator(mode="after")
    def _validate_population_rate_semantics(self) -> "PopulationRateRecord":
        if self.value_kind is not PopulationValueKind.RATE:
            raise ValueError("population_rate records must use rate semantics")
        if not (self.prefecture or self.municipality):
            raise ValueError("population rate records require at least one geography label")
        return self


class PopulationGeographyKey(BaseModel):
    """Deterministic geography key used for QA and downstream joins."""

    model_config = ConfigDict(extra="forbid")

    key_id: str = Field(min_length=1)
    source_record_ids: list[str] = Field(min_length=1)
    source_file_path: Path | None = None
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    raw_prefecture_label: str | None = None
    raw_municipality_label: str | None = None
    normalized_prefecture_label: str | None = None
    normalized_municipality_label: str | None = None
    year: int | str | None = None
    age_group: str | None = None
    value_kind: PopulationValueKind
    provenance: list[ProvenanceRecord] = Field(min_length=1)

    @field_validator("key_id")
    @classmethod
    def _key_id_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("key_id must be non-empty")
        return text

    @field_validator("source_record_ids")
    @classmethod
    def _source_record_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source record IDs must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _normalized_labels_must_be_present(self) -> "PopulationGeographyKey":
        if not self.provenance:
            raise ValueError("population geography keys require provenance")
        if not self.source_file_hash:
            raise ValueError("population geography keys require source_file_hash")
        if not (self.raw_prefecture_label or self.raw_municipality_label):
            raise ValueError("population geography keys require raw geography labels")
        if not (self.normalized_prefecture_label or self.normalized_municipality_label):
            raise ValueError("population geography keys require normalized geography labels")
        return self


class PopulationGeographyIssue(BaseModel):
    """Issue discovered while validating population geography keys."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: PopulationQASeverity
    issue_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_record_ids: list[str] = Field(default_factory=list)
    source_file_hash: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    recommended_action: str = Field(min_length=1)

    @field_validator("issue_id", "issue_type", "message", "recommended_action")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("required text fields must be non-empty")
        return text

    @field_validator("source_record_ids")
    @classmethod
    def _source_record_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source record IDs must be non-empty")
        return cleaned


class PopulationQAManifest(BaseModel):
    """Manifest for one deterministic population geography QA run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    generated_at: datetime
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
