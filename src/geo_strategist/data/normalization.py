"""Contracts and helpers for deterministic source normalization."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.provenance import ProvenanceRecord, SourceRef


class MappingStatus(str, Enum):
    """Status of a mapping decision."""

    MANUAL_VERIFIED = "manual_verified"
    INFERRED = "inferred"
    UNRESOLVED = "unresolved"


class ValueType(str, Enum):
    """Normalized source-cell value type."""

    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    FORMULA = "formula"


class SourceTable(BaseModel):
    """Workbook sheet or detected table region used for normalization."""

    model_config = ConfigDict(extra="forbid")

    table_id: str = Field(min_length=1)
    workbook_path: Path
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    sheet_name: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    detected_header_row: int | None = Field(default=None, ge=1)
    column_names: list[str] = Field(default_factory=list)
    table_role: str = Field(min_length=1)
    mapping_status: MappingStatus
    warnings: list[str] = Field(default_factory=list)

    @field_validator("table_id", "sheet_name", "table_role")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("source table text fields must be non-empty")
        return text


class ExtractionMapping(BaseModel):
    """Mapping decision for one source table."""

    model_config = ConfigDict(extra="forbid")

    mapping_id: str = Field(min_length=1)
    source_table_id: str = Field(min_length=1)
    sheet_name: str = Field(min_length=1)
    status: MappingStatus
    confidence: float = Field(ge=0, le=1)
    header_row: int | None = Field(default=None, ge=1)
    value_columns: list[int] = Field(default_factory=list)
    label_columns: list[int] = Field(default_factory=list)
    year_columns: list[int] = Field(default_factory=list)
    age_columns: list[int] = Field(default_factory=list)
    notes: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("value_columns", "label_columns", "year_columns", "age_columns")
    @classmethod
    def _columns_must_be_positive(cls, values: list[int]) -> list[int]:
        if any(value < 1 for value in values):
            raise ValueError("column indexes must be positive")
        return values

    @model_validator(mode="after")
    def _unresolved_mapping_must_not_select_columns(self) -> "ExtractionMapping":
        selected = self.value_columns or self.label_columns or self.year_columns or self.age_columns
        if self.status is MappingStatus.UNRESOLVED and selected:
            raise ValueError("unresolved mappings cannot select extraction columns")
        return self


class NormalizedRecord(BaseModel):
    """One normalized value with source-cell provenance."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    source_table_id: str = Field(min_length=1)
    source_file_path: Path
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    source_sheet: str = Field(min_length=1)
    source_row_number: int = Field(ge=1)
    source_column_number: int = Field(ge=1)
    normalized_field_name: str = Field(min_length=1)
    normalized_value: str | float | int | bool
    original_column: str = Field(min_length=1)
    original_header: str | None = None
    value_type: ValueType
    unit: str | None = None
    value_role: str | None = None
    mapping_override_id: str | None = None
    geography_labels: dict[str, str] = Field(default_factory=dict)
    time_labels: dict[str, str] = Field(default_factory=dict)
    age_labels: dict[str, str] = Field(default_factory=dict)
    provenance: list[ProvenanceRecord] = Field(min_length=1)

    @field_validator("record_id", "source_table_id", "source_sheet", "normalized_field_name")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("normalized record text fields must be non-empty")
        return text

    @field_validator("geography_labels", "time_labels", "age_labels")
    @classmethod
    def _labels_must_not_be_blank(cls, labels: dict[str, str]) -> dict[str, str]:
        cleaned = {key.strip(): value.strip() for key, value in labels.items()}
        if any(not key or not value for key, value in cleaned.items()):
            raise ValueError("label keys and values must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_value_and_provenance(self) -> "NormalizedRecord":
        if not self.source_file_hash:
            raise ValueError("normalized records extracted from files require source_file_hash")
        if not self.provenance:
            raise ValueError("normalized records require provenance")
        if self.value_type is ValueType.NUMBER and (
            isinstance(self.normalized_value, bool)
            or not isinstance(self.normalized_value, (int, float))
        ):
            raise ValueError("number records require a numeric source-cell value")
        return self


class NormalizationManifest(BaseModel):
    """Manifest for one deterministic normalization run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    source_type: str = Field(min_length=1)
    started_at: datetime
    ended_at: datetime | None = None
    input_files: list[SourceRef] = Field(default_factory=list)
    output_files: list[Path] = Field(default_factory=list)
    source_table_count: int = Field(ge=0)
    normalized_record_count: int = Field(ge=0)
    unresolved_mapping_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)
    code_version: str | None = None


class NormalizationResult(BaseModel):
    """Runtime result returned by normalizer entry points."""

    model_config = ConfigDict(extra="forbid")

    found: bool
    source_type: str
    source_tables: list[SourceTable] = Field(default_factory=list)
    normalized_records: list[NormalizedRecord] = Field(default_factory=list)
    mappings: list[ExtractionMapping] = Field(default_factory=list)
    manifest: NormalizationManifest
    output_paths: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @property
    def unresolved_mapping_count(self) -> int:
        """Return unresolved mapping count."""

        return sum(1 for mapping in self.mappings if mapping.status is MappingStatus.UNRESOLVED)


def now_utc() -> datetime:
    """Return current UTC time with timezone metadata."""

    return datetime.now(UTC)


def safe_text(value: Any, limit: int = 120) -> str:
    """Return a bounded text representation for headers and labels."""

    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    return text[:limit]


def normalize_field_name(header: str, column_index: int) -> str:
    """Convert a source header into a stable field identifier."""

    text = safe_text(header).lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9a-zA-Z_\u3040-\u30ff\u3400-\u9fff]+", "_", text)
    text = text.strip("_")
    return text or f"column_{column_index}"


def infer_value_type(value: Any) -> ValueType | None:
    """Infer value type from the source cell value without semantic conversion."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        return ValueType.BOOLEAN
    if isinstance(value, (int, float)):
        return ValueType.NUMBER
    if isinstance(value, (date, datetime)):
        return ValueType.DATE
    if isinstance(value, str) and value.startswith("="):
        return ValueType.FORMULA
    return ValueType.STRING


def normalize_cell_value(value: Any, value_type: ValueType) -> str | float | int | bool:
    """Normalize one source-cell value without inventing or calculating values."""

    if value_type is ValueType.DATE:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, date):
            return value.isoformat()
    if value_type is ValueType.STRING or value_type is ValueType.FORMULA:
        return safe_text(value, limit=500)
    return value


def source_table_id(source_type: str, workbook_path: Path, sheet_name: str) -> str:
    """Build a deterministic source table ID."""

    path_part = "_".join(workbook_path.parts).replace(" ", "_")
    sheet_part = normalize_field_name(sheet_name, 1)
    return f"{source_type}:{path_part}:{sheet_part}"
