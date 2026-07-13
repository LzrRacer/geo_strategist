"""Analysis-ready source view contracts.

These models describe conservative intermediate views. They are not demand
scores, land scores, cash-flow projections, candidate sites, or proposals.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.normalization import ValueType
from geo_strategist.data.provenance import ProvenanceRecord
from geo_strategist.data.population_views import PopulationLongRecord


class IssueSeverity(str, Enum):
    """Source-quality issue severity."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SourceQualityIssue(BaseModel):
    """Source-quality issue found while building analysis-ready views."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: IssueSeverity
    source_file: Path | None = None
    sheet: str | None = None
    issue_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    recommended_action: str = Field(min_length=1)


class AnalysisViewManifest(BaseModel):
    """Manifest for one deterministic analysis-view build."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    view_name: str = Field(min_length=1)
    generated_at: datetime
    input_files: list[Path] = Field(default_factory=list)
    output_files: list[Path] = Field(default_factory=list)
    record_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    unresolved_mapping_count: int = Field(ge=0)

    @field_validator("record_counts")
    @classmethod
    def _counts_must_be_nonnegative(cls, counts: dict[str, int]) -> dict[str, int]:
        if any(value < 0 for value in counts.values()):
            raise ValueError("record counts must be nonnegative")
        return counts


class _ProvenancedViewRow(BaseModel):
    """Common validation for derived view rows."""

    model_config = ConfigDict(extra="forbid")

    source_record_ids: list[str] = Field(min_length=1)
    source_file_hash: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    provenance: list[ProvenanceRecord] = Field(min_length=1)

    @field_validator("source_record_ids")
    @classmethod
    def _source_record_ids_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("source record IDs must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _require_provenance_and_hash(self) -> "_ProvenancedViewRow":
        if not self.provenance:
            raise ValueError("analysis-view rows require provenance")
        if not self.source_file_hash:
            raise ValueError("analysis-view rows require source_file_hash")
        return self


class HospitalWorkbookFact(_ProvenancedViewRow):
    """Conservative fact copied from normalized hospital workbook records."""

    fact_id: str = Field(min_length=1)
    source_file_path: Path
    source_sheet: str = Field(min_length=1)
    field_name: str = Field(min_length=1)
    value: str | float | int | bool
    value_type: ValueType
    unit: str | None = None
    source_row_number: int = Field(ge=1)
    source_column_number: int = Field(ge=1)
    original_column: str = Field(min_length=1)
    original_header: str | None = None
