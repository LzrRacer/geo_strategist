"""Pydantic schemas for data contracts and future workflow artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from geo_strategist.data.normalization import (
    ExtractionMapping,
    MappingStatus,
    NormalizationManifest,
    NormalizedRecord,
    SourceTable,
    ValueType,
)
from geo_strategist.data.provenance import ProvenanceRecord, SourceKind, SourceRef


class LocalStorePaths(BaseModel):
    """Local-only storage locations used by future workflows."""

    model_config = ConfigDict(extra="forbid")

    manual_inputs: Path = Path(".data/manual")
    api_raw: Path = Path(".data/api_raw")
    cache: Path = Path(".cache")
    runs: Path = Path(".runs")
    references_local: Path = Path("references/local")


class ProposalType(str, Enum):
    """Supported hospital strategy proposal types."""

    NEW_CONSTRUCTION = "new_construction"
    CONSOLIDATION = "consolidation"
    REORGANIZATION = "reorganization"
    CLOSURE_RELOCATION = "closure_relocation"
    MIXED = "mixed"


class ProposalStatus(str, Enum):
    """Lifecycle states for future proposal artifacts."""

    DRAFT = "draft"
    REVIEWED = "reviewed"
    REVISED = "revised"
    SELECTED = "selected"
    REJECTED = "rejected"


class ReviewerType(str, Enum):
    """Supported reviewer or scorer sources."""

    HUMAN_REVIEWER = "human_reviewer"
    LLM_AS_JUDGE = "llm_as_judge"
    DETERMINISTIC_VALIDATOR = "deterministic_validator"


class ActivityPattern(str, Enum):
    """Activity-pattern labels reserved for later analysis."""

    CONTINUOUS = "Continuous"
    EXPLORATORY = "Exploratory"
    INTERMITTENT = "Intermittent"
    EARLY_CONCENTRATED = "Early-Concentrated"
    SPECIAL = "Special"


class RunStatus(str, Enum):
    """Minimal run manifest states."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SCAFFOLD = "scaffold"


class CaseScope(BaseModel):
    """High-level case scope without numeric assumptions."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    candidate_regions: list[str] = Field(min_length=1)
    notes: str | None = None


class NumericObservation(BaseModel):
    """Base contract for one numeric value with required provenance."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    notes: str | None = None

    @field_validator("observation_id", "metric", "unit")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("required text fields must be non-empty")
        return text

    @model_validator(mode="after")
    def _numeric_value_requires_provenance(self) -> "NumericObservation":
        if not self.provenance:
            raise ValueError("numeric observations require provenance")
        return self


class CandidateSite(BaseModel):
    """Candidate site identity and source references."""

    model_config = ConfigDict(extra="forbid")

    site_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    prefecture: str = Field(min_length=1)
    municipality: str = Field(min_length=1)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    source_refs: list[SourceRef] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("site_id", "name", "prefecture", "municipality")
    @classmethod
    def _site_text_must_not_be_blank(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("site fields must be non-empty")
        return text

    @field_validator("tags")
    @classmethod
    def _tags_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("tags must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _coordinates_must_be_pair(self) -> "CandidateSite":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        return self


class SiteObservation(NumericObservation):
    """Numeric observation tied to a candidate site."""

    site_id: str = Field(min_length=1)


class PopulationObservation(NumericObservation):
    """Population observation from real population source files."""

    prefecture: str = Field(min_length=1)
    municipality: str | None = None
    year: int | None = None
    age_group: str | None = None


class LandObservation(NumericObservation):
    """Land-related observation such as price, area, or distance."""

    site_id: str = Field(min_length=1)
    currency: str | None = None


class HealthcareFacilityObservation(NumericObservation):
    """Healthcare facility observation from real source files or API captures."""

    facility_id: str = Field(min_length=1)
    facility_name: str | None = None
    prefecture: str | None = None
    municipality: str | None = None


class CostAssumption(BaseModel):
    """Explicit cost assumption with basis and provenance."""

    model_config = ConfigDict(extra="forbid")

    assumption_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    basis: str = Field(min_length=1)
    source_refs: list[SourceRef] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _assumption_value_requires_basis_and_provenance(self) -> "CostAssumption":
        if not self.basis.strip():
            raise ValueError("cost assumptions require basis metadata")
        if not self.provenance:
            raise ValueError("cost assumptions require provenance")
        return self


class CashflowLineItem(BaseModel):
    """One numeric line item in a future annual cash-flow row."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    provenance: list[ProvenanceRecord] = Field(min_length=1)


class CashflowRow(BaseModel):
    """Annual row container for future cash-flow projections."""

    model_config = ConfigDict(extra="forbid")

    year_index: int = Field(ge=0)
    calendar_year: int | None = None
    line_items: list[CashflowLineItem] = Field(default_factory=list)


class CashflowProjection(BaseModel):
    """Cash-flow projection contract; this model does not calculate rows."""

    model_config = ConfigDict(extra="forbid")

    projection_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)
    input_refs: list[str] = Field(min_length=1)
    rows: list[CashflowRow] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("input_refs")
    @classmethod
    def _input_refs_must_not_be_blank(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("input references must be non-empty")
        return cleaned


class Proposal(BaseModel):
    """Future proposal artifact contract without proposal generation logic."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1)
    proposal_type: ProposalType
    title: str = Field(min_length=1)
    site_ids: list[str] = Field(default_factory=list)
    summary: str | None = None
    source_refs: list[SourceRef] = Field(min_length=1)
    cost_assumption_ids: list[str] = Field(default_factory=list)
    cashflow_projection_ids: list[str] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.DRAFT
    notes: str | None = None


class Review(BaseModel):
    """Review-board output contract for human, judge, or validator reviews."""

    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    reviewer_type: ReviewerType
    reviewer_id: str | None = None
    findings: list[str] = Field(default_factory=list)
    requested_revisions: list[str] = Field(default_factory=list)
    score_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class ActivityEvent(BaseModel):
    """Activity trace event for later activity-pattern analysis."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    condition_id: str = Field(pattern=r"^C[0-7]$")
    timestamp: datetime
    actor: str | None = None
    event_type: str = Field(min_length=1)
    pattern_label: ActivityPattern | None = None
    summary: str | None = None
    input_refs: list[str] = Field(default_factory=list)
    output_refs: list[str] = Field(default_factory=list)


class RunManifest(BaseModel):
    """Minimal run manifest contract for future condition runners."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    condition_id: str = Field(pattern=r"^C[0-7]$")
    started_at: datetime | None = None
    ended_at: datetime | None = None
    config_files: list[Path] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(default_factory=list)
    input_provenance: list[ProvenanceRecord] = Field(default_factory=list)
    activity_events: list[ActivityEvent] = Field(default_factory=list)
    output_directory: Path
    status: RunStatus = RunStatus.SCAFFOLD
    notes: str | None = None


class CaseInputBundle(BaseModel):
    """Structured case inputs assembled from normalized local data."""

    model_config = ConfigDict(extra="forbid")

    case_scope: CaseScope
    source_refs: list[SourceRef]
    records: list[dict[str, Any]]
    provenance: list[ProvenanceRecord]
    validation_errors: list[str] = Field(default_factory=list)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _source_ref(source_id: str, path: Path, kind: SourceKind = SourceKind.RUN_OUTPUT) -> SourceRef:
    return SourceRef(source_id=source_id, kind=kind, path=path)


def _provenance_for_record(source_ref: SourceRef, claim: str, locator: str | None = None) -> ProvenanceRecord:
    return ProvenanceRecord(
        provenance_id=f"case_input_prov:{uuid.uuid5(uuid.NAMESPACE_URL, source_ref.source_id + claim + str(locator))}",
        source_ref=source_ref,
        claim=claim,
        locator=locator,
    )


def build_hospital_case_inputs(scope: CaseScope, repo_root: str | Path = ".") -> CaseInputBundle:
    """Build hospital case inputs from existing normalized/workbook-derived data."""

    root = Path(repo_root).resolve()
    paths = [
        root / ".data/interim/views/hospital_facts.jsonl",
        root / ".data/interim/study_area/tokyo_aichi_osaka/population_base_records.jsonl",
        root / ".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_feature_records.jsonl",
        root / ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base_records.jsonl",
    ]
    source_refs: list[SourceRef] = []
    records: list[dict[str, Any]] = []
    provenance: list[ProvenanceRecord] = []
    errors: list[str] = []
    for path in paths:
        rows = _load_jsonl(path)
        if not rows:
            errors.append(f"missing_or_empty_input:{path.relative_to(root)}")
            continue
        ref = _source_ref(path.stem, path)
        source_refs.append(ref)
        for row in rows:
            if scope.candidate_regions and row.get("prefecture") not in scope.candidate_regions:
                continue
            records.append({"source_id": ref.source_id, **row})
            provenance.append(_provenance_for_record(ref, f"case input row from {path.name}", row.get("feature_id") or row.get("fact_id") or row.get("master_id")))
    return CaseInputBundle(case_scope=scope, source_refs=source_refs, records=records, provenance=provenance, validation_errors=errors)


def build_franchise_case_inputs(scope: CaseScope, repo_root: str | Path = ".") -> CaseInputBundle:
    """Build generic expansion/franchise case inputs from existing site-selection records."""

    root = Path(repo_root).resolve()
    paths = [
        root / ".data/interim/study_area/tokyo_aichi_osaka/population_base_records.jsonl",
        root / ".data/interim/study_area/tokyo_aichi_osaka/municipality_land_feature_records.jsonl",
        root / ".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_feature_records.jsonl",
    ]
    source_refs: list[SourceRef] = []
    records: list[dict[str, Any]] = []
    provenance: list[ProvenanceRecord] = []
    errors: list[str] = []
    for path in paths:
        rows = _load_jsonl(path)
        if not rows:
            errors.append(f"missing_or_empty_input:{path.relative_to(root)}")
            continue
        ref = _source_ref(path.stem, path)
        source_refs.append(ref)
        for row in rows:
            if scope.candidate_regions and row.get("prefecture") not in scope.candidate_regions:
                continue
            records.append({"source_id": ref.source_id, **row})
            provenance.append(_provenance_for_record(ref, f"franchise case row from {path.name}", row.get("feature_id") or row.get("record_id")))
    return CaseInputBundle(case_scope=scope, source_refs=source_refs, records=records, provenance=provenance, validation_errors=errors)
