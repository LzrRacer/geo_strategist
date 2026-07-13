"""Provenance contracts for real-world data workflows."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SourceKind(str, Enum):
    """Allowed source categories for claims and inputs."""

    MANUAL_FILE = "manual_file"
    API_RAW = "api_raw"
    CACHE = "cache"
    RUN_OUTPUT = "run_output"
    DERIVED = "derived"
    REFERENCE = "reference"
    CONFIG_ASSUMPTION = "config_assumption"


class SourceRef(BaseModel):
    """Reference to a source artifact without embedding its contents."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    kind: SourceKind
    path: Path | None = None
    url: str | None = None
    title: str | None = None
    publisher: str | None = None
    retrieved_at: datetime | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    notes: str | None = None

    @field_validator("source_id")
    @classmethod
    def _source_id_must_not_be_blank(cls, value: str) -> str:
        source_id = value.strip()
        if not source_id:
            raise ValueError("source_id must be non-empty")
        return source_id

    @model_validator(mode="after")
    def _require_reference_target(self) -> "SourceRef":
        if self.path is None and self.url is None and self.notes is None:
            raise ValueError("source reference requires path, url, or notes")
        return self


class ProvenanceRecord(BaseModel):
    """Provenance for an observed, calculated, or assumed claim."""

    model_config = ConfigDict(extra="forbid")

    provenance_id: str = Field(min_length=1)
    source_ref: SourceRef
    claim: str = Field(min_length=1)
    locator: str | None = None
    input_refs: list[str] = Field(default_factory=list)
    calculated_from: list[str] = Field(default_factory=list)
    calculation_method: str | None = None
    notes: str | None = None

    @field_validator("provenance_id")
    @classmethod
    def _provenance_id_must_not_be_blank(cls, value: str) -> str:
        provenance_id = value.strip()
        if not provenance_id:
            raise ValueError("provenance_id must be non-empty")
        return provenance_id

    @field_validator("input_refs", "calculated_from")
    @classmethod
    def _refs_must_be_non_empty(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("reference IDs must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _calculated_values_need_inputs(self) -> "ProvenanceRecord":
        if self.source_ref.kind is SourceKind.DERIVED and not (
            self.input_refs or self.calculated_from
        ):
            raise ValueError("derived provenance requires input_refs or calculated_from")
        return self

    @property
    def source_kind(self) -> SourceKind:
        """Backward-compatible access to the source kind."""

        return self.source_ref.kind

    @property
    def source_path(self) -> Path | None:
        """Backward-compatible access to the source path."""

        return self.source_ref.path


class NumericClaim(BaseModel):
    """One numeric claim extracted from generated text or structured output."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    value_text: str
    surrounding_text: str
    claim_class: str
    provenance_status: str
    source_reference: str | None = None
    calculation_trace: str | None = None
    issue: str | None = None


class NumericClaimVerificationResult(BaseModel):
    """Summary of numeric-claim provenance verification."""

    model_config = ConfigDict(extra="forbid")

    checked_claim_count: int
    supported_claim_count: int
    unsupported_claim_count: int
    claims: list[NumericClaim]
    issues: list[dict[str, Any]] = Field(default_factory=list)


def find_claims_without_provenance(
    claims: Iterable[str],
    provenance_records: Sequence[ProvenanceRecord],
) -> list[str]:
    """Return claims that do not have a matching provenance record."""

    supported_claims = {record.claim for record in provenance_records}
    return [claim for claim in claims if claim not in supported_claims]


_NUMERIC_RE = re.compile(
    r"(?<![\w])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?:\s?(?:%|km|m2|㎡|JPY|円|人|床|beds?|years?|年|件|count|per\s?100k))?",
    re.IGNORECASE,
)
_SOURCE_RE = re.compile(r"(source|source_ref|source_refs|provenance|evidence_ref|根拠|出典|source_artifact)", re.IGNORECASE)
_ASSUMPTION_RE = re.compile(r"(assumption|scenario|proxy|estimate|仮定|推定|前提)", re.IGNORECASE)
_DERIVED_RE = re.compile(r"(formula|calculation|calculated_from|derived|計算|算出)", re.IGNORECASE)
_UNSUPPORTED_RE = re.compile(r"(unsupported|not_available|missing|unverified|未確認|欠落)", re.IGNORECASE)


def _to_text(payload: object) -> str:
    if isinstance(payload, Path):
        return payload.read_text(encoding="utf-8")
    if isinstance(payload, str):
        try:
            maybe_path = Path(payload)
            if len(payload) < 4096 and maybe_path.exists() and maybe_path.is_file():
                return maybe_path.read_text(encoding="utf-8")
        except OSError:
            pass
        return payload
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _classify_numeric_context(context: str) -> tuple[str, str, str | None]:
    if _SOURCE_RE.search(context):
        return "source-derived value", "supported", None
    if _DERIVED_RE.search(context):
        return "model-derived estimate", "supported", None
    if _ASSUMPTION_RE.search(context):
        if "proxy" in context.lower():
            return "proxy estimate", "supported", None
        return "scenario assumption", "supported", None
    if _UNSUPPORTED_RE.search(context):
        return "unsupported claim", "explicitly_unsupported", "numeric claim is explicitly unsupported or missing"
    return "unsupported claim", "unsupported", "numeric claim lacks source reference, assumption label, calculation trace, or unsupported flag"


def extract_numeric_claim_provenance(payload: object) -> list[NumericClaim]:
    """Extract numeric claims and classify visible provenance context."""

    text = _to_text(payload)
    claims: list[NumericClaim] = []
    for index, match in enumerate(_NUMERIC_RE.finditer(text), start=1):
        start = max(0, match.start() - 160)
        end = min(len(text), match.end() + 160)
        context = text[start:end]
        claim_class, status, issue = _classify_numeric_context(context)
        claims.append(NumericClaim(
            claim_id=f"numeric_claim:{index}",
            value_text=match.group(0),
            surrounding_text=context.strip(),
            claim_class=claim_class,
            provenance_status=status,
            source_reference="contextual_source_reference_present" if _SOURCE_RE.search(context) else None,
            calculation_trace="contextual_calculation_trace_present" if _DERIVED_RE.search(context) else None,
            issue=issue,
        ))
    return claims


def verify_no_unproven_numeric_claims(payload: object) -> NumericClaimVerificationResult:
    """Verify that every numeric claim is sourced, derived, assumed, or flagged."""

    claims = extract_numeric_claim_provenance(payload)
    issues: list[dict[str, Any]] = []
    supported = 0
    for claim in claims:
        if claim.provenance_status in {"supported", "explicitly_unsupported"}:
            supported += 1
            continue
        issues.append({
            "issue_code": "numeric_claim_missing_provenance",
            "severity": "error",
            "claim_id": claim.claim_id,
            "value_text": claim.value_text,
            "message": claim.issue,
        })
    return NumericClaimVerificationResult(
        checked_claim_count=len(claims),
        supported_claim_count=supported,
        unsupported_claim_count=len(claims) - supported,
        claims=claims,
        issues=issues,
    )
