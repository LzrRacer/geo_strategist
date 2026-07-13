"""Schemas for source-traceable proposal records.

Concrete facts (facility name, address, coordinates) must either carry a
source-evidence reference or be explicitly graded as unverified/estimated.
Proposals themselves are never blocked from existing; cautionary notes are
consolidated in ``required_due_diligence`` and the report footer.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ActionType = Literal["build", "reorganize", "consolidate"]
ProposalStatus = Literal["draft_proposal", "revised_proposal"]

# Grades that let a concrete field be stated without a source reference,
# as long as the grade itself is declared (see data_sources.evidence_grade).
UNVERIFIED_FIELD_GRADES = frozenset(
    {"scenario_assumption", "model_estimate", "unverified_candidate", "not_available"}
)


class SourceEvidenceRef(BaseModel):
    """Reference proving that a concrete string came from a source artifact."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    source_artifact: str
    source_record_id: str
    source_field: str
    evidence_status: str = "source_traceable"


class ExperimentalProposal(BaseModel):
    """Proposal output with per-field evidence grading."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    proposal_status: ProposalStatus = "draft_proposal"
    requires_human_due_diligence: bool = True
    condition_id: str | None = None
    condition_group: str | None = None
    action_type: ActionType
    prefecture: str
    municipality: str
    target_facility_name: str | None = None
    target_facility_address: str | None = "not_available"
    target_coordinates: dict[str, Any] | str | None = "not_available"
    exact_address_status: Literal["source_traceable", "not_verified", "not_available"] = "not_verified"
    source_evidence_refs: list[SourceEvidenceRef] = Field(default_factory=list)
    evidence_grades: dict[str, str] = Field(default_factory=dict)
    unsupported_fields: list[str] = Field(default_factory=list)
    evidence_gaps: list[str] = Field(default_factory=list)
    required_due_diligence: list[str] = Field(default_factory=list)
    novelty_check_status: str = "not_available"
    reviewer_scores: dict[str, float] = Field(default_factory=dict)
    aggregate_score: float | None = None
    keep_discard_decision: Literal["keep", "discard", "pending"] = "pending"
    review_summary: str = "not_available"
    revision_summary: str = "not_available"
    exact_site_readiness_status: str = "not_available"

    @model_validator(mode="after")
    def validate_source_traceability(self) -> "ExperimentalProposal":
        refs = {ref.field_name for ref in self.source_evidence_refs}
        concrete_fields = {
            "target_facility_name": self.target_facility_name,
            "target_facility_address": self.target_facility_address,
            "target_coordinates": self.target_coordinates,
        }
        for field_name, value in concrete_fields.items():
            if value in (None, "", {}, [], "not_available"):
                continue
            if field_name in refs:
                continue
            if self.evidence_grades.get(field_name) in UNVERIFIED_FIELD_GRADES:
                continue
            raise ValueError(
                f"{field_name} requires source_evidence_refs provenance or an explicit "
                f"unverified evidence grade"
            )
        return self


def source_refs_for_field(refs: list[SourceEvidenceRef], field_name: str) -> list[SourceEvidenceRef]:
    return [ref for ref in refs if ref.field_name == field_name]
