"""Proposal facade over the evidence-graded stage outputs."""

from __future__ import annotations

from typing import Any

from geo_strategist.agent.schemas import ExperimentalProposal, SourceEvidenceRef


def build_experimental_proposal(
    *,
    proposal_id: str,
    action_type: str,
    prefecture: str,
    municipality: str,
    condition_id: str | None = None,
    condition_group: str | None = None,
    target_facility_name: str | None = None,
    target_facility_address: str | None = None,
    target_coordinates: dict[str, Any] | str | None = None,
    exact_address_status: str = "not_verified",
    source_evidence_refs: list[SourceEvidenceRef] | None = None,
    evidence_grades: dict[str, str] | None = None,
    unsupported_fields: list[str] | None = None,
    evidence_gaps: list[str] | None = None,
    required_due_diligence: list[str] | None = None,
    novelty_check_status: str = "not_available",
    reviewer_scores: dict[str, float] | None = None,
    aggregate_score: float | None = None,
    keep_discard_decision: str = "pending",
    review_summary: str = "not_available",
    revision_summary: str = "not_available",
    exact_site_readiness_status: str = "not_available",
    proposal_status: str = "draft_proposal",
) -> ExperimentalProposal:
    """Build a proposal with provenance/grade validation."""

    return ExperimentalProposal(
        proposal_id=proposal_id,
        proposal_status=proposal_status,  # type: ignore[arg-type]
        condition_id=condition_id,
        condition_group=condition_group,
        action_type=action_type,  # type: ignore[arg-type]
        prefecture=prefecture,
        municipality=municipality,
        target_facility_name=target_facility_name,
        target_facility_address=target_facility_address,
        target_coordinates=target_coordinates,
        exact_address_status=exact_address_status,  # type: ignore[arg-type]
        source_evidence_refs=source_evidence_refs or [],
        evidence_grades=evidence_grades or {},
        unsupported_fields=unsupported_fields or [],
        evidence_gaps=evidence_gaps or [],
        required_due_diligence=required_due_diligence or [],
        novelty_check_status=novelty_check_status,
        reviewer_scores=reviewer_scores or {},
        aggregate_score=aggregate_score,
        keep_discard_decision=keep_discard_decision,  # type: ignore[arg-type]
        review_summary=review_summary,
        revision_summary=revision_summary,
        exact_site_readiness_status=exact_site_readiness_status,
    )
