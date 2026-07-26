"""Structured schemas for the candidate-level deliberation pipeline.

Every LLM-facing step in the deliberation pipeline (reviewer data requests,
reviewer findings, author responses) is parsed into one of these models
before it touches a report — free-form LLM text is never rendered directly.
A ``CandidateReviewPacket`` is the unit stored on
``LiveConditionResult.candidate_review_packets`` and read by
``live_report.py``; nothing downstream re-derives semantics from prose.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CandidateDossier(BaseModel):
    candidate_id: str
    prefecture: str
    municipality: str
    action_type: str
    rank: int | None = None
    composite_score: float | None = None
    score_components: dict[str, float | None] = Field(default_factory=dict)
    municipal_facts: dict[str, Any] = Field(default_factory=dict)
    cost_model: dict[str, Any] = Field(default_factory=dict)
    facility_records: list[dict[str, Any]] = Field(default_factory=list)
    evidence_grades: dict[str, str] = Field(default_factory=dict)
    evidence_gaps: list[str] = Field(default_factory=list)
    required_due_diligence: list[str] = Field(default_factory=list)
    local_comparators: list[dict[str, Any]] = Field(default_factory=list)


class DataRequest(BaseModel):
    reviewer_id: str
    candidate_id: str
    question: str
    requested_fields: list[str] = Field(default_factory=list)
    rationale: str = ""


class DataObservation(BaseModel):
    reviewer_id: str
    candidate_id: str
    question: str
    observations: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[str] = Field(default_factory=list)


class ReviewerFinding(BaseModel):
    finding_id: str
    reviewer_id: str
    candidate_id: str
    severity: Literal["blocking", "major", "moderate", "minor"]
    issue: str
    evidence_refs: list[str] = Field(default_factory=list)
    data_observation: str = ""
    recommendation: Literal[
        "accept_with_due_diligence",
        "revise_rationale",
        "replace_candidate",
        "reject",
    ]
    required_response: str = ""


class AuthorResponse(BaseModel):
    finding_id: str
    candidate_id: str
    response_status: Literal["accepted", "partially_accepted", "rejected"]
    response: str
    why_still_proceed: str | None = None
    mitigation: str | None = None
    residual_risk: str | None = None
    added_due_diligence: list[str] = Field(default_factory=list)


class CandidateReviewPacket(BaseModel):
    candidate_id: str
    reviewer_findings: list[ReviewerFinding] = Field(default_factory=list)
    author_responses: list[AuthorResponse] = Field(default_factory=list)
    judge_flags: list[str] = Field(default_factory=list)
    invalidated_findings: list[ReviewerFinding] = Field(default_factory=list)
    final_candidate_position: Literal[
        "retain",
        "retain_with_major_due_diligence",
        "replace",
        "drop",
    ] = "retain"
    final_reason: str = ""
    replacement_decision: dict[str, Any] | None = None


class ReviewThread(BaseModel):
    candidate_id: str
    reviewer_id: str
    data_requests: list[DataRequest] = Field(default_factory=list)
    data_observations: list[DataObservation] = Field(default_factory=list)
    findings: list[ReviewerFinding] = Field(default_factory=list)
    is_completed: bool = False
    error: str | None = None
