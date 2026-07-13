"""Deterministic reviewer ensemble for hospital-strategy proposals.

Adapted from AI Scientist-v2 ``perform_llm_review.py``: an ensemble of
reviewer personas with different stances (base / negative / positive) each
produce a structured review (strengths, weaknesses, questions, score,
decision), which is then meta-aggregated. Here the personas are deterministic
rule-based reviewers over proposal records instead of live LLM calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from geo_strategist.agent.schemas import UNVERIFIED_FIELD_GRADES


# Persona ensemble required by the C6 review loop; stances mirror the
# reference's reviewer_system_prompt_{base,neg,pos} variants.
REVIEWER_ROLES: tuple[tuple[str, str], ...] = (
    ("healthcare_strategy_reviewer", "base"),
    ("real_estate_reviewer", "negative"),
    ("finance_reviewer", "negative"),
    ("operations_reviewer", "base"),
    ("regulatory_risk_reviewer", "negative"),
    ("data_provenance_reviewer", "negative"),
    ("skeptical_investment_committee_reviewer", "negative"),
    ("executive_summary_reviewer", "positive"),
)

_CONCRETE_FIELDS = ("target_facility_name", "target_facility_address", "target_coordinates")


@dataclass(frozen=True)
class ReviewerScore:
    reviewer_id: str
    proposal_id: str
    score: float
    stance: str = "base"
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    blocking_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    decision: str = "revise"
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _source_ref_fields(proposal: dict[str, Any]) -> set[str]:
    return {str(ref.get("field_name")) for ref in proposal.get("source_evidence_refs") or []}


def _evidence_issues(proposal: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Evidence-integrity findings shared by every persona.

    Blocking issues are limited to fabrication risks: concrete facts asserted
    without a source reference or an explicit unverified grade.
    """

    blocking: list[str] = []
    warnings: list[str] = []
    refs = _source_ref_fields(proposal)
    grades = proposal.get("evidence_grades") or {}
    for field_name in _CONCRETE_FIELDS:
        value = proposal.get(field_name)
        if value in (None, "", "not_available", {}, []):
            continue
        if field_name in refs:
            continue
        if grades.get(field_name) in UNVERIFIED_FIELD_GRADES:
            warnings.append(f"unverified_grade_for_{field_name}")
            continue
        blocking.append(f"missing_source_or_grade_for_{field_name}")
    if not proposal.get("requires_human_due_diligence"):
        blocking.append("missing_human_due_diligence_flag")
    if not proposal.get("evidence_gaps") and not proposal.get("required_due_diligence"):
        warnings.append("missing_evidence_gap_disclosure")
    return sorted(set(blocking)), sorted(set(warnings))


def _persona_findings(reviewer_id: str, proposal: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Persona-specific strengths / weaknesses / questions from record fields."""

    strengths: list[str] = []
    weaknesses: list[str] = []
    questions: list[str] = []
    scores = proposal.get("score_components") or {}
    grades = proposal.get("evidence_grades") or {}

    if reviewer_id == "healthcare_strategy_reviewer":
        if scores.get("demand") is not None:
            strengths.append("demand reasoning is quantified from population data")
        else:
            weaknesses.append("no quantified demand basis")
        questions.append("How does the catchment interact with neighbouring municipalities?")
    elif reviewer_id == "real_estate_reviewer":
        if proposal.get("action_type") == "build" and proposal.get("target_facility_address") in (None, "", "not_available"):
            questions.append("Which parcels inside the municipality are actually buildable?")
        if grades.get("land") in (None, "not_available"):
            weaknesses.append("land-price basis missing or unverified")
        elif scores.get("land") is not None:
            strengths.append("land cost is reflected in the score")
    elif reviewer_id == "finance_reviewer":
        if scores.get("financial") is None:
            weaknesses.append("no financial plausibility estimate")
        else:
            strengths.append("financial plausibility is scored")
        questions.append("What capex/opex scenario underlies the financial score?")
    elif reviewer_id == "operations_reviewer":
        if proposal.get("action_type") in ("reorganize", "consolidate") and proposal.get("target_facility_name") in (None, "", "not_available"):
            weaknesses.append("reorganization target facility unresolved")
        questions.append("What staffing transition is implied by this action?")
    elif reviewer_id == "regulatory_risk_reviewer":
        if scores.get("demographic_risk") is None:
            weaknesses.append("no risk adjustment applied")
        questions.append("Does the regional healthcare plan permit this bed change?")
    elif reviewer_id == "data_provenance_reviewer":
        if proposal.get("source_evidence_refs"):
            strengths.append("concrete fields carry source references")
        if scores.get("evidence_completeness") is not None and float(scores["evidence_completeness"]) < 0.5:
            weaknesses.append("evidence completeness below 0.5")
    elif reviewer_id == "skeptical_investment_committee_reviewer":
        if scores.get("evidence_completeness") is None:
            weaknesses.append("evidence completeness not reported")
        questions.append("What would falsify this candidate ranking?")
    elif reviewer_id == "executive_summary_reviewer":
        if proposal.get("municipality") not in (None, "", "not_available"):
            strengths.append("candidate is locatable at municipality level")

    return strengths, weaknesses, questions


def review_proposal(proposal: dict[str, Any]) -> list[ReviewerScore]:
    pid = str(proposal.get("proposal_id"))
    base_blocking, base_warnings = _evidence_issues(proposal)
    reviews: list[ReviewerScore] = []
    for reviewer_id, stance in REVIEWER_ROLES:
        strengths, weaknesses, questions = _persona_findings(reviewer_id, proposal)
        blocking = list(base_blocking)
        warnings = list(base_warnings)
        stance_bias = {"negative": -0.25, "positive": 0.25}.get(stance, 0.0)
        penalty = len(set(blocking)) * 1.5 + (len(set(warnings)) + len(weaknesses)) * 0.25
        score = max(0.0, min(5.0, round(4.0 + stance_bias + len(strengths) * 0.25 - penalty, 2)))
        if blocking:
            decision = "reject_pending_evidence"
        elif weaknesses:
            decision = "revise"
        else:
            decision = "accept_draft"
        reviews.append(ReviewerScore(
            reviewer_id=reviewer_id,
            proposal_id=pid,
            score=score,
            stance=stance,
            strengths=strengths,
            weaknesses=weaknesses,
            questions=questions,
            blocking_issues=sorted(set(blocking)),
            warnings=sorted(set(warnings)),
            decision=decision,
            rationale="Deterministic persona review of provenance, scores, and disclosure fields.",
        ))
    return reviews


def aggregate_review_score(scores: list[ReviewerScore]) -> float:
    """Meta-aggregation across the ensemble (mean, fabrication-gated)."""

    if not scores:
        return 0.0
    if any("missing_source_or_grade" in issue for review in scores for issue in review.blocking_issues):
        return 0.0
    return round(mean(review.score for review in scores), 4)


def has_blocking_review(scores: list[ReviewerScore]) -> bool:
    return any(score.blocking_issues for score in scores)


def revision_requests(scores: list[ReviewerScore]) -> list[str]:
    """Distinct revision items the revision loop must address."""

    items: list[str] = []
    for review in scores:
        items.extend(review.blocking_issues)
        items.extend(review.weaknesses)
    return sorted(set(items))
