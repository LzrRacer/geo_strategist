"""Shared condition-output contract for C0-C13 workflow-condition records.

This module defines the common proposal-producing output surface that every
workflow condition (C0-C13) must emit, plus reusable validation helpers so that
each condition module does not need to reimplement the same evidence checks.

The contract enforces evidence handling, not output blocking:

- Every condition may produce its own proposal report.
- Concrete facility fields (name/address/coordinates) require source-evidence
  provenance whenever they are asserted as verified.
- Fields without a real source must be explicitly marked (``not_available``,
  ``model_estimate``, ``scenario_assumption``, or ``unverified_candidate``),
  never silently asserted.
- Every proposal carries a ``required_due_diligence`` list; cautionary notes
  are consolidated there (and in the report footer), not spread through the
  record.

Nothing in this module calls a live API, reads secrets, or fabricates data.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from geo_strategist.reporting.footer import standard_due_diligence_items


JUDGMENT_SCOPE = "proposal_and_workflow_quality"

REQUIRED_CONDITION_OUTPUT_FIELDS: tuple[str, ...] = (
    "condition_id",
    "condition_group",
    "workflow_type",
    "provider_family",
    "agentic_loop",
    "tree_search",
    "llm_required",
    "live_api_calls_made",
    "raw_llm_outputs_read",
    "judgment_scope",
    "concrete_proposals_count",
    "proposals_with_facility_name",
    "proposals_with_facility_address",
    "proposals_with_coordinates",
    "source_artifacts",
    "eligible_for_judge",
    "exclusion_reason",
    "proposal_report_path",
    "required_due_diligence",
)

# Fields stripped before a condition record may be shown to a comparison
# judge, so that provider/brand identity does not bias scoring.
PROVIDER_IDENTIFYING_FIELDS: tuple[str, ...] = (
    "condition_id",
    "condition_group",
    "provider_family",
    "provider",
    "model",
    "model_roles",
    "harness",
    "label",
    "model_call_summary",
    "source_run_dir",
    "source_artifacts",
    "proposal_report_path",
)

_CONCRETE_PROPOSAL_FIELDS: tuple[str, ...] = (
    "target_facility_name",
    "target_facility_address",
    "target_coordinates",
)

_UNSET_VALUES: tuple[Any, ...] = (None, "", "not_available")

# Grades a concrete field may carry without a matching source-evidence ref.
_UNVERIFIED_GRADES: frozenset[str] = frozenset(
    {"scenario_assumption", "model_estimate", "unverified_candidate", "not_available"}
)


def _is_set(value: Any) -> bool:
    return value not in _UNSET_VALUES and value not in ({}, [])


def concrete_fields_have_provenance(proposal: dict[str, Any]) -> bool:
    """True if every asserted concrete field is either source-backed or
    explicitly marked with an unverified evidence grade."""

    refs = {ref.get("field_name") for ref in proposal.get("source_evidence_refs") or []}
    grades = proposal.get("evidence_grades") or {}
    for field_name in _CONCRETE_PROPOSAL_FIELDS:
        value = proposal.get(field_name)
        if not _is_set(value):
            continue
        if field_name in refs:
            continue
        if grades.get(field_name) in _UNVERIFIED_GRADES:
            continue
        return False
    return True


def build_proposal_site_claims_graded(proposal: dict[str, Any]) -> bool:
    """Build-action proposals may only claim an exact address or coordinates
    when the claim is source-backed or explicitly graded as unverified."""

    if proposal.get("action_type") != "build":
        return True
    refs = {ref.get("field_name") for ref in proposal.get("source_evidence_refs") or []}
    grades = proposal.get("evidence_grades") or {}
    for field_name in ("target_facility_address", "target_coordinates"):
        value = proposal.get(field_name)
        if not _is_set(value):
            continue
        if field_name in refs:
            continue
        if grades.get(field_name) in _UNVERIFIED_GRADES:
            continue
        return False
    return True


def proposal_evidence_passed(proposals: list[dict[str, Any]]) -> tuple[bool, str | None]:
    """Check the shared proposal-level evidence gate for a list of proposal rows.

    The gate never blocks a proposal for existing; it only rejects records
    that assert concrete facts without a source or an explicit unverified mark.
    """

    for row in proposals:
        if row.get("requires_human_due_diligence") is not True:
            return False, "proposal_missing_due_diligence_flag"
        if not concrete_fields_have_provenance(row):
            return False, "concrete_field_missing_provenance_or_grade"
        if not build_proposal_site_claims_graded(row):
            return False, "build_site_claim_missing_provenance_or_grade"
    return True, None


def condition_record_evidence_passed(
    record: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    """Check the shared condition-record-level evidence gate.

    This does not check condition_id/condition_group membership; callers that
    need to restrict eligibility to a specific condition family should check
    that separately before calling this helper.
    """

    if not record.get("judgment_scope"):
        return False, "judgment_scope_missing"
    passed, reason = proposal_evidence_passed(proposals)
    if not passed:
        return False, reason
    return True, None


def _count_concrete(proposals: list[dict[str, Any]], field_name: str) -> int:
    return sum(1 for row in proposals if _is_set(row.get(field_name)))


def make_condition_record(
    *,
    condition_id: str,
    condition_group: str,
    workflow_type: str,
    provider_family: str,
    agentic_loop: bool,
    tree_search: bool,
    llm_required: bool,
    live_api_calls_made: bool,
    raw_llm_outputs_read: bool,
    proposals: list[dict[str, Any]] | None = None,
    source_artifacts: list[str] | None = None,
    eligible_for_judge: bool = True,
    exclusion_reason: str | None = None,
    proposal_report_path: str | None = None,
    required_due_diligence: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a condition record that satisfies the common C0-C13 output contract."""

    proposals = proposals or []
    record: dict[str, Any] = {
        "condition_id": condition_id,
        "condition_group": condition_group,
        "workflow_type": workflow_type,
        "provider_family": provider_family,
        "agentic_loop": bool(agentic_loop),
        "tree_search": bool(tree_search),
        "llm_required": bool(llm_required),
        "live_api_calls_made": bool(live_api_calls_made),
        "raw_llm_outputs_read": bool(raw_llm_outputs_read),
        "judgment_scope": JUDGMENT_SCOPE,
        "concrete_proposals_count": len(proposals),
        "proposals_with_facility_name": _count_concrete(proposals, "target_facility_name"),
        "proposals_with_facility_address": _count_concrete(proposals, "target_facility_address"),
        "proposals_with_coordinates": _count_concrete(proposals, "target_coordinates"),
        "source_artifacts": list(source_artifacts or []),
        "eligible_for_judge": bool(eligible_for_judge),
        "exclusion_reason": exclusion_reason,
        "proposal_report_path": proposal_report_path,
        "required_due_diligence": list(required_due_diligence or standard_due_diligence_items()),
    }
    if extra:
        record.update(extra)
    return record


def merge_condition_record(
    existing: dict[str, Any] | None,
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Precedence rule shared by every condition-record merge path.

    An incoming record replaces the existing one unless it is a
    ``waiting_for_manual_harness`` placeholder and the existing record is a
    completed comparable run (``live`` / ``live_manual_harness`` /
    ``deterministic_baseline``) — an ingested result must never silently
    regress to a waiting placeholder just because the manual-result file
    moved.
    """

    if existing is None:
        return incoming
    completed_modes = ("live", "live_manual_harness", "deterministic_baseline")
    if (incoming.get("execution_mode") == "waiting_for_manual_harness"
            and existing.get("execution_mode") in completed_modes
            and existing.get("comparable_for_e13")):
        return existing
    return incoming


def anonymize_condition_record_for_judge(record: dict[str, Any], alias: str) -> dict[str, Any]:
    """Return a copy of a condition record with provider/condition identity stripped.

    The alias (e.g. "Method A") replaces the identifying fields so a live judge
    prompt cannot infer brand identity from condition_id, condition_group, or
    provider_family. Structural/workflow signal fields used for scoring are kept.
    """

    anonymized = dict(record)
    for field_name in PROVIDER_IDENTIFYING_FIELDS:
        anonymized.pop(field_name, None)
    # Identity also hides in nested structures: per-proposal condition ids and
    # the method_summary narrative ("Manual codex harness run ingested from
    # .../runs/C02/..."). Strip those too, or the alias is decorative.
    anonymized["proposals"] = [
        {key: value for key, value in proposal.items()
         if key not in ("condition_id", "condition_group", "proposal_id",
                        "source_artifact_refs",
                        # free text authored by the (identified) harness; the
                        # judge reads the structured qualitative_discussion
                        "manual_harness_discussion")}
        for proposal in (anonymized.get("proposals") or [])
    ]
    anonymized.pop("narrative_sections", None)
    anonymized.pop("artifacts", None)
    # The C11-C13 candidate-level deliberation artifacts carry a model_name
    # field for audit bookkeeping; strip it here too so the anonymized E13
    # judge input cannot recover provider/model identity through them.
    for list_field in ("candidate_qualitative_assessments", "candidate_assessment_reviews"):
        rows = anonymized.get(list_field)
        if isinstance(rows, list):
            anonymized[list_field] = [
                {key: value for key, value in row.items() if key not in ("model_name", "condition_id")}
                for row in rows if isinstance(row, dict)
            ]
    anonymized["judge_alias"] = alias
    anonymized["method_alias"] = alias
    anonymized["judgment_scope"] = JUDGMENT_SCOPE
    return anonymized


def stable_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, canonical)}"
