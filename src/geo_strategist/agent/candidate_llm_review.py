"""Candidate-level critical LLM reviewer.

Each reviewer persona looks at exactly one candidate's dossier, may request
a bounded number of additional data lookups through the
``CandidateDataExplorer`` (which can only answer for that candidate or its
explicitly listed comparators), and then returns structured
``ReviewerFinding`` objects. The prompt is deliberately adversarial: the
reviewer's job is to find reasons the candidate might not deserve its slot,
not to restate the proposal's own rationale.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from pydantic import ValidationError

from geo_strategist.agent.candidate_review_schemas import (
    CandidateDossier,
    CandidateQualitativeAssessment,
    DataObservation,
    DataRequest,
    ReviewerFinding,
    ReviewScope,
    ReviewThread,
)
from geo_strategist.experiments.candidate_data_explorer import CandidateDataExplorer
from geo_strategist.experiments.condition_utils import _stable_id
from geo_strategist.providers.base import ChatResult

LlmCall = Callable[[str, "str | None", str], ChatResult]

DEFAULT_REVIEWERS: list[str] = [
    "healthcare_strategy",
    "emergency_access",
    "real_estate_site_feasibility",
    "hospital_operations",
    "finance_investment",
    "regulatory_policy",
    "data_provenance",
    "skeptical_investment_committee",
]

_REVIEWER_FOCUS: dict[str, str] = {
    "healthcare_strategy": "whether demand/aging reasoning actually supports this action here",
    "emergency_access": "emergency-access and catchment plausibility given only proxy distance data",
    "real_estate_site_feasibility": "land, zoning, and buildability risk for this exact municipality",
    "hospital_operations": "staffing, bed-transition, and operational feasibility of the action",
    "finance_investment": "financial plausibility of the prefecture-median cost model applied here",
    "regulatory_policy": "regional healthcare-plan / regulatory permissibility of the action",
    "data_provenance": "whether every concrete claim about this candidate is source-traceable",
    "skeptical_investment_committee": "what would falsify this candidate's inclusion in the slate",
}

_CRITICAL_SYSTEM_PROMPT = """\
You are a skeptical reviewer evaluating exactly one hospital location/reorganization candidate.

You must actively look for weaknesses, missing evidence, overclaims, fragile assumptions, and reasons this candidate might not deserve to remain in the slate.

You may use only:
1. the candidate dossier,
2. data observations returned by the local explorer,
3. explicitly listed local comparators.

Do not introduce unrelated municipalities.
Do not discuss study-area-wide missingness unless it directly affects this candidate.
Do not invent facts.
Every finding must cite dossier fields or data observations.
Return JSON only.
"""


def _dossier_json(dossier: CandidateDossier) -> str:
    return json.dumps(dossier.model_dump(), ensure_ascii=False)


def _extract_json(text: str) -> Any | None:
    from geo_strategist.experiments.live_common import extract_json_block

    return extract_json_block(text)


def _request_data_requests(
    llm: LlmCall,
    reviewer_id: str,
    dossier: CandidateDossier,
    *,
    max_data_requests: int,
) -> list[DataRequest]:
    focus = _REVIEWER_FOCUS.get(reviewer_id, "a critical review of this candidate")
    prompt = (
        f"Reviewer persona: {reviewer_id} — focus on {focus}.\n\n"
        "Candidate dossier (JSON):\n" + _dossier_json(dossier) + "\n\n"
        f"Before writing findings, list up to {max_data_requests} specific data "
        "requests you need answered about THIS candidate or its listed "
        "local_comparators (never another municipality).\n"
        'Reply as JSON: {"data_requests": [{"question": "...", '
        '"requested_fields": ["..."], "rationale": "..."}]}'
    )
    result = llm(prompt, _CRITICAL_SYSTEM_PROMPT, "candidate_review_data_requests")
    if not result.ok:
        return []
    parsed = _extract_json(result.text) or {}
    raw_requests = parsed.get("data_requests") if isinstance(parsed, dict) else None
    requests: list[DataRequest] = []
    for row in (raw_requests or [])[:max_data_requests]:
        if not isinstance(row, dict):
            continue
        try:
            requests.append(DataRequest(
                reviewer_id=reviewer_id,
                candidate_id=dossier.candidate_id,
                question=str(row.get("question") or ""),
                requested_fields=[str(f) for f in row.get("requested_fields") or []],
                rationale=str(row.get("rationale") or ""),
            ))
        except ValidationError:
            continue
    return requests


_FINDINGS_REPLY_SCHEMA = (
    '{"findings": [{"issue": "...", '
    '"severity": "blocking|major|moderate|minor", "evidence_refs": ["..."], '
    '"data_observation": "...", '
    '"recommendation": "accept_with_due_diligence|revise_rationale|replace_candidate|reject", '
    '"required_response": "..."}]}'
)


def _parse_findings_result(
    result: ChatResult,
    reviewer_id: str,
    dossier: CandidateDossier,
) -> tuple[list[ReviewerFinding], str | None]:
    if not result.ok:
        return [], f"llm_error:{result.error_class}:{result.error_detail}"
    parsed = _extract_json(result.text)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("findings"), list):
        return [], "findings_parse_error: no JSON findings array in reviewer reply"
    findings: list[ReviewerFinding] = []
    parse_errors = 0
    for row in parsed["findings"]:
        if not isinstance(row, dict):
            parse_errors += 1
            continue
        candidate_id = str(row.get("candidate_id") or dossier.candidate_id)
        try:
            findings.append(ReviewerFinding(
                finding_id=_stable_id(
                    "candidate_review_finding",
                    {"reviewer_id": reviewer_id, "candidate_id": candidate_id,
                     "issue": row.get("issue"), "index": len(findings)}),
                reviewer_id=reviewer_id,
                candidate_id=candidate_id,
                severity=row.get("severity"),
                issue=str(row.get("issue") or ""),
                evidence_refs=[str(r) for r in row.get("evidence_refs") or []],
                data_observation=str(row.get("data_observation") or ""),
                recommendation=row.get("recommendation"),
                required_response=str(row.get("required_response") or ""),
            ))
        except ValidationError:
            parse_errors += 1
    if not findings and parse_errors:
        return [], f"findings_parse_error: {parse_errors} finding(s) failed schema validation"
    return findings, None


def _request_findings(
    llm: LlmCall,
    reviewer_id: str,
    dossier: CandidateDossier,
    observations: list[DataObservation],
) -> tuple[list[ReviewerFinding], str | None]:
    prompt = (
        f"Reviewer persona: {reviewer_id}.\n\n"
        "Candidate dossier (JSON):\n" + _dossier_json(dossier) + "\n\n"
        "Data observations you requested (JSON):\n"
        + json.dumps([o.model_dump() for o in observations], ensure_ascii=False) + "\n\n"
        "Now produce your critical findings for THIS candidate only. Every "
        "finding must be groundable in the dossier or an observation above.\n"
        "Reply as JSON: " + _FINDINGS_REPLY_SCHEMA
    )
    result = llm(prompt, _CRITICAL_SYSTEM_PROMPT, "candidate_review_findings")
    return _parse_findings_result(result, reviewer_id, dossier)


# ---------------------------------------------------------------------------
# Assessment-aware review (C11-C13 candidate qualitative deliberation layer)
#
# The reviewer additionally sees the condition's own agent qualitative
# assessment of this candidate and is asked to critique it alongside the
# dossier. The reviewer is explicitly told it has not seen the full
# candidate universe and does not know which model/condition produced the
# candidate or assessment -- ``ReviewScope`` records that bookkeeping so
# downstream artifacts/tests can verify the scope limits held.
# ---------------------------------------------------------------------------

_CRITICAL_SYSTEM_PROMPT_WITH_ASSESSMENT = """\
You are a skeptical reviewer evaluating exactly one hospital location/reorganization candidate \
AND the report-authoring agent's own qualitative assessment of that candidate.

You must actively look for weaknesses, missing evidence, overclaims, fragile assumptions, and \
reasons this candidate (or its stated assessment) might not deserve to remain in the slate.

You may use only:
1. the candidate dossier,
2. the agent's qualitative assessment of this candidate,
3. data observations returned by the local explorer,
4. explicitly listed local comparators.

You have NOT been shown the full candidate universe, only this candidate. You do not know which \
model or condition produced this candidate or its assessment -- do not speculate about it.

Do not introduce unrelated municipalities.
Do not discuss study-area-wide missingness unless it directly affects this candidate.
Do not invent facts.
Every finding must cite dossier fields, the assessment, or a data observation.
Return JSON only.
"""


def _request_findings_with_assessment(
    llm: LlmCall,
    reviewer_id: str,
    dossier: CandidateDossier,
    assessment: CandidateQualitativeAssessment,
    observations: list[DataObservation],
) -> tuple[list[ReviewerFinding], str | None]:
    prompt = (
        f"Reviewer persona: {reviewer_id}.\n\n"
        "Candidate dossier (JSON):\n" + _dossier_json(dossier) + "\n\n"
        "Agent's qualitative assessment of this candidate (JSON):\n"
        + json.dumps(assessment.assessment.model_dump(), ensure_ascii=False)
        + "\nAssessment evidence_refs: "
        + json.dumps(assessment.evidence_refs, ensure_ascii=False) + "\n\n"
        "Data observations you requested (JSON):\n"
        + json.dumps([o.model_dump() for o in observations], ensure_ascii=False) + "\n\n"
        "Critique BOTH the candidate and the agent's assessment above. Every "
        "finding must be groundable in the dossier, the assessment, or an "
        "observation above.\n"
        "Reply as JSON: " + _FINDINGS_REPLY_SCHEMA
    )
    result = llm(prompt, _CRITICAL_SYSTEM_PROMPT_WITH_ASSESSMENT,
                 "candidate_review_findings_with_assessment")
    return _parse_findings_result(result, reviewer_id, dossier)


def run_candidate_reviewer_with_assessment(
    llm: LlmCall,
    reviewer_id: str,
    dossier: CandidateDossier,
    assessment: CandidateQualitativeAssessment,
    explorer: CandidateDataExplorer,
    *,
    max_data_requests: int = 3,
) -> tuple[ReviewThread, ReviewScope]:
    thread = ReviewThread(candidate_id=dossier.candidate_id, reviewer_id=reviewer_id)
    try:
        data_requests = (
            _request_data_requests(
                llm, reviewer_id, dossier, max_data_requests=max_data_requests)
            if max_data_requests > 0 else []
        )
        thread.data_requests = data_requests

        observations = [explorer.answer_data_request(request, dossier)
                        for request in data_requests]
        thread.data_observations = observations

        findings, error = _request_findings_with_assessment(
            llm, reviewer_id, dossier, assessment, observations)
        thread.findings = findings
        if error and not findings:
            thread.error = error
            thread.is_completed = False
        else:
            thread.is_completed = True
    except Exception as exc:  # a crashed reviewer is a recorded failure, not a crash
        thread.error = f"{type(exc).__name__}: {exc}"
        thread.is_completed = False
    review_scope = ReviewScope(
        saw_candidate_dossier=True,
        saw_agent_assessment=True,
        saw_full_candidate_universe=False,
        saw_model_identity=False,
        saw_condition_identity=False,
    )
    return thread, review_scope


def run_candidate_reviewer(
    llm: LlmCall,
    reviewer_id: str,
    dossier: CandidateDossier,
    explorer: CandidateDataExplorer,
    *,
    max_data_requests: int = 3,
) -> ReviewThread:
    thread = ReviewThread(candidate_id=dossier.candidate_id, reviewer_id=reviewer_id)
    try:
        data_requests = (
            _request_data_requests(
                llm, reviewer_id, dossier, max_data_requests=max_data_requests)
            if max_data_requests > 0 else []
        )
        thread.data_requests = data_requests

        observations = [explorer.answer_data_request(request, dossier)
                        for request in data_requests]
        thread.data_observations = observations

        findings, error = _request_findings(llm, reviewer_id, dossier, observations)
        thread.findings = findings
        if error and not findings:
            thread.error = error
            thread.is_completed = False
        else:
            thread.is_completed = True
    except Exception as exc:  # a crashed reviewer is a recorded failure, not a crash
        thread.error = f"{type(exc).__name__}: {exc}"
        thread.is_completed = False
    return thread
