"""Candidate-level agent qualitative assessment (C11-C13 deliberation layer).

Runs once per selected candidate, before reviewer critique: the condition's
own author/agent model writes a structured self-assessment of why the
candidate was selected. The assessment is grounded ONLY in the candidate's
own ``CandidateDossier`` and proposal record -- never in the full candidate
universe, other conditions' output, or unverified outside knowledge -- so
that reviewer personas (``candidate_llm_review.run_candidate_reviewer_with_assessment``)
have a bounded, auditable claim set to critique.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from geo_strategist.agent.candidate_review_schemas import (
    AssessmentContent,
    CandidateDossier,
    CandidateQualitativeAssessment,
)
from geo_strategist.providers.base import ChatResult

LlmCall = Callable[[str, "str | None", str], ChatResult]

_ASSESSMENT_SYSTEM_PROMPT = """\
You are the report-authoring agent writing a qualitative self-assessment for exactly one \
candidate that was selected for the final slate.

Ground every claim ONLY in the candidate dossier and proposal record given below.

You must NEVER:
- invent a hospital name that is not already in the dossier's facility_records,
- invent a land parcel,
- invent a facility address,
- refer to any municipality other than this candidate's own municipality or its explicitly \
listed local_comparators,
- make a claim the dossier does not support,
- cite an evidence_refs path for a field that is not actually present in the dossier.

If information is not present in the dossier, say so explicitly (for example "not available in \
the dossier") instead of inventing it.

Return JSON only.
"""

_ALLOWED_REF_PREFIXES: tuple[str, ...] = (
    "score_components.",
    "municipal_facts.",
    "cost_model.",
    "facility_records",
    "evidence_grades.",
    "evidence_gaps",
    "required_due_diligence",
    "local_comparators",
)


def valid_dossier_evidence_ref(ref: str) -> bool:
    """True if ``ref`` names a field family that actually exists on a dossier.

    Accepts both the bare dotted-path convention used by candidate
    assessments (``score_components.demand``) and the ``dossier:``-prefixed
    convention already used by reviewer findings (``dossier:evidence_grades.financial``).
    A cheap, deterministic grounding check -- not a substitute for the
    provenance/consistency judge, but enough to flag a path that could not
    possibly be a dossier field.
    """

    candidate = ref[len("dossier:"):] if ref.startswith("dossier:") else ref
    return any(candidate.startswith(prefix) for prefix in _ALLOWED_REF_PREFIXES)


def _dossier_json(dossier: CandidateDossier) -> str:
    return json.dumps(dossier.model_dump(), ensure_ascii=False)


def _extract_json(text: str) -> Any | None:
    from geo_strategist.experiments.live_common import extract_json_block

    return extract_json_block(text)


def generate_candidate_assessment(
    llm: LlmCall,
    condition_id: str,
    dossier: CandidateDossier,
    proposal: dict[str, Any],
    *,
    model_name: str = "",
) -> CandidateQualitativeAssessment:
    prompt = (
        "Candidate dossier (JSON, the ONLY source of facts you may use):\n"
        + _dossier_json(dossier) + "\n\n"
        "This candidate's proposal record (JSON):\n"
        + json.dumps({
            "candidate_id": proposal.get("candidate_id"),
            "rank": proposal.get("rank"),
            "action_type": proposal.get("action_type"),
            "composite_score": proposal.get("composite_score"),
            "llm_rationale": proposal.get("llm_rationale"),
            "required_due_diligence": proposal.get("required_due_diligence"),
        }, ensure_ascii=False) + "\n\n"
        "Write a structured qualitative assessment of why this candidate was selected.\n"
        'Reply as JSON: {"why_selected": "...", "demand_supply_case": "...", '
        '"financial_case": "...", "access_case": "...", "site_feasibility_case": "...", '
        '"main_uncertainties": ["..."], "what_would_falsify": ["..."], '
        '"evidence_refs": ["score_components.demand", "municipal_facts.<field>"]}'
    )
    result = llm(prompt, _ASSESSMENT_SYSTEM_PROMPT, "candidate_qualitative_assessment")
    parsed: dict[str, Any] = {}
    if result.ok:
        maybe_parsed = _extract_json(result.text)
        if isinstance(maybe_parsed, dict):
            parsed = maybe_parsed

    assessment = AssessmentContent(
        why_selected=str(parsed.get("why_selected") or ""),
        demand_supply_case=str(parsed.get("demand_supply_case") or ""),
        financial_case=str(parsed.get("financial_case") or ""),
        access_case=str(parsed.get("access_case") or ""),
        site_feasibility_case=str(parsed.get("site_feasibility_case") or ""),
        main_uncertainties=[str(u) for u in parsed.get("main_uncertainties") or []],
        what_would_falsify=[str(u) for u in parsed.get("what_would_falsify") or []],
    )
    # Recorded as-is (not silently filtered) so the grounding-rate metric
    # computed downstream reflects what the model actually cited; an invalid
    # ref here is a data point about the model's grounding quality, not
    # something to hide.
    evidence_refs = [str(ref) for ref in parsed.get("evidence_refs") or []][:20]
    return CandidateQualitativeAssessment(
        condition_id=condition_id,
        candidate_id=dossier.candidate_id,
        rank=dossier.rank,
        assessment=assessment,
        evidence_refs=evidence_refs,
        model_role="author",
        model_name=model_name,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
