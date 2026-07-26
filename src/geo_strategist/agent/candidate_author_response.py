"""Report-author responses to candidate-level reviewer findings.

One LLM call handles every finding for a candidate at once (cheaper than
per-finding calls, and lets the author reason about the finding set as a
whole). Every finding is guaranteed exactly one response: if the LLM omits
one, a conservative fallback response is synthesized so downstream code
never has to special-case a missing response.
"""

from __future__ import annotations

import json
from typing import Callable

from pydantic import ValidationError

from geo_strategist.agent.candidate_review_schemas import (
    AuthorResponse,
    CandidateDossier,
    ReviewerFinding,
)
from geo_strategist.providers.base import ChatResult

LlmCall = Callable[[str, "str | None", str], ChatResult]

_AUTHOR_SYSTEM_PROMPT = """\
You are the report author responding to skeptical reviewer findings.

For each finding:
- accept, partially accept, or reject the criticism;
- state what due diligence or mitigation is required;
- if the candidate remains recommended, explain why the evidence still supports retaining it;
- state residual risk.

Do not dismiss a finding without citing the dossier or data observation.
Return JSON only.
"""


def _extract_json(text: str):
    from geo_strategist.experiments.live_common import extract_json_block

    return extract_json_block(text)


def _fallback_response(finding: ReviewerFinding) -> AuthorResponse:
    return AuthorResponse(
        finding_id=finding.finding_id,
        candidate_id=finding.candidate_id,
        response_status="partially_accepted",
        response=(
            "No author response was returned for this finding; treat it as "
            "unresolved pending manual follow-up."),
        why_still_proceed=(
            "Not yet assessed — requires manual reviewer follow-up before "
            "this candidate can be considered fully addressed."
            if finding.severity in ("blocking", "major") else None),
        mitigation="Manual follow-up required." if finding.severity in ("blocking", "major") else None,
        residual_risk="Unresolved reviewer finding." if finding.severity in ("blocking", "major") else None,
        added_due_diligence=["Manually resolve unaddressed reviewer finding: " + finding.issue],
    )


def generate_author_responses(
    llm: LlmCall,
    dossier: CandidateDossier,
    findings: list[ReviewerFinding],
) -> list[AuthorResponse]:
    if not findings:
        return []

    prompt = (
        "Candidate dossier (JSON):\n" + json.dumps(dossier.model_dump(), ensure_ascii=False) + "\n\n"
        "Reviewer findings to respond to (JSON):\n"
        + json.dumps([f.model_dump() for f in findings], ensure_ascii=False) + "\n\n"
        "Respond to EVERY finding_id listed above, referencing it exactly.\n"
        'Reply as JSON: {"responses": [{"finding_id": "...", '
        '"response_status": "accepted|partially_accepted|rejected", "response": "...", '
        '"why_still_proceed": "...", "mitigation": "...", "residual_risk": "...", '
        '"added_due_diligence": ["..."]}]}'
    )
    result = llm(prompt, _AUTHOR_SYSTEM_PROMPT, "candidate_author_response")

    by_id: dict[str, AuthorResponse] = {}
    if result.ok:
        parsed = _extract_json(result.text)
        rows = parsed.get("responses") if isinstance(parsed, dict) else None
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            finding_id = str(row.get("finding_id") or "")
            if finding_id not in {f.finding_id for f in findings}:
                continue
            try:
                by_id[finding_id] = AuthorResponse(
                    finding_id=finding_id,
                    candidate_id=dossier.candidate_id,
                    response_status=row.get("response_status"),
                    response=str(row.get("response") or ""),
                    why_still_proceed=row.get("why_still_proceed"),
                    mitigation=row.get("mitigation"),
                    residual_risk=row.get("residual_risk"),
                    added_due_diligence=[str(d) for d in row.get("added_due_diligence") or []],
                )
            except ValidationError:
                continue

    responses: list[AuthorResponse] = []
    for finding in findings:
        responses.append(by_id.get(finding.finding_id) or _fallback_response(finding))
    return responses
