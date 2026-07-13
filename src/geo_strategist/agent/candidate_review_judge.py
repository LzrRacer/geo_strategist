"""Provenance / consistency judge for candidate review packets.

Runs after every reviewer finding has an author response. It never trusts a
finding or response at face value: findings that drift outside the
candidate's own scope, lack evidence references, or get no response at all
are surfaced as judge flags and (for scope violations) moved to
``invalidated_findings`` so they never reach the rendered report.
"""

from __future__ import annotations

from geo_strategist.agent.candidate_review_schemas import (
    CandidateDossier,
    CandidateReviewPacket,
)

_MIN_GROUNDED_REJECTION_LENGTH = 20


def judge_candidate_review_packet(
    dossier: CandidateDossier,
    packet: CandidateReviewPacket,
    *,
    disallowed_municipalities: frozenset[str] = frozenset(),
) -> CandidateReviewPacket:
    valid_findings = []
    invalidated = list(packet.invalidated_findings)
    flags: list[str] = list(packet.judge_flags)
    known_finding_ids = {f.finding_id for f in packet.reviewer_findings}

    for finding in packet.reviewer_findings:
        issues: list[str] = []
        if finding.candidate_id != dossier.candidate_id:
            issues.append("candidate_id_mismatch")
        if not finding.evidence_refs:
            issues.append("missing_evidence_refs")
        text = f"{finding.issue} {finding.data_observation}"
        mentioned = sorted({m for m in disallowed_municipalities if m and m in text})
        if mentioned:
            issues.append("mentions_out_of_scope_municipality:" + ",".join(mentioned[:3]))
        if issues:
            invalidated.append(finding)
            flags.append(f"finding {finding.finding_id} invalidated: {'; '.join(issues)}")
        else:
            valid_findings.append(finding)

    for response in packet.author_responses:
        if response.finding_id not in known_finding_ids:
            flags.append(
                f"author response references unknown finding_id {response.finding_id}")

    responses_by_id = {r.finding_id: r for r in packet.author_responses}
    for finding in valid_findings:
        response = responses_by_id.get(finding.finding_id)
        if finding.severity in ("blocking", "major"):
            if response is None:
                flags.append(
                    f"finding {finding.finding_id} ({finding.severity}) has no author response")
                continue
            if not response.why_still_proceed:
                flags.append(
                    f"finding {finding.finding_id} retained without why_still_proceed")
            if not response.mitigation:
                flags.append(f"finding {finding.finding_id} missing mitigation")
            if not response.residual_risk:
                flags.append(f"finding {finding.finding_id} missing residual_risk")
        if response is not None and response.response_status == "rejected":
            if len(response.response.strip()) < _MIN_GROUNDED_REJECTION_LENGTH:
                flags.append(
                    f"finding {finding.finding_id} rejected without a data-grounded reason")

    blocking_valid = [f for f in valid_findings if f.severity == "blocking"]
    major_valid = [f for f in valid_findings if f.severity == "major"]
    if not valid_findings:
        final_position = "retain"
        final_reason = "No reviewer findings survived provenance/consistency checks."
    elif blocking_valid or major_valid:
        final_position = "retain_with_major_due_diligence"
        final_reason = (
            f"{len(blocking_valid)} blocking and {len(major_valid)} major "
            "finding(s) require due diligence before adoption.")
    else:
        final_position = "retain"
        final_reason = "Only moderate/minor findings recorded; no blocking issues."

    return packet.model_copy(update={
        "reviewer_findings": valid_findings,
        "invalidated_findings": invalidated,
        "judge_flags": flags,
        "final_candidate_position": final_position,
        "final_reason": final_reason,
    })
