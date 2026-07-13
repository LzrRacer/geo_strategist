"""Candidate-level qualitative deliberation layer for C11-C13.

Extends the candidate-level deliberation pipeline in
``candidate_deliberation_runtime.py`` with an explicit "agent writes a
qualitative assessment first" step, per the C11-C13 requirement:

    final slate -> build_live_proposals()
    -> agent qualitative assessment (this module)
    -> reviewer critique of dossier + assessment (candidate_llm_review)
    -> author response to reviewer findings (candidate_author_response, reused)
    -> provenance/consistency judge (candidate_review_judge, reused)
    -> proposal artifacts / report / E13 comparison

C0-C10 keep using ``candidate_deliberation_runtime.run_candidate_deliberation``
unchanged; this module is only wired in for C11-C13 (see
``run_condition_proposals.py`` and ``c13_fugu_router.py``). Building blocks
(dossier construction, author-response generation, the provenance judge, and
figure/artifact writing) are imported and reused rather than reimplemented.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from geo_strategist.agent.candidate_llm_review import (
    DEFAULT_REVIEWERS,
    run_candidate_reviewer_with_assessment,
)
from geo_strategist.agent.candidate_qualitative_assessment import (
    generate_candidate_assessment,
    valid_dossier_evidence_ref,
)
from geo_strategist.agent.candidate_review_schemas import (
    CandidateAssessmentReview,
    CandidateDossier,
    CandidateQualitativeAssessment,
    CandidateReviewPacket,
    ReviewThread,
)
from geo_strategist.experiments.candidate_data_explorer import CandidateDataExplorer
from geo_strategist.experiments.candidate_deliberation_runtime import (
    _attempt_replacement,
    _max_review_candidates,
    _municipality_universe,
    _write_artifacts,
    aggregate_findings_by_candidate,
    build_candidate_dossier,
    build_packet_for_candidate,
    condition_supports_candidate_deliberation,
    should_replace_candidate,
)
from geo_strategist.experiments.condition_utils import _read_json, _write_json, _write_jsonl
from geo_strategist.experiments.deterministic_evaluation_engine import DataBundle
from geo_strategist.experiments.live_common import LiveConditionResult
from geo_strategist.providers.base import ChatResult

LlmCall = Callable[[str, "str | None", str], ChatResult]

_FALLBACK_RESPONSE_MARKER = "No author response was returned for this finding"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Step 1: agent qualitative assessment, one call per selected candidate
# ---------------------------------------------------------------------------

def generate_assessments_for_slate(
    llm: LlmCall,
    condition_id: str,
    slate: list[dict[str, Any]],
    dossier_map: dict[str, CandidateDossier],
    *,
    model_name: str = "",
) -> list[CandidateQualitativeAssessment]:
    assessments: list[CandidateQualitativeAssessment] = []
    for proposal in slate:
        candidate_id = str(proposal.get("candidate_id"))
        dossier = dossier_map.get(candidate_id)
        if dossier is None:
            continue
        assessments.append(generate_candidate_assessment(
            llm, condition_id, dossier, proposal, model_name=model_name))
    return assessments


# ---------------------------------------------------------------------------
# Step 2: reviewer critique of dossier + assessment
# ---------------------------------------------------------------------------

def run_reviews_for_slate(
    llm: LlmCall,
    slate: list[dict[str, Any]],
    dossier_map: dict[str, CandidateDossier],
    assessment_by_candidate: dict[str, CandidateQualitativeAssessment],
    explorer: CandidateDataExplorer,
    reviewers: list[str],
    *,
    condition_id: str,
    model_name: str = "",
    max_workers: int = 8,
    max_data_requests: int = 3,
) -> tuple[list[ReviewThread], list[CandidateAssessmentReview]]:
    tasks = [
        (str(proposal.get("candidate_id")), reviewer_id)
        for proposal in slate
        for reviewer_id in reviewers
        if str(proposal.get("candidate_id")) in dossier_map
        and str(proposal.get("candidate_id")) in assessment_by_candidate
    ]

    def _run_one(candidate_id: str, reviewer_id: str) -> tuple[ReviewThread, Any]:
        return run_candidate_reviewer_with_assessment(
            llm, reviewer_id, dossier_map[candidate_id],
            assessment_by_candidate[candidate_id], explorer,
            max_data_requests=max_data_requests)

    threads: list[ReviewThread] = []
    assessment_reviews: list[CandidateAssessmentReview] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(_run_one, candidate_id, reviewer_id): (candidate_id, reviewer_id)
            for candidate_id, reviewer_id in tasks
        }
        for future in as_completed(futures):
            candidate_id, reviewer_id = futures[future]
            thread, scope = future.result()
            threads.append(thread)
            assessment_reviews.append(CandidateAssessmentReview(
                condition_id=condition_id,
                candidate_id=candidate_id,
                reviewer=reviewer_id,
                findings=thread.findings,
                review_scope=scope,
                model_role="reviewer",
                model_name=model_name,
                created_at=_now_iso(),
            ))
    return threads, assessment_reviews


def disallowed_municipalities_for_slate(
    data: DataBundle,
    slate: list[dict[str, Any]],
    dossier_map: dict[str, CandidateDossier],
) -> frozenset[str]:
    return frozenset(
        _municipality_universe(data)
        - {p.get("municipality") for p in slate}
        - {str(c.get("municipality")) for dossier in dossier_map.values()
           for c in dossier.local_comparators}
    )


# ---------------------------------------------------------------------------
# Step 3 + 4: author response to findings, then provenance/consistency judge
# (both fully reused from candidate_deliberation_runtime.build_packet_for_candidate)
# ---------------------------------------------------------------------------

def build_packets_for_slate(
    llm: LlmCall,
    slate: list[dict[str, Any]],
    dossier_map: dict[str, CandidateDossier],
    threads: list[ReviewThread],
    *,
    disallowed_municipalities: frozenset[str] = frozenset(),
) -> list[CandidateReviewPacket]:
    findings_by_candidate = aggregate_findings_by_candidate(threads)
    packets: list[CandidateReviewPacket] = []
    for proposal in slate:
        candidate_id = str(proposal.get("candidate_id"))
        dossier = dossier_map.get(candidate_id)
        if dossier is None:
            continue
        findings = findings_by_candidate.get(candidate_id, [])
        packets.append(build_packet_for_candidate(
            candidate_id, findings, dossier, llm,
            disallowed_municipalities=disallowed_municipalities))
    return packets


# ---------------------------------------------------------------------------
# Summary metrics consumed by the report writer and E13 (kept structurally
# separate from the core algorithm's own fitness/ranking).
# ---------------------------------------------------------------------------

def compute_deliberation_summary(
    slate: list[dict[str, Any]],
    assessments: list[CandidateQualitativeAssessment],
    assessment_reviews: list[CandidateAssessmentReview],
    packets: list[CandidateReviewPacket],
) -> dict[str, Any]:
    slate_size = len(slate)
    assessed_with_content = sum(
        1 for a in assessments if a.assessment.why_selected.strip())

    raw_findings = [f for review in assessment_reviews for f in review.findings]
    supported_findings = [f for p in packets for f in p.reviewer_findings]
    invalidated_findings = [f for p in packets for f in p.invalidated_findings]

    severity_counts = {"blocking": 0, "major": 0, "moderate": 0, "minor": 0}
    for finding in supported_findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1

    candidates_with_supported_finding = len({
        p.candidate_id for p in packets if p.reviewer_findings
    })

    major_blocking_supported = [
        f for f in supported_findings if f.severity in ("blocking", "major")
    ]
    responses_by_finding: dict[str, Any] = {
        r.finding_id: r for p in packets for r in p.author_responses
    }
    major_blocking_handled = sum(
        1 for f in major_blocking_supported
        if (response := responses_by_finding.get(f.finding_id)) is not None
        and response.response_status in ("accepted", "partially_accepted")
        and (response.mitigation or response.residual_risk)
    )
    non_fallback_responses = sum(
        1 for f in supported_findings
        if (response := responses_by_finding.get(f.finding_id)) is not None
        and _FALLBACK_RESPONSE_MARKER not in response.response
    )
    candidates_with_added_due_diligence = len({
        p.candidate_id for p in packets
        if any(r.added_due_diligence for r in p.author_responses)
    })

    all_evidence_refs = (
        [ref for a in assessments for ref in a.evidence_refs]
        + [ref for f in raw_findings for ref in f.evidence_refs]
    )
    valid_refs = sum(1 for ref in all_evidence_refs if valid_dossier_evidence_ref(ref))

    return {
        "candidates_in_slate": slate_size,
        "candidates_assessed": len(assessments),
        "candidate_level_assessment_coverage": (
            round(assessed_with_content / slate_size, 4) if slate_size else 0.0),
        "reviewer_finding_coverage": (
            round(candidates_with_supported_finding / len(assessments), 4)
            if assessments else 0.0),
        "total_raw_reviewer_findings": len(raw_findings),
        "total_supported_reviewer_findings": len(supported_findings),
        "total_unsupported_findings_filtered": len(invalidated_findings),
        "unsupported_finding_filter_rate": (
            round(len(invalidated_findings) / len(raw_findings), 4) if raw_findings else 0.0),
        "blocking_findings": severity_counts["blocking"],
        "major_findings": severity_counts["major"],
        "moderate_minor_findings": severity_counts["moderate"] + severity_counts["minor"],
        "major_blocking_issue_handling": (
            round(major_blocking_handled / len(major_blocking_supported), 4)
            if major_blocking_supported else 1.0),
        "author_response_quality": (
            round(non_fallback_responses / len(supported_findings), 4)
            if supported_findings else 1.0),
        "review_revision_quality": (
            round(candidates_with_added_due_diligence / len(packets), 4)
            if packets else 0.0),
        "evidence_ref_grounding_rate": (
            round(valid_refs / len(all_evidence_refs), 4) if all_evidence_refs else 1.0),
    }


def _merge_summary_json(summary_path: Path, extra: dict[str, Any]) -> None:
    existing = _read_json(summary_path) if summary_path.exists() else {}
    existing = existing if isinstance(existing, dict) else {}
    existing["candidate_qualitative_deliberation"] = extra
    _write_json(summary_path, existing)


# ---------------------------------------------------------------------------
# Top-level orchestrator (C11/C12 post-hoc use; C13 uses the granular
# functions above directly so each step can be logged as its own routed task)
# ---------------------------------------------------------------------------

def run_candidate_qualitative_deliberation(
    result: LiveConditionResult,
    data: DataBundle,
    llm: LlmCall,
    *,
    reviewers: list[str] = DEFAULT_REVIEWERS,
    allow_replacement: bool = False,
    max_workers: int = 8,
    max_data_requests_per_reviewer: int = 3,
    run_dir: Path | None = None,
    model_name: str = "",
) -> LiveConditionResult:
    if not result.proposals:
        return result
    if not condition_supports_candidate_deliberation(result.spec):
        return result

    slate = result.proposals
    if max_candidates := _max_review_candidates():
        slate = slate[:max_candidates]
    condition_id = result.spec.condition_id
    dossier_map = {
        str(p.get("candidate_id")): build_candidate_dossier(p, slate, data)
        for p in slate
    }
    explorer = CandidateDataExplorer(data)

    assessments = generate_assessments_for_slate(
        llm, condition_id, slate, dossier_map, model_name=model_name)
    assessment_by_candidate = {a.candidate_id: a for a in assessments}

    threads, assessment_reviews = run_reviews_for_slate(
        llm, slate, dossier_map, assessment_by_candidate, explorer, reviewers,
        condition_id=condition_id, model_name=model_name,
        max_workers=max_workers, max_data_requests=max_data_requests_per_reviewer)

    disallowed = disallowed_municipalities_for_slate(data, slate, dossier_map)
    packets = build_packets_for_slate(
        llm, slate, dossier_map, threads, disallowed_municipalities=disallowed)

    if allow_replacement:
        by_candidate_id = {p.get("candidate_id"): p for p in slate}
        updated_packets: list[CandidateReviewPacket] = []
        for packet in packets:
            proposal = by_candidate_id.get(packet.candidate_id)
            if proposal is not None and should_replace_candidate(packet):
                replacement_proposal, packet = _attempt_replacement(
                    result.spec, proposal, packet, slate, data, llm,
                    reviewers=reviewers, max_workers=max_workers,
                    max_data_requests=max_data_requests_per_reviewer,
                    disallowed_municipalities=disallowed)
                if replacement_proposal is not None:
                    index = slate.index(proposal)
                    slate[index] = replacement_proposal
            updated_packets.append(packet)
        packets = updated_packets
        result.proposals = slate

    result.candidate_review_packets = [p.model_dump() for p in packets]
    result.candidate_review_threads = [t.model_dump() for t in threads]
    result.candidate_qualitative_assessments = [a.model_dump() for a in assessments]
    result.candidate_assessment_reviews = [r.model_dump() for r in assessment_reviews]

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_artifacts(result, packets, threads, run_dir)
        assessments_path = run_dir / "candidate_qualitative_assessments.jsonl"
        reviews_path = run_dir / "candidate_assessment_reviews.jsonl"
        _write_jsonl(assessments_path, result.candidate_qualitative_assessments)
        _write_jsonl(reviews_path, result.candidate_assessment_reviews)
        result.artifacts["candidate_qualitative_assessments"] = str(assessments_path)
        result.artifacts["candidate_assessment_reviews"] = str(reviews_path)

    summary = compute_deliberation_summary(slate, assessments, assessment_reviews, packets)
    result.candidate_deliberation_summary = summary
    if run_dir is not None:
        _merge_summary_json(run_dir / "candidate_review_summary.json", summary)

    return result
