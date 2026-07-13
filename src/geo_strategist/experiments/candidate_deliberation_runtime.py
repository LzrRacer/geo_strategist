"""Candidate-level deliberation pipeline: plan -> execute -> review -> report.

Runs after a condition's base slate exists. For every (candidate, reviewer)
pair a ``ReviewThread`` captures the reviewer's data requests, the
explorer's scope-limited observations, and the reviewer's findings; findings
are aggregated per candidate, the report author responds to each one, and a
provenance/consistency judge isolates anything that drifted out of scope
before the packet is attached to the condition's result.

This stage never replaces the base proposal-producing algorithm — it is a
post-review augmentation. Deterministic-only conditions (C0) are skipped by
``condition_supports_candidate_deliberation``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from geo_strategist.agent.candidate_author_response import generate_author_responses
from geo_strategist.agent.candidate_llm_review import (
    DEFAULT_REVIEWERS,
    run_candidate_reviewer,
)
from geo_strategist.agent.candidate_review_judge import judge_candidate_review_packet
from geo_strategist.agent.candidate_review_schemas import (
    CandidateDossier,
    CandidateReviewPacket,
    ReviewerFinding,
    ReviewThread,
)
from geo_strategist.experiments.candidate_data_explorer import CandidateDataExplorer
from geo_strategist.experiments.candidate_dossier import build_candidate_dossier
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _write_json, _write_jsonl
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DEFAULT_WEIGHTS,
    DataBundle,
    proposal_for_candidate,
    rank_candidates,
)
from geo_strategist.experiments.live_common import LiveConditionResult
from geo_strategist.providers.base import ChatResult

LlmCall = Callable[[str, "str | None", str], ChatResult]

_DETERMINISTIC_RUNNERS: frozenset[str] = frozenset({"c0_deterministic"})
_DETERMINISTIC_ALGORITHMS: frozenset[str] = frozenset({"deterministic_baseline"})

_REPLACEMENT_MIN_AVAILABILITY = 0.5


def condition_supports_candidate_deliberation(spec: ConditionSpec) -> bool:
    if spec.runner in _DETERMINISTIC_RUNNERS:
        return False
    if spec.algorithm in _DETERMINISTIC_ALGORITHMS:
        return False
    return True


# ---------------------------------------------------------------------------
# Plan -> execute
# ---------------------------------------------------------------------------

def generate_review_plan(
    proposals: list[dict[str, Any]],
    reviewers: list[str],
) -> list[dict[str, str]]:
    tasks = []
    for proposal in proposals:
        for reviewer_id in reviewers:
            tasks.append({
                "candidate_id": str(proposal.get("candidate_id")),
                "reviewer_id": reviewer_id,
            })
    return tasks


def execute_review_task(
    task: dict[str, str],
    dossier: CandidateDossier,
    explorer: CandidateDataExplorer,
    llm: LlmCall,
    *,
    max_data_requests: int,
) -> ReviewThread:
    return run_candidate_reviewer(
        llm, task["reviewer_id"], dossier, explorer,
        max_data_requests=max_data_requests)


def run_all_review_tasks(
    tasks: list[dict[str, str]],
    dossier_map: dict[str, CandidateDossier],
    explorer: CandidateDataExplorer,
    llm: LlmCall,
    *,
    max_workers: int = 8,
    max_data_requests: int = 3,
) -> list[ReviewThread]:
    threads: list[ReviewThread] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [
            executor.submit(
                execute_review_task, task, dossier_map[task["candidate_id"]],
                explorer, llm, max_data_requests=max_data_requests)
            for task in tasks if task["candidate_id"] in dossier_map
        ]
        for future in as_completed(futures):
            threads.append(future.result())
    return threads


# ---------------------------------------------------------------------------
# Review aggregation -> packets
# ---------------------------------------------------------------------------

def aggregate_findings_by_candidate(
    threads: list[ReviewThread],
) -> dict[str, list[ReviewerFinding]]:
    result: dict[str, list[ReviewerFinding]] = {}
    for thread in threads:
        if thread.error and not thread.findings:
            continue
        result.setdefault(thread.candidate_id, []).extend(thread.findings)
    return result


def _municipality_universe(data: DataBundle) -> set[str]:
    return {str(municipality) for _prefecture, municipality in data.scores_by_key}


def build_packet_for_candidate(
    candidate_id: str,
    findings: list[ReviewerFinding],
    dossier: CandidateDossier,
    llm: LlmCall,
    *,
    disallowed_municipalities: frozenset[str],
) -> CandidateReviewPacket:
    responses = generate_author_responses(llm, dossier, findings)
    packet = CandidateReviewPacket(
        candidate_id=candidate_id,
        reviewer_findings=findings,
        author_responses=responses,
    )
    return judge_candidate_review_packet(
        dossier, packet, disallowed_municipalities=disallowed_municipalities)


def generate_candidate_review_packets(
    threads: list[ReviewThread],
    dossier_map: dict[str, CandidateDossier],
    llm: LlmCall,
    *,
    disallowed_municipalities: frozenset[str] = frozenset(),
) -> list[CandidateReviewPacket]:
    findings_by_candidate = aggregate_findings_by_candidate(threads)
    packets: list[CandidateReviewPacket] = []
    for candidate_id, findings in findings_by_candidate.items():
        dossier = dossier_map.get(candidate_id)
        if dossier is None:
            continue
        packets.append(build_packet_for_candidate(
            candidate_id, findings, dossier, llm,
            disallowed_municipalities=disallowed_municipalities))
    return packets


# ---------------------------------------------------------------------------
# Replacement (optional, off by default)
# ---------------------------------------------------------------------------

def should_replace_candidate(packet: CandidateReviewPacket) -> bool:
    blocking_replace = [
        f for f in packet.reviewer_findings
        if f.severity == "blocking" and f.recommendation in ("replace_candidate", "reject")
    ]
    return len(blocking_replace) >= 2


def find_replacement_candidates(
    proposal: dict[str, Any],
    data: DataBundle,
    *,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    ranking = data.runtime_cache.get("default_ranking")
    if ranking is None:
        ranking = rank_candidates(data, DEFAULT_WEIGHTS)
        data.runtime_cache["default_ranking"] = ranking
    own_id = proposal.get("candidate_id")
    rows = [r for r in ranking
            if r["candidate_id"] != own_id
            and r["action_type"] == proposal.get("action_type")
            and r["component_availability"] >= _REPLACEMENT_MIN_AVAILABILITY]
    same_prefecture = [r for r in rows if r["prefecture"] == proposal.get("prefecture")]
    pool = same_prefecture or rows
    pool = sorted(pool, key=lambda r: -r["composite_score"])
    return pool[:top_n]


def _attempt_replacement(
    spec: ConditionSpec,
    proposal: dict[str, Any],
    packet: CandidateReviewPacket,
    slate: list[dict[str, Any]],
    data: DataBundle,
    llm: LlmCall,
    *,
    reviewers: list[str],
    max_workers: int,
    max_data_requests: int,
    disallowed_municipalities: frozenset[str],
) -> tuple[dict[str, Any] | None, CandidateReviewPacket]:
    """Returns (replacement_proposal_or_None, updated_packet)."""

    existing_ids = {p.get("candidate_id") for p in slate}
    candidates = find_replacement_candidates(proposal, data)
    candidate_row = next(
        (row for row in candidates if row["candidate_id"] not in existing_ids), None)
    if candidate_row is None:
        packet = packet.model_copy(update={
            "replacement_decision": {
                "attempted": True, "original_candidate_id": proposal.get("candidate_id"),
                "replacement_candidate_id": None,
                "rationale": "No eligible replacement candidate found "
                             "(same action type, sufficient data availability, not already in slate).",
            },
        })
        return None, packet

    replacement_base = dict(candidate_row)
    replacement_base["rank"] = proposal.get("rank")
    replacement_proposal = proposal_for_candidate(
        replacement_base, data,
        condition_id=spec.condition_id, condition_group=spec.group)
    replacement_proposal["rank"] = proposal.get("rank")

    replacement_dossier = build_candidate_dossier(replacement_proposal, slate, data)
    explorer = CandidateDataExplorer(data)
    tasks = generate_review_plan([replacement_proposal], reviewers)
    threads = run_all_review_tasks(
        tasks, {replacement_dossier.candidate_id: replacement_dossier}, explorer, llm,
        max_workers=max_workers, max_data_requests=max_data_requests)
    replacement_findings = aggregate_findings_by_candidate(threads).get(
        replacement_dossier.candidate_id, [])
    replacement_packet = build_packet_for_candidate(
        replacement_dossier.candidate_id, replacement_findings, replacement_dossier, llm,
        disallowed_municipalities=disallowed_municipalities)

    original_blocking = len([f for f in packet.reviewer_findings if f.severity == "blocking"])
    replacement_blocking = len(
        [f for f in replacement_packet.reviewer_findings if f.severity == "blocking"])
    decision = {
        "attempted": True,
        "original_candidate_id": proposal.get("candidate_id"),
        "replacement_candidate_id": replacement_dossier.candidate_id,
        "original_blocking_findings": original_blocking,
        "replacement_blocking_findings": replacement_blocking,
    }
    if replacement_blocking < original_blocking:
        decision["rationale"] = (
            f"Replacement {replacement_dossier.candidate_id} carries fewer blocking "
            f"findings ({replacement_blocking} vs {original_blocking}); swapped.")
        replacement_packet = replacement_packet.model_copy(update={
            "final_candidate_position": "replace",
            "replacement_decision": decision,
        })
        return replacement_proposal, replacement_packet
    decision["rationale"] = (
        f"Replacement {replacement_dossier.candidate_id} did not clear fewer blocking "
        f"findings ({replacement_blocking} vs {original_blocking}); original candidate retained.")
    packet = packet.model_copy(update={"replacement_decision": decision})
    return None, packet


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _write_artifacts(
    result: LiveConditionResult,
    packets: list[CandidateReviewPacket],
    threads: list[ReviewThread],
    run_dir: Path,
) -> None:
    packets_path = run_dir / "candidate_review_packets.json"
    threads_path = run_dir / "candidate_review_threads.jsonl"
    summary_path = run_dir / "candidate_review_summary.json"
    _write_json(packets_path, [p.model_dump() for p in packets])
    _write_jsonl(threads_path, [t.model_dump() for t in threads])
    summary = {
        "candidate_count": len(packets),
        "total_findings": sum(len(p.reviewer_findings) for p in packets),
        "total_invalidated_findings": sum(len(p.invalidated_findings) for p in packets),
        "total_judge_flags": sum(len(p.judge_flags) for p in packets),
        "positions": {packet.candidate_id: packet.final_candidate_position for packet in packets},
    }
    _write_json(summary_path, summary)
    result.artifacts["candidate_review_packets"] = str(packets_path)
    result.artifacts["candidate_review_threads"] = str(threads_path)
    result.artifacts["candidate_review_summary"] = str(summary_path)

    from geo_strategist.reporting.review_figures import (
        author_response_status_figure,
        residual_risk_figure,
        review_severity_figure,
        reviewer_coverage_figure,
    )
    from geo_strategist.reporting.figures import write_figure_data

    # Siblings of run_dir under the shared output_dir (out_dir/runs/<Cx> and
    # out_dir/figures), matching the slug-prefixed naming every other report
    # figure uses so live_report.py can reference them the same way.
    figures_dir = run_dir.parent.parent / "figures"
    slug = result.spec.report_slug
    figure_calls = (
        (f"{slug}_candidate_review_severity", review_severity_figure,
         f"{result.spec.group}: reviewer finding severity per candidate"),
        (f"{slug}_candidate_review_coverage", reviewer_coverage_figure,
         f"{result.spec.group}: reviewer coverage per candidate"),
        (f"{slug}_candidate_author_response_status", author_response_status_figure,
         f"{result.spec.group}: author response status per candidate"),
        (f"{slug}_candidate_residual_risk", residual_risk_figure,
         f"{result.spec.group}: unresolved major/blocking risk per candidate"),
    )
    for name, fn, title in figure_calls:
        figure_path = figures_dir / f"{name}.png"
        write_figure_data(figure_path, {
            "figure": name,
            "condition_group": result.spec.group,
            "candidate_review_packets": [p.model_dump() for p in packets],
            "candidate_review_summary": summary,
        })
        path = fn(packets, figure_path, title=title)
        if path is not None:
            result.artifacts[name] = str(path)


def run_candidate_deliberation(
    result: LiveConditionResult,
    data: DataBundle,
    llm: LlmCall,
    *,
    reviewers: list[str] = DEFAULT_REVIEWERS,
    allow_replacement: bool = False,
    max_workers: int = 8,
    max_data_requests_per_reviewer: int = 3,
    run_dir: Path | None = None,
) -> LiveConditionResult:
    if not result.proposals:
        return result
    if not condition_supports_candidate_deliberation(result.spec):
        return result

    slate = result.proposals
    if max_candidates := _max_review_candidates():
        slate = slate[:max_candidates]
    dossier_map = {
        str(p.get("candidate_id")): build_candidate_dossier(p, slate, data)
        for p in slate
    }
    explorer = CandidateDataExplorer(data)
    tasks = generate_review_plan(slate, reviewers)
    threads = run_all_review_tasks(
        tasks, dossier_map, explorer, llm,
        max_workers=max_workers, max_data_requests=max_data_requests_per_reviewer)

    disallowed = frozenset(
        _municipality_universe(data)
        - {p.get("municipality") for p in slate}
        - {str(c.get("municipality")) for dossier in dossier_map.values()
           for c in dossier.local_comparators}
    )
    packets = generate_candidate_review_packets(
        threads, dossier_map, llm, disallowed_municipalities=disallowed)

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

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_artifacts(result, packets, threads, run_dir)

    return result


def _max_review_candidates() -> int:
    import os

    try:
        return max(0, int(os.environ.get("CANDIDATE_REVIEW_MAX_CANDIDATES", "1")))
    except ValueError:
        return 1
