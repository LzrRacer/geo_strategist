"""Per-candidate dossier builder for the candidate-level deliberation pipeline.

A dossier is deliberately narrow: it carries only facts about *this*
candidate plus a short, explicit list of comparator candidates (same
prefecture / same action / rank-adjacent within the current slate). It never
carries study-area-wide missing-data lists — those belong in the
report-level "Study-area data quality note", not in any candidate's review
scope.
"""

from __future__ import annotations

from typing import Any

from geo_strategist.agent.candidate_review_schemas import CandidateDossier
from geo_strategist.experiments.deterministic_evaluation_engine import DataBundle
from geo_strategist.experiments.location_costing import location_cost_model

_SCORE_COMPONENT_KEYS: tuple[str, ...] = (
    "demand",
    "aging",
    "supply_shortage",
    "financial",
    "land",
    "demographic_risk",
    "evidence_completeness",
)


def _comparator_row(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": proposal.get("candidate_id"),
        "prefecture": proposal.get("prefecture"),
        "municipality": proposal.get("municipality"),
        "action_type": proposal.get("action_type"),
        "rank": proposal.get("rank"),
        "composite_score": proposal.get("composite_score"),
    }


def _local_comparators(
    proposal: dict[str, Any],
    slate: list[dict[str, Any]],
    *,
    n_comparators: int,
) -> list[dict[str, Any]]:
    """Same-prefecture / same-action peers first, then rank neighbors, all
    drawn only from the current slate (never the full candidate universe) so
    the comparator pool stays an explicit, bounded list."""

    own_id = proposal.get("candidate_id")
    own_rank = proposal.get("rank") or 0
    peers = [p for p in slate if p.get("candidate_id") != own_id]

    def _peer_key(p: dict[str, Any]) -> tuple[int, int, int]:
        same_prefecture = 0 if p.get("prefecture") == proposal.get("prefecture") else 1
        same_action = 0 if p.get("action_type") == proposal.get("action_type") else 1
        rank_distance = abs((p.get("rank") or 0) - own_rank)
        return (same_prefecture + same_action, rank_distance, p.get("rank") or 0)

    ordered = sorted(peers, key=_peer_key)
    seen: set[str] = set()
    comparators: list[dict[str, Any]] = []
    for peer in ordered:
        cid = peer.get("candidate_id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        comparators.append(_comparator_row(peer))
        if len(comparators) >= n_comparators:
            break
    return comparators


def build_candidate_dossier(
    proposal: dict[str, Any],
    slate: list[dict[str, Any]],
    data: DataBundle,
    *,
    n_comparators: int = 5,
) -> CandidateDossier:
    key = (proposal.get("prefecture"), proposal.get("municipality"))
    facts = dict(data.municipal_facts_by_key.get(key, {}))
    cost_model = dict(data.cost_model_by_prefecture.get(proposal.get("prefecture"), {}))
    cost_model["candidate_location_cost_scenario"] = location_cost_model(
        facts, cost_model
    )
    facilities = data.facilities_by_key.get(key, [])
    components = proposal.get("score_components") or {}
    score_components = {k: components.get(k) for k in _SCORE_COMPONENT_KEYS}

    return CandidateDossier(
        candidate_id=str(proposal.get("candidate_id")),
        prefecture=str(proposal.get("prefecture")),
        municipality=str(proposal.get("municipality")),
        action_type=str(proposal.get("action_type")),
        rank=proposal.get("rank"),
        composite_score=proposal.get("composite_score"),
        score_components=score_components,
        municipal_facts=facts,
        cost_model=cost_model,
        facility_records=[record.to_dict() for record in facilities],
        evidence_grades=dict(proposal.get("evidence_grades") or {}),
        evidence_gaps=list(proposal.get("evidence_gaps") or []),
        required_due_diligence=list(proposal.get("required_due_diligence") or []),
        local_comparators=_local_comparators(proposal, slate, n_comparators=n_comparators),
    )
