"""Deterministic, scope-limited data lookups for candidate-level reviewers.

The explorer never receives the whole ``DataBundle`` and never accepts an
arbitrary municipality name from LLM free text: every ``query_*`` method
takes only the candidate's own ``(prefecture, municipality)`` or one of the
dossier's explicitly listed ``local_comparators``. A request that names
anything else comes back ``out_of_scope`` rather than being answered.
"""

from __future__ import annotations

from typing import Any

from geo_strategist.agent.candidate_review_schemas import (
    CandidateDossier,
    DataObservation,
    DataRequest,
)
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DEFAULT_WEIGHTS,
    DataBundle,
    rank_candidates,
)

_SCORE_COMPONENT_KEYS: tuple[str, ...] = (
    "demand",
    "aging",
    "supply_shortage",
    "financial",
    "land",
    "demographic_risk",
    "evidence_completeness",
)


class CandidateDataExplorer:
    def __init__(self, data: DataBundle) -> None:
        self.data = data

    # -- primitive, scope-bound lookups -----------------------------------

    def query_municipal_facts(self, prefecture: str, municipality: str) -> dict[str, Any]:
        facts = self.data.municipal_facts_by_key.get((prefecture, municipality), {})
        return {
            "prefecture": prefecture,
            "municipality": municipality,
            "facts": dict(facts),
            "evidence_refs": [f"municipal_facts:{prefecture}:{municipality}"] if facts else [],
        }

    def query_facilities(self, prefecture: str, municipality: str) -> list[dict[str, Any]]:
        records = self.data.facilities_by_key.get((prefecture, municipality), [])
        return [record.to_dict() for record in records]

    def _ranking(self) -> list[dict[str, Any]]:
        cached = self.data.runtime_cache.get("default_ranking")
        if cached is not None:
            return cached
        ranking = rank_candidates(self.data, DEFAULT_WEIGHTS)
        self.data.runtime_cache["default_ranking"] = ranking
        return ranking

    def query_candidate_rank_context(self, candidate_id: str) -> dict[str, Any]:
        ranking = self._ranking()
        by_id = {row["candidate_id"]: row for row in ranking}
        row = by_id.get(candidate_id)
        if row is None:
            return {"candidate_id": candidate_id, "found": False, "evidence_refs": []}
        total = len(ranking)
        neighbors = [
            {"candidate_id": r["candidate_id"], "municipality": r["municipality"],
             "rank": r["rank"], "composite_score": r["composite_score"]}
            for r in ranking if abs(r["rank"] - row["rank"]) <= 2 and r["candidate_id"] != candidate_id
        ]
        return {
            "candidate_id": candidate_id,
            "found": True,
            "rank": row["rank"],
            "total_candidates": total,
            "composite_score": row["composite_score"],
            "component_availability": row["component_availability"],
            "percentile": round(100.0 * (total - row["rank"] + 1) / total, 1),
            "rank_neighbors": neighbors,
            "evidence_refs": [f"default_ranking:candidate:{candidate_id}"],
        }

    def query_missingness(self, prefecture: str, municipality: str) -> dict[str, Any]:
        """This municipality's own null score components — never the
        study-area-wide list of affected municipalities."""

        row = next(
            (r for r in self._ranking()
             if r["prefecture"] == prefecture and r["municipality"] == municipality),
            None,
        )
        grades = (row or {}).get("component_evidence_grades") or {}
        missing = [k for k in _SCORE_COMPONENT_KEYS if grades.get(k) == "not_available"]
        return {
            "prefecture": prefecture, "municipality": municipality,
            "missing_components": missing,
            "evidence_refs": [f"scores_by_key:{prefecture}:{municipality}"] if row else [],
        }

    def query_local_comparators(
        self,
        proposal: dict[str, Any],
        *,
        same_prefecture: bool = True,
        same_action: bool = True,
        top_n: int = 5,
    ) -> list[dict[str, Any]]:
        ranking = self._ranking()
        own_id = proposal.get("candidate_id")
        rows = [r for r in ranking if r["candidate_id"] != own_id]
        if same_prefecture:
            rows = [r for r in rows if r["prefecture"] == proposal.get("prefecture")] or rows
        if same_action:
            rows = [r for r in rows if r["action_type"] == proposal.get("action_type")] or rows
        rows = sorted(rows, key=lambda r: abs(r["rank"] - (proposal.get("rank") or 0)))
        return [
            {"candidate_id": r["candidate_id"], "municipality": r["municipality"],
             "prefecture": r["prefecture"], "action_type": r["action_type"],
             "rank": r["rank"], "composite_score": r["composite_score"]}
            for r in rows[:top_n]
        ]

    # -- reviewer-facing entry point ---------------------------------------

    def answer_data_request(
        self,
        request: DataRequest,
        dossier: CandidateDossier,
    ) -> DataObservation:
        if request.candidate_id and request.candidate_id != dossier.candidate_id:
            return DataObservation(
                reviewer_id=request.reviewer_id,
                candidate_id=dossier.candidate_id,
                question=request.question,
                observations={"out_of_scope": True,
                              "reason": "request targets a different candidate_id"},
                evidence_refs=[],
            )

        question = (request.question or "").lower()
        fields = {f.lower() for f in request.requested_fields}
        observations: dict[str, Any] = {}
        refs: list[str] = []

        def _wants(*keywords: str) -> bool:
            return any(k in question for k in keywords) or bool(fields & set(keywords))

        if _wants("facilit", "hospital", "yahoo"):
            facilities = self.query_facilities(dossier.prefecture, dossier.municipality)
            observations["facilities"] = facilities
            refs.append(f"facilities_by_key:{dossier.prefecture}:{dossier.municipality}")
        if _wants("municip", "census", "population", "land", "cost", "facts"):
            result = self.query_municipal_facts(dossier.prefecture, dossier.municipality)
            observations["municipal_facts"] = result["facts"]
            refs.extend(result["evidence_refs"])
        if _wants("rank", "comparator", "neighbor", "peer", "percentile"):
            context = self.query_candidate_rank_context(dossier.candidate_id)
            observations["rank_context"] = context
            refs.extend(context.get("evidence_refs") or [])
            observations["local_comparators"] = dossier.local_comparators
        if _wants("missing", "gap", "availability", "complete"):
            missing = self.query_missingness(dossier.prefecture, dossier.municipality)
            observations["missingness"] = missing
            refs.extend(missing["evidence_refs"])
        if not observations:
            # Default: hand back the dossier's own facts so a vague question
            # still gets a grounded, in-scope answer instead of nothing.
            observations["dossier_summary"] = {
                "score_components": dossier.score_components,
                "evidence_grades": dossier.evidence_grades,
                "evidence_gaps": dossier.evidence_gaps,
            }
            refs.append(f"dossier:{dossier.candidate_id}")

        return DataObservation(
            reviewer_id=request.reviewer_id,
            candidate_id=dossier.candidate_id,
            question=request.question,
            observations=observations,
            evidence_refs=sorted(set(refs)),
        )
