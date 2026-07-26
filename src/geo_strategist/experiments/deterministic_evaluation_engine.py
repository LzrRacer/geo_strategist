"""Deterministic evaluation-model engine over the local evidence base.

This module is the *data* layer of the condition track, not a condition
strategy multiplexer: it loads the shared interim datasets (municipality
scores, candidate actions, evidence bundles, facility records, hospital
financial features), computes evidence-graded evaluation-model components,
and builds provenance-checked proposal records.

It is used three ways:

- C0 runs it directly as the deterministic non-LLM baseline.
- Live conditions (C1/C4, C13-C14) use its components as the *external* metric
  that steers and scores LLM-driven search, and its proposal builder to turn
  validated LLM slates into evidence-graded proposals.
- Debug-only fallback reports reuse it, always marked
  ``deterministic_fallback`` and never comparable for E13.

Earlier versions dispatched every C-condition through per-condition
"strategy profiles" here, which collapsed conditions into nearly identical
deterministic rankings; that layer has been removed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any

from geo_strategist.agent.concrete_evidence import (
    FacilityEvidenceRecord,
    evidence_refs_for_facility,
    load_facility_evidence_records,
    records_usable_for_concrete_proposals,
)
from geo_strategist.agent.proposal_agent import build_experimental_proposal
from geo_strategist.agent.reviewer_ensemble import (
    review_proposal,
    revision_requests,
)
from geo_strategist.experiments.condition_utils import _read_jsonl, _stable_id

STUDY_AREA_DIR = Path(".data/interim/study_area/tokyo_aichi_osaka")

# Generic default weights over evaluation-model components; the data decides
# the ranking. Branch objectives apply emphasis multipliers over these.
DEFAULT_WEIGHTS: dict[str, float] = {
    "demand": 0.20,
    "aging": 0.15,
    "supply_shortage": 0.20,
    "financial": 0.15,
    "land": 0.10,
    "demographic_risk": 0.10,
    "evidence_completeness": 0.10,
}


@dataclass
class DataBundle:
    scores_by_key: dict[tuple[str, str], dict[str, Any]]
    candidates: list[dict[str, Any]]
    bundles_by_candidate: dict[str, dict[str, Any]]
    features_by_key: dict[tuple[str, str], dict[str, Any]]
    financial_by_prefecture: dict[str, dict[str, Any]]
    facilities_by_key: dict[tuple[str, str], list[FacilityEvidenceRecord]]
    enriched_scores_used: bool
    data_notes: list[str] = field(default_factory=list)
    # Raw per-municipality facts (real sourced figures for the report tables):
    # census population projections, MLIT land prices, Yahoo facility counts.
    municipal_facts_by_key: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    # Prefecture-level hospital cost-model medians from the cash-flow workbook
    # (data_basis = model estimate; used for indicative cost tables only).
    cost_model_by_prefecture: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Nationwide census-view coverage counts for the selection-funnel table.
    national_coverage: dict[str, Any] = field(default_factory=dict)
    # Per-bundle memo for derived values that are identical across a run
    # (default ranking, municipality anchors); never serialized.
    runtime_cache: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class DeterministicOutcome:
    condition_id: str
    condition_group: str
    weights_used: dict[str, float]
    candidates_ranked: list[dict[str, Any]]
    proposals: list[dict[str, Any]]
    review_rows: list[dict[str, Any]]
    due_diligence_items: list[str]
    data_notes: list[str]


def load_data_bundle(repo_root: str | Path = ".") -> DataBundle:
    root = Path(repo_root).resolve()
    area = root / STUDY_AREA_DIR
    notes: list[str] = []

    enriched = _read_jsonl(area / "municipality_scores_enriched.jsonl")
    plain = _read_jsonl(area / "municipality_scores.jsonl")
    score_rows = enriched or plain
    if not enriched:
        notes.append("enriched municipality scores unavailable; using base score layer")
    scores_by_key = {(row["prefecture"], row["municipality"]): row for row in score_rows}

    candidates = _read_jsonl(area / "candidate_actions.jsonl")
    if not candidates:
        notes.append("candidate_actions.jsonl missing; no upstream candidates available")
    bundles = _read_jsonl(area / "candidate_evidence_bundles.jsonl")
    bundles_by_candidate = {row["candidate_id"]: row for row in bundles}

    features = _read_jsonl(area / "municipality_feature_base.jsonl")
    features_by_key = {(row["prefecture"], row["municipality"]): row for row in features}

    financial_by_prefecture = _prefecture_financials(_read_jsonl(area / "hospital_features.jsonl"), notes)

    facility_records, _issues = load_facility_evidence_records(root)
    usable = records_usable_for_concrete_proposals(facility_records)
    facilities_by_key: dict[tuple[str, str], list[FacilityEvidenceRecord]] = {}
    for record in usable:
        facilities_by_key.setdefault((record.prefecture, record.municipality), []).append(record)
    if not usable:
        notes.append("no source-traceable facility records; reorganize/consolidate targets stay unverified")

    return DataBundle(
        scores_by_key=scores_by_key,
        candidates=candidates,
        bundles_by_candidate=bundles_by_candidate,
        features_by_key=features_by_key,
        financial_by_prefecture=financial_by_prefecture,
        facilities_by_key=facilities_by_key,
        enriched_scores_used=bool(enriched),
        data_notes=notes,
        municipal_facts_by_key=_load_municipal_facts(area, notes),
        cost_model_by_prefecture=_prefecture_cost_model(
            _read_jsonl(area / "hospital_features.jsonl")),
        national_coverage=_national_census_coverage(root),
    )


# (path, mtime_ns, size) -> coverage counts; the 68k-line nationwide view is
# parsed once per file version instead of on every bundle load.
_NATIONAL_COVERAGE_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _national_census_coverage(root: Path) -> dict[str, Any]:
    """Nationwide coverage counts of the normalized census projection view.

    Feeds the selection-funnel table: how many prefectures/municipalities the
    e-Stat census projection workbook covers before the study-area filter.
    Counts only — the nationwide view itself stays out of the bundle.
    """

    path = root / ".data/interim/views/population_long.jsonl"
    if not path.exists():
        return {}
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _NATIONAL_COVERAGE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)
    prefectures: set[str] = set()
    municipalities: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            prefecture = row.get("prefecture") or row.get("raw_prefecture")
            municipality = row.get("municipality") or row.get("normalized_municipality")
            if prefecture:
                prefectures.add(str(prefecture))
            if prefecture and municipality:
                municipalities.add((str(prefecture), str(municipality)))
    if not prefectures:
        return {}
    coverage = {
        "prefecture_count": len(prefectures),
        "municipality_count": len(municipalities),
        "source": ".data/interim/views/population_long.jsonl "
                  "(normalized census projection workbook)",
    }
    _NATIONAL_COVERAGE_CACHE[cache_key] = coverage
    return dict(coverage)


def _load_municipal_facts(area: Path, notes: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """Real per-municipality figures for the report data tables.

    Sources: census population projections (population_features.jsonl),
    MLIT land prices (municipality_land_features.jsonl), and Yahoo Local
    Search facility counts (municipality_healthcare_supply_features.jsonl).
    Values are reproduced as-is with their source labels; nothing is imputed.
    """

    population = {(r["prefecture"], r["municipality"]): r
                  for r in _read_jsonl(area / "population_features.jsonl")}
    enriched = {(r["prefecture"], r["municipality"]): r
                for r in _read_jsonl(area / "municipality_feature_base_enriched.jsonl")}
    land = {(r["prefecture"], r["municipality"]): r
            for r in _read_jsonl(area / "municipality_land_features.jsonl")}
    supply = {(r["prefecture"], r["municipality"]): r
              for r in _read_jsonl(area / "municipality_healthcare_supply_features.jsonl")}
    if not population:
        notes.append("population_features.jsonl unavailable; raw population figures marked not_available")

    facts: dict[tuple[str, str], dict[str, Any]] = {}
    for key in set(population) | set(enriched) | set(land) | set(supply):
        pop = population.get(key, {})
        enr = enriched.get(key, {})
        land_row = land.get(key, {})
        supply_row = supply.get(key, {})
        totals = pop.get("population_total_by_year") or {}
        seniors = pop.get("population_65_plus_by_year") or {}

        def _year_value(by_year: dict[str, Any], year: str) -> float | None:
            value = by_year.get(year)
            return float(value) if isinstance(value, (int, float)) else None

        facts[key] = {
            "population_total_2025": _year_value(totals, "2025"),
            "population_total_2050": _year_value(totals, "2050"),
            "population_65_plus_2025": _year_value(seniors, "2025"),
            "share_65_plus_2025_pct": _share(seniors, totals, "2025"),
            "share_65_plus_2050_pct": _share(seniors, totals, "2050"),
            "population_pct_change_2020_2050": enr.get("population_total_pct_change"),
            "population_source": "census projection workbook (kekkahyo, 2020-2050)",
            "land_price_median_jpy_per_sqm": land_row.get("land_price_median"),
            "land_price_sample_count": land_row.get("land_price_record_count"),
            "land_price_year": land_row.get("land_price_latest_year"),
            "land_price_source": "MLIT Real Estate Information Library (reinfolib)",
            "hospital_count": supply_row.get("hospital_count"),
            "supply_density_per_100k": supply_row.get("supply_density_per_100k"),
            "supply_source": "Yahoo Local Search API facility records",
            "evidence_grade": "verified_source",
        }
    return facts


def _share(seniors: dict[str, Any], totals: dict[str, Any], year: str) -> float | None:
    senior = seniors.get(year)
    total = totals.get(year)
    if isinstance(senior, (int, float)) and isinstance(total, (int, float)) and total:
        return round(100.0 * float(senior) / float(total), 1)
    return None


def _prefecture_cost_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Indicative per-prefecture cost medians from the hospital cash-flow
    workbook (data_basis: model estimate). Never a substitute for candidate-
    specific cost due diligence; reports label every figure model_estimate."""

    per_pref: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        prefecture = row.get("prefecture")
        beds = row.get("beds_used_in_model")
        if not prefecture or not isinstance(beds, (int, float)) or beds <= 0:
            continue
        bucket = per_pref.setdefault(prefecture, {
            "land_price_jpy_per_sqm": [], "construction_cost_per_bed_jpy_mm": [],
            "equipment_it_per_bed_jpy_mm": [], "initial_investment_per_bed_jpy_mm": [],
            "payback_years": [],
        })
        if isinstance(row.get("land_price_jpy_per_sqm"), (int, float)):
            bucket["land_price_jpy_per_sqm"].append(float(row["land_price_jpy_per_sqm"]))
        for source_field, target in (
            ("construction_cost_jpy_mm", "construction_cost_per_bed_jpy_mm"),
            ("equipment_it_jpy_mm", "equipment_it_per_bed_jpy_mm"),
            ("initial_investment_jpy_mm", "initial_investment_per_bed_jpy_mm"),
        ):
            value = row.get(source_field)
            if isinstance(value, (int, float)) and value > 0:
                bucket[target].append(float(value) / float(beds))
        if isinstance(row.get("payback_years"), (int, float)) and row["payback_years"] > 0:
            bucket["payback_years"].append(float(row["payback_years"]))

    result: dict[str, dict[str, Any]] = {}
    for prefecture, buckets in per_pref.items():
        row = {"evidence_grade": "model_estimate",
               "source": "hospital cash-flow workbook (data_basis: モデル推計)",
               "sample_size": max((len(v) for v in buckets.values()), default=0)}
        for name, values in buckets.items():
            row[f"median_{name}"] = round(median(values), 1) if values else None
        result[prefecture] = row
    return result


def _prefecture_financials(rows: list[dict[str, Any]], notes: list[str]) -> dict[str, dict[str, Any]]:
    """Financial-plausibility model estimate per prefecture.

    Derived from the hospital cash-flow workbook features (data_basis is a
    model estimate); shorter median payback maps to a higher score.
    """

    by_pref: dict[str, list[float]] = {}
    for row in rows:
        payback = row.get("payback_years")
        prefecture = row.get("prefecture")
        if prefecture and isinstance(payback, (int, float)) and payback > 0:
            by_pref.setdefault(prefecture, []).append(float(payback))
    if not by_pref:
        notes.append("hospital financial features unavailable; financial component marked not_available")
        return {}
    result: dict[str, dict[str, Any]] = {}
    for prefecture, values in by_pref.items():
        med = median(values)
        # 10y payback -> ~1.0, 50y -> ~0.0 (clamped); model estimate only.
        score = max(0.0, min(1.0, 1.0 - (med - 10.0) / 40.0))
        result[prefecture] = {
            "median_payback_years": round(med, 2),
            "financial_score": round(score, 4),
            "sample_size": len(values),
            "evidence_grade": "model_estimate",
        }
    return result


def components_for(candidate: dict[str, Any], data: DataBundle) -> tuple[dict[str, float | None], dict[str, str]]:
    """Evaluation-model components + per-component evidence grades."""

    key = (candidate["prefecture"], candidate["municipality"])
    score_row = data.scores_by_key.get(key, {})
    feature_row = data.features_by_key.get(key, {})
    financial = data.financial_by_prefecture.get(candidate["prefecture"], {})

    hcs = score_row.get("healthcare_supply_score")
    action = candidate.get("candidate_action")
    supply_shortage: float | None
    if hcs is None:
        supply_shortage = None
    elif action == "build":
        supply_shortage = 1.0 - float(hcs)
    else:
        # reorganize/consolidate favor areas with dense existing supply
        supply_shortage = float(hcs)

    decline = feature_row.get("population_total_pct_change")
    demographic_risk: float | None = None
    if isinstance(decline, (int, float)):
        # growing/stable population -> low risk -> high score
        demographic_risk = max(0.0, min(1.0, 1.0 + float(decline)))

    components: dict[str, float | None] = {
        "demand": score_row.get("demand_pressure_score"),
        "aging": score_row.get("population_aging_pressure_score"),
        "supply_shortage": supply_shortage,
        "financial": financial.get("financial_score"),
        "land": score_row.get("land_score") if score_row.get("land_score_available") else None,
        "demographic_risk": demographic_risk,
        "evidence_completeness": score_row.get("data_completeness_score"),
    }
    grades = {
        "demand": "derived_from_verified_source" if components["demand"] is not None else "not_available",
        "aging": "derived_from_verified_source" if components["aging"] is not None else "not_available",
        "supply_shortage": "derived_from_verified_source" if supply_shortage is not None else "not_available",
        "financial": financial.get("evidence_grade", "not_available"),
        "land": "verified_source" if components["land"] is not None else "not_available",
        "demographic_risk": "derived_from_verified_source" if demographic_risk is not None else "not_available",
        "evidence_completeness": "derived_from_verified_source" if components["evidence_completeness"] is not None else "not_available",
    }
    return components, grades


def composite_score(components: dict[str, float | None], weights: dict[str, float]) -> tuple[float, float]:
    """Weighted composite over available components, renormalized.

    Returns (composite, availability_ratio); missing components lower the
    availability ratio and are surfaced as evidence gaps, but never abort
    the ranking.
    """

    total_weight = 0.0
    acc = 0.0
    available = 0
    for name, weight in weights.items():
        value = components.get(name)
        if value is None:
            continue
        acc += weight * float(value)
        total_weight += weight
        available += 1
    if total_weight == 0.0:
        return 0.0, 0.0
    return round(acc / total_weight, 6), round(available / len(weights), 4)


def rank_candidates(
    data: DataBundle,
    weights: dict[str, float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in data.candidates:
        components, grades = components_for(candidate, data)
        score, availability = composite_score(components, weights)
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "prefecture": candidate["prefecture"],
            "municipality": candidate["municipality"],
            "action_type": candidate["candidate_action"],
            "composite_score": score,
            "component_availability": availability,
            "score_components": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in components.items()},
            "component_evidence_grades": grades,
            "upstream_priority_score": candidate.get("priority_score"),
        })
    rows.sort(key=lambda row: (-row["composite_score"], row["candidate_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return rows


def slate_objective(ranked: list[dict[str, Any]], top_k: int) -> float:
    """Quality of a top-k slate: mean composite weighted by mean availability,
    so a slate cannot win by ranking data-poor candidates highly."""

    head = ranked[:top_k]
    if not head:
        return 0.0
    return round(
        mean(row["composite_score"] for row in head) * mean(row["component_availability"] for row in head),
        6,
    )


_NON_HOSPITAL_NAME_RE = re.compile(r"動物|ペット|獣医|犬猫|わんにゃん")


def match_facility(candidate_row: dict[str, Any], data: DataBundle) -> FacilityEvidenceRecord | None:
    if candidate_row["action_type"] == "build":
        return None
    records = data.facilities_by_key.get((candidate_row["prefecture"], candidate_row["municipality"])) or []
    human = [r for r in records if r.facility_name and not _NON_HOSPITAL_NAME_RE.search(r.facility_name)]
    hospitals = [r for r in human if "病院" in r.facility_name]
    if hospitals:
        return hospitals[0]
    return human[0] if human else None


def proposal_for_candidate(
    candidate_row: dict[str, Any],
    data: DataBundle,
    *,
    condition_id: str,
    condition_group: str,
) -> dict[str, Any]:
    """Build an evidence-graded proposal record for one ranked candidate."""

    facility = match_facility(candidate_row, data)
    bundle = data.bundles_by_candidate.get(candidate_row["candidate_id"], {})
    grades: dict[str, str] = dict(candidate_row["component_evidence_grades"])
    unsupported: list[str] = []
    gaps: list[str] = []
    due_diligence: list[str] = []

    if facility is not None:
        refs = evidence_refs_for_facility(facility)
        name = facility.facility_name
        address = facility.facility_address if "facility_address" in facility.source_fields else "not_available"
        if facility.latitude is not None and "coordinates" in facility.source_fields:
            coordinates: dict[str, Any] | str = {"latitude": facility.latitude, "longitude": facility.longitude}
        else:
            coordinates = "not_available"
        exact_address_status = "source_traceable" if address != "not_available" else "not_available"
        grades["target_facility_name"] = "verified_source"
        if address != "not_available":
            grades["target_facility_address"] = "verified_source"
        due_diligence.append(
            f"Confirm on site that {name} is a suitable {candidate_row['action_type']} target; "
            "source is a public facility-search record, not an operator agreement."
        )
    else:
        refs = []
        name = None
        address = "not_available"
        coordinates = "not_available"
        exact_address_status = "not_available"
        unsupported = ["target_facility_name", "target_facility_address", "target_coordinates"]
        if candidate_row["action_type"] == "build":
            due_diligence.append(
                f"Identify and verify buildable parcels within {candidate_row['municipality']} "
                "(zoning, ownership, disaster risk); this proposal is municipality-level."
            )
        else:
            gaps.append("no_source_traceable_facility_in_municipality")
            due_diligence.append(
                f"Identify the concrete {candidate_row['action_type']} target facility in "
                f"{candidate_row['municipality']} through a facility registry search."
            )

    for component, grade in candidate_row["component_evidence_grades"].items():
        if grade == "not_available":
            gaps.append(f"{component}_component_not_available")
        elif grade == "model_estimate":
            due_diligence.append(f"Validate the {component} model estimate against audited financials.")

    for limitation in bundle.get("known_limitations") or []:
        item = limitation if isinstance(limitation, str) else str(limitation)
        due_diligence.append(item)

    proposal = build_experimental_proposal(
        proposal_id=_stable_id(f"{condition_group.lower()}_proposal", {
            "condition_id": condition_id,
            "candidate_id": candidate_row["candidate_id"],
        }),
        condition_id=condition_id,
        condition_group=condition_group,
        action_type=candidate_row["action_type"],
        prefecture=candidate_row["prefecture"],
        municipality=candidate_row["municipality"],
        target_facility_name=name,
        target_facility_address=address,
        target_coordinates=coordinates,
        exact_address_status=exact_address_status,
        source_evidence_refs=refs,
        evidence_grades=grades,
        unsupported_fields=unsupported,
        evidence_gaps=sorted(set(gaps)),
        required_due_diligence=due_diligence,
    ).model_dump()
    proposal["candidate_id"] = candidate_row["candidate_id"]
    proposal["rank"] = candidate_row["rank"]
    proposal["composite_score"] = candidate_row["composite_score"]
    proposal["score_components"] = candidate_row["score_components"]
    proposal["source_artifact_refs"] = bundle.get("source_artifact_refs") or []
    return proposal


def apply_reviews(
    proposals: list[dict[str, Any]],
    review_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Persona reviewer-ensemble passes; findings extend the due-diligence
    list rather than blocking the proposal."""

    review_rows: list[dict[str, Any]] = []
    for round_index in range(max(review_rounds, 0)):
        for proposal in proposals:
            reviews = review_proposal(proposal)
            requests = revision_requests(reviews)
            review_rows.extend({**review.to_dict(), "review_round": round_index + 1} for review in reviews)
            proposal["reviewer_scores"] = {review.reviewer_id: review.score for review in reviews}
            proposal["aggregate_score"] = round(
                mean(review.score for review in reviews), 4
            ) if reviews else None
            proposal["review_summary"] = (
                f"round {round_index + 1}: {len(requests)} revision items from "
                f"{len(reviews)} persona reviewers"
            )
            if requests:
                existing = set(proposal.get("required_due_diligence") or [])
                for request in requests:
                    item = f"Reviewer follow-up: {request.replace('_', ' ')}"
                    if item not in existing:
                        proposal.setdefault("required_due_diligence", []).append(item)
                        existing.add(item)
                proposal["revision_summary"] = f"revised to disclose {len(requests)} reviewer findings"
                proposal["proposal_status"] = "revised_proposal"
            else:
                proposal["revision_summary"] = "no revision required"
            proposal["keep_discard_decision"] = "keep"
    if review_rounds <= 0:
        for proposal in proposals:
            proposal["keep_discard_decision"] = "keep"
            proposal["review_summary"] = "no self-review in this condition"
    return proposals, review_rows


def build_deterministic_outcome(
    repo_root: str | Path,
    *,
    condition_id: str = "deterministic_python_baseline",
    condition_group: str = "C0",
    top_k: int = 5,
    review_rounds: int = 0,
    data: DataBundle | None = None,
) -> DeterministicOutcome:
    """The C0 baseline (also reused, explicitly labeled, for debug fallbacks)."""

    bundle = data or load_data_bundle(repo_root)
    ranked = rank_candidates(bundle, DEFAULT_WEIGHTS)
    proposals = [
        proposal_for_candidate(row, bundle, condition_id=condition_id, condition_group=condition_group)
        for row in ranked[:top_k]
    ]
    proposals, review_rows = apply_reviews(proposals, review_rounds)

    due_diligence: list[str] = []
    for proposal in proposals:
        due_diligence.extend(proposal.get("required_due_diligence") or [])
    due_diligence.extend(bundle.data_notes)

    return DeterministicOutcome(
        condition_id=condition_id,
        condition_group=condition_group,
        weights_used=dict(DEFAULT_WEIGHTS),
        candidates_ranked=ranked,
        proposals=proposals,
        review_rows=review_rows,
        due_diligence_items=list(dict.fromkeys(due_diligence)),
        data_notes=list(bundle.data_notes),
    )
