"""S4 agentic tree search over real candidate sites.

Unlike the old E7 tree search (which branched over rewritten proposal text
for a small set of municipality/action pairs), this tree explores concrete
candidate-site hypotheses from S2/S3:

    root (region + investment objective)
      -> municipality branch
        -> site branch (real candidate_site_id, real address/coords when available)
          -> facility-concept / service-line hypothesis branch
            -> financial-scenario branch (base/downside/upside)

Every node score is computed deterministically from S3 feature values; no
LLM call invents a site fact, an address, or a financial figure. Multiple
named expansion strategies re-rank the same real candidate pool from
different angles; best-first selection keeps the top `beam_width` sites per
municipality and the overall `top_k_sites` leaf scenarios.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from geo_strategist.data_sources.evidence_grade import worst_grade
from geo_strategist.experiments.s3_site_feature_engineering import OUTPUT_ROOT as S3_OUTPUT_ROOT


OUTPUT_ROOT = Path(".runs/experiments/s4_site_tree_search")
CASH_FLOW_ASSUMPTIONS_PATH = Path("configs/cash_flow_assumptions.yaml")

SERVICE_LINE_HYPOTHESES = {
    "build": "new_emergency_and_acute_care_center_concept",
    "rebuild": "rebuilt_acute_care_facility_concept",
    "relocate": "relocated_acute_care_facility_concept",
    "consolidate": "consolidated_acute_care_hub_concept",
    "acquire": "acquired_facility_conversion_concept",
    "lease": "leased_facility_conversion_concept",
    "reject": "not_applicable_rejected",
    "defer": "not_applicable_deferred",
}

EXPANSION_STRATEGIES = (
    "demand_gap_expansion",
    "accessibility_expansion",
    "financial_return_expansion",
    "risk_conservative_expansion",
    "portfolio_diversification_expansion",
    "competitive_white_space_expansion",
    "due_diligence_expansion",
)

SCORE_WEIGHTS = {
    "demand_need_score": 0.17,
    "supply_gap_score": 0.13,
    "catchment_pressure_score": 0.10,
    "emergency_coverage_gap_score": 0.05,
    "financial_plausibility_score": 0.18,
    "capex_efficiency_score": 0.14,
    "operational_feasibility_score": 0.10,
    "risk_adjusted_score": 0.08,
    "evidence_completeness_score": 0.05,
}


@dataclass(frozen=True)
class S4Result:
    run_id: str
    output_dir: Path
    node_count: int
    selected_count: int
    top_k_candidate_site_ids: list[str]
    output_paths: dict[str, str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, canonical)}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _latest_s3_run(repo_root: Path) -> Path | None:
    root = repo_root / S3_OUTPUT_ROOT
    if not root.exists():
        return None
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "s3_site_feature_records.jsonl").exists():
            return candidate
    return None


def _load_cash_flow_assumptions(repo_root: Path) -> dict[str, Any]:
    path = repo_root / CASH_FLOW_ASSUMPTIONS_PATH
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _min_max_normalize(values: list[float | None]) -> list[float]:
    finite = [value for value in values if value is not None]
    if not finite:
        return [0.5 for _ in values]
    lo, hi = min(finite), max(finite)
    if hi == lo:
        return [0.5 if value is not None else 0.0 for value in values]
    return [round((value - lo) / (hi - lo), 4) if value is not None else 0.0 for value in values]


def _payback_score(payback_years: float | None, target_payback_years: float) -> float:
    if payback_years is None or payback_years <= 0:
        return 0.0
    return round(min(1.0, target_payback_years / payback_years), 4)


def _feature_value(features: dict[str, Any], group: str, key: str) -> Any:
    return (features.get(group, {}).get(key) or {}).get("value")


def _emergency_gap_value(features: dict[str, Any]) -> float | None:
    distance = _feature_value(features, "supply_features", "distance_to_nearest_emergency_hospital_km")
    if distance is not None:
        return float(distance)
    count = _feature_value(features, "supply_features", "ambulance_emergency_catchment_count_15km")
    if count is not None:
        return -float(count)
    return None


def _catchment_pressure_value(features: dict[str, Any]) -> float | None:
    density = _feature_value(features, "supply_features", "hospital_density_per_100k_population")
    medical_count = _feature_value(features, "supply_features", "medical_catchment_hospital_count_10km")
    underserved = _feature_value(features, "supply_features", "underserved_area_signal")
    parts: list[float] = []
    if density is not None:
        parts.append(-float(density))
    if medical_count is not None:
        parts.append(-float(medical_count))
    if underserved is not None:
        parts.append(1.0 if underserved else 0.0)
    if not parts:
        return None
    return sum(parts) / len(parts)


def _apply_scenario_multipliers(financial_features: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    revenue = financial_features["estimated_annual_revenue_JPY_mm"]["value"]
    opex = financial_features["estimated_annual_opex_JPY_mm"]["value"]
    capex = financial_features["estimated_capex_JPY_mm"]["value"]
    revenue_mult = scenario.get("revenue_multiplier", 1.0)
    expense_mult = scenario.get("expense_multiplier", 1.0)
    capex_mult = scenario.get("capex_multiplier", 1.0)
    scenario_revenue = round(revenue * revenue_mult, 4) if revenue is not None else None
    scenario_opex = round(opex * expense_mult, 4) if opex is not None else None
    scenario_capex = round(capex * capex_mult, 4) if capex is not None else None
    scenario_ebitda = round(scenario_revenue - scenario_opex, 4) if scenario_revenue is not None and scenario_opex is not None else None
    scenario_payback = round(scenario_capex / scenario_ebitda, 4) if scenario_capex and scenario_ebitda and scenario_ebitda > 0 else None
    return {
        "estimated_annual_revenue_JPY_mm": scenario_revenue,
        "estimated_annual_opex_JPY_mm": scenario_opex,
        "estimated_capex_JPY_mm": scenario_capex,
        "estimated_ebitda_JPY_mm": scenario_ebitda,
        "payback_years_scenario": scenario_payback,
        "scenario_multipliers": scenario,
        "formula": "value = base_feature_value * scenario_multiplier (see configs/cash_flow_assumptions.yaml#sensitivity_scenarios)",
        "assumption_refs": ["configs/cash_flow_assumptions.yaml#sensitivity_scenarios"],
    }


def _expansion_strategy_votes(
    candidate_id: str,
    demand_need: float,
    supply_gap: float,
    financial_plausibility: float,
    risk_adjusted: float,
    evidence_completeness: float,
    prefecture: str,
    top_prefecture_pick: dict[str, str],
) -> list[str]:
    votes: list[str] = []
    if demand_need >= 0.7:
        votes.append("demand_gap_expansion")
    if financial_plausibility >= 0.7:
        votes.append("financial_return_expansion")
    if risk_adjusted >= 0.7 and evidence_completeness >= 0.7:
        votes.append("risk_conservative_expansion")
    if supply_gap >= 0.7:
        votes.append("competitive_white_space_expansion")
    if evidence_completeness < 0.5:
        votes.append("due_diligence_expansion")
    if top_prefecture_pick.get(prefecture) == candidate_id:
        votes.append("portfolio_diversification_expansion")
    votes.append("accessibility_expansion")  # always neutral: no accessibility dataset in this environment
    return votes


def run_s4_site_tree_search(
    repo_root: str | Path = ".",
    *,
    s2_run_dir: str | Path | None = None,
    s3_run_dir: str | Path | None = None,
    beam_width: int = 5,
    max_depth: int = 4,
    top_k_sites: int = 10,
    min_evidence_score: float = 0.0,
    output_root: str | Path | None = None,
) -> S4Result:
    repo_root = Path(repo_root).resolve()

    if s3_run_dir:
        s3_dir = Path(s3_run_dir)
        if not s3_dir.is_absolute():
            s3_dir = repo_root / s3_dir
    else:
        s3_dir = _latest_s3_run(repo_root)
    s3_manifest = _read_json(s3_dir / "s3_manifest.json") if s3_dir else {}
    if s2_run_dir:
        s2_dir = Path(s2_run_dir)
        if not s2_dir.is_absolute():
            s2_dir = repo_root / s2_dir
    elif s3_manifest.get("input_s2_run_dir"):
        s2_dir = repo_root / s3_manifest["input_s2_run_dir"]
    else:
        s2_dir = None

    candidates = _read_jsonl(s2_dir / "s2_candidate_site_records.jsonl") if s2_dir else []
    feature_records = {row["candidate_site_id"]: row for row in _read_jsonl(s3_dir / "s3_site_feature_records.jsonl")} if s3_dir else {}
    assumptions = _load_cash_flow_assumptions(repo_root)
    target_payback_years = float(assumptions.get("payback_status", {}).get("target_payback_years", {}).get("value", 20.0))
    sensitivity_scenarios = assumptions.get("sensitivity_scenarios", {"base": {"capex_multiplier": 1.0, "revenue_multiplier": 1.0, "expense_multiplier": 1.0}})

    run_id = str(uuid.uuid4())
    out_root = Path(output_root) if output_root else repo_root / OUTPUT_ROOT
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / run_id
    generated_at = _now_iso()

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    root_node_id = _stable_id("site_tree_root", {"run_id": run_id})
    nodes.append({
        "node_id": root_node_id,
        "parent_node_id": None,
        "depth": 0,
        "node_kind": "root",
        "candidate_site_id": None,
        "action_type": None,
        "service_line_hypothesis": None,
        "investment_hypothesis": "Identify and evaluate real candidate hospital sites across the study area for decision-support review.",
        "evidence_bundle": {},
        "financial_model_snapshot": {},
        "operational_plan_snapshot": {},
        "risk_register": [],
        "reviewer_feedback": None,
        "score_vector": {},
        "status": "expanded",
        "status_reason": None,
        "expansion_strategies_favoring_this_node": [],
        "generated_at": generated_at,
    })

    # Pre-compute pool-wide metrics for normalization.
    pool_demand = []
    pool_supply_inverse = []
    pool_capex_efficiency = []
    pool_catchment_pressure = []
    pool_emergency_gap = []
    candidate_ids_in_order = []
    for candidate in candidates:
        features = feature_records.get(candidate["candidate_site_id"], {})
        demand = _feature_value(features, "demand_features", "elderly_demand_proxy_score")
        nearby = _feature_value(features, "supply_features", "nearby_hospital_like_count_3km")
        revenue = _feature_value(features, "financial_features", "estimated_annual_revenue_JPY_mm")
        capex = _feature_value(features, "financial_features", "estimated_capex_JPY_mm")
        capex_efficiency = (revenue / capex) if revenue is not None and capex else None
        pool_demand.append(demand)
        pool_supply_inverse.append(-nearby if nearby is not None else None)
        pool_capex_efficiency.append(capex_efficiency)
        pool_catchment_pressure.append(_catchment_pressure_value(features))
        pool_emergency_gap.append(_emergency_gap_value(features))
        candidate_ids_in_order.append(candidate["candidate_site_id"])

    demand_need_scores = dict(zip(candidate_ids_in_order, _min_max_normalize(pool_demand)))
    supply_gap_scores = dict(zip(candidate_ids_in_order, _min_max_normalize(pool_supply_inverse)))
    capex_efficiency_scores = dict(zip(candidate_ids_in_order, _min_max_normalize(pool_capex_efficiency)))
    catchment_pressure_scores = dict(zip(candidate_ids_in_order, _min_max_normalize(pool_catchment_pressure)))
    emergency_gap_scores = dict(zip(candidate_ids_in_order, _min_max_normalize(pool_emergency_gap)))

    # Determine one "top pick" per prefecture for the diversification strategy.
    top_prefecture_pick: dict[str, str] = {}
    for candidate_id in candidate_ids_in_order:
        candidate = next(row for row in candidates if row["candidate_site_id"] == candidate_id)
        prefecture = candidate.get("prefecture") or "unknown"
        current_best = top_prefecture_pick.get(prefecture)
        if current_best is None or demand_need_scores[candidate_id] > demand_need_scores.get(current_best, -1):
            top_prefecture_pick[prefecture] = candidate_id

    municipality_node_ids: dict[str, str] = {}
    leaf_scores: list[tuple[str, float, str]] = []  # (node_id, overall_score, candidate_site_id)

    for candidate in candidates:
        candidate_id = candidate["candidate_site_id"]
        features = feature_records.get(candidate_id, {})
        municipality_key = f"{candidate.get('prefecture')}::{candidate.get('municipality') or 'unspecified'}"
        if municipality_key not in municipality_node_ids:
            muni_node_id = _stable_id("site_tree_municipality", {"run_id": run_id, "key": municipality_key})
            municipality_node_ids[municipality_key] = muni_node_id
            nodes.append({
                "node_id": muni_node_id,
                "parent_node_id": root_node_id,
                "depth": 1,
                "node_kind": "municipality",
                "candidate_site_id": None,
                "action_type": None,
                "service_line_hypothesis": None,
                "investment_hypothesis": f"Evaluate hospital-planning opportunities in {candidate.get('municipality') or 'an unspecified municipality'}, {candidate.get('prefecture')}.",
                "evidence_bundle": {},
                "financial_model_snapshot": {},
                "operational_plan_snapshot": {},
                "risk_register": [],
                "reviewer_feedback": None,
                "score_vector": {},
                "status": "expanded",
                "status_reason": None,
                "expansion_strategies_favoring_this_node": [],
                "generated_at": generated_at,
            })
            edges.append({
                "edge_id": _stable_id("site_tree_edge", {"parent": root_node_id, "child": muni_node_id}),
                "parent_node_id": root_node_id,
                "child_node_id": muni_node_id,
                "edge_kind": "region_to_municipality",
            })
        muni_node_id = municipality_node_ids[municipality_key]

        risk_features = features.get("risk_features", {})
        evidence_completeness = (risk_features.get("source_completeness_score") or {}).get("value") or 0.0
        risk_adjusted = round(1 - ((risk_features.get("uncertainty_score") or {}).get("value") or 0.0), 4)

        site_node_id = _stable_id("site_tree_site", {"run_id": run_id, "candidate_site_id": candidate_id})
        component_grades = [candidate.get("site_evidence_grade", "unverified_candidate")]
        site_status = "needs_more_data" if evidence_completeness < min_evidence_score else "expanded"
        nodes.append({
            "node_id": site_node_id,
            "parent_node_id": muni_node_id,
            "depth": 2,
            "node_kind": "site",
            "candidate_site_id": candidate_id,
            "action_type": candidate.get("action_type"),
            "service_line_hypothesis": None,
            "investment_hypothesis": f"{candidate.get('action_type')} hypothesis for candidate site {candidate_id}.",
            "evidence_bundle": {"candidate_site_record": candidate, "site_feature_record": features},
            "financial_model_snapshot": {},
            "operational_plan_snapshot": {},
            "risk_register": [{"issue": issue} for issue in candidate.get("blocking_issues", [])],
            "reviewer_feedback": None,
            "score_vector": {},
            "status": site_status,
            "status_reason": "evidence_completeness_below_min_evidence_score" if site_status == "needs_more_data" else None,
            "expansion_strategies_favoring_this_node": [],
            "generated_at": generated_at,
        })
        edges.append({
            "edge_id": _stable_id("site_tree_edge", {"parent": muni_node_id, "child": site_node_id}),
            "parent_node_id": muni_node_id,
            "child_node_id": site_node_id,
            "edge_kind": "municipality_to_site",
        })

        concept_node_id = _stable_id("site_tree_concept", {"run_id": run_id, "candidate_site_id": candidate_id})
        service_line = SERVICE_LINE_HYPOTHESES.get(candidate.get("action_type"), "unspecified_concept")
        nodes.append({
            "node_id": concept_node_id,
            "parent_node_id": site_node_id,
            "depth": 3,
            "node_kind": "facility_concept",
            "candidate_site_id": candidate_id,
            "action_type": candidate.get("action_type"),
            "service_line_hypothesis": service_line,
            "investment_hypothesis": f"{service_line} at {candidate.get('municipality') or 'municipality-level site (exact site pending)'}.",
            "evidence_bundle": {},
            "financial_model_snapshot": {},
            "operational_plan_snapshot": {
                "concept": service_line,
                "catchment_area_definition": candidate.get("catchment_area_definition"),
            },
            "risk_register": [],
            "reviewer_feedback": None,
            "score_vector": {},
            "status": "expanded",
            "status_reason": None,
            "expansion_strategies_favoring_this_node": [],
            "generated_at": generated_at,
        })
        edges.append({
            "edge_id": _stable_id("site_tree_edge", {"parent": site_node_id, "child": concept_node_id}),
            "parent_node_id": site_node_id,
            "child_node_id": concept_node_id,
            "edge_kind": "site_to_facility_concept",
        })

        demand_need = demand_need_scores.get(candidate_id, 0.5)
        supply_gap = supply_gap_scores.get(candidate_id, 0.5)
        catchment_pressure = catchment_pressure_scores.get(candidate_id, 0.5)
        emergency_gap = emergency_gap_scores.get(candidate_id, 0.5)
        capex_efficiency = capex_efficiency_scores.get(candidate_id, 0.5)
        strategies = _expansion_strategy_votes(
            candidate_id, demand_need, supply_gap, 0.5, risk_adjusted, evidence_completeness,
            candidate.get("prefecture") or "unknown", top_prefecture_pick,
        )

        best_scenario_node_id = None
        best_scenario_score = -1.0
        for scenario_name, scenario in sensitivity_scenarios.items():
            financial_features = features.get("financial_features", {})
            if not financial_features:
                continue
            scenario_snapshot = _apply_scenario_multipliers(financial_features, scenario)
            financial_plausibility = _payback_score(scenario_snapshot["payback_years_scenario"], target_payback_years)
            score_vector = {
                "demand_need_score": demand_need,
                "supply_gap_score": supply_gap,
                "catchment_pressure_score": catchment_pressure,
                "emergency_coverage_gap_score": emergency_gap,
                "accessibility_score": 0.5,
                "financial_plausibility_score": financial_plausibility,
                "capex_efficiency_score": capex_efficiency,
                "operational_feasibility_score": evidence_completeness,
                "risk_adjusted_score": risk_adjusted,
                "evidence_completeness_score": evidence_completeness,
                "strategic_fit_score": round((demand_need + supply_gap + catchment_pressure + emergency_gap) / 4, 4),
            }
            overall = round(sum(score_vector[key] * weight for key, weight in SCORE_WEIGHTS.items()), 4)
            score_vector["overall_site_selection_score"] = overall
            score_vector["catchment_metrics_used"] = {
                "hospital_density_per_100k_population": _feature_value(features, "supply_features", "hospital_density_per_100k_population"),
                "distance_to_nearest_major_hospital_km": _feature_value(features, "supply_features", "distance_to_nearest_major_hospital_km"),
                "distance_to_nearest_emergency_hospital_km": _feature_value(features, "supply_features", "distance_to_nearest_emergency_hospital_km"),
                "medical_catchment_hospital_count_10km": _feature_value(features, "supply_features", "medical_catchment_hospital_count_10km"),
                "ambulance_emergency_catchment_count_15km": _feature_value(features, "supply_features", "ambulance_emergency_catchment_count_15km"),
                "competition_intensity": _feature_value(features, "supply_features", "competition_intensity"),
                "underserved_area_signal": _feature_value(features, "supply_features", "underserved_area_signal"),
                "metric_basis": "geographic_distance_proxy_when_travel_time_unavailable",
            }

            scenario_node_id = _stable_id("site_tree_scenario", {"run_id": run_id, "candidate_site_id": candidate_id, "scenario": scenario_name})
            nodes.append({
                "node_id": scenario_node_id,
                "parent_node_id": concept_node_id,
                "depth": 4,
                "node_kind": "financial_scenario",
                "candidate_site_id": candidate_id,
                "action_type": candidate.get("action_type"),
                "service_line_hypothesis": service_line,
                "investment_hypothesis": f"{scenario_name} financial scenario for {service_line} at candidate {candidate_id}.",
                "evidence_bundle": {"site_evidence_grade": worst_grade(component_grades)},
                "financial_model_snapshot": {"scenario_name": scenario_name, **scenario_snapshot},
                "operational_plan_snapshot": {},
                "risk_register": [],
                "reviewer_feedback": None,
                "score_vector": score_vector,
                "status": "expanded",
                "status_reason": None,
                "expansion_strategies_favoring_this_node": strategies,
                "generated_at": generated_at,
            })
            edges.append({
                "edge_id": _stable_id("site_tree_edge", {"parent": concept_node_id, "child": scenario_node_id}),
                "parent_node_id": concept_node_id,
                "child_node_id": scenario_node_id,
                "edge_kind": "facility_concept_to_financial_scenario",
            })
            if scenario_name == "base":
                leaf_scores.append((scenario_node_id, overall, candidate_id))
            if overall > best_scenario_score:
                best_scenario_score = overall
                best_scenario_node_id = scenario_node_id

    # Best-first / beam selection: rank base-scenario leaves by overall score,
    # apply beam_width per municipality implicitly via top_k_sites overall cap.
    ranked_leaves = sorted(leaf_scores, key=lambda row: row[1], reverse=True)
    selected_candidate_ids: list[str] = []
    seen_municipalities: dict[str, int] = {}
    selected_node_ids: set[str] = set()
    for node_id, score, candidate_id in ranked_leaves:
        if len(selected_candidate_ids) >= top_k_sites:
            break
        candidate = next(row for row in candidates if row["candidate_site_id"] == candidate_id)
        muni_key = candidate.get("municipality") or "unspecified"
        if seen_municipalities.get(muni_key, 0) >= beam_width:
            continue
        seen_municipalities[muni_key] = seen_municipalities.get(muni_key, 0) + 1
        selected_candidate_ids.append(candidate_id)
        selected_node_ids.add(node_id)

    for node in nodes:
        if node["node_kind"] == "financial_scenario" and node["financial_model_snapshot"].get("scenario_name") == "base":
            if node["node_id"] in selected_node_ids:
                node["status"] = "selected"
            elif node["status"] == "expanded":
                node["status"] = "rejected"
                node["status_reason"] = "below_beam_width_or_top_k_sites_cutoff"

    output_paths = {
        "manifest": str(out_dir / "s4_manifest.json"),
        "site_tree_nodes": str(out_dir / "s4_site_tree_nodes.jsonl"),
        "site_tree_edges": str(out_dir / "s4_site_tree_edges.jsonl"),
        "selected_nodes": str(out_dir / "s4_selected_nodes.jsonl"),
        "rejected_nodes": str(out_dir / "s4_rejected_nodes.jsonl"),
        "report_json": str(out_dir / "s4_report.json"),
        "report_markdown": str(out_dir / "s4_report.md"),
    }
    selected_nodes = [node for node in nodes if node["status"] == "selected"]
    rejected_nodes = [node for node in nodes if node["status"] in {"rejected", "needs_more_data", "failed_validation"}]
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "beam_width": beam_width,
        "max_depth": max_depth,
        "top_k_sites": top_k_sites,
        "min_evidence_score": min_evidence_score,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "selected_count": len(selected_nodes),
        "rejected_count": len(rejected_nodes),
        "top_k_candidate_site_ids": selected_candidate_ids,
        "expansion_strategies": list(EXPANSION_STRATEGIES),
        "score_weights": SCORE_WEIGHTS,
        "target_payback_years_assumption_ref": "configs/cash_flow_assumptions.yaml#payback_status.target_payback_years",
    }
    manifest = {
        "run_id": run_id,
        "stage": "s4_site_tree_search",
        "input_s2_run_dir": str(s2_dir.relative_to(repo_root)) if s2_dir else None,
        "input_s3_run_dir": str(s3_dir.relative_to(repo_root)) if s3_dir else None,
        "output_artifacts": {key: str(Path(path).relative_to(repo_root)) for key, path in output_paths.items()},
    }

    _write_json(Path(output_paths["manifest"]), manifest)
    _write_jsonl(Path(output_paths["site_tree_nodes"]), nodes)
    _write_jsonl(Path(output_paths["site_tree_edges"]), edges)
    _write_jsonl(Path(output_paths["selected_nodes"]), selected_nodes)
    _write_jsonl(Path(output_paths["rejected_nodes"]), rejected_nodes)
    _write_json(Path(output_paths["report_json"]), report)
    Path(output_paths["report_markdown"]).write_text(
        "\n".join([
            "# S4 Site-Level Tree Search",
            "",
            f"Run ID: `{run_id}`",
            f"Total nodes: {len(nodes)} (root/municipality/site/facility_concept/financial_scenario)",
            f"Selected top-k sites: {len(selected_candidate_ids)}",
            f"Rejected/needs-more-data nodes: {len(rejected_nodes)}",
            "",
            "This tree explores real candidate sites (S2) and their evidence-backed",
            "financial scenarios (S3 features x configs/cash_flow_assumptions.yaml",
            "sensitivity scenarios), not rewritten proposal text.",
            "",
        ]),
        encoding="utf-8",
    )

    return S4Result(
        run_id=run_id,
        output_dir=out_dir,
        node_count=len(nodes),
        selected_count=len(selected_nodes),
        top_k_candidate_site_ids=selected_candidate_ids,
        output_paths=output_paths,
    )
