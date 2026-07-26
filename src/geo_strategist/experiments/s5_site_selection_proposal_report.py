"""S5 decision-support site-selection/investment proposal report generation.

Builds one proposal per S4 top-k selected site plus a portfolio-level report
(executive summary, rejected sites, disclaimers). Every concrete claim
(address, parcel, financial figure) is evidence-graded; `final_recommendation`
is only reached when every *populated* concrete claim carries a grade other
than `unverified_candidate`/`rejected_or_blocked`. This is decision-support
material, not a certified investment recommendation: regulatory, medical,
legal, and financial due diligence remain human-reviewed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_strategist.data_sources.evidence_grade import NOT_RECOMMENDABLE_GRADES
from geo_strategist.experiments.s4_site_tree_search import OUTPUT_ROOT as S4_OUTPUT_ROOT


OUTPUT_ROOT = Path(".runs/experiments/s5_site_selection_proposal_report")

DECISION_SUPPORT_DISCLAIMER = (
    "This is a research decision-support proposal, not a certified investment "
    "recommendation, and not regulatory, medical, legal, or financial advice. "
    "Human expert due diligence is required before any real-world action."
)


@dataclass(frozen=True)
class S5Result:
    run_id: str
    output_dir: Path
    proposal_count: int
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


def _latest_s4_run(repo_root: Path) -> Path | None:
    root = repo_root / S4_OUTPUT_ROOT
    if not root.exists():
        return None
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "s4_site_tree_nodes.jsonl").exists():
            return candidate
    return None


def _confidence_level(overall_score: float) -> str:
    if overall_score >= 0.6:
        return "high"
    if overall_score >= 0.4:
        return "medium"
    return "low"


def _evidence_table(candidate: dict[str, Any], features: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(candidate.get("source_refs") or [])
    for group in ("demand_features", "supply_features", "land_build_features", "financial_features"):
        for field_name, feature in (features.get(group) or {}).items():
            for ref in feature.get("source_refs") or []:
                if ref not in rows:
                    rows.append(ref)
    return rows


def _assumption_table(features: dict[str, Any]) -> list[str]:
    assumptions: list[str] = []
    for group in ("demand_features", "supply_features", "land_build_features", "financial_features", "risk_features"):
        for feature in (features.get(group) or {}).values():
            for ref in feature.get("assumption_refs") or []:
                if ref not in assumptions:
                    assumptions.append(ref)
    return assumptions


def _concrete_claim_grades(
    candidate: dict[str, Any],
    features: dict[str, Any],
    *,
    require_parcel_id: bool = False,
) -> dict[str, tuple[Any, str]]:
    claims: dict[str, tuple[Any, str]] = {}
    if candidate.get("address"):
        claims["address"] = (candidate["address"], candidate.get("address_evidence_grade", "unverified_candidate"))
    if candidate.get("site_area_m2") is not None:
        claims["site_area_m2"] = (candidate["site_area_m2"], (features.get("land_build_features", {}).get("site_area_m2") or {}).get("evidence_grade", "unverified_candidate"))
    for field_name in ("estimated_annual_revenue_JPY_mm", "estimated_annual_opex_JPY_mm", "estimated_capex_JPY_mm", "payback_years_scenario"):
        feature = features.get("financial_features", {}).get(field_name)
        if feature and feature.get("value") is not None:
            claims[field_name] = (feature["value"], feature.get("evidence_grade", "unverified_candidate"))
    if require_parcel_id:
        # No parcel/lot-number registry is configured in this environment, so
        # this flag will realistically keep every candidate below the
        # recommendable tier until a real registry is added.
        claims["parcel_id"] = (candidate.get("parcel_id"), "unverified_candidate" if not candidate.get("parcel_id") else "verified_source")
    return claims


def _final_recommendation(
    candidate: dict[str, Any],
    features: dict[str, Any],
    overall_score: float,
    *,
    deny_grades: frozenset[str] = NOT_RECOMMENDABLE_GRADES,
    require_parcel_id: bool = False,
) -> dict[str, Any]:
    claims = _concrete_claim_grades(candidate, features, require_parcel_id=require_parcel_id)
    blocking = [name for name, (_, grade) in claims.items() if grade in deny_grades]
    confidence = _confidence_level(overall_score)
    if blocking:
        return {
            "status": "not_recommended_insufficient_evidence",
            "confidence_level": "low",
            "rationale": f"Concrete claim(s) {blocking} do not carry a recommendable evidence grade.",
            "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
        }
    if candidate.get("address"):
        return {
            "status": "site_specific_recommendation",
            "confidence_level": confidence,
            "rationale": "All populated concrete claims (address, financial figures) carry a recommendable evidence grade.",
            "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
        }
    return {
        "status": "municipality_level_recommendation_site_tbd",
        "confidence_level": confidence,
        "rationale": "Demand/financial evidence is recommendable, but no source-traceable exact address/parcel was available; exact-site search remains open.",
        "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
    }


def _due_diligence_checklist(candidate: dict[str, Any]) -> list[str]:
    checklist = list(candidate.get("blocking_issues") or [])
    checklist.append("regulatory_healthcare_licensing_review_required")
    checklist.append("medical_staffing_and_service_line_feasibility_review_required")
    checklist.append("legal_title_and_zoning_confirmation_required")
    checklist.append("financial_model_independent_audit_required")
    return checklist


def _human_validation_items(candidate: dict[str, Any], final_recommendation: dict[str, Any]) -> list[str]:
    items = [
        "Independent site visit and physical due diligence.",
        "Licensed real-estate/zoning confirmation of buildability.",
        "Independent financial model review by a qualified analyst.",
        "Regulatory and healthcare-licensing review by qualified counsel.",
    ]
    if not candidate.get("address"):
        items.append("Exact-site identification and geocoding confirmation (currently municipality-level only).")
    if not candidate.get("parcel_id"):
        items.append("Parcel/lot-number (地番) confirmation via a licensed registry.")
    if final_recommendation["status"] == "not_recommended_insufficient_evidence":
        items.append("Resolve missing/insufficient evidence before any further consideration.")
    return items


def _build_proposal(
    rank: int,
    node: dict[str, Any],
    scenario_nodes: list[dict[str, Any]],
    site_node: dict[str, Any],
    generated_at: str,
    *,
    deny_grades: frozenset[str] = NOT_RECOMMENDABLE_GRADES,
    require_parcel_id: bool = False,
) -> dict[str, Any]:
    candidate = site_node["evidence_bundle"]["candidate_site_record"]
    features = site_node["evidence_bundle"]["site_feature_record"]
    overall_score = node["score_vector"].get("overall_site_selection_score", 0.0)

    cash_flow_scenarios = {
        scenario_node["financial_model_snapshot"]["scenario_name"]: scenario_node["financial_model_snapshot"]
        for scenario_node in scenario_nodes
    }
    base = cash_flow_scenarios.get("base", {})
    sensitivity_analysis = {
        name: {
            "estimated_annual_revenue_JPY_mm": snapshot.get("estimated_annual_revenue_JPY_mm"),
            "estimated_ebitda_JPY_mm": snapshot.get("estimated_ebitda_JPY_mm"),
            "payback_years_scenario": snapshot.get("payback_years_scenario"),
            "delta_ebitda_vs_base_JPY_mm": (
                round(snapshot.get("estimated_ebitda_JPY_mm", 0) - base.get("estimated_ebitda_JPY_mm", 0), 4)
                if snapshot.get("estimated_ebitda_JPY_mm") is not None and base.get("estimated_ebitda_JPY_mm") is not None
                else None
            ),
        }
        for name, snapshot in cash_flow_scenarios.items()
    }

    final_recommendation = _final_recommendation(
        candidate, features, overall_score, deny_grades=deny_grades, require_parcel_id=require_parcel_id,
    )
    proposal_id = _stable_id("site_selection_proposal", {"candidate_site_id": candidate["candidate_site_id"]})

    return {
        "proposal_id": proposal_id,
        "candidate_site_id": candidate["candidate_site_id"],
        "rank": rank,
        "action_type": candidate.get("action_type"),
        "service_line_hypothesis": node.get("service_line_hypothesis"),
        "site_fact_sheet": {
            "facility_name": candidate.get("facility_name"),
            "prefecture": candidate.get("prefecture"),
            "municipality": candidate.get("municipality"),
            "address": candidate.get("address"),
            "latitude": candidate.get("latitude"),
            "longitude": candidate.get("longitude"),
            "parcel_id": candidate.get("parcel_id"),
            "parcel_id_not_available_reason": candidate.get("parcel_id_not_available_reason"),
            "site_area_m2": candidate.get("site_area_m2"),
            "zoning_or_land_use": candidate.get("zoning_or_land_use"),
            "nearest_transport_access": candidate.get("nearest_transport_access"),
            "site_evidence_grade": candidate.get("site_evidence_grade"),
        },
        "demand_and_catchment_analysis": features.get("demand_features", {}),
        "competitor_supply_analysis": {
            **features.get("supply_features", {}),
            "nearby_existing_facilities": candidate.get("nearby_existing_facilities", []),
        },
        "facility_concept": {
            "service_line_hypothesis": node.get("service_line_hypothesis"),
            "catchment_area_definition": candidate.get("catchment_area_definition"),
        },
        "capex_assumptions": features.get("financial_features", {}).get("estimated_capex_JPY_mm", {}),
        "revenue_assumptions": features.get("financial_features", {}).get("estimated_annual_revenue_JPY_mm", {}),
        "opex_assumptions": features.get("financial_features", {}).get("estimated_annual_opex_JPY_mm", {}),
        "cash_flow_scenarios": cash_flow_scenarios,
        "sensitivity_analysis": sensitivity_analysis,
        "operational_feasibility": features.get("risk_features", {}),
        "regulatory_and_due_diligence_checklist": _due_diligence_checklist(candidate),
        "risk_register": node.get("risk_register", []),
        "evidence_table": _evidence_table(candidate, features),
        "assumption_table": _assumption_table(features),
        "final_recommendation": final_recommendation,
        "items_requiring_human_expert_validation": _human_validation_items(candidate, final_recommendation),
        "score_vector": node.get("score_vector", {}),
        "generated_at": generated_at,
    }


def _report_markdown(report: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    lines = [
        "# Site-Selection / Investment Decision-Support Proposal Report",
        "",
        f"Run ID: `{report['run_id']}`",
        "",
        DECISION_SUPPORT_DISCLAIMER,
        "",
        "## Executive Summary",
        "",
        report["executive_summary"],
        "",
        "## Recommended Top Candidate Sites",
        "",
        "| rank | candidate_site_id | action | status | confidence | overall_score |",
        "|---:|---|---|---|---|---:|",
    ]
    for proposal in proposals:
        lines.append(
            f"| {proposal['rank']} | `{proposal['candidate_site_id']}` | {proposal['action_type']} | "
            f"{proposal['final_recommendation']['status']} | {proposal['final_recommendation']['confidence_level']} | "
            f"{proposal['score_vector'].get('overall_site_selection_score')} |"
        )
    lines.append("")
    lines.append("## Rejected Candidate Sites and Reasons")
    lines.append("")
    for row in report["rejected_candidate_sites"][:20]:
        lines.append(f"- `{row.get('candidate_site_id')}`: {row.get('status_reason')}")
    lines.append("")
    return "\n".join(lines)


def run_s5_site_selection_proposal_report(
    repo_root: str | Path = ".",
    *,
    s4_run_dir: str | Path | None = None,
    top_k_sites: int | None = None,
    allow_scenario_assumptions: bool = True,
    require_parcel_id: bool = False,
    output_root: str | Path | None = None,
) -> S5Result:
    repo_root = Path(repo_root).resolve()
    input_dir = Path(s4_run_dir) if s4_run_dir else _latest_s4_run(repo_root)
    if input_dir and not input_dir.is_absolute():
        input_dir = repo_root / input_dir

    run_id = str(uuid.uuid4())
    out_root = Path(output_root) if output_root else repo_root / OUTPUT_ROOT
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / run_id
    generated_at = _now_iso()

    all_nodes = _read_jsonl(input_dir / "s4_site_tree_nodes.jsonl") if input_dir else []
    s4_report = _read_json(input_dir / "s4_report.json") if input_dir else {}
    selected_candidate_ids = s4_report.get("top_k_candidate_site_ids", [])
    if top_k_sites is not None:
        selected_candidate_ids = selected_candidate_ids[:top_k_sites]

    nodes_by_candidate: dict[str, list[dict[str, Any]]] = {}
    site_node_by_candidate: dict[str, dict[str, Any]] = {}
    for node in all_nodes:
        if node["node_kind"] == "financial_scenario" and node.get("candidate_site_id"):
            nodes_by_candidate.setdefault(node["candidate_site_id"], []).append(node)
        elif node["node_kind"] == "site" and node.get("candidate_site_id"):
            site_node_by_candidate[node["candidate_site_id"]] = node

    deny_grades = frozenset(NOT_RECOMMENDABLE_GRADES | ({"scenario_assumption", "model_estimate"} if not allow_scenario_assumptions else set()))

    proposals: list[dict[str, Any]] = []
    for rank, candidate_id in enumerate(selected_candidate_ids, start=1):
        scenario_nodes = nodes_by_candidate.get(candidate_id, [])
        base_node = next((node for node in scenario_nodes if node["financial_model_snapshot"].get("scenario_name") == "base"), None)
        site_node = site_node_by_candidate.get(candidate_id)
        if base_node is None or site_node is None:
            continue
        proposals.append(_build_proposal(
            rank, base_node, scenario_nodes, site_node, generated_at,
            deny_grades=deny_grades, require_parcel_id=require_parcel_id,
        ))

    rejected_candidate_sites = [
        {"candidate_site_id": node.get("candidate_site_id"), "status_reason": node.get("status_reason")}
        for node in all_nodes
        if node["node_kind"] == "financial_scenario" and node.get("status") in {"rejected", "needs_more_data", "failed_validation"}
    ]

    site_specific = sum(1 for p in proposals if p["final_recommendation"]["status"] == "site_specific_recommendation")
    municipality_level = sum(1 for p in proposals if p["final_recommendation"]["status"] == "municipality_level_recommendation_site_tbd")
    not_recommended = sum(1 for p in proposals if p["final_recommendation"]["status"] == "not_recommended_insufficient_evidence")

    executive_summary = (
        f"{len(proposals)} candidate sites were reviewed. {site_specific} reached a site-specific "
        f"recommendation tier, {municipality_level} reached a municipality-level (exact site TBD) tier, "
        f"and {not_recommended} did not reach a recommendable evidence tier. "
        f"{DECISION_SUPPORT_DISCLAIMER}"
    )

    output_paths = {
        "manifest": str(out_dir / "s5_manifest.json"),
        "site_selection_proposals": str(out_dir / "s5_site_selection_proposals.jsonl"),
        "report_json": str(out_dir / "s5_report.json"),
        "report_markdown": str(out_dir / "s5_report.md"),
    }
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "input_s4_run_dir": str(input_dir.relative_to(repo_root)) if input_dir else None,
        "proposal_count": len(proposals),
        "site_specific_recommendation_count": site_specific,
        "municipality_level_recommendation_count": municipality_level,
        "not_recommended_count": not_recommended,
        "rejected_candidate_sites": rejected_candidate_sites,
        "executive_summary": executive_summary,
        "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
    }
    manifest = {
        "run_id": run_id,
        "stage": "s5_site_selection_proposal_report",
        "input_s4_run_dir": report["input_s4_run_dir"],
        "output_artifacts": {key: str(Path(path).relative_to(repo_root)) for key, path in output_paths.items()},
    }

    _write_json(Path(output_paths["manifest"]), manifest)
    _write_jsonl(Path(output_paths["site_selection_proposals"]), proposals)
    _write_json(Path(output_paths["report_json"]), report)
    Path(output_paths["report_markdown"]).write_text(_report_markdown(report, proposals), encoding="utf-8")

    return S5Result(
        run_id=run_id,
        output_dir=out_dir,
        proposal_count=len(proposals),
        output_paths=output_paths,
    )
