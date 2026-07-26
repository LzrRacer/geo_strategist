"""E14 AI-Scientist-style proposal-quality judge for site-selection proposals.

Unlike E13 (which scores C0-C14 *workflow-control* quality — whether a
condition preserves safety gates, provenance, and blockers — and remains a
secondary audit for that comparison), E14 reviews the *content* of the S5
decision-support proposal reports as a rigorous investment/site-selection
review board would: site-selection quality, demand/supply reasoning,
evidence strength, financial-model quality, operational feasibility, risk
identification, assumption transparency, sensitivity-analysis quality,
decision usefulness, recommendation clarity, comparative-ranking quality,
and implementation readiness. Workflow-control concerns (provenance,
fabrication risk, uncertainty labeling, regulatory caution, workflow
traceability) are folded in as a secondary audit only.

Scoring is deterministic and structural (following the
project's established pattern), not a live LLM call, consistent with this
being a research decision-support prototype: rankings and recommendations
here are never a certified investment decision.
"""

from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_strategist.data_sources.evidence_grade import NOT_RECOMMENDABLE_GRADES
from geo_strategist.experiments.s5_site_selection_proposal_report import (
    DECISION_SUPPORT_DISCLAIMER,
    OUTPUT_ROOT as S5_OUTPUT_ROOT,
)


OUTPUT_ROOT = Path(".runs/experiments/e14_site_selection_proposal_judge")

PRIMARY_RUBRIC_DIMENSIONS: tuple[str, ...] = (
    "site_selection_quality",
    "demand_supply_reasoning",
    "evidence_strength",
    "financial_model_quality",
    "operational_feasibility",
    "risk_identification",
    "assumption_transparency",
    "sensitivity_analysis_quality",
    "decision_usefulness",
    "recommendation_clarity",
    "comparative_ranking_quality",
    "implementation_readiness",
)

SECONDARY_AUDIT_DIMENSIONS: tuple[str, ...] = (
    "provenance_completeness",
    "fabrication_risk",
    "uncertainty_labeling",
    "regulatory_caution",
    "workflow_traceability",
)

REVIEWER_ROLES: tuple[str, ...] = (
    "healthcare_strategy_reviewer",
    "real_estate_site_reviewer",
    "financial_model_reviewer",
    "operations_reviewer",
    "regulatory_risk_reviewer",
    "data_provenance_reviewer",
    "skeptical_investment_committee_reviewer",
)

ROLE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "healthcare_strategy_reviewer": ("site_selection_quality", "demand_supply_reasoning", "decision_usefulness", "recommendation_clarity"),
    "real_estate_site_reviewer": ("site_selection_quality", "evidence_strength", "implementation_readiness", "assumption_transparency"),
    "financial_model_reviewer": ("financial_model_quality", "sensitivity_analysis_quality", "assumption_transparency", "decision_usefulness"),
    "operations_reviewer": ("operational_feasibility", "implementation_readiness", "risk_identification", "recommendation_clarity"),
    "regulatory_risk_reviewer": ("risk_identification", "assumption_transparency", "implementation_readiness", "recommendation_clarity"),
    "data_provenance_reviewer": ("evidence_strength", "assumption_transparency", "comparative_ranking_quality", "recommendation_clarity"),
    "skeptical_investment_committee_reviewer": ("financial_model_quality", "risk_identification", "decision_usefulness", "comparative_ranking_quality"),
}


@dataclass(frozen=True)
class E14Result:
    run_id: str
    output_dir: Path
    proposal_count: int
    review_count: int
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


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _latest_s5_run(repo_root: Path) -> Path | None:
    root = repo_root / S5_OUTPUT_ROOT
    if not root.exists():
        return None
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "s5_site_selection_proposals.jsonl").exists():
            return candidate
    return None


def _scale_0_1_to_1_5(value: float) -> int:
    value = max(0.0, min(1.0, value))
    return max(1, min(5, round(1 + value * 4)))


def _primary_dimension_scores(proposal: dict[str, Any], total_proposals: int) -> dict[str, int]:
    final_rec = proposal.get("final_recommendation", {})
    score_vector = proposal.get("score_vector", {})
    evidence_table = proposal.get("evidence_table") or []
    assumption_table = proposal.get("assumption_table") or []
    sensitivity = proposal.get("sensitivity_analysis") or {}
    due_diligence = proposal.get("regulatory_and_due_diligence_checklist") or []
    risk_register = proposal.get("risk_register") or []

    status = final_rec.get("status")
    evidence_strength = 5 if status == "site_specific_recommendation" else 3 if status == "municipality_level_recommendation_site_tbd" else 1
    confidence = final_rec.get("confidence_level")
    decision_usefulness = 5 if confidence == "high" else 3 if confidence == "medium" else 2
    rank = proposal.get("rank", total_proposals)
    comparative_ranking_quality = _scale_0_1_to_1_5(1 - (rank - 1) / max(1, total_proposals - 1)) if total_proposals > 1 else 3

    return {
        "site_selection_quality": _scale_0_1_to_1_5(score_vector.get("overall_site_selection_score", 0.0)),
        "demand_supply_reasoning": _scale_0_1_to_1_5((score_vector.get("demand_need_score", 0.0) + score_vector.get("supply_gap_score", 0.0)) / 2),
        "evidence_strength": evidence_strength,
        "financial_model_quality": _scale_0_1_to_1_5(score_vector.get("financial_plausibility_score", 0.0)),
        "operational_feasibility": _scale_0_1_to_1_5(score_vector.get("operational_feasibility_score", 0.0)),
        "risk_identification": 5 if (len(risk_register) + len(due_diligence)) >= 4 else 3 if (risk_register or due_diligence) else 1,
        "assumption_transparency": 5 if len(assumption_table) >= 2 else 3 if assumption_table else 1,
        "sensitivity_analysis_quality": 5 if len(sensitivity) >= 3 else 3 if sensitivity else 1,
        "decision_usefulness": decision_usefulness,
        "recommendation_clarity": 5 if (final_rec.get("rationale") and final_rec.get("decision_support_disclaimer")) else 2,
        "comparative_ranking_quality": comparative_ranking_quality,
        "implementation_readiness": 5 if (len(evidence_table) >= 3 and status != "not_recommended_insufficient_evidence") else 2,
    }


def _secondary_audit_scores(proposal: dict[str, Any]) -> dict[str, int]:
    final_rec = proposal.get("final_recommendation", {})
    evidence_table = proposal.get("evidence_table") or []
    assumption_table = proposal.get("assumption_table") or []
    sensitivity = proposal.get("sensitivity_analysis") or {}
    due_diligence = proposal.get("regulatory_and_due_diligence_checklist") or []
    site_grade = proposal.get("site_fact_sheet", {}).get("site_evidence_grade")

    fabrication_risk_score = 1 if site_grade in NOT_RECOMMENDABLE_GRADES and final_rec.get("status") != "not_recommended_insufficient_evidence" else 5
    return {
        "provenance_completeness": 5 if len(evidence_table) >= 3 else 3 if evidence_table else 1,
        "fabrication_risk": fabrication_risk_score,
        "uncertainty_labeling": 5 if (assumption_table and sensitivity) else 3 if (assumption_table or sensitivity) else 1,
        "regulatory_caution": 5 if due_diligence else 2,
        "workflow_traceability": 5 if (proposal.get("candidate_site_id") and evidence_table) else 2,
    }


def _decision_from_score(mean_score: float) -> str:
    if mean_score >= 4.5:
        return "accept"
    if mean_score >= 3.5:
        return "weak_accept"
    if mean_score >= 2.5:
        return "borderline"
    if mean_score >= 1.5:
        return "weak_reject"
    return "reject"


def _reviewer_comments(role: str, proposal: dict[str, Any], dims: dict[str, int]) -> tuple[list[str], list[str], list[str], list[str]]:
    comments: list[str] = [f"{role} reviewed proposal {proposal['proposal_id']} (rank {proposal.get('rank')})."]
    strengths: list[str] = []
    weaknesses: list[str] = []
    revisions: list[str] = []
    for dimension in ROLE_DIMENSIONS[role]:
        score = dims[dimension]
        if score >= 4:
            strengths.append(f"Strong {dimension.replace('_', ' ')} (score {score}/5).")
        elif score <= 2:
            weaknesses.append(f"Weak {dimension.replace('_', ' ')} (score {score}/5).")
            revisions.append(f"Improve {dimension.replace('_', ' ')} before the next iteration.")
    return comments, strengths, weaknesses, revisions


def _build_reviews(proposal: dict[str, Any], total_proposals: int, generated_at: str) -> list[dict[str, Any]]:
    primary = _primary_dimension_scores(proposal, total_proposals)
    reviews = []
    for role in REVIEWER_ROLES:
        role_dims = {dimension: primary[dimension] for dimension in ROLE_DIMENSIONS[role]}
        mean_score = round(statistics.mean(role_dims.values()), 4)
        comments, strengths, weaknesses, revisions = _reviewer_comments(role, proposal, primary)
        reviews.append({
            "review_id": _stable_id("e14_review", {"proposal_id": proposal["proposal_id"], "role": role}),
            "proposal_id": proposal["proposal_id"],
            "reviewer_role": role,
            "dimension_scores": role_dims,
            "mean_score": mean_score,
            "comments": comments,
            "major_strengths": strengths,
            "major_weaknesses": weaknesses,
            "required_revisions": revisions,
            "decision": _decision_from_score(mean_score),
            "generated_at": generated_at,
        })
    return reviews


def _pairwise_rows(scored_proposals: list[dict[str, Any]], generated_at: str) -> list[dict[str, Any]]:
    eligible = [row for row in scored_proposals if row["aggregate_decision"] in {"accept", "weak_accept", "borderline"}]
    rows: list[dict[str, Any]] = []
    for i in range(len(eligible)):
        for j in range(i + 1, len(eligible)):
            left, right = eligible[i], eligible[j]
            delta = round(left["aggregate_score"] - right["aggregate_score"], 4)
            winner = left["proposal_id"] if delta > 0 else right["proposal_id"] if delta < 0 else "tie"
            rows.append({
                "pairwise_comparison_id": _stable_id("e14_pairwise", {"left": left["proposal_id"], "right": right["proposal_id"]}),
                "left_proposal_id": left["proposal_id"],
                "right_proposal_id": right["proposal_id"],
                "winner_proposal_id": winner,
                "aggregate_score_delta": delta,
                "comparison_scope": "proposal_quality_site_selection_investment_decision_support",
                "generated_at": generated_at,
            })
    return rows


def run_e14_site_selection_proposal_judge(
    repo_root: str | Path = ".",
    *,
    s5_run_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> E14Result:
    repo_root = Path(repo_root).resolve()
    input_dir = Path(s5_run_dir) if s5_run_dir else _latest_s5_run(repo_root)
    if input_dir and not input_dir.is_absolute():
        input_dir = repo_root / input_dir

    run_id = str(uuid.uuid4())
    out_root = Path(output_root) if output_root else repo_root / OUTPUT_ROOT
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / run_id
    generated_at = _now_iso()

    proposals = _read_jsonl(input_dir / "s5_site_selection_proposals.jsonl") if input_dir else []
    total = len(proposals)

    all_reviews: list[dict[str, Any]] = []
    scored_proposals: list[dict[str, Any]] = []
    revision_requests: list[dict[str, Any]] = []
    for proposal in proposals:
        reviews = _build_reviews(proposal, total, generated_at)
        all_reviews.extend(reviews)
        aggregate_score = round(statistics.mean(review["mean_score"] for review in reviews), 4)
        aggregate_decision = _decision_from_score(aggregate_score)
        secondary = _secondary_audit_scores(proposal)
        scored_proposals.append({
            "proposal_id": proposal["proposal_id"],
            "candidate_site_id": proposal["candidate_site_id"],
            "rank": proposal.get("rank"),
            "aggregate_score": aggregate_score,
            "aggregate_decision": aggregate_decision,
            "secondary_audit_scores": secondary,
            "final_recommendation_status": proposal.get("final_recommendation", {}).get("status"),
        })
        for review in reviews:
            for revision in review["required_revisions"]:
                revision_requests.append({
                    "revision_request_id": _stable_id("e14_revision", {"proposal_id": proposal["proposal_id"], "role": review["reviewer_role"], "text": revision}),
                    "proposal_id": proposal["proposal_id"],
                    "reviewer_role": review["reviewer_role"],
                    "instruction": revision,
                    "generated_at": generated_at,
                })

    rankings = sorted(scored_proposals, key=lambda row: row["aggregate_score"], reverse=True)
    for rank, row in enumerate(rankings, start=1):
        row["quality_rank"] = rank
    pairwise = _pairwise_rows(rankings, generated_at)

    output_paths = {
        "manifest": str(out_dir / "e14_manifest.json"),
        "reviewer_scores": str(out_dir / "e14_reviewer_scores.jsonl"),
        "pairwise_comparisons": str(out_dir / "e14_pairwise_comparisons.jsonl"),
        "rankings": str(out_dir / "e14_rankings.jsonl"),
        "revision_requests": str(out_dir / "e14_revision_requests.jsonl"),
        "report_json": str(out_dir / "e14_site_selection_report.json"),
        "report_markdown": str(out_dir / "e14_site_selection_report.md"),
    }
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "input_s5_run_dir": str(input_dir.relative_to(repo_root)) if input_dir else None,
        "reviewer_roles": list(REVIEWER_ROLES),
        "primary_rubric_dimensions": list(PRIMARY_RUBRIC_DIMENSIONS),
        "secondary_audit_dimensions": list(SECONDARY_AUDIT_DIMENSIONS),
        "proposal_count": total,
        "reviewed_proposal_ids": [proposal["proposal_id"] for proposal in proposals],
        "rankings": rankings,
        "pairwise_comparison_count": len(pairwise),
        "revision_request_count": len(revision_requests),
        "decision_support_disclaimer": DECISION_SUPPORT_DISCLAIMER,
    }
    manifest = {
        "run_id": run_id,
        "stage": "e14_site_selection_proposal_judge",
        "input_s5_run_dir": report["input_s5_run_dir"],
        "output_artifacts": {key: str(Path(path).relative_to(repo_root)) for key, path in output_paths.items()},
    }

    _write_json(Path(output_paths["manifest"]), manifest)
    _write_jsonl(Path(output_paths["reviewer_scores"]), all_reviews)
    _write_jsonl(Path(output_paths["pairwise_comparisons"]), pairwise)
    _write_jsonl(Path(output_paths["rankings"]), rankings)
    _write_jsonl(Path(output_paths["revision_requests"]), revision_requests)
    _write_json(Path(output_paths["report_json"]), report)
    Path(output_paths["report_markdown"]).write_text(
        "\n".join([
            "# E14 Site-Selection Proposal Judge",
            "",
            f"Run ID: `{run_id}`",
            f"Proposals reviewed: {total}",
            f"Revision requests: {len(revision_requests)}",
            "",
            DECISION_SUPPORT_DISCLAIMER,
            "",
            "## Rankings (by proposal quality)",
            "",
            "| quality_rank | proposal_id | aggregate_score | decision |",
            "|---:|---|---:|---|",
            *[f"| {row['quality_rank']} | `{row['proposal_id']}` | {row['aggregate_score']} | {row['aggregate_decision']} |" for row in rankings],
            "",
        ]),
        encoding="utf-8",
    )

    return E14Result(
        run_id=run_id,
        output_dir=out_dir,
        proposal_count=total,
        review_count=len(all_reviews),
        output_paths=output_paths,
    )
