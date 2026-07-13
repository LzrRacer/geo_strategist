"""Deterministic candidate municipality/action generation (Phase 7, E0).

Candidate actions are restricted to exactly: build, reorganize, consolidate.
No site selection, parcel claims, cash-flow projections, or LLM narrative.
"""

from __future__ import annotations

import json
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class CandidateAction(str, Enum):
    build = "build"
    reorganize = "reorganize"
    consolidate = "consolidate"


class CandidateActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    study_area_id: str
    prefecture: str
    municipality: str
    candidate_action: CandidateAction
    priority_score: float = Field(ge=0.0, le=1.0)
    score_rank_within_action: int = Field(ge=1)
    score_rank_overall: int = Field(ge=1)
    selection_basis: str
    input_score_refs: dict[str, Any]
    feature_refs: dict[str, bool]
    coverage_flags: dict[str, Any]
    issue_refs: list[str] = Field(default_factory=list)
    provenance: dict[str, str] = Field(default_factory=dict)
    run_id: str
    generated_at: str


class CandidateEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str = Field(min_length=1)
    candidate_action: CandidateAction
    prefecture: str
    municipality: str
    population_features_summary: dict[str, Any]
    land_features_summary: dict[str, Any]
    healthcare_supply_features_summary: dict[str, Any]
    score_summary: dict[str, Any]
    coverage_summary: dict[str, Any]
    known_limitations: list[str] = Field(default_factory=list)
    source_artifact_refs: list[str] = Field(default_factory=list)
    issue_refs: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


class CandidateGenerationIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    issue_code: str
    severity: str
    prefecture: str | None
    municipality: str | None
    detail: str
    attempted_action: str | None = None
    missing_fields: list[str] = Field(default_factory=list)


class CandidateGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_area_id: str
    candidates_written: int = 0
    evidence_bundles_written: int = 0
    candidate_counts_by_action: dict[str, int] = Field(default_factory=dict)
    municipalities_evaluated: int = 0
    municipalities_with_no_action: int = 0
    issue_count: int = 0
    blocking_error_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score_val(row: dict, field: str) -> float | None:
    v = row.get(field)
    return float(v) if v is not None else None


def _qualifies_build(
    score_row: dict[str, Any],
    feat_row: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[bool, str]:
    t = thresholds.get("build", {})
    overall = _score_val(score_row, "overall_pre_candidate_priority_score")
    demand = _score_val(score_row, "demand_pressure_score")
    aging = _score_val(score_row, "population_aging_pressure_score")
    supply_gap = _score_val(score_row, "supply_gap_score")
    hcs_score = _score_val(score_row, "healthcare_supply_score")
    bed_score = _score_val(score_row, "bed_supply_pressure_score")
    hcs_available = bool(score_row.get("healthcare_supply_score_available", False))
    land_available = bool(score_row.get("land_score_available", False))

    if overall is None:
        return False, "overall_pre_candidate_priority_score is None"
    if overall < t.get("overall_min", 0.50):
        return False, f"overall {overall:.3f} < {t.get('overall_min', 0.50)}"

    demand_ok = demand is not None and demand >= t.get("demand_min", 0.35)
    aging_ok = aging is not None and aging >= t.get("aging_min", 0.40)
    if not (demand_ok or aging_ok):
        return False, f"demand {demand} / aging {aging} below build thresholds"

    # Supply pressure: prefer municipality-level healthcare_supply_score when available.
    # Supply_gap_score is prefecture-level and often 0 for well-served urban prefectures.
    hcs_bed_min = t.get("healthcare_supply_or_bed_min", 0.40)
    supply_gap_fallback = t.get("supply_gap_min_fallback", 0.20)
    if hcs_available:
        hcs_ok = (hcs_score is not None and hcs_score >= hcs_bed_min) or (
            bed_score is not None and bed_score >= hcs_bed_min
        )
        if not hcs_ok:
            return (
                False,
                f"hcs_score={hcs_score} and bed_score={bed_score} < {hcs_bed_min} threshold",
            )
    else:
        if supply_gap is None or supply_gap < supply_gap_fallback:
            return False, f"supply_gap {supply_gap} < {supply_gap_fallback} (fallback, no hcs data)"

    if t.get("land_score_available_required", True) and not land_available:
        return False, "land_score_available required but missing"

    reason_parts = []
    if demand_ok:
        reason_parts.append(f"demand={demand:.3f}")
    if aging_ok:
        reason_parts.append(f"aging={aging:.3f}")
    supply_info = (
        f"hcs={hcs_score:.3f}" if hcs_score is not None
        else f"supply_gap={supply_gap}"
    )
    reason = (
        f"overall={overall:.3f}, {', '.join(reason_parts)}, "
        f"{supply_info}, land_available={land_available}"
    )
    return True, reason


def _qualifies_reorganize(
    score_row: dict[str, Any],
    feat_row: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[bool, str]:
    t = thresholds.get("reorganize", {})
    overall = _score_val(score_row, "overall_pre_candidate_priority_score")
    demand = _score_val(score_row, "demand_pressure_score")
    aging = _score_val(score_row, "population_aging_pressure_score")
    hcs_available = bool(score_row.get("healthcare_supply_score_available", False))

    if overall is None:
        return False, "overall_pre_candidate_priority_score is None"
    if overall < t.get("overall_min", 0.30):
        return False, f"overall {overall:.3f} < {t.get('overall_min', 0.30)}"

    demand_thresh = t.get("demand_or_aging_demand_min", 0.15)
    aging_thresh = t.get("demand_or_aging_aging_min", 0.25)
    demand_ok = demand is not None and demand >= demand_thresh
    aging_ok = aging is not None and aging >= aging_thresh
    if not (demand_ok or aging_ok):
        return False, f"demand {demand} / aging {aging} below reorganize thresholds"

    if t.get("healthcare_supply_score_available_required", True) and not hcs_available:
        return False, "healthcare_supply_score_available required but missing"

    reason_parts = []
    if demand_ok:
        reason_parts.append(f"demand={demand:.3f}")
    if aging_ok:
        reason_parts.append(f"aging={aging:.3f}")
    reason = (
        f"overall={overall:.3f}, {', '.join(reason_parts)}, "
        f"hcs_available={hcs_available}; rebalancing signal rather than new build"
    )
    return True, reason


def _qualifies_consolidate(
    score_row: dict[str, Any],
    feat_row: dict[str, Any],
    thresholds: dict[str, Any],
) -> tuple[bool, str]:
    t = thresholds.get("consolidate", {})
    overall = _score_val(score_row, "overall_pre_candidate_priority_score")
    demand = _score_val(score_row, "demand_pressure_score")
    land_available = bool(score_row.get("land_score_available", False))
    hcs_available = bool(score_row.get("healthcare_supply_score_available", False))

    if overall is None:
        return False, "overall_pre_candidate_priority_score is None"
    if overall > t.get("overall_max", 0.30):
        return False, f"overall {overall:.3f} > {t.get('overall_max', 0.30)}"

    demand_max = t.get("demand_max", 0.25)
    if demand is not None and demand > demand_max:
        return False, f"demand {demand:.3f} > {demand_max}"

    if t.get("any_real_source_available_required", True) and not (land_available or hcs_available):
        return False, "no real source data available for consolidate evidence"

    reason = (
        f"overall={overall:.3f}, demand={demand}, "
        f"land_available={land_available}, hcs_available={hcs_available}; "
        "low demand + real data supports consolidation review"
    )
    return True, reason


def _build_provenance(
    study_area_config_path: str,
    score_config_path: str,
    enriched_feat_path: str,
    enriched_score_path: str,
) -> dict[str, str]:
    return {
        "study_area_config": study_area_config_path,
        "candidate_generation_config": "configs/candidate_generation.yaml",
        "enriched_feature_base": enriched_feat_path,
        "enriched_score_layer": enriched_score_path,
    }


def _build_evidence_bundle(
    candidate_id: str,
    action: CandidateAction,
    score_row: dict[str, Any],
    feat_row: dict[str, Any],
    provenance: dict[str, str],
    issues: list[str],
) -> CandidateEvidenceBundle:
    pref = score_row.get("prefecture", "")
    muni = score_row.get("municipality", "")

    # Population features summary — summarize from enriched feature base
    pop_summary: dict[str, Any] = {
        "population_feature_available": feat_row.get("population_feature_available"),
        "year_earliest": feat_row.get("year_earliest"),
        "year_latest": feat_row.get("year_latest"),
        "population_total_latest": feat_row.get("population_total_latest"),
        "share_65_plus_latest": feat_row.get("share_65_plus_latest"),
        "share_75_plus_latest": feat_row.get("share_75_plus_latest"),
        "population_total_pct_change": feat_row.get("population_total_pct_change"),
        "population_65_plus_pct_change": feat_row.get("population_65_plus_pct_change"),
    }

    # Land features summary
    land_summary: dict[str, Any] = {
        "land_feature_available": feat_row.get("land_feature_available"),
        "land_price_coverage_status": feat_row.get("land_price_coverage_status"),
        "land_price_median": feat_row.get("land_price_median"),
        "land_price_mean": feat_row.get("land_price_mean"),
        "land_price_min": feat_row.get("land_price_min"),
        "land_price_max": feat_row.get("land_price_max"),
        "land_price_latest_year": feat_row.get("land_price_latest_year"),
        "land_price_unit": feat_row.get("land_price_unit"),
        "land_price_record_count": feat_row.get("land_price_record_count"),
    }

    # Healthcare supply features summary
    hcs_summary: dict[str, Any] = {
        "healthcare_supply_feature_available": feat_row.get("healthcare_supply_feature_available"),
        "healthcare_supply_coverage_status": feat_row.get("healthcare_supply_coverage_status"),
        "supply_density_per_100k": feat_row.get("supply_density_per_100k"),
        "hospital_count_municipality": feat_row.get("hospital_count_municipality"),
        "clinic_count_municipality": feat_row.get("clinic_count_municipality"),
        "healthcare_supply_record_count": feat_row.get("healthcare_supply_record_count"),
    }

    # Score summary
    score_summary: dict[str, Any] = {
        "overall_pre_candidate_priority_score": score_row.get(
            "overall_pre_candidate_priority_score"
        ),
        "demand_pressure_score": score_row.get("demand_pressure_score"),
        "population_aging_pressure_score": score_row.get("population_aging_pressure_score"),
        "supply_gap_score": score_row.get("supply_gap_score"),
        "data_completeness_score": score_row.get("data_completeness_score"),
        "land_score": score_row.get("land_score"),
        "land_score_available": score_row.get("land_score_available"),
        "healthcare_supply_score": score_row.get("healthcare_supply_score"),
        "healthcare_supply_score_available": score_row.get("healthcare_supply_score_available"),
        "bed_supply_pressure_score": score_row.get("bed_supply_pressure_score"),
        "bed_supply_pressure_score_available": score_row.get("bed_supply_pressure_score_available"),
        "score_components_used": score_row.get("score_components_used"),
    }

    # Coverage summary
    coverage_summary: dict[str, Any] = {
        "land_price": feat_row.get("land_price_coverage_status", "unavailable"),
        "healthcare_supply": feat_row.get("healthcare_supply_coverage_status", "unavailable"),
        "population": "available" if feat_row.get("population_feature_available") else "unavailable",
        "hospital_join_level": feat_row.get("hospital_join_level"),
    }

    # Determine known limitations
    limitations: list[str] = []
    if not feat_row.get("land_feature_available"):
        limitations.append("Land price data unavailable; land-based claims must not be made.")
    if not feat_row.get("healthcare_supply_feature_available"):
        limitations.append(
            "Healthcare supply data unavailable; supply-density claims must not be made."
        )
    if not feat_row.get("population_feature_available"):
        limitations.append("Population feature data unavailable.")
    if feat_row.get("hospital_join_level") != "municipality":
        limitations.append(
            "Hospital data joined at prefecture level, not municipality level; "
            "municipality-grain hospital counts are estimates."
        )

    # Forbidden claims
    forbidden: list[str] = [
        "Exact parcel or land coordinates",
        "Specific site addresses or cadastral identifiers",
        "Cash-flow projections or financial return estimates",
        "LLM-generated narrative or unsupported numeric claims",
        "Any candidate action other than build, reorganize, or consolidate",
    ]
    if not feat_row.get("land_feature_available"):
        forbidden.append("Land price claims (land data unavailable for this municipality)")
    if not feat_row.get("healthcare_supply_feature_available"):
        forbidden.append(
            "Specific facility density claims (HC supply data unavailable for this municipality)"
        )

    return CandidateEvidenceBundle(
        candidate_id=candidate_id,
        candidate_action=action,
        prefecture=pref,
        municipality=muni,
        population_features_summary=pop_summary,
        land_features_summary=land_summary,
        healthcare_supply_features_summary=hcs_summary,
        score_summary=score_summary,
        coverage_summary=coverage_summary,
        known_limitations=limitations,
        source_artifact_refs=list(provenance.values()),
        issue_refs=issues,
        forbidden_claims=forbidden,
    )


def run_candidate_generation(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
    candidate_config_path: str = "configs/candidate_generation.yaml",
) -> CandidateGenerationResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path
    cand_cfg_path = root / candidate_config_path

    if not cfg_path.exists() or not cand_cfg_path.exists():
        return CandidateGenerationResult(run_id=run_id, study_area_id="unknown")

    study_area, config = load_study_area_config(cfg_path)
    cand_cfg = _load_yaml(cand_cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id
    generated_at = now_utc().isoformat()
    thresholds = cand_cfg.get("thresholds", {})

    # Input paths
    enriched_feat_path = root / out.get(
        "municipality_feature_base_enriched",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base_enriched.jsonl",
    )
    enriched_score_path = root / out.get(
        "municipality_scores_enriched",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_scores_enriched.jsonl",
    )

    # Output paths
    candidates_path = root / out.get(
        "candidate_actions",
        ".data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl",
    )
    bundles_path = root / out.get(
        "candidate_evidence_bundles",
        ".data/interim/study_area/tokyo_aichi_osaka/candidate_evidence_bundles.jsonl",
    )
    manifest_path = root / out.get(
        "candidate_generation_manifest",
        ".cache/study_area/tokyo_aichi_osaka/candidate_generation_manifest.json",
    )
    issues_path = root / out.get(
        "candidate_generation_issues",
        ".cache/study_area/tokyo_aichi_osaka/candidate_generation_issues.jsonl",
    )
    report_json_path = root / out.get(
        "candidate_generation_report_json",
        ".cache/study_area/tokyo_aichi_osaka/candidate_generation_report.json",
    )
    report_md_path = root / out.get(
        "candidate_generation_report_markdown",
        ".cache/study_area/tokyo_aichi_osaka/candidate_generation_report.md",
    )

    provenance = _build_provenance(
        study_area_config_path=str(cfg_path.relative_to(root)),
        score_config_path="configs/scoring/demand_supply_score.yaml",
        enriched_feat_path=str(enriched_feat_path.relative_to(root)),
        enriched_score_path=str(enriched_score_path.relative_to(root)),
    )

    # Load input data
    score_rows: dict[tuple[str, str], dict] = {}
    for r in _load_jsonl(enriched_score_path):
        score_rows[(r["prefecture"], r["municipality"])] = r

    feat_rows: dict[tuple[str, str], dict] = {}
    for r in _load_jsonl(enriched_feat_path):
        feat_rows[(r["prefecture"], r["municipality"])] = r

    all_keys = sorted(set(score_rows.keys()) | set(feat_rows.keys()))

    issues: list[CandidateGenerationIssue] = []
    # Tracking per-action candidates before ranking
    action_candidates: dict[str, list[tuple[float, dict, dict, str]]] = {
        "build": [], "reorganize": [], "consolidate": [],
    }
    selected_keys: set[tuple[str, str]] = set()
    municipalities_with_no_action = 0

    if not enriched_score_path.exists():
        issues.append(CandidateGenerationIssue(
            issue_id=str(uuid.uuid4()),
            issue_code="input_artifact_missing",
            severity="warning",
            prefecture=None,
            municipality=None,
            detail=f"Enriched score artifact not found: {enriched_score_path}; run build-enriched-score-layer first.",
        ))

    for key in all_keys:
        pref, muni = key
        score_row = score_rows.get(key, {})
        feat_row = feat_rows.get(key, {})

        # Try build first (highest priority)
        build_ok, build_reason = _qualifies_build(score_row, feat_row, thresholds)
        if build_ok:
            overall = _score_val(score_row, "overall_pre_candidate_priority_score") or 0.0
            action_candidates["build"].append((overall, score_row, feat_row, build_reason))
            selected_keys.add(key)
            continue

        # Try reorganize next
        reorg_ok, reorg_reason = _qualifies_reorganize(score_row, feat_row, thresholds)
        if reorg_ok:
            overall = _score_val(score_row, "overall_pre_candidate_priority_score") or 0.0
            action_candidates["reorganize"].append((overall, score_row, feat_row, reorg_reason))
            selected_keys.add(key)
            continue

        # Try consolidate
        consol_ok, consol_reason = _qualifies_consolidate(score_row, feat_row, thresholds)
        if consol_ok:
            overall = _score_val(score_row, "overall_pre_candidate_priority_score") or 0.0
            action_candidates["consolidate"].append((overall, score_row, feat_row, consol_reason))
            selected_keys.add(key)
            continue

        # No qualifying action — issue info
        municipalities_with_no_action += 1
        issues.append(CandidateGenerationIssue(
            issue_id=str(uuid.uuid4()),
            issue_code="no_qualifying_action_for_municipality",
            severity="info",
            prefecture=pref,
            municipality=muni,
            detail=(
                f"No candidate action qualifies for {pref} {muni}. "
                f"build: {build_reason}; reorg: {reorg_reason}; consol: {consol_reason}"
            ),
        ))

    # Sort within each action group and assign ranks
    action_candidates["build"].sort(key=lambda t: (-t[0], t[1].get("municipality", "")))
    action_candidates["reorganize"].sort(key=lambda t: (-t[0], t[1].get("municipality", "")))
    action_candidates["consolidate"].sort(key=lambda t: (t[0], t[1].get("municipality", "")))

    all_candidates: list[CandidateActionRecord] = []
    all_bundles: list[CandidateEvidenceBundle] = []
    overall_rank = 0

    for action_str, entries in [
        ("build", action_candidates["build"]),
        ("reorganize", action_candidates["reorganize"]),
        ("consolidate", action_candidates["consolidate"]),
    ]:
        action = CandidateAction(action_str)
        for rank_within, (overall_score, score_row, feat_row, selection_reason) in enumerate(
            entries, start=1
        ):
            overall_rank += 1
            pref = score_row.get("prefecture", "")
            muni = score_row.get("municipality", "")
            candidate_id = f"cand:{action_str}:{pref}:{muni}"

            input_score_refs = {
                field: score_row.get(field)
                for field in [
                    "overall_pre_candidate_priority_score",
                    "demand_pressure_score",
                    "population_aging_pressure_score",
                    "supply_gap_score",
                    "data_completeness_score",
                    "land_score",
                    "healthcare_supply_score",
                    "bed_supply_pressure_score",
                ]
            }
            feature_refs = {
                "land_score_available": bool(score_row.get("land_score_available", False)),
                "healthcare_supply_score_available": bool(
                    score_row.get("healthcare_supply_score_available", False)
                ),
                "bed_supply_pressure_score_available": bool(
                    score_row.get("bed_supply_pressure_score_available", False)
                ),
                "population_feature_available": bool(
                    feat_row.get("population_feature_available", False)
                ),
            }
            coverage_flags = {
                "land_price_coverage_status": feat_row.get(
                    "land_price_coverage_status", "unavailable"
                ),
                "healthcare_supply_coverage_status": feat_row.get(
                    "healthcare_supply_coverage_status", "unavailable"
                ),
            }

            rec = CandidateActionRecord(
                candidate_id=candidate_id,
                study_area_id=sid,
                prefecture=pref,
                municipality=muni,
                candidate_action=action,
                priority_score=overall_score,
                score_rank_within_action=rank_within,
                score_rank_overall=overall_rank,
                selection_basis=selection_reason,
                input_score_refs=input_score_refs,
                feature_refs=feature_refs,
                coverage_flags=coverage_flags,
                issue_refs=[],
                provenance=provenance,
                run_id=run_id,
                generated_at=generated_at,
            )
            bundle = _build_evidence_bundle(
                candidate_id=candidate_id,
                action=action,
                score_row=score_row,
                feat_row=feat_row,
                provenance=provenance,
                issues=[],
            )
            all_candidates.append(rec)
            all_bundles.append(bundle)

    counts_by_action = {
        "build": len(action_candidates["build"]),
        "reorganize": len(action_candidates["reorganize"]),
        "consolidate": len(action_candidates["consolidate"]),
    }
    issue_severity_counts = {"error": 0, "warning": 0, "info": 0}
    for iss in issues:
        issue_severity_counts[iss.severity] = issue_severity_counts.get(iss.severity, 0) + 1

    write_jsonl(candidates_path, all_candidates)
    write_jsonl(bundles_path, all_bundles)
    write_jsonl(issues_path, issues)

    report = {
        "study_area_id": sid,
        "total_candidates": len(all_candidates),
        "total_evidence_bundles": len(all_bundles),
        "candidate_counts_by_action": counts_by_action,
        "municipalities_evaluated": len(all_keys),
        "municipalities_with_no_action": municipalities_with_no_action,
        "issue_count": len(issues),
        "issue_counts_by_severity": issue_severity_counts,
        "blocking_errors": issue_severity_counts["error"],
        "candidate_generation_passed": issue_severity_counts["error"] == 0,
        "allowed_candidate_actions": ["build", "reorganize", "consolidate"],
        "forbidden_actions_emitted": False,
        "config_version": cand_cfg.get("version", "1.0"),
    }
    write_json(report_json_path, report)

    md_lines = [
        f"# Candidate Generation Report — {sid}",
        "",
        f"Total candidates: **{len(all_candidates)}**",
        f"Evidence bundles: **{len(all_bundles)}**",
        "",
        "## Candidates by action",
        "",
        f"| Action | Count |",
        f"|--------|------:|",
        f"| build | {counts_by_action['build']} |",
        f"| reorganize | {counts_by_action['reorganize']} |",
        f"| consolidate | {counts_by_action['consolidate']} |",
        "",
        f"Municipalities evaluated: {len(all_keys)}",
        f"Municipalities with no qualifying action: {municipalities_with_no_action}",
        "",
        "## Issue counts",
        "",
        f"- Errors: {issue_severity_counts['error']}",
        f"- Warnings: {issue_severity_counts['warning']}",
        f"- Info: {issue_severity_counts['info']}",
        "",
        "## Constraints",
        "",
        "- Allowed actions: build, reorganize, consolidate only",
        "- No site selection, parcel claims, or cash-flow projections",
        "- No LLM-generated narrative",
        "- All numeric values derived from real data artifacts",
    ]
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    write_json(manifest_path, {
        "run_id": run_id,
        "generated_at": generated_at,
        "study_area_id": sid,
        "config_version": cand_cfg.get("version", "1.0"),
        "candidate_counts_by_action": counts_by_action,
        "total_candidates": len(all_candidates),
        "total_evidence_bundles": len(all_bundles),
        "issue_counts_by_severity": issue_severity_counts,
        "record_counts": {
            "candidate_actions": len(all_candidates),
            "evidence_bundles": len(all_bundles),
            "issues": len(issues),
        },
        "municipalities_evaluated": len(all_keys),
        "municipalities_with_no_action": municipalities_with_no_action,
    })

    return CandidateGenerationResult(
        run_id=run_id,
        study_area_id=sid,
        candidates_written=len(all_candidates),
        evidence_bundles_written=len(all_bundles),
        candidate_counts_by_action=counts_by_action,
        municipalities_evaluated=len(all_keys),
        municipalities_with_no_action=municipalities_with_no_action,
        issue_count=len(issues),
        blocking_error_count=issue_severity_counts["error"],
        output_paths={
            "candidate_actions": str(candidates_path),
            "candidate_evidence_bundles": str(bundles_path),
            "candidate_generation_manifest": str(manifest_path),
            "candidate_generation_issues": str(issues_path),
            "candidate_generation_report_json": str(report_json_path),
            "candidate_generation_report_markdown": str(report_md_path),
        },
    )
