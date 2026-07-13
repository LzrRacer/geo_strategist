"""Enriched deterministic score layer using land and healthcare supply features (Phase 6)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


class EnrichedMunicipalityScoreRecord(BaseModel):
    """Enriched deterministic pre-candidate scores for one municipality."""

    model_config = ConfigDict(extra="forbid")

    score_id: str = Field(min_length=1)
    study_area_id: str
    prefecture: str
    municipality: str
    score_config_version: str

    # Prior scores (preserved)
    demand_pressure_score: float | None = None
    population_aging_pressure_score: float | None = None
    supply_gap_score: float | None = None
    data_completeness_score: float | None = None

    # New or updated scores
    land_score: float | None = None
    land_score_available: bool = False
    land_score_unavailable_reason: str | None = None

    healthcare_supply_score: float | None = None
    healthcare_supply_score_available: bool = False
    healthcare_supply_score_unavailable_reason: str | None = None

    bed_supply_pressure_score: float | None = None
    bed_supply_pressure_score_available: bool = False
    bed_supply_pressure_score_unavailable_reason: str | None = None

    cash_flow_score_available: bool = False
    cash_flow_score_unavailable_reason: str = (
        "No validated site-specific finance artifact in current phase"
    )

    overall_pre_candidate_priority_score: float | None = None
    score_components_used: list[str]
    input_availability_flags: dict[str, bool]


class EnrichedScoreLayerResult(BaseModel):
    """Summary of enriched score layer run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_area_id: str
    scores_written: int = 0
    issue_count: int = 0
    blocking_error_count: int = 0
    newly_available_components: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_score_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _compute_enriched_score(
    enriched_row: dict[str, Any],
    prior_score: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> EnrichedMunicipalityScoreRecord:
    study_area_id = enriched_row.get("study_area_id", "")
    pref = enriched_row.get("prefecture", "")
    muni = enriched_row.get("municipality", "")
    version = str(cfg.get("version", "1.0"))
    ceilings = cfg.get("normalization_ceilings", {})
    weights = cfg.get("weights", {})
    missing_data_policy = cfg.get("missing_data_policy", "zero_score")

    # Carry forward prior scores
    demand_score = prior_score.get("demand_pressure_score") if prior_score else None
    aging_score = prior_score.get("population_aging_pressure_score") if prior_score else None
    supply_gap_score = prior_score.get("supply_gap_score") if prior_score else None
    completeness_score = prior_score.get("data_completeness_score") if prior_score else None

    flags: dict[str, bool] = {}
    components_used: list[str] = []

    # --- Land score ---
    land_available = enriched_row.get("land_feature_available", False)
    land_median = enriched_row.get("land_price_median")
    land_score: float | None = None
    land_score_reason: str | None = None

    if land_available and land_median is not None:
        # Higher land price → higher construction cost pressure → higher need for careful site selection
        # Normalize: higher land price = higher land pressure score
        ceiling_land = ceilings.get("land_price_jpy_per_m2", 1_000_000.0)
        land_score = _clamp01(land_median / ceiling_land)
        flags["land_price_median_available"] = True
        components_used.append("land_score")
    else:
        flags["land_price_median_available"] = False
        land_score_reason = (
            "No real land-price data for this municipality. "
            "Set REINFOLIB_API_KEY and re-run ingest_land_prices.py."
        )

    # --- Healthcare supply score ---
    hcs_available = enriched_row.get("healthcare_supply_feature_available", False)
    supply_density = enriched_row.get("supply_density_per_100k")
    hcs_score: float | None = None
    hcs_reason: str | None = None

    if hcs_available and supply_density is not None:
        # Lower density = higher supply gap = higher need
        ceiling_density = ceilings.get("healthcare_supply_density_per_100k_adequate", 10.0)
        hcs_score = _clamp01(1.0 - supply_density / ceiling_density)
        flags["supply_density_per_100k_available"] = True
        components_used.append("healthcare_supply_score")
    else:
        flags["supply_density_per_100k_available"] = False
        hcs_reason = (
            "No real healthcare supply data for this municipality. "
            "Set YAHOO_CLIENT_ID and re-run ingest_healthcare_supply.py."
        )

    # --- Bed supply pressure (municipality-grain now possible if healthcare supply available) ---
    bed_pressure: float | None = None
    bed_pressure_available = False
    bed_pressure_reason: str | None = None

    if hcs_available:
        hosp_count = enriched_row.get("hospital_count_municipality", 0) or 0
        pop = enriched_row.get("population_total_earliest")
        if pop and pop > 0:
            # beds per 100k using hospital count as a supply proxy
            # (real bed count from Yahoo not available; use hospital count)
            hosp_per_100k = hosp_count / pop * 100000
            ceiling_hosp = ceilings.get("hospitals_per_100k_adequate", 5.0)
            bed_pressure = _clamp01(1.0 - hosp_per_100k / ceiling_hosp)
            bed_pressure_available = True
            flags["hospital_count_municipality_available"] = True
            components_used.append("bed_supply_pressure_score")
        else:
            bed_pressure_reason = "Population denominator unavailable for beds per 100k."
    else:
        bed_pressure_reason = (
            "Municipality-grain hospital count unavailable. "
            "Set YAHOO_CLIENT_ID and re-run ingest_healthcare_supply.py."
        )
    flags["bed_supply_pressure_available"] = bed_pressure_available

    # --- Carry forward prior scores into components ---
    if demand_score is not None:
        components_used.append("demand_pressure_score")
        flags["demand_pressure_score_carried"] = True
    if aging_score is not None:
        components_used.append("population_aging_pressure_score")
        flags["aging_pressure_score_carried"] = True
    if supply_gap_score is not None:
        components_used.append("supply_gap_score")
    if completeness_score is not None:
        components_used.append("data_completeness_score")

    # --- Overall score (expanded weights) ---
    w_demand = weights.get("demand_pressure_score", 0.25)
    w_aging = weights.get("population_aging_pressure_score", 0.20)
    w_supply_gap = weights.get("supply_gap_score", 0.15)
    w_land = weights.get("land_score", 0.15)
    w_hcs = weights.get("healthcare_supply_score", 0.15)
    w_bed = weights.get("bed_supply_pressure_score", 0.10)

    weighted_sum = 0.0
    total_weight = 0.0

    def _add(score: float | None, weight: float) -> None:
        nonlocal weighted_sum, total_weight
        if score is not None:
            weighted_sum += score * weight
            total_weight += weight
        elif missing_data_policy == "zero_score":
            total_weight += weight

    _add(demand_score, w_demand)
    _add(aging_score, w_aging)
    _add(supply_gap_score, w_supply_gap)
    _add(land_score, w_land)
    _add(hcs_score, w_hcs)
    _add(bed_pressure, w_bed)

    overall: float | None = None
    if total_weight > 0:
        overall = _clamp01(weighted_sum / total_weight)
        components_used.append("overall_pre_candidate_priority_score")

    return EnrichedMunicipalityScoreRecord(
        score_id=f"enr_score:{pref}:{muni}",
        study_area_id=study_area_id,
        prefecture=pref,
        municipality=muni,
        score_config_version=version,
        demand_pressure_score=demand_score,
        population_aging_pressure_score=aging_score,
        supply_gap_score=supply_gap_score,
        data_completeness_score=completeness_score,
        land_score=land_score,
        land_score_available=land_score is not None,
        land_score_unavailable_reason=land_score_reason,
        healthcare_supply_score=hcs_score,
        healthcare_supply_score_available=hcs_score is not None,
        healthcare_supply_score_unavailable_reason=hcs_reason,
        bed_supply_pressure_score=bed_pressure,
        bed_supply_pressure_score_available=bed_pressure_available,
        bed_supply_pressure_score_unavailable_reason=bed_pressure_reason,
        cash_flow_score_available=False,
        cash_flow_score_unavailable_reason=(
            "No validated site-specific finance artifact in current phase"
        ),
        overall_pre_candidate_priority_score=overall,
        score_components_used=sorted(set(components_used)),
        input_availability_flags=flags,
    )


def run_enriched_score_layer(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
    score_config_path: str = "configs/scoring/demand_supply_score.yaml",
) -> EnrichedScoreLayerResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path

    if not cfg_path.exists():
        return EnrichedScoreLayerResult(run_id=run_id, study_area_id="unknown")

    study_area, config = load_study_area_config(cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id
    generated_at = now_utc().isoformat()

    # Paths
    enriched_feat_path = root / out.get(
        "municipality_feature_base_enriched",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base_enriched.jsonl",
    )
    prior_scores_path = root / out.get(
        "municipality_scores",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_scores.jsonl",
    )
    scores_path = root / out.get(
        "municipality_scores_enriched",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_scores_enriched.jsonl",
    )
    manifest_path = root / out.get(
        "score_layer_enriched_manifest",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_enriched_manifest.json",
    )
    issues_path = root / out.get(
        "score_layer_enriched_issues",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_enriched_issues.jsonl",
    )
    report_json_path = root / out.get(
        "score_layer_enriched_report_json",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_enriched_report.json",
    )
    report_md_path = root / out.get(
        "score_layer_enriched_report_markdown",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_enriched_report.md",
    )

    # Load score config
    score_cfg_path = root / score_config_path
    if not score_cfg_path.exists():
        return EnrichedScoreLayerResult(run_id=run_id, study_area_id=sid)
    score_cfg = _load_score_config(score_cfg_path)

    # Load enriched feature base
    enriched_rows: dict[tuple[str, str], dict] = {}
    if enriched_feat_path.exists():
        for line in enriched_feat_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                enriched_rows[(r["prefecture"], r["municipality"])] = r

    # Load prior scores
    prior_scores: dict[tuple[str, str], dict] = {}
    if prior_scores_path.exists():
        for line in prior_scores_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                prior_scores[(r["prefecture"], r["municipality"])] = r

    scores: list[EnrichedMunicipalityScoreRecord] = []
    for key in sorted(enriched_rows.keys()):
        enriched = enriched_rows[key]
        prior = prior_scores.get(key)
        score = _compute_enriched_score(enriched, prior, score_cfg)
        scores.append(score)

    # Compute which components are newly available
    newly_available: list[str] = []
    if any(s.land_score_available for s in scores):
        newly_available.append("land_score")
    if any(s.healthcare_supply_score_available for s in scores):
        newly_available.append("healthcare_supply_score")
    if any(s.bed_supply_pressure_score_available for s in scores):
        newly_available.append("bed_supply_pressure_score")

    write_jsonl(scores_path, scores)
    write_jsonl(issues_path, [])

    report = {
        "study_area_id": sid,
        "enriched_score_count": len(scores),
        "newly_available_components": newly_available,
        "unavailable_components": [
            c for c in ("land_score", "healthcare_supply_score", "bed_supply_pressure_score")
            if c not in newly_available
        ],
        "cash_flow_score_available": False,
        "cash_flow_score_unavailable_reason": "No validated site-specific finance artifact in current phase",
        "blocking_errors": 0,
        "stage_2_enriched_passed": True,
    }
    write_json(report_json_path, report)

    md_lines = [
        f"# Enriched Score Layer Report — {sid}",
        "",
        f"Enriched municipality scores: {len(scores)}",
        f"Newly available components: {newly_available or ['(none — source data unavailable)']}",
        "",
        "## Score availability",
    ]
    for comp in ("land_score", "healthcare_supply_score", "bed_supply_pressure_score"):
        status = "available" if comp in newly_available else "unavailable"
        md_lines.append(f"- `{comp}`: {status}")
    md_lines += [
        "- `cash_flow_score`: unavailable (no validated site-specific finance artifact)",
        "",
        "## Note",
        "Prior population/demand/aging/data-completeness scores are preserved.",
    ]
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    write_json(manifest_path, {
        "run_id": run_id,
        "generated_at": generated_at,
        "study_area_id": sid,
        "newly_available_components": newly_available,
        "record_counts": {"enriched_scores": len(scores), "issues": 0},
        "issue_counts_by_severity": {"error": 0, "warning": 0, "info": 0},
    })

    return EnrichedScoreLayerResult(
        run_id=run_id,
        study_area_id=sid,
        scores_written=len(scores),
        issue_count=0,
        blocking_error_count=0,
        newly_available_components=newly_available,
        output_paths={
            "municipality_scores_enriched": str(scores_path),
            "score_layer_enriched_manifest": str(manifest_path),
            "score_layer_enriched_issues": str(issues_path),
            "score_layer_enriched_report_json": str(report_json_path),
            "score_layer_enriched_report_markdown": str(report_md_path),
        },
    )
