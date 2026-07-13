"""Deterministic municipality score layer (Phase 5, Stage 2)."""

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


# ---------------------------------------------------------------------------
# Score record model
# ---------------------------------------------------------------------------


class MunicipalityScoreRecord(BaseModel):
    """Deterministic pre-candidate priority scores for one municipality."""

    model_config = ConfigDict(extra="forbid")

    score_id: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    prefecture: str = Field(min_length=1)
    municipality: str = Field(min_length=1)
    score_config_version: str

    demand_pressure_score: float | None = None
    population_aging_pressure_score: float | None = None
    supply_gap_score: float | None = None
    bed_supply_pressure_score: float | None = None
    data_completeness_score: float | None = None
    overall_pre_candidate_priority_score: float | None = None

    land_score_available: bool = False
    land_score_unavailable_reason: str = (
        "No validated real land-price artifact in current phase"
    )
    cash_flow_score_available: bool = False
    cash_flow_score_unavailable_reason: str = (
        "No validated site-specific finance artifact in current phase"
    )

    score_components_used: list[str]
    input_availability_flags: dict[str, bool]


class ScoreIssue(BaseModel):
    """Issue discovered during score computation."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: str
    issue_code: str
    message: str
    study_area_id: str
    context: dict[str, str | int | float | None] = Field(default_factory=dict)
    recommended_action: str


class ScoreLayerResult(BaseModel):
    """Summary of score layer run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_area_id: str
    stage_1_passed: bool
    scores_written: int = 0
    issue_count: int = 0
    blocking_error_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_score_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _validate_score_config(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = ["version", "normalization_ceilings", "weights"]
    for key in required:
        if key not in cfg:
            errors.append(f"Missing required key: {key}")
    weights = cfg.get("weights", {})
    if weights:
        unavailable = cfg.get("unavailable_scores", {})
        base_components = {
            "demand_pressure_score",
            "population_aging_pressure_score",
            "supply_gap_score",
            "data_completeness_score",
        }
        total = sum(
            value for key, value in weights.items()
            if key in base_components
            if unavailable.get(key, {}).get("available", True) is not False
        )
        if abs(total - 1.0) > 0.001:
            errors.append(f"Active score weights sum to {total:.4f}, expected 1.0")
    return errors


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _compute_scores(
    muni_row: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[MunicipalityScoreRecord, list[str]]:
    """Compute all scores for one municipality feature base row."""

    study_area_id = muni_row.get("study_area_id", "")
    prefecture = muni_row.get("prefecture", "")
    municipality = muni_row.get("municipality", "")
    version = str(cfg.get("version", "1.0"))
    ceilings = cfg.get("normalization_ceilings", {})
    weights = cfg.get("weights", {})
    missing_data_policy = cfg.get("missing_data_policy", "zero_score")

    flags: dict[str, bool] = {}
    components_used: list[str] = []
    issues: list[str] = []

    # --- population_aging_pressure_score ---
    # Based on 65+ share at latest year. Higher = more pressure.
    # Ceiling: configurable, default 0.40
    share_65_latest = muni_row.get("share_65_plus_latest")
    ceiling_65 = ceilings.get("share_65_plus", 0.40)
    aging_score: float | None = None
    if share_65_latest is not None:
        aging_score = _clamp01(share_65_latest / ceiling_65)
        flags["share_65_plus_latest_available"] = True
        components_used.append("population_aging_pressure_score")
    else:
        flags["share_65_plus_latest_available"] = False

    # --- demand_pressure_score ---
    # Based on 65+ population percentage change from earliest to latest year.
    # Higher growth = more demand pressure.
    pop_65_pct_change = muni_row.get("population_65_plus_pct_change")
    ceiling_65_growth = ceilings.get("population_65_plus_pct_change", 1.5)
    demand_score: float | None = None
    if pop_65_pct_change is not None:
        demand_score = _clamp01(pop_65_pct_change / ceiling_65_growth)
        flags["population_65_plus_pct_change_available"] = True
        components_used.append("demand_pressure_score")
    else:
        flags["population_65_plus_pct_change_available"] = False

    # --- supply_gap_score ---
    # Based on prefecture-level beds per 100k using municipality's 2020 population.
    # This is approximate: lower beds per 100k = higher gap.
    # We compute approximate beds_per_100k using prefecture total beds and muni pop.
    # Since we only have prefecture-level beds, we can only use prefecture-level ratio.
    hosp_count = muni_row.get("hospital_count_prefecture", 0)
    total_beds = muni_row.get("total_known_beds_prefecture")
    pop_earliest = muni_row.get("population_total_earliest")
    beds_per_100k: float | None = None
    supply_gap_score: float | None = None
    ceiling_beds_per_100k = ceilings.get("beds_per_100k_adequate", 200.0)
    if total_beds is not None and pop_earliest is not None and pop_earliest > 0:
        # Use municipality's own population against prefecture total beds as a proxy
        # This is intentionally approximate — emit info issue
        beds_per_100k = total_beds / pop_earliest * 100000
        supply_gap_score = _clamp01(1.0 - beds_per_100k / ceiling_beds_per_100k)
        flags["beds_per_100k_available"] = True
        components_used.append("supply_gap_score")
    else:
        flags["beds_per_100k_available"] = False

    # --- bed_supply_pressure_score ---
    # How much of the prefecture's demand this municipality contributes.
    # = municipality population share of prefecture (not available at muni level)
    # We approximate using muni pop / prefecture pop
    # Since we don't have prefecture total population here, skip this score.
    # It will be computed in a later phase when prefecture aggregates are available.
    bed_supply_score: float | None = None
    flags["bed_supply_pressure_available"] = False

    # --- data_completeness_score ---
    coverage_flags = muni_row.get("age_group_coverage_flags", {})
    required_count_keys = [
        "total_count", "age_0_14_count", "age_15_64_count",
        "age_65_plus_count", "age_75_plus_count",
    ]
    available = sum(1 for k in required_count_keys if coverage_flags.get(k, False))
    completeness_score = _clamp01(available / len(required_count_keys))
    flags["data_completeness_available"] = True
    components_used.append("data_completeness_score")

    # --- overall_pre_candidate_priority_score ---
    w_demand = weights.get("demand_pressure_score", 0.35)
    w_aging = weights.get("population_aging_pressure_score", 0.25)
    w_supply = weights.get("supply_gap_score", 0.25)
    w_complete = weights.get("data_completeness_score", 0.15)

    weighted_sum = 0.0
    total_weight = 0.0

    if demand_score is not None:
        weighted_sum += demand_score * w_demand
        total_weight += w_demand
    elif missing_data_policy == "zero_score":
        total_weight += w_demand  # contributes zero

    if aging_score is not None:
        weighted_sum += aging_score * w_aging
        total_weight += w_aging
    elif missing_data_policy == "zero_score":
        total_weight += w_aging

    if supply_gap_score is not None:
        weighted_sum += supply_gap_score * w_supply
        total_weight += w_supply
    elif missing_data_policy == "zero_score":
        total_weight += w_supply

    weighted_sum += completeness_score * w_complete
    total_weight += w_complete

    overall: float | None = None
    if total_weight > 0:
        overall = _clamp01(weighted_sum / total_weight)
        components_used.append("overall_pre_candidate_priority_score")

    return MunicipalityScoreRecord(
        score_id=f"score:{prefecture}:{municipality}",
        study_area_id=study_area_id,
        prefecture=prefecture,
        municipality=municipality,
        score_config_version=version,
        demand_pressure_score=demand_score,
        population_aging_pressure_score=aging_score,
        supply_gap_score=supply_gap_score,
        bed_supply_pressure_score=bed_supply_score,
        data_completeness_score=completeness_score,
        overall_pre_candidate_priority_score=overall,
        land_score_available=False,
        land_score_unavailable_reason=(
            "No validated real land-price artifact in current phase"
        ),
        cash_flow_score_available=False,
        cash_flow_score_unavailable_reason=(
            "No validated site-specific finance artifact in current phase"
        ),
        score_components_used=sorted(set(components_used)),
        input_availability_flags=flags,
    ), issues


# ---------------------------------------------------------------------------
# Report / manifest
# ---------------------------------------------------------------------------


def _write_score_manifest(
    path: Path,
    run_id: str,
    study_area_id: str,
    input_files: dict[str, str],
    output_files: dict[str, str],
    counts: dict[str, int],
    issue_counts: dict[str, int],
) -> None:
    write_json(path, {
        "run_id": run_id,
        "generated_at": now_utc().isoformat(),
        "study_area_id": study_area_id,
        "input_files": input_files,
        "output_files": output_files,
        "record_counts": counts,
        "issue_counts_by_severity": issue_counts,
    })


def _write_score_report(
    json_path: Path,
    md_path: Path,
    study_area_id: str,
    scores: list[MunicipalityScoreRecord],
    issues: list[ScoreIssue],
    cfg: dict[str, Any],
) -> None:
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos = sum(1 for i in issues if i.severity == "info")

    available_components = sorted(
        set(c for s in scores for c in s.score_components_used)
    )
    report = {
        "study_area_id": study_area_id,
        "score_count": len(scores),
        "issue_count": len(issues),
        "issue_counts_by_severity": {"error": errors, "warning": warnings, "info": infos},
        "blocking_errors": errors,
        "stage_2_passed": errors == 0,
        "available_score_components": available_components,
        "land_score_available": False,
        "cash_flow_score_available": False,
        "score_config_version": str(cfg.get("version", "1.0")),
        "weights_used": cfg.get("weights", {}),
    }
    write_json(json_path, report)

    lines = [
        f"# Score Layer Report — {study_area_id}",
        "",
        f"Municipality scores written: {len(scores)}",
        f"Issues: {len(issues)} (errors={errors}, warnings={warnings}, info={infos})",
        f"Stage 2 passed (zero blocking errors): {errors == 0}",
        "",
        "## Available score components",
    ]
    for c in available_components:
        lines.append(f"- `{c}`")
    lines += [
        "",
        "## Unavailable scores (real inputs not yet available)",
        "- `land_score`: No validated real land-price artifact in current phase.",
        "- `cash_flow_score`: No validated site-specific finance artifact in current phase.",
        "- `bed_supply_pressure_score`: Prefecture population aggregate not available at "
        "municipality grain; deferred to Phase 6.",
        "",
        "## Score weights used",
    ]
    for k, v in cfg.get("weights", {}).items():
        lines.append(f"- `{k}`: {v}")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_score_layer(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
    score_config_path: str = "configs/scoring/demand_supply_score.yaml",
    require_stage1_clean: bool = True,
) -> ScoreLayerResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path

    if not cfg_path.exists():
        return ScoreLayerResult(
            run_id=run_id, study_area_id="unknown", stage_1_passed=False
        )

    study_area, config = load_study_area_config(cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id

    # Check Stage 1 report
    report_path = root / out.get(
        "feature_engineering_report_json",
        ".cache/study_area/tokyo_aichi_osaka/feature_engineering_report.json",
    )
    stage_1_passed = False
    if report_path.exists():
        stage1_report = json.loads(report_path.read_text(encoding="utf-8"))
        stage_1_passed = bool(stage1_report.get("stage_1_passed", False))

    if require_stage1_clean and not stage_1_passed:
        return ScoreLayerResult(
            run_id=run_id,
            study_area_id=sid,
            stage_1_passed=False,
        )

    # Load municipality feature base
    muni_feat_path = root / out.get(
        "municipality_feature_base",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base.jsonl",
    )
    if not muni_feat_path.exists():
        return ScoreLayerResult(
            run_id=run_id, study_area_id=sid, stage_1_passed=stage_1_passed
        )

    muni_rows = [
        json.loads(line)
        for line in muni_feat_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Load score config
    score_cfg_path = root / score_config_path
    if not score_cfg_path.exists():
        return ScoreLayerResult(
            run_id=run_id, study_area_id=sid, stage_1_passed=stage_1_passed
        )

    score_cfg = _load_score_config(score_cfg_path)
    cfg_errors = _validate_score_config(score_cfg)
    issues: list[ScoreIssue] = []
    issue_seq = 0

    if cfg_errors:
        for err in cfg_errors:
            issue_seq += 1
            issues.append(ScoreIssue(
                issue_id=f"score:config_invalid:{issue_seq}",
                severity="error",
                issue_code="score_config_invalid",
                message=err,
                study_area_id=sid,
                recommended_action="Fix configs/scoring/demand_supply_score.yaml.",
            ))
        error_count = len(issues)
        return ScoreLayerResult(
            run_id=run_id,
            study_area_id=sid,
            stage_1_passed=stage_1_passed,
            issue_count=error_count,
            blocking_error_count=error_count,
        )

    # Emit info: beds_per_100k approximation
    issue_seq += 1
    issues.append(ScoreIssue(
        issue_id=f"score:beds_per_100k_approximation:{issue_seq}",
        severity="info",
        issue_code="beds_per_100k_approximation",
        message=(
            "supply_gap_score uses prefecture total beds against each municipality's "
            "own population as an approximation, not a true catchment-level ratio. "
            "Refine in Phase 6 when prefecture population aggregates are available."
        ),
        study_area_id=sid,
        recommended_action="Compute true catchment-level beds/100k in Phase 6.",
    ))

    scores: list[MunicipalityScoreRecord] = []
    for row in sorted(muni_rows, key=lambda r: (r.get("prefecture", ""), r.get("municipality", ""))):
        score, row_issues = _compute_scores(row, score_cfg)
        scores.append(score)
        for code in row_issues:
            issue_seq += 1
            issues.append(ScoreIssue(
                issue_id=f"score:compute:{code}:{issue_seq}",
                severity="warning",
                issue_code=code,
                message=f"Score computation issue for {row.get('prefecture')}/{row.get('municipality')}: {code}",
                study_area_id=sid,
                recommended_action="Review feature substrate for this municipality.",
            ))

    # Output paths
    scores_path = root / out.get(
        "municipality_scores",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_scores.jsonl",
    )
    score_manifest_path = root / out.get(
        "score_layer_manifest",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_manifest.json",
    )
    score_issues_path = root / out.get(
        "score_layer_issues",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_issues.jsonl",
    )
    score_report_json_path = root / out.get(
        "score_layer_report_json",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_report.json",
    )
    score_report_md_path = root / out.get(
        "score_layer_report_markdown",
        ".cache/study_area/tokyo_aichi_osaka/score_layer_report.md",
    )

    write_jsonl(scores_path, scores)
    write_jsonl(score_issues_path, issues)

    sev_counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for iss in issues:
        sev_counts[iss.severity] = sev_counts.get(iss.severity, 0) + 1

    _write_score_manifest(
        score_manifest_path,
        run_id=run_id,
        study_area_id=sid,
        input_files={"municipality_feature_base": str(muni_feat_path)},
        output_files={"municipality_scores": str(scores_path)},
        counts={"municipality_scores": len(scores), "issues": len(issues)},
        issue_counts=sev_counts,
    )
    _write_score_report(
        score_report_json_path, score_report_md_path, sid, scores, issues, score_cfg
    )

    return ScoreLayerResult(
        run_id=run_id,
        study_area_id=sid,
        stage_1_passed=stage_1_passed,
        scores_written=len(scores),
        issue_count=len(issues),
        blocking_error_count=sev_counts.get("error", 0),
        output_paths={
            "municipality_scores": str(scores_path),
            "score_layer_manifest": str(score_manifest_path),
            "score_layer_issues": str(score_issues_path),
            "score_layer_report_json": str(score_report_json_path),
            "score_layer_report_markdown": str(score_report_md_path),
        },
    )
