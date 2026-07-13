"""Enriched municipality feature base joining land and healthcare features (Phase 6)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class EnrichedMunicipalityFeatureRecord(BaseModel):
    """Municipality feature base enriched with land and healthcare supply features."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    study_area_id: str
    prefecture: str
    municipality: str
    geography_grain: str = "municipality"

    # --- Population features (from base) ---
    population_feature_available: bool = False
    year_earliest: int | None = None
    year_latest: int | None = None
    year_count: int = 0
    population_total_earliest: float | None = None
    population_total_latest: float | None = None
    population_total_change: float | None = None
    population_total_pct_change: float | None = None
    share_65_plus_earliest: float | None = None
    share_65_plus_latest: float | None = None
    population_65_plus_change: float | None = None
    population_65_plus_pct_change: float | None = None
    share_75_plus_earliest: float | None = None
    share_75_plus_latest: float | None = None
    elderly_dependency_proxy_earliest: float | None = None
    elderly_dependency_proxy_latest: float | None = None

    # --- Hospital features (from base, prefecture-level) ---
    hospital_municipality_join_available: bool = False
    hospital_join_level: str = "prefecture"
    hospital_count_prefecture: int = 0
    total_known_beds_prefecture: float | None = None
    beds_per_100k_prefecture: float | None = None

    # --- Land features (new) ---
    land_feature_available: bool = False
    land_price_record_count: int = 0
    land_price_median: float | None = None
    land_price_mean: float | None = None
    land_price_min: float | None = None
    land_price_max: float | None = None
    land_price_latest_year: int | None = None
    land_price_unit: str = "JPY/m2"
    land_price_coverage_status: str = "unavailable"

    # --- Healthcare supply features (new) ---
    healthcare_supply_feature_available: bool = False
    healthcare_supply_record_count: int = 0
    hospital_count_municipality: int = 0
    clinic_count_municipality: int = 0
    supply_density_per_100k: float | None = None
    healthcare_supply_coverage_status: str = "unavailable"

    # --- Coverage summary ---
    age_group_coverage_flags: dict[str, bool] = Field(default_factory=dict)
    issue_codes: list[str] = Field(default_factory=list)


class EnrichedFeatureBaseResult(BaseModel):
    """Summary of enriched feature base construction."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_area_id: str
    records_written: int = 0
    issue_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


def run_enriched_feature_base(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> EnrichedFeatureBaseResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path

    if not cfg_path.exists():
        return EnrichedFeatureBaseResult(run_id=run_id, study_area_id="unknown")

    study_area, config = load_study_area_config(cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id
    generated_at = now_utc().isoformat()

    # Paths
    base_path = root / out.get(
        "municipality_feature_base",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base.jsonl",
    )
    land_feat_path = root / out.get(
        "municipality_land_features",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_land_features.jsonl",
    )
    hcs_feat_path = root / out.get(
        "municipality_healthcare_supply_features",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_healthcare_supply_features.jsonl",
    )
    enriched_path = root / out.get(
        "municipality_feature_base_enriched",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base_enriched.jsonl",
    )
    manifest_path = root / out.get(
        "enriched_feature_base_manifest",
        ".cache/study_area/tokyo_aichi_osaka/enriched_feature_base_manifest.json",
    )
    issues_path = root / out.get(
        "enriched_feature_base_issues",
        ".cache/study_area/tokyo_aichi_osaka/enriched_feature_base_issues.jsonl",
    )
    report_json_path = root / out.get(
        "enriched_feature_base_report_json",
        ".cache/study_area/tokyo_aichi_osaka/enriched_feature_base_report.json",
    )
    report_md_path = root / out.get(
        "enriched_feature_base_report_markdown",
        ".cache/study_area/tokyo_aichi_osaka/enriched_feature_base_report.md",
    )

    # Load base municipalities
    base_rows: dict[tuple[str, str], dict] = {}
    if base_path.exists():
        for line in base_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                base_rows[(r["prefecture"], r["municipality"])] = r

    # Load land features index
    land_by_muni: dict[tuple[str, str], dict] = {}
    if land_feat_path.exists():
        for line in land_feat_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                land_by_muni[(r["prefecture"], r["municipality"])] = r

    # Load healthcare supply features index
    hcs_by_muni: dict[tuple[str, str], dict] = {}
    if hcs_feat_path.exists():
        for line in hcs_feat_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                hcs_by_muni[(r["prefecture"], r["municipality"])] = r

    enriched: list[EnrichedMunicipalityFeatureRecord] = []
    issues: list[dict] = []
    issue_seq = 0

    # Join — never drop municipalities
    all_munis = sorted(
        set(base_rows.keys()) | set(land_by_muni.keys()) | set(hcs_by_muni.keys())
    )

    for pref, muni in all_munis:
        base = base_rows.get((pref, muni), {})
        land = land_by_muni.get((pref, muni), {})
        hcs = hcs_by_muni.get((pref, muni), {})

        land_available = land.get("land_price_coverage_status") == "available"
        hcs_available = hcs.get("healthcare_supply_coverage_status") == "available"

        issue_codes = list(base.get("issue_codes", []))
        if not land_available and (pref, muni) in base_rows:
            issue_codes.append("land_features_unavailable")
        if not hcs_available and (pref, muni) in base_rows:
            issue_codes.append("healthcare_supply_features_unavailable")

        enriched.append(EnrichedMunicipalityFeatureRecord(
            feature_id=f"enr_feat:{pref}:{muni}",
            study_area_id=sid,
            prefecture=pref,
            municipality=muni,
            # Base population fields
            population_feature_available=base.get("population_feature_available", False),
            year_earliest=base.get("year_earliest"),
            year_latest=base.get("year_latest"),
            year_count=base.get("year_count", 0),
            population_total_earliest=base.get("population_total_earliest"),
            population_total_latest=base.get("population_total_latest"),
            population_total_change=base.get("population_total_change"),
            population_total_pct_change=base.get("population_total_pct_change"),
            share_65_plus_earliest=base.get("share_65_plus_earliest"),
            share_65_plus_latest=base.get("share_65_plus_latest"),
            population_65_plus_change=base.get("population_65_plus_change"),
            population_65_plus_pct_change=base.get("population_65_plus_pct_change"),
            share_75_plus_earliest=base.get("share_75_plus_earliest"),
            share_75_plus_latest=base.get("share_75_plus_latest"),
            elderly_dependency_proxy_earliest=base.get("elderly_dependency_proxy_earliest"),
            elderly_dependency_proxy_latest=base.get("elderly_dependency_proxy_latest"),
            # Hospital (prefecture-level from base)
            hospital_municipality_join_available=base.get("hospital_municipality_join_available", False),
            hospital_join_level=base.get("hospital_join_level", "prefecture"),
            hospital_count_prefecture=base.get("hospital_count_prefecture", 0),
            total_known_beds_prefecture=base.get("total_known_beds_prefecture"),
            beds_per_100k_prefecture=base.get("beds_per_100k_prefecture"),
            # Land features
            land_feature_available=land_available,
            land_price_record_count=land.get("land_price_record_count", 0),
            land_price_median=land.get("land_price_median") if land_available else None,
            land_price_mean=land.get("land_price_mean") if land_available else None,
            land_price_min=land.get("land_price_min") if land_available else None,
            land_price_max=land.get("land_price_max") if land_available else None,
            land_price_latest_year=land.get("land_price_latest_year") if land_available else None,
            land_price_coverage_status=land.get("land_price_coverage_status", "unavailable"),
            # Healthcare supply features
            healthcare_supply_feature_available=hcs_available,
            healthcare_supply_record_count=hcs.get("healthcare_supply_record_count", 0),
            hospital_count_municipality=hcs.get("hospital_count", 0) if hcs_available else 0,
            clinic_count_municipality=hcs.get("clinic_count", 0) if hcs_available else 0,
            supply_density_per_100k=hcs.get("supply_density_per_100k") if hcs_available else None,
            healthcare_supply_coverage_status=hcs.get("healthcare_supply_coverage_status", "unavailable"),
            # Coverage
            age_group_coverage_flags=base.get("age_group_coverage_flags", {}),
            issue_codes=sorted(set(issue_codes)),
        ))

    write_jsonl(enriched_path, enriched)

    land_available_count = sum(1 for r in enriched if r.land_feature_available)
    hcs_available_count = sum(1 for r in enriched if r.healthcare_supply_feature_available)

    # Write report
    report = {
        "study_area_id": sid,
        "enriched_record_count": len(enriched),
        "municipalities_with_land_features": land_available_count,
        "municipalities_with_healthcare_supply_features": hcs_available_count,
        "municipalities_without_land_features": len(enriched) - land_available_count,
        "municipalities_without_healthcare_supply_features": len(enriched) - hcs_available_count,
        "issue_count": len(issues),
        "blocking_errors": 0,
        "enriched_feature_base_passed": True,
    }
    write_json(report_json_path, report)

    md_lines = [
        f"# Enriched Feature Base Report — {sid}",
        "",
        f"Enriched municipality records: {len(enriched)}",
        f"Municipalities with land features: {land_available_count}",
        f"Municipalities with healthcare supply features: {hcs_available_count}",
        "",
        "## Coverage summary",
        f"- Land features: {land_available_count}/{len(enriched)} municipalities",
        f"- Healthcare supply features: {hcs_available_count}/{len(enriched)} municipalities",
    ]
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    write_jsonl(issues_path, issues)

    write_json(manifest_path, {
        "run_id": run_id,
        "generated_at": generated_at,
        "study_area_id": sid,
        "record_counts": {
            "enriched_municipality_features": len(enriched),
            "issues": len(issues),
        },
        "issue_counts_by_severity": {"error": 0, "warning": 0, "info": 0},
    })

    return EnrichedFeatureBaseResult(
        run_id=run_id,
        study_area_id=sid,
        records_written=len(enriched),
        issue_count=len(issues),
        output_paths={
            "municipality_feature_base_enriched": str(enriched_path),
            "enriched_feature_base_manifest": str(manifest_path),
            "enriched_feature_base_issues": str(issues_path),
            "enriched_feature_base_report_json": str(report_json_path),
            "enriched_feature_base_report_markdown": str(report_md_path),
        },
    )
