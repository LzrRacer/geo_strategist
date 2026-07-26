"""S3 site-level feature engineering.

Computes demand, supply, accessibility, land/build, financial, and risk
features for each S2 candidate site. Every feature value carries its
formula, input fields, source refs, assumption refs, unit, and evidence
grade. Features for source categories that are not configured in this
environment (accessibility, zoning) are explicitly `not_available` rather
than guessed.
"""

from __future__ import annotations

import json
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from geo_strategist.data_sources.connectors import (
    financial_workbook_connector,
    healthcare_facility_connector,
    population_demand_connector,
)
from geo_strategist.experiments.s2_candidate_site_generation import OUTPUT_ROOT as S2_OUTPUT_ROOT
from geo_strategist.experiments.site_catchment import build_site_catchment_metrics


OUTPUT_ROOT = Path(".runs/experiments/s3_site_feature_engineering")
CASH_FLOW_ASSUMPTIONS_PATH = Path("configs/cash_flow_assumptions.yaml")


@dataclass(frozen=True)
class S3Result:
    run_id: str
    output_dir: Path
    feature_record_count: int
    output_paths: dict[str, str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _feature(value: Any, unit: str, formula: str, input_fields: list[str], source_refs: list[dict[str, Any]], assumption_refs: list[str], evidence_grade: str) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "formula": formula,
        "input_fields": input_fields,
        "source_refs": source_refs,
        "assumption_refs": assumption_refs,
        "evidence_grade": evidence_grade,
    }


def _not_available_feature(unit: str, formula: str, reason: str) -> dict[str, Any]:
    return _feature(None, unit, formula, [], [], [reason], "unverified_candidate")


def _latest_s2_run(repo_root: Path) -> Path | None:
    root = repo_root / S2_OUTPUT_ROOT
    if not root.exists():
        return None
    candidates = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "s2_candidate_site_records.jsonl").exists():
            return candidate
    return None


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


def _demand_features(candidate: dict[str, Any], population_by_municipality: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    key = (candidate.get("prefecture"), candidate.get("municipality"))
    row = population_by_municipality.get(key)
    if not row:
        reason = "municipality_not_available_or_no_population_match"
        return {
            "catchment_population_total": _not_available_feature("persons", "population_total for municipality", reason),
            "elderly_population_ratio_65_plus": _not_available_feature("ratio", "population_age_65_plus / population_total", reason),
            "elderly_demand_proxy_score": _not_available_feature("index", "population_total * elderly_ratio_65_plus", reason),
        }
    source_refs = [{
        "source_artifact": row["source_artifact"],
        "source_record_id": row["source_record_ids"][0] if row.get("source_record_ids") else key[1],
        "source_field": "population_total,population_age_65_plus",
    }]
    total = row.get("population_total")
    ratio = row.get("elderly_ratio_65_plus")
    proxy = round(total * ratio, 2) if total is not None and ratio is not None else None
    return {
        "catchment_population_total": _feature(
            total, "persons", "population_total for the candidate's municipality (catchment proxy)",
            ["population_total"], source_refs, [], "verified_source",
        ),
        "elderly_population_ratio_65_plus": _feature(
            ratio, "ratio", "population_age_65_plus / population_total",
            ["population_age_65_plus", "population_total"], source_refs, [], "derived_from_verified_source",
        ),
        "elderly_demand_proxy_score": _feature(
            proxy, "index", "population_total * elderly_ratio_65_plus (higher = larger elderly-driven demand proxy)",
            ["population_total", "elderly_ratio_65_plus"], source_refs,
            ["elderly_demand_proxy_is_a_project_defined_index_not_an_official_demand_forecast"],
            "derived_from_verified_source",
        ),
    }


def _supply_features(
    candidate: dict[str, Any],
    *,
    facility_records: list[dict[str, Any]],
    workbook_records: list[dict[str, Any]],
    population_row: dict[str, Any] | None,
) -> dict[str, Any]:
    nearby = candidate.get("nearby_existing_facilities") or []
    has_coordinates = candidate.get("latitude") is not None and candidate.get("longitude") is not None
    source_refs = [{
        "source_artifact": ".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_records.jsonl",
        "source_record_id": candidate["candidate_site_id"],
        "source_field": "nearby_existing_facilities",
    }] if has_coordinates else []
    catchment = build_site_catchment_metrics(
        candidate,
        facility_records,
        workbook_records=workbook_records,
        population_row=population_row,
    )
    catchment_source_refs = [{
        "source_artifact": ".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_records.jsonl",
        "source_record_id": candidate["candidate_site_id"],
        "source_field": "facility coordinates, municipality, taxonomy",
    }, {
        "source_artifact": ".data/manual/hospital_cf_workbook/hospital_rough_cf_payback_model_tokyo_aichi_osaka_beds_updated.xlsx",
        "source_record_id": "hospital_master_68",
        "source_field": "hospital_name,prefecture,beds",
    }]
    catchment_assumptions = [
        "travel_time_unavailable_haversine_distance_proxy",
        "catchment_radius_proxy_3km_5km_10km_15km",
    ]
    catchment_features = {
        "hospital_density_same_municipality_count": _feature(catchment["hospital_density_same_municipality_count"], "count", "count hospital-classified facility records in the same municipality", ["facility_records.municipality"], catchment_source_refs, [], "derived_from_verified_source"),
        "hospital_density_per_100k_population": _feature(catchment["hospital_density_per_100k_population"], "count_per_100k", "hospital_density_same_municipality_count / municipality_population * 100000", ["facility_records.municipality", "population_total"], catchment_source_refs, catchment_assumptions, "proxy_estimate"),
        "distance_to_nearest_major_hospital_km": _feature(catchment["distance_to_nearest_major_hospital_km"], "km", "min(haversine distance) to major hospital proxy", ["facility_coordinates", "candidate_coordinates"], catchment_source_refs, catchment_assumptions, "proxy_estimate") if has_coordinates else _not_available_feature("km", "min distance to major hospital; coordinates unavailable", "candidate_has_no_source_traceable_coordinates"),
        "distance_to_nearest_emergency_hospital_km": _feature(catchment["distance_to_nearest_emergency_hospital_km"], "km", "min(haversine distance) to emergency-capable hospital proxy", ["facility_coordinates", "candidate_coordinates", "facility_taxonomy"], catchment_source_refs, catchment_assumptions, "proxy_estimate") if has_coordinates else _not_available_feature("km", "min distance to emergency hospital; coordinates unavailable", "candidate_has_no_source_traceable_coordinates"),
        "medical_catchment_hospital_count_10km": _feature(catchment["medical_catchment_hospital_count_10km"], "count", "count hospital-classified facilities within 10km", ["facility_coordinates", "candidate_coordinates"], catchment_source_refs, catchment_assumptions, "proxy_estimate") if has_coordinates else _not_available_feature("count", "count hospitals within 10km; coordinates unavailable", "candidate_has_no_source_traceable_coordinates"),
        "outpatient_catchment_facility_count_5km": _feature(catchment["outpatient_catchment_facility_count_5km"], "count", "count hospital-classified facilities within 5km outpatient catchment proxy", ["facility_coordinates", "candidate_coordinates"], catchment_source_refs, catchment_assumptions, "proxy_estimate") if has_coordinates else _not_available_feature("count", "count facilities within 5km; coordinates unavailable", "candidate_has_no_source_traceable_coordinates"),
        "ambulance_emergency_catchment_count_15km": _feature(catchment["ambulance_emergency_catchment_count_15km"], "count", "count emergency-classified hospitals within 15km ambulance proxy", ["facility_coordinates", "candidate_coordinates", "facility_taxonomy"], catchment_source_refs, catchment_assumptions, "proxy_estimate") if has_coordinates else _not_available_feature("count", "count emergency hospitals within 15km; coordinates unavailable", "candidate_has_no_source_traceable_coordinates"),
        "neighboring_municipality_supply_effect_count": _feature(catchment["neighboring_municipality_hospital_supply_count_prefecture_proxy"], "count", "prefecture hospital count outside candidate municipality", ["facility_records.prefecture", "facility_records.municipality"], catchment_source_refs, catchment_assumptions, "proxy_estimate"),
        "competition_intensity": _feature(catchment["competition_intensity"], "category", "competition tier from local and catchment hospital counts", ["catchment_counts"], catchment_source_refs, catchment_assumptions, "proxy_estimate"),
        "underserved_area_signal": _feature(catchment["underserved_area_signal"], "boolean", "true when density/distance proxies indicate low local supply", ["hospital_density", "nearest_hospital_distance"], catchment_source_refs, catchment_assumptions, "proxy_estimate"),
        "catchment_metric_confidence_score": _feature(catchment["confidence_score"], "ratio", "confidence from coordinates and source coverage", ["candidate_coordinates", "facility_record_coverage"], catchment_source_refs, catchment_assumptions, "proxy_estimate"),
    }
    if not has_coordinates:
        reason = "candidate_has_no_source_traceable_coordinates"
        return {
            "nearby_facility_count_3km": _not_available_feature("count", "count of geocoded facilities within 3km", reason),
            "nearby_hospital_like_count_3km": _not_available_feature("count", "count of hospital-like facilities within 3km", reason),
            "distance_to_nearest_hospital_km": _not_available_feature("km", "min distance to a hospital-like facility", reason),
            **catchment_features,
        }
    hospital_like = [row for row in nearby if row.get("is_hospital_like")]
    nearest = min((row["distance_km"] for row in hospital_like), default=None)
    return {
        "nearby_facility_count_3km": _feature(
            len(nearby), "count", "count of geocoded facility records within 3km",
            ["nearby_existing_facilities"], source_refs, [], "derived_from_verified_source",
        ),
        "nearby_hospital_like_count_3km": _feature(
            len(hospital_like), "count", "count of hospital-like facility records within 3km",
            ["nearby_existing_facilities"], source_refs, [], "derived_from_verified_source",
        ),
        "distance_to_nearest_hospital_km": _feature(
            nearest, "km", "min(distance_km) over hospital-like nearby facilities",
            ["nearby_existing_facilities"], source_refs, [], "derived_from_verified_source",
        ),
        **catchment_features,
    }


def _accessibility_features() -> dict[str, Any]:
    reason = "transport_accessibility_source_not_configured_in_this_environment"
    return {
        "nearest_transport_access": _not_available_feature("text", "nearest station/line name", reason),
        "travel_time_proxy_minutes": _not_available_feature("minutes", "estimated travel time to catchment center", reason),
    }


def _land_build_features(candidate: dict[str, Any], financial: dict[str, Any] | None) -> dict[str, Any]:
    if not financial:
        reason = "no_matched_financial_workbook_row_for_this_candidate"
        return {
            "site_area_m2": _not_available_feature("m2", "land_area_sqm from financial workbook", reason),
            "land_price_JPY_per_sqm": _not_available_feature("JPY_per_m2", "land_price_JPY_per_sqm from financial workbook", reason),
            "zoning_or_land_use": _not_available_feature("text", "zoning classification", "zoning_source_not_configured_in_this_environment"),
            "buildability_flag": _not_available_feature("boolean", "site_area_m2 is not None", reason),
        }
    source_refs = [{
        "source_artifact": financial["source_artifact"],
        "source_record_id": financial["source_record_id"],
        "source_field": "land_area_sqm,land_price_JPY_per_sqm",
    }]
    grade = financial.get("land_construction_evidence_grade", "model_estimate")
    site_area = financial.get("land_area_sqm")
    return {
        "site_area_m2": _feature(site_area, "m2", "land_area_sqm from the workbook's rough cash-flow/payback model", ["land_area_sqm"], source_refs, [], grade),
        "land_price_JPY_per_sqm": _feature(financial.get("land_price_JPY_per_sqm"), "JPY_per_m2", "land_price_JPY_per_sqm from the workbook model", ["land_price_JPY_per_sqm"], source_refs, [], grade),
        "zoning_or_land_use": _not_available_feature("text", "zoning classification", "zoning_source_not_configured_in_this_environment"),
        "buildability_flag": _feature(site_area is not None, "boolean", "true iff a land_area_sqm figure exists in the cost model", ["land_area_sqm"], source_refs, [], grade),
    }


def _financial_features(financial: dict[str, Any] | None, benchmark: dict[str, Any]) -> dict[str, Any]:
    if financial:
        grade = financial.get("financial_evidence_grade", "model_estimate")
        capex_grade = financial.get("land_construction_evidence_grade", "model_estimate")
        source_refs = [{
            "source_artifact": financial["source_artifact"],
            "source_record_id": financial["source_record_id"],
            "source_field": "estimated_revenue_JPY_mm,cash_expenses_JPY_mm,hand_CF_JPY_mm,initial_investment_JPY_mm,payback_years",
        }]
        revenue = financial.get("estimated_revenue_JPY_mm") if financial.get("estimated_revenue_JPY_mm") is not None else financial.get("actual_revenue_JPY_mm")
        return {
            "estimated_annual_revenue_JPY_mm": _feature(revenue, "JPY_million_per_year", "estimated_revenue_JPY_mm (or actual_revenue_JPY_mm when data_basis is facility-level actuals)", ["estimated_revenue_JPY_mm", "actual_revenue_JPY_mm"], source_refs, [], grade),
            "estimated_annual_opex_JPY_mm": _feature(financial.get("cash_expenses_JPY_mm"), "JPY_million_per_year", "cash_expenses_JPY_mm from the workbook model", ["cash_expenses_JPY_mm"], source_refs, [], grade),
            "estimated_ebitda_cf_JPY_mm": _feature(financial.get("hand_cf_JPY_mm"), "JPY_million_per_year", "hand_CF_JPY_mm from the workbook model", ["hand_cf_JPY_mm"], source_refs, [], grade),
            "estimated_capex_JPY_mm": _feature(financial.get("initial_investment_JPY_mm"), "JPY_million", "initial_investment_JPY_mm (construction + equipment + land + working capital + contingency) from the workbook model", ["initial_investment_JPY_mm"], source_refs, [], capex_grade),
            "payback_years_scenario": _feature(financial.get("payback_years"), "years", "payback_years from the workbook model", ["payback_years"], source_refs, [], capex_grade),
        }

    # No matched financial anchor (greenfield "build" candidate): use a
    # documented benchmark derived from the real 68-hospital sample plus the
    # existing cash_flow_assumptions.yaml build scope factor, never an
    # invented figure.
    assumption_refs = [
        "configs/cash_flow_assumptions.yaml#action_scope_factors.build",
        "benchmark_median_across_68_hospital_financial_workbook_sample",
    ]
    source_refs = [{
        "source_artifact": ".data/manual/hospital_cf_workbook/hospital_rough_cf_payback_model_tokyo_aichi_osaka_beds_updated.xlsx",
        "source_record_id": "cf_payback_model_68:median_benchmark",
        "source_field": "estimated_revenue_JPY_mm,cash_expenses_JPY_mm,hand_CF_JPY_mm,initial_investment_JPY_mm,payback_years",
    }]
    return {
        "estimated_annual_revenue_JPY_mm": _feature(benchmark.get("median_revenue"), "JPY_million_per_year", "median(estimated_revenue_JPY_mm) across the 68-hospital sample, applied by analogy", ["estimated_revenue_JPY_mm"], source_refs, assumption_refs, "model_estimate"),
        "estimated_annual_opex_JPY_mm": _feature(benchmark.get("median_opex"), "JPY_million_per_year", "median(cash_expenses_JPY_mm) across the 68-hospital sample, applied by analogy", ["cash_expenses_JPY_mm"], source_refs, assumption_refs, "model_estimate"),
        "estimated_ebitda_cf_JPY_mm": _feature(benchmark.get("median_ebitda_cf"), "JPY_million_per_year", "median(hand_CF_JPY_mm) across the 68-hospital sample, applied by analogy", ["hand_cf_JPY_mm"], source_refs, assumption_refs, "model_estimate"),
        "estimated_capex_JPY_mm": _feature(benchmark.get("median_capex"), "JPY_million", "median(initial_investment_JPY_mm) across the 68-hospital sample x build action-scope factor (1.0)", ["initial_investment_JPY_mm"], source_refs, assumption_refs, "model_estimate"),
        "payback_years_scenario": _feature(benchmark.get("median_payback"), "years", "median(payback_years) across the 68-hospital sample, applied by analogy", ["payback_years"], source_refs, assumption_refs, "model_estimate"),
    }


def _risk_features(candidate: dict[str, Any]) -> dict[str, Any]:
    blocking = candidate.get("blocking_issues") or []
    total_expected_fields = 6  # address, parcel, zoning, transport, site_area, financials
    present_fields = sum([
        bool(candidate.get("address")),
        bool(candidate.get("parcel_id")),
        candidate.get("zoning_or_land_use") is not None,
        candidate.get("nearest_transport_access") is not None,
        candidate.get("site_area_m2") is not None,
        candidate.get("anchor_master_id") is not None,
    ])
    completeness = round(present_fields / total_expected_fields, 4)
    return {
        "hazard_or_disaster_risk_flag": _not_available_feature("boolean", "hazard/disaster-risk dataset lookup", "hazard_source_not_configured_in_this_environment"),
        "regulatory_due_diligence_flag": _feature(
            bool(blocking), "boolean", "true iff any blocking_issues are present on the candidate",
            ["blocking_issues"], [], [], "derived_from_verified_source",
        ),
        "source_completeness_score": _feature(
            completeness, "ratio", "count(present key fields) / 6", ["address", "parcel_id", "zoning_or_land_use", "nearest_transport_access", "site_area_m2", "anchor_master_id"],
            [], [], "derived_from_verified_source",
        ),
        "uncertainty_score": _feature(
            round(1 - completeness, 4), "ratio", "1 - source_completeness_score", ["source_completeness_score"], [], [], "derived_from_verified_source",
        ),
    }


def _financial_benchmark(financial_records: list[dict[str, Any]]) -> dict[str, Any]:
    def _median(field: str) -> float | None:
        values = [row[field] for row in financial_records if row.get(field) is not None]
        return round(statistics.median(values), 4) if values else None

    return {
        "median_revenue": _median("estimated_revenue_JPY_mm"),
        "median_opex": _median("cash_expenses_JPY_mm"),
        "median_ebitda_cf": _median("hand_cf_JPY_mm"),
        "median_capex": _median("initial_investment_JPY_mm"),
        "median_payback": _median("payback_years"),
    }


def run_s3_site_feature_engineering(
    repo_root: str | Path = ".",
    *,
    s2_run_dir: str | Path | None = None,
    output_root: str | Path | None = None,
) -> S3Result:
    repo_root = Path(repo_root).resolve()
    input_dir = Path(s2_run_dir) if s2_run_dir else _latest_s2_run(repo_root)
    if input_dir and not input_dir.is_absolute():
        input_dir = repo_root / input_dir

    run_id = str(uuid.uuid4())
    out_root = Path(output_root) if output_root else repo_root / OUTPUT_ROOT
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / run_id
    generated_at = _now_iso()

    candidates = _read_jsonl(input_dir / "s2_candidate_site_records.jsonl") if input_dir else []
    financial_records, _ = financial_workbook_connector.load_records(repo_root)
    facility_records, facility_issues = healthcare_facility_connector.load_records(repo_root)
    financial_by_master_id = {row["master_id"]: row for row in financial_records}
    population_records, _ = population_demand_connector.load_records(repo_root)
    population_by_municipality = {(row["prefecture"], row["municipality"]): row for row in population_records}
    benchmark = _financial_benchmark(financial_records)

    feature_records: list[dict[str, Any]] = []
    for candidate in candidates:
        financial = financial_by_master_id.get(candidate.get("anchor_master_id"))
        feature_records.append({
            "candidate_site_id": candidate["candidate_site_id"],
            "demand_features": _demand_features(candidate, population_by_municipality),
            "supply_features": _supply_features(
                candidate,
                facility_records=facility_records,
                workbook_records=financial_records,
                population_row=population_by_municipality.get((candidate.get("prefecture"), candidate.get("municipality"))),
            ),
            "accessibility_features": _accessibility_features(),
            "land_build_features": _land_build_features(candidate, financial),
            "financial_features": _financial_features(financial, benchmark),
            "risk_features": _risk_features(candidate),
            "generated_at": generated_at,
        })

    output_paths = {
        "manifest": str(out_dir / "s3_manifest.json"),
        "site_feature_records": str(out_dir / "s3_site_feature_records.jsonl"),
        "report_json": str(out_dir / "s3_report.json"),
        "report_markdown": str(out_dir / "s3_report.md"),
    }
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "input_s2_run_dir": str(input_dir.relative_to(repo_root)) if input_dir else None,
        "feature_record_count": len(feature_records),
        "financial_benchmark_used_for_greenfield_candidates": benchmark,
        "healthcare_facility_source_issue_count": len(facility_issues),
    }
    manifest = {
        "run_id": run_id,
        "stage": "s3_site_feature_engineering",
        "input_s2_run_dir": report["input_s2_run_dir"],
        "output_artifacts": {key: str(Path(path).relative_to(repo_root)) for key, path in output_paths.items()},
    }

    _write_json(Path(output_paths["manifest"]), manifest)
    _write_jsonl(Path(output_paths["site_feature_records"]), feature_records)
    _write_json(Path(output_paths["report_json"]), report)
    Path(output_paths["report_markdown"]).write_text(
        "\n".join([
            "# S3 Site-Level Feature Engineering",
            "",
            f"Run ID: `{run_id}`",
            f"Feature records: {len(feature_records)}",
            "",
            "Every feature value carries formula, input_fields, source_refs,",
            "assumption_refs, unit, and evidence_grade. Accessibility and zoning",
            "features stay `not_available` because no dataset is configured for",
            "them in this environment.",
            "",
        ]),
        encoding="utf-8",
    )

    return S3Result(
        run_id=run_id,
        output_dir=out_dir,
        feature_record_count=len(feature_records),
        output_paths=output_paths,
    )
