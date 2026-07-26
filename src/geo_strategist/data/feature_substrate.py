"""Population and hospital feature substrate for model input (Phase 5)."""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PopulationFeatureRecord(BaseModel):
    """Deterministic population features for one municipality across all years."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    prefecture: str = Field(min_length=1)
    municipality: str = Field(min_length=1)
    geography_grain: str = "municipality"

    years_available: list[int]
    year_earliest: int | None = None
    year_latest: int | None = None
    year_count: int = 0

    population_total_by_year: dict[str, float | None]
    population_0_14_by_year: dict[str, float | None]
    population_15_64_by_year: dict[str, float | None]
    population_65_plus_by_year: dict[str, float | None]
    population_75_plus_by_year: dict[str, float | None]

    share_total_by_year: dict[str, float | None]
    share_0_14_by_year: dict[str, float | None]
    share_15_64_by_year: dict[str, float | None]
    share_65_plus_by_year: dict[str, float | None]
    share_75_plus_by_year: dict[str, float | None]

    population_total_change: float | None = None
    population_total_pct_change: float | None = None
    population_65_plus_change: float | None = None
    population_65_plus_pct_change: float | None = None
    population_75_plus_change: float | None = None
    population_75_plus_pct_change: float | None = None

    elderly_dependency_proxy_by_year: dict[str, float | None]

    age_group_coverage_flags: dict[str, bool]
    source_record_count: int = 0
    source_file_hash: str | None = None


class HospitalFeatureRecord(BaseModel):
    """Deterministic features for one hospital from the normalized workbook."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    master_id: str = Field(min_length=1)
    hospital_name: str | None = None
    prefecture: str | None = None
    geography_level: str = "prefecture"
    municipality: None = None

    pref_order: int | None = None
    operator_type: str | None = None
    operator_type_normalized: str | None = None
    model_archetype: str | None = None
    data_basis: str | None = None

    beds_used_in_model: float | None = None
    official_beds_total: float | None = None
    general_beds: float | None = None

    land_area_sqm: float | None = None
    land_cost_jpy_mm: float | None = None
    land_price_jpy_per_sqm: float | None = None
    construction_cost_jpy_mm: float | None = None
    equipment_it_jpy_mm: float | None = None
    contingency_jpy_mm: float | None = None
    working_capital_jpy_mm: float | None = None
    initial_investment_jpy_mm: float | None = None

    estimated_revenue_jpy_mm: float | None = None
    cash_expenses_jpy_mm: float | None = None
    hand_cf_jpy_mm: float | None = None
    hand_cf_margin: float | None = None
    ebitda_cf_margin: float | None = None
    payback_years: float | None = None
    payback_flag: str | None = None

    missing_fields: list[str]
    source_file_hash: str | None = None
    source_fact_ids: list[str]


class MunicipalityFeatureBaseRecord(BaseModel):
    """Joined feature summary for one municipality."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    prefecture: str = Field(min_length=1)
    municipality: str = Field(min_length=1)
    geography_grain: str = "municipality"

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

    hospital_municipality_join_available: bool = False
    hospital_join_level: str = "prefecture"
    hospital_count_prefecture: int = 0
    total_known_beds_prefecture: float | None = None
    beds_per_100k_prefecture: float | None = None

    age_group_coverage_flags: dict[str, bool]
    issue_codes: list[str]


class FeatureIssue(BaseModel):
    """Issue discovered during feature substrate construction."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: str
    issue_code: str
    message: str
    study_area_id: str
    context: dict[str, str | int | float | None] = Field(default_factory=dict)
    recommended_action: str


class FeatureSubstrateResult(BaseModel):
    """Summary of feature substrate construction run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_area_id: str
    input_found: bool
    population_records_read: int = 0
    hospital_facts_read: int = 0
    population_features_written: int = 0
    hospital_features_written: int = 0
    municipality_feature_base_written: int = 0
    issue_count: int = 0
    blocking_error_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hospital fact helpers
# ---------------------------------------------------------------------------

_HOSPITAL_NUM_FIELDS = {
    "pref_order", "beds_used_in_model", "official_beds_total", "general_beds",
    "land_area_sqm", "land_cost_jpy_mm", "land_price_jpy_per_sqm",
    "construction_cost_jpy_mm", "equipment_it_jpy_mm", "contingency_jpy_mm",
    "working_capital_jpy_mm", "initial_investment_jpy_mm",
    "estimated_revenue_jpy_mm", "cash_expenses_jpy_mm",
    "hand_cf_jpy_mm", "hand_cf_margin", "ebitda_cf_margin",
    "payback_years",
}
_HOSPITAL_STR_FIELDS = {
    "hospital_name", "prefecture", "operator_type", "operator_type_normalized",
    "model_archetype", "data_basis", "payback_flag",
}
_HOSPITAL_ALL_TRACKED = _HOSPITAL_NUM_FIELDS | _HOSPITAL_STR_FIELDS


def _extract_hospitals(
    rows: list[dict[str, Any]], study_area_id: str
) -> tuple[list[HospitalFeatureRecord], list[FeatureIssue]]:
    """Build one HospitalFeatureRecord per master_id from cf_payback_model_68."""

    cf_rows = [r for r in rows if r.get("source_sheet") == "cf_payback_model_68"]
    # Collect normalized fact cells by workbook row; each row contains one
    # source master_id plus the numeric/model fields used by downstream views.
    by_row: dict[int, dict[str, Any]] = defaultdict(dict)
    row_to_facts: dict[int, list[str]] = defaultdict(list)
    row_to_hash: dict[int, str] = {}

    for r in cf_rows:
        rn = r.get("source_row_number", -1)
        fn = r.get("field_name", "")
        val = r.get("value")
        by_row[rn][fn] = val
        row_to_facts[rn].append(r.get("fact_id", ""))
        if r.get("source_file_hash"):
            row_to_hash[rn] = r["source_file_hash"]

    records: list[HospitalFeatureRecord] = []
    issues: list[FeatureIssue] = []
    issue_seq = 0

    for rn, fields in sorted(by_row.items()):
        master_id = fields.get("master_id")
        if not master_id:
            continue
        master_id = str(master_id)

        def _num(fn: str) -> float | None:
            v = fields.get(fn)
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        def _str(fn: str) -> str | None:
            v = fields.get(fn)
            return str(v) if v is not None else None

        pref_order_raw = _num("pref_order")
        pref_order = int(pref_order_raw) if pref_order_raw is not None else None

        missing: list[str] = [
            fn for fn in _HOSPITAL_ALL_TRACKED if fields.get(fn) is None
        ]
        if missing:
            issue_seq += 1
            issues.append(FeatureIssue(
                issue_id=f"feature:hospital_missing_fields:{master_id}:{issue_seq}",
                severity="warning",
                issue_code="hospital_missing_fields",
                message=f"Hospital {master_id} missing fields: {sorted(missing)}",
                study_area_id=study_area_id,
                context={"master_id": master_id, "missing_count": len(missing)},
                recommended_action="Review source workbook for missing values.",
            ))

        records.append(HospitalFeatureRecord(
            feature_id=f"hosp_feat:{master_id}",
            study_area_id=study_area_id,
            master_id=master_id,
            hospital_name=_str("hospital_name"),
            prefecture=_str("prefecture"),
            geography_level="prefecture",
            municipality=None,
            pref_order=pref_order,
            operator_type=_str("operator_type"),
            operator_type_normalized=_str("operator_type_normalized"),
            model_archetype=_str("model_archetype"),
            data_basis=_str("data_basis"),
            beds_used_in_model=_num("beds_used_in_model"),
            official_beds_total=_num("official_beds_total"),
            general_beds=_num("general_beds"),
            land_area_sqm=_num("land_area_sqm"),
            land_cost_jpy_mm=_num("land_cost_jpy_mm"),
            land_price_jpy_per_sqm=_num("land_price_jpy_per_sqm"),
            construction_cost_jpy_mm=_num("construction_cost_jpy_mm"),
            equipment_it_jpy_mm=_num("equipment_it_jpy_mm"),
            contingency_jpy_mm=_num("contingency_jpy_mm"),
            working_capital_jpy_mm=_num("working_capital_jpy_mm"),
            initial_investment_jpy_mm=_num("initial_investment_jpy_mm"),
            estimated_revenue_jpy_mm=_num("estimated_revenue_jpy_mm"),
            cash_expenses_jpy_mm=_num("cash_expenses_jpy_mm"),
            hand_cf_jpy_mm=_num("hand_cf_jpy_mm"),
            hand_cf_margin=_num("hand_cf_margin"),
            ebitda_cf_margin=_num("ebitda_cf_margin"),
            payback_years=_num("payback_years"),
            payback_flag=_str("payback_flag"),
            missing_fields=sorted(missing),
            source_file_hash=row_to_hash.get(rn),
            source_fact_ids=sorted(set(row_to_facts.get(rn, []))),
        ))

    return records, issues


# ---------------------------------------------------------------------------
# Population feature helpers
# ---------------------------------------------------------------------------

_AGE_GROUP_FIELD_MAP = {
    "total": "population_total",
    "age_0_14": "population_0_14",
    "age_15_64": "population_15_64",
    "age_65_plus": "population_65_plus",
    "age_75_plus": "population_75_plus",
}


def _extract_population_features(
    rows: list[dict[str, Any]], study_area_id: str
) -> tuple[list[PopulationFeatureRecord], list[FeatureIssue]]:
    """Build one PopulationFeatureRecord per (prefecture, municipality)."""

    by_muni: dict[tuple[str, str], dict[tuple[int, str, str], dict[str, Any]]] = defaultdict(dict)

    for r in rows:
        pref = r.get("matched_target_prefecture", "")
        muni = r.get("municipality", "") or ""
        year = r.get("year")
        age_id = r.get("canonical_age_group_id", "")
        vkind = r.get("value_kind", "")
        if not (pref and muni and year and age_id and vkind):
            continue
        by_muni[(pref, muni)][(int(year), age_id, vkind)] = r

    records: list[PopulationFeatureRecord] = []
    issues: list[FeatureIssue] = []
    issue_seq = 0

    for (pref, muni), cell_map in sorted(by_muni.items()):
        years_set: set[int] = {k[0] for k in cell_map}
        years = sorted(years_set)
        year_earliest = years[0] if years else None
        year_latest = years[-1] if years else None

        def _count(yr: int, age_id: str) -> float | None:
            r = cell_map.get((yr, age_id, "count"))
            if r is None:
                return None
            v = r.get("population_value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        def _rate(yr: int, age_id: str) -> float | None:
            r = cell_map.get((yr, age_id, "rate"))
            if r is None:
                return None
            v = r.get("rate_value") or r.get("population_value")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        total_by_yr: dict[str, float | None] = {str(y): _count(y, "total") for y in years}
        age_0_14_by_yr: dict[str, float | None] = {str(y): _count(y, "age_0_14") for y in years}
        age_15_64_by_yr: dict[str, float | None] = {str(y): _count(y, "age_15_64") for y in years}
        age_65_plus_by_yr: dict[str, float | None] = {str(y): _count(y, "age_65_plus") for y in years}
        age_75_plus_by_yr: dict[str, float | None] = {str(y): _count(y, "age_75_plus") for y in years}

        share_total_by_yr: dict[str, float | None] = {str(y): _rate(y, "total") for y in years}
        share_0_14_by_yr: dict[str, float | None] = {str(y): _rate(y, "age_0_14") for y in years}
        share_15_64_by_yr: dict[str, float | None] = {str(y): _rate(y, "age_15_64") for y in years}
        share_65_plus_by_yr: dict[str, float | None] = {str(y): _rate(y, "age_65_plus") for y in years}
        share_75_plus_by_yr: dict[str, float | None] = {str(y): _rate(y, "age_75_plus") for y in years}

        # Cross-year derived fields
        pop_total_earliest = total_by_yr.get(str(year_earliest)) if year_earliest else None
        pop_total_latest = total_by_yr.get(str(year_latest)) if year_latest else None
        pop_65_earliest = age_65_plus_by_yr.get(str(year_earliest)) if year_earliest else None
        pop_65_latest = age_65_plus_by_yr.get(str(year_latest)) if year_latest else None
        pop_75_earliest = age_75_plus_by_yr.get(str(year_earliest)) if year_earliest else None
        pop_75_latest = age_75_plus_by_yr.get(str(year_latest)) if year_latest else None

        def _change(e: float | None, l: float | None) -> float | None:
            if e is None or l is None:
                return None
            return l - e

        def _pct(e: float | None, l: float | None) -> float | None:
            if e is None or l is None or e == 0:
                return None
            return (l - e) / e

        # Elderly dependency proxy: 65+ / 15-64
        dep_by_yr: dict[str, float | None] = {}
        for y in years:
            e = age_65_plus_by_yr.get(str(y))
            w = age_15_64_by_yr.get(str(y))
            dep_by_yr[str(y)] = e / w if (e is not None and w is not None and w > 0) else None

        # Coverage flags
        coverage: dict[str, bool] = {}
        for age_id in ("total", "age_0_14", "age_15_64", "age_65_plus", "age_75_plus"):
            coverage[f"{age_id}_count"] = any(
                cell_map.get((y, age_id, "count")) is not None for y in years
            )
            coverage[f"{age_id}_rate"] = any(
                cell_map.get((y, age_id, "rate")) is not None for y in years
            )

        all_groups_present = all(
            coverage.get(f"{aid}_count", False)
            for aid in ("total", "age_0_14", "age_15_64", "age_65_plus", "age_75_plus")
        )
        if not all_groups_present:
            missing_groups = [
                aid for aid in ("total", "age_0_14", "age_15_64", "age_65_plus", "age_75_plus")
                if not coverage.get(f"{aid}_count", False)
            ]
            issue_seq += 1
            issues.append(FeatureIssue(
                issue_id=f"feature:pop_incomplete_age_coverage:{pref}:{muni}:{issue_seq}",
                severity="warning",
                issue_code="population_incomplete_age_group_coverage",
                message=f"Municipality {pref}/{muni} missing count data for: {missing_groups}",
                study_area_id=study_area_id,
                context={"prefecture": pref, "municipality": muni},
                recommended_action="Check upstream age-group normalization for this municipality.",
            ))

        source_file_hash: str | None = None
        for r in cell_map.values():
            if r.get("source_file_hash"):
                source_file_hash = r["source_file_hash"]
                break

        records.append(PopulationFeatureRecord(
            feature_id=f"pop_feat:{pref}:{muni}",
            study_area_id=study_area_id,
            prefecture=pref,
            municipality=muni,
            geography_grain="municipality",
            years_available=years,
            year_earliest=year_earliest,
            year_latest=year_latest,
            year_count=len(years),
            population_total_by_year=total_by_yr,
            population_0_14_by_year=age_0_14_by_yr,
            population_15_64_by_year=age_15_64_by_yr,
            population_65_plus_by_year=age_65_plus_by_yr,
            population_75_plus_by_year=age_75_plus_by_yr,
            share_total_by_year=share_total_by_yr,
            share_0_14_by_year=share_0_14_by_yr,
            share_15_64_by_year=share_15_64_by_yr,
            share_65_plus_by_year=share_65_plus_by_yr,
            share_75_plus_by_year=share_75_plus_by_yr,
            population_total_change=_change(pop_total_earliest, pop_total_latest),
            population_total_pct_change=_pct(pop_total_earliest, pop_total_latest),
            population_65_plus_change=_change(pop_65_earliest, pop_65_latest),
            population_65_plus_pct_change=_pct(pop_65_earliest, pop_65_latest),
            population_75_plus_change=_change(pop_75_earliest, pop_75_latest),
            population_75_plus_pct_change=_pct(pop_75_earliest, pop_75_latest),
            elderly_dependency_proxy_by_year=dep_by_yr,
            age_group_coverage_flags=coverage,
            source_record_count=len(cell_map),
            source_file_hash=source_file_hash,
        ))

    return records, issues


# ---------------------------------------------------------------------------
# Municipality feature base
# ---------------------------------------------------------------------------


def _build_municipality_feature_base(
    pop_features: list[PopulationFeatureRecord],
    hosp_features: list[HospitalFeatureRecord],
    study_area_id: str,
) -> tuple[list[MunicipalityFeatureBaseRecord], list[FeatureIssue]]:
    issues: list[FeatureIssue] = []
    issue_seq = 0

    # Hospital stats per prefecture
    hosp_by_pref: dict[str, list[HospitalFeatureRecord]] = defaultdict(list)
    for h in hosp_features:
        if h.prefecture:
            hosp_by_pref[h.prefecture].append(h)

    # Issue: no municipality data in hospital workbook
    if hosp_features:
        issue_seq += 1
        issues.append(FeatureIssue(
            issue_id=f"feature:hospital_municipality_join_unavailable:{issue_seq}",
            severity="info",
            issue_code="hospital_municipality_join_unavailable",
            message=(
                "Hospital workbook does not contain municipality-level geography. "
                "Hospital features are joined at prefecture level only."
            ),
            study_area_id=study_area_id,
            context={"hospital_count": len(hosp_features)},
            recommended_action=(
                "Add municipality field to hospital workbook when real municipality "
                "data becomes available."
            ),
        ))

    records: list[MunicipalityFeatureBaseRecord] = []
    for pf in sorted(pop_features, key=lambda x: (x.prefecture, x.municipality)):
        pref_hosps = hosp_by_pref.get(pf.prefecture, [])
        hosp_count = len(pref_hosps)
        total_beds: float | None = None
        if pref_hosps:
            beds_vals = [h.beds_used_in_model for h in pref_hosps if h.beds_used_in_model is not None]
            total_beds = sum(beds_vals) if beds_vals else None

        pop_latest_yr = str(pf.year_latest) if pf.year_latest else None
        pop_earliest_yr = str(pf.year_earliest) if pf.year_earliest else None

        pop_total_latest = pf.population_total_by_year.get(pop_latest_yr) if pop_latest_yr else None
        pop_total_earliest = pf.population_total_by_year.get(pop_earliest_yr) if pop_earliest_yr else None
        share_65_latest = pf.share_65_plus_by_year.get(pop_latest_yr) if pop_latest_yr else None
        share_65_earliest = pf.share_65_plus_by_year.get(pop_earliest_yr) if pop_earliest_yr else None
        share_75_latest = pf.share_75_plus_by_year.get(pop_latest_yr) if pop_latest_yr else None
        share_75_earliest = pf.share_75_plus_by_year.get(pop_earliest_yr) if pop_earliest_yr else None
        dep_latest = pf.elderly_dependency_proxy_by_year.get(pop_latest_yr) if pop_latest_yr else None
        dep_earliest = pf.elderly_dependency_proxy_by_year.get(pop_earliest_yr) if pop_earliest_yr else None

        # Beds per 100k at prefecture level (approximate — use pop_total_earliest as 2020 anchor)
        beds_per_100k: float | None = None
        if total_beds is not None and pop_total_earliest is not None and pop_total_earliest > 0:
            # Sum prefecture population (from all municipalities' earliest year)
            beds_per_100k = None  # Computed at muni level: use muni pop, pref beds

        muni_issue_codes: list[str] = []
        if not pref_hosps:
            muni_issue_codes.append("no_prefecture_hospitals")
        if total_beds is None and pref_hosps:
            muni_issue_codes.append("beds_unavailable_for_prefecture_hospitals")

        records.append(MunicipalityFeatureBaseRecord(
            feature_id=f"muni_feat:{pf.prefecture}:{pf.municipality}",
            study_area_id=study_area_id,
            prefecture=pf.prefecture,
            municipality=pf.municipality,
            geography_grain="municipality",
            population_feature_available=True,
            year_earliest=pf.year_earliest,
            year_latest=pf.year_latest,
            year_count=pf.year_count,
            population_total_earliest=pop_total_earliest,
            population_total_latest=pop_total_latest,
            population_total_change=pf.population_total_change,
            population_total_pct_change=pf.population_total_pct_change,
            share_65_plus_earliest=share_65_earliest,
            share_65_plus_latest=share_65_latest,
            population_65_plus_change=pf.population_65_plus_change,
            population_65_plus_pct_change=pf.population_65_plus_pct_change,
            share_75_plus_earliest=share_75_earliest,
            share_75_plus_latest=share_75_latest,
            elderly_dependency_proxy_earliest=dep_earliest,
            elderly_dependency_proxy_latest=dep_latest,
            hospital_municipality_join_available=False,
            hospital_join_level="prefecture",
            hospital_count_prefecture=hosp_count,
            total_known_beds_prefecture=total_beds,
            beds_per_100k_prefecture=beds_per_100k,
            age_group_coverage_flags=pf.age_group_coverage_flags,
            issue_codes=muni_issue_codes,
        ))

    return records, issues


# ---------------------------------------------------------------------------
# Manifest and report
# ---------------------------------------------------------------------------


def _write_manifest(
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


def _write_report(
    json_path: Path,
    md_path: Path,
    study_area_id: str,
    pop_features: list[PopulationFeatureRecord],
    hosp_features: list[HospitalFeatureRecord],
    muni_features: list[MunicipalityFeatureBaseRecord],
    issues: list[FeatureIssue],
) -> None:
    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")
    infos = sum(1 for i in issues if i.severity == "info")

    report = {
        "study_area_id": study_area_id,
        "population_feature_count": len(pop_features),
        "hospital_feature_count": len(hosp_features),
        "municipality_feature_base_count": len(muni_features),
        "issue_count": len(issues),
        "issue_counts_by_severity": {"error": errors, "warning": warnings, "info": infos},
        "blocking_errors": errors,
        "stage_1_passed": errors == 0,
        "land_score_available": False,
        "land_score_unavailable_reason": "No validated real land-price artifact in current phase",
        "cash_flow_score_available": False,
        "cash_flow_score_unavailable_reason": "No validated site-specific finance artifact in current phase",
    }
    write_json(json_path, report)

    prefs = sorted(set(f.prefecture for f in pop_features))
    lines = [
        f"# Feature Substrate Report — {study_area_id}",
        "",
        f"Population feature records: {len(pop_features)} municipalities",
        f"Hospital feature records: {len(hosp_features)} hospitals",
        f"Municipality feature base records: {len(muni_features)} municipalities",
        f"Issues: {len(issues)} (errors={errors}, warnings={warnings}, info={infos})",
        f"Stage 1 passed (zero blocking errors): {errors == 0}",
        "",
        "## Scope",
        f"Prefectures: {', '.join(prefs)}",
        "",
        "## Unavailable scores",
        "- `land_score`: No validated real land-price artifact in current phase.",
        "- `cash_flow_score`: No validated site-specific finance artifact in current phase.",
        "",
        "## Issues by code",
    ]
    from collections import Counter
    for code, cnt in Counter(i.issue_code for i in issues).most_common():
        lines.append(f"- `{code}`: {cnt}")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_feature_substrate(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> FeatureSubstrateResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path

    if not cfg_path.exists():
        return FeatureSubstrateResult(
            run_id=run_id, study_area_id="unknown", input_found=False
        )

    study_area, config = load_study_area_config(cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id

    pop_path = root / out.get("model_input_ready_population_base", "")
    hosp_path = root / ".data/interim/views/hospital_workbook_facts.jsonl"

    pop_feat_path = root / out.get("population_features", ".data/interim/study_area/tokyo_aichi_osaka/population_features.jsonl")
    hosp_feat_path = root / out.get("hospital_features", ".data/interim/study_area/tokyo_aichi_osaka/hospital_features.jsonl")
    muni_feat_path = root / out.get("municipality_feature_base", ".data/interim/study_area/tokyo_aichi_osaka/municipality_feature_base.jsonl")
    manifest_path = root / out.get("feature_engineering_manifest", ".cache/study_area/tokyo_aichi_osaka/feature_engineering_manifest.json")
    issues_path = root / out.get("feature_engineering_issues", ".cache/study_area/tokyo_aichi_osaka/feature_engineering_issues.jsonl")
    report_json_path = root / out.get("feature_engineering_report_json", ".cache/study_area/tokyo_aichi_osaka/feature_engineering_report.json")
    report_md_path = root / out.get("feature_engineering_report_markdown", ".cache/study_area/tokyo_aichi_osaka/feature_engineering_report.md")

    if not pop_path.exists():
        return FeatureSubstrateResult(
            run_id=run_id, study_area_id=sid, input_found=False
        )

    pop_rows = [
        json.loads(line) for line in pop_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    hosp_rows: list[dict[str, Any]] = []
    if hosp_path.exists():
        hosp_rows = [
            json.loads(line) for line in hosp_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

    pop_features, pop_issues = _extract_population_features(pop_rows, sid)
    hosp_features, hosp_issues = _extract_hospitals(hosp_rows, sid)
    muni_features, muni_issues = _build_municipality_feature_base(pop_features, hosp_features, sid)

    all_issues = sorted(pop_issues + hosp_issues + muni_issues, key=lambda i: i.issue_id)

    write_jsonl(pop_feat_path, pop_features)
    write_jsonl(hosp_feat_path, hosp_features)
    write_jsonl(muni_feat_path, muni_features)
    write_jsonl(issues_path, all_issues)

    issue_counts_by_sev: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for iss in all_issues:
        issue_counts_by_sev[iss.severity] = issue_counts_by_sev.get(iss.severity, 0) + 1

    _write_manifest(
        manifest_path,
        run_id=run_id,
        study_area_id=sid,
        input_files={
            "model_input_ready_population_base": str(pop_path),
            "hospital_workbook_facts": str(hosp_path),
        },
        output_files={
            "population_features": str(pop_feat_path),
            "hospital_features": str(hosp_feat_path),
            "municipality_feature_base": str(muni_feat_path),
            "feature_engineering_issues": str(issues_path),
        },
        counts={
            "population_features": len(pop_features),
            "hospital_features": len(hosp_features),
            "municipality_feature_base": len(muni_features),
            "issues": len(all_issues),
        },
        issue_counts=issue_counts_by_sev,
    )
    _write_report(
        report_json_path, report_md_path, sid,
        pop_features, hosp_features, muni_features, all_issues,
    )

    output_paths = {
        "population_features": str(pop_feat_path),
        "hospital_features": str(hosp_feat_path),
        "municipality_feature_base": str(muni_feat_path),
        "feature_engineering_manifest": str(manifest_path),
        "feature_engineering_issues": str(issues_path),
        "feature_engineering_report_json": str(report_json_path),
        "feature_engineering_report_markdown": str(report_md_path),
    }

    return FeatureSubstrateResult(
        run_id=run_id,
        study_area_id=sid,
        input_found=True,
        population_records_read=len(pop_rows),
        hospital_facts_read=len(hosp_rows),
        population_features_written=len(pop_features),
        hospital_features_written=len(hosp_features),
        municipality_feature_base_written=len(muni_features),
        issue_count=len(all_issues),
        blocking_error_count=issue_counts_by_sev.get("error", 0),
        output_paths=output_paths,
    )
