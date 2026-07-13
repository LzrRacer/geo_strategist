"""Real healthcare-supply ingestion from Yahoo Local Search API (Phase 6)."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


# ---------------------------------------------------------------------------
# Yahoo Local Search constants
# ---------------------------------------------------------------------------

YAHOO_LOCAL_SEARCH_URL = "https://map.yahooapis.jp/search/local/V1/localSearch"
YAHOO_CLIENT_ID_ENV = "YAHOO_CLIENT_ID"

# Hospital search queries per prefecture
HOSPITAL_QUERY_TEMPLATE = "{prefecture} 病院"
CLINIC_QUERY_TEMPLATE = "{prefecture} クリニック 診療所"

# Yahoo Local Search category codes for medical facilities
MEDICAL_CATEGORY_CODES = {"0120": "医療施設", "01200000": "総合病院・診療所"}

# Results per page (Yahoo max is 100)
YAHOO_PAGE_SIZE = 100
# Max pages per query to avoid rate limit exhaustion
YAHOO_MAX_PAGES = 3


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class HealthcareSupplyRecord(BaseModel):
    """Normalized healthcare supply record from Yahoo Local Search."""

    model_config = ConfigDict(extra="forbid")

    supply_record_id: str = Field(min_length=1)
    source_type: str = "yahoo_local_search"
    source_name: str = "Yahoo Local Search API"
    source_url: str = YAHOO_LOCAL_SEARCH_URL
    retrieved_at: str
    facility_name: str | None = None
    facility_category: str | None = None
    prefecture: str | None = None
    municipality: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    source_record_id: str | None = None
    response_hash: str
    query_used: str | None = None


class MunicipalityHealthcareSupplyFeatureRecord(BaseModel):
    """Aggregated healthcare-supply features for one municipality."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    study_area_id: str
    prefecture: str
    municipality: str

    healthcare_supply_record_count: int = 0
    hospital_count: int = 0
    clinic_count: int = 0
    other_medical_count: int = 0
    supply_density_per_100k: float | None = None
    healthcare_supply_coverage_status: str = "unavailable"

    population_denominator_year: int | None = None
    population_denominator_used: float | None = None


class HealthcareIngestionIssue(BaseModel):
    """Issue from healthcare supply ingestion."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: str
    issue_code: str
    message: str
    study_area_id: str
    context: dict[str, str | int | float | None] = Field(default_factory=dict)
    recommended_action: str


class HealthcareIngestionResult(BaseModel):
    """Summary of healthcare supply ingestion run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    study_area_id: str
    source_available: bool
    records_ingested: int = 0
    municipality_features_written: int = 0
    issue_count: int = 0
    blocking_error_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------


def _get_api_key() -> str | None:
    return os.environ.get(YAHOO_CLIENT_ID_ENV)


def _response_hash(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_key(query: str, start: int) -> str:
    digest = hashlib.md5(f"{query}_{start}".encode()).hexdigest()[:12]
    safe = query.replace(" ", "_").replace("/", "_")[:60]
    return f"yahoo_{safe}_{start}_{digest}.json"


def _fetch_yahoo_local(
    query: str,
    api_key: str,
    start: int,
    cache_dir: Path,
    sleep_sec: float = 0.6,
    allow_network: bool = True,
) -> dict[str, Any] | None:
    """Fetch one page of Yahoo Local Search; returns parsed JSON or None."""
    import requests

    cache_path = cache_dir / _cache_key(query, start)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    if not allow_network:
        return None

    try:
        resp = requests.get(
            YAHOO_LOCAL_SEARCH_URL,
            params={
                "appid": api_key,
                "query": query,
                "results": YAHOO_PAGE_SIZE,
                "start": start,
                "output": "json",
                # No gc (genre) filter — genre codes restrict too aggressively
                # and the query text already targets medical facilities.
            },
            headers={"User-Agent": "geo-strategist/0.1 (research)"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        time.sleep(sleep_sec)
        return data
    except Exception:
        return None


def _parse_municipality_from_address(address: str | None, known_munis: list[str]) -> str | None:
    if not address:
        return None
    for muni in sorted(known_munis, key=len, reverse=True):
        if muni in address:
            return muni
    return None


def _parse_prefecture_from_address(address: str | None, target_prefectures: list[str]) -> str | None:
    if not address:
        return None
    for pref in target_prefectures:
        if pref in address:
            return pref
    return None


def _infer_facility_category(raw_category: str | None, name: str | None) -> tuple[str, str]:
    """Returns (facility_category, facility_type: hospital|clinic|other)."""
    category = raw_category or ""
    name_str = name or ""
    if "病院" in name_str or "病院" in category:
        return category or "病院", "hospital"
    if any(x in name_str or x in category for x in ("クリニック", "診療所", "医院", "内科", "外科")):
        return category or "診療所・クリニック", "clinic"
    return category or "医療施設", "other"


def _normalize_yahoo_record(
    feature: dict[str, Any],
    query: str,
    retrieved_at: str,
    response_hash: str,
    target_prefectures: list[str],
    known_munis: list[str],
) -> HealthcareSupplyRecord | None:
    try:
        prop = feature.get("Property", {}) or {}
        geometry = feature.get("Geometry", {}) or {}

        name = feature.get("Name")
        address = prop.get("Address")
        source_id = feature.get("Id") or feature.get("Uid", "")

        coords = geometry.get("Coordinates", "")
        lat: float | None = None
        lon: float | None = None
        if "," in str(coords):
            parts = str(coords).split(",", 1)
            try:
                lon = float(parts[0])
                lat = float(parts[1])
            except ValueError:
                pass

        genres = prop.get("Genre") or []
        if isinstance(genres, list):
            raw_category = ";".join(g.get("Name", "") for g in genres if isinstance(g, dict))
        else:
            raw_category = str(genres)

        facility_category, _ = _infer_facility_category(raw_category, name)
        pref = _parse_prefecture_from_address(address, target_prefectures)
        muni = _parse_municipality_from_address(address, known_munis) if pref else None

        rhash = response_hash
        return HealthcareSupplyRecord(
            supply_record_id=f"supply:yahoo:{pref or 'unknown'}:{muni or 'unknown'}:{source_id}",
            retrieved_at=retrieved_at,
            facility_name=name,
            facility_category=facility_category,
            prefecture=pref,
            municipality=muni,
            address=address,
            latitude=lat,
            longitude=lon,
            source_record_id=str(source_id),
            response_hash=rhash,
            query_used=query,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------


def _build_healthcare_features(
    records: list[HealthcareSupplyRecord],
    all_municipalities: list[tuple[str, str]],
    pop_by_muni: dict[tuple[str, str], float],
    study_area_id: str,
    pop_year: int | None,
) -> list[MunicipalityHealthcareSupplyFeatureRecord]:
    by_muni: dict[tuple[str, str], list[HealthcareSupplyRecord]] = {}
    for pref, muni in all_municipalities:
        by_muni[(pref, muni)] = []

    for rec in records:
        if rec.prefecture and rec.municipality:
            key = (rec.prefecture, rec.municipality)
            if key in by_muni:
                by_muni[key].append(rec)

    results: list[MunicipalityHealthcareSupplyFeatureRecord] = []
    for (pref, muni), recs in sorted(by_muni.items()):
        if not recs:
            results.append(MunicipalityHealthcareSupplyFeatureRecord(
                feature_id=f"hcs_feat:{pref}:{muni}",
                study_area_id=study_area_id,
                prefecture=pref,
                municipality=muni,
                healthcare_supply_record_count=0,
                healthcare_supply_coverage_status="unavailable",
            ))
            continue

        hospital_count = sum(
            1 for r in recs
            if r.facility_name and "病院" in r.facility_name
        )
        clinic_count = sum(
            1 for r in recs
            if r.facility_name and any(
                x in r.facility_name for x in ("クリニック", "診療所", "医院")
            )
        )
        other_count = len(recs) - hospital_count - clinic_count

        pop = pop_by_muni.get((pref, muni))
        density: float | None = None
        if pop and pop > 0:
            density = len(recs) / pop * 100000

        results.append(MunicipalityHealthcareSupplyFeatureRecord(
            feature_id=f"hcs_feat:{pref}:{muni}",
            study_area_id=study_area_id,
            prefecture=pref,
            municipality=muni,
            healthcare_supply_record_count=len(recs),
            hospital_count=hospital_count,
            clinic_count=clinic_count,
            other_medical_count=max(0, other_count),
            supply_density_per_100k=density,
            healthcare_supply_coverage_status="available",
            population_denominator_year=pop_year,
            population_denominator_used=pop,
        ))

    return results


# ---------------------------------------------------------------------------
# Manifest / report helpers
# ---------------------------------------------------------------------------


def _write_healthcare_report(
    json_path: Path,
    md_path: Path,
    study_area_id: str,
    records: list[HealthcareSupplyRecord],
    features: list[MunicipalityHealthcareSupplyFeatureRecord],
    issues: list[HealthcareIngestionIssue],
    source_available: bool,
) -> None:
    errors = sum(1 for i in issues if i.severity == "error")
    available_count = sum(1 for f in features if f.healthcare_supply_coverage_status == "available")

    report = {
        "study_area_id": study_area_id,
        "source_available": source_available,
        "healthcare_supply_record_count": len(records),
        "municipality_healthcare_feature_count": len(features),
        "municipalities_with_healthcare_data": available_count,
        "municipalities_without_healthcare_data": len(features) - available_count,
        "issue_count": len(issues),
        "issue_counts_by_severity": {
            "error": errors,
            "warning": sum(1 for i in issues if i.severity == "warning"),
            "info": sum(1 for i in issues if i.severity == "info"),
        },
        "blocking_errors": errors,
        "healthcare_supply_score_available": available_count > 0,
    }
    write_json(json_path, report)

    lines = [
        f"# Healthcare Supply Ingestion Report — {study_area_id}",
        "",
        f"Source available: {source_available}",
        f"Healthcare supply records: {len(records)}",
        f"Municipality healthcare features: {len(features)}",
        f"Municipalities with data: {available_count}",
        f"Issues: {len(issues)} (errors={errors})",
        "",
        "## Status",
        (
            "Healthcare supply scoring available for municipalities with real data."
            if available_count > 0
            else (
                "Healthcare supply score unavailable. "
                "Set `YAHOO_CLIENT_ID` and re-run `scripts/ingest_healthcare_supply.py` "
                "to enable real healthcare supply features."
            )
        ),
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_healthcare_supply_ingestion(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
    allow_network: bool = False,
    prefecture_limit: int | None = None,
    cache_only: bool = False,
) -> HealthcareIngestionResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path

    if not cfg_path.exists():
        return HealthcareIngestionResult(
            run_id=run_id, study_area_id="unknown", source_available=False
        )

    study_area, config = load_study_area_config(cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id
    target_prefectures = list(study_area.target_prefectures)

    records_path = root / out.get(
        "healthcare_supply_records",
        ".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_records.jsonl",
    )
    features_path = root / out.get(
        "municipality_healthcare_supply_features",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_healthcare_supply_features.jsonl",
    )
    manifest_path = root / out.get(
        "healthcare_supply_ingestion_manifest",
        ".cache/study_area/tokyo_aichi_osaka/healthcare_supply_ingestion_manifest.json",
    )
    issues_path = root / out.get(
        "healthcare_supply_ingestion_issues",
        ".cache/study_area/tokyo_aichi_osaka/healthcare_supply_ingestion_issues.jsonl",
    )
    report_json_path = root / out.get(
        "healthcare_supply_ingestion_report_json",
        ".cache/study_area/tokyo_aichi_osaka/healthcare_supply_ingestion_report.json",
    )
    report_md_path = root / out.get(
        "healthcare_supply_ingestion_report_markdown",
        ".cache/study_area/tokyo_aichi_osaka/healthcare_supply_ingestion_report.md",
    )

    # Load population features for municipality list and population denominators
    pop_feat_path = root / out.get(
        "population_features",
        ".data/interim/study_area/tokyo_aichi_osaka/population_features.jsonl",
    )
    all_municipalities: list[tuple[str, str]] = []
    pop_by_muni: dict[tuple[str, str], float] = {}
    pop_year: int | None = None
    known_munis: list[str] = []

    if pop_feat_path.exists():
        for line in pop_feat_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                key = (r["prefecture"], r["municipality"])
                all_municipalities.append(key)
                known_munis.append(r["municipality"])
                # Use 2020 population as denominator
                pop_val = r.get("population_total_by_year", {}).get("2020")
                if pop_val is not None:
                    pop_by_muni[key] = float(pop_val)
                    pop_year = 2020

    issues: list[HealthcareIngestionIssue] = []
    issue_seq = 0
    records: list[HealthcareSupplyRecord] = []
    retrieved_at = now_utc().isoformat()

    api_key = _get_api_key()
    cache_dir = root / ".data" / "api_raw" / "yahoo"

    source_available = False

    # Apply prefecture limit for smoke runs
    if prefecture_limit is not None:
        target_prefectures = target_prefectures[:prefecture_limit]

    has_cache = any(cache_dir.glob("yahoo_*.json"))

    if not api_key and not has_cache:
        issue_seq += 1
        issues.append(HealthcareIngestionIssue(
            issue_id=f"hcs:source_credentials_unavailable:{issue_seq}",
            severity="warning",
            issue_code="source_credentials_unavailable",
            message=(
                f"Yahoo Local Search API key not found in environment variable "
                f"'{YAHOO_CLIENT_ID_ENV}' and no cached responses found at {cache_dir}. "
                "Healthcare supply data unavailable for this run."
            ),
            study_area_id=sid,
            context={"env_var": YAHOO_CLIENT_ID_ENV, "cache_dir": str(cache_dir)},
            recommended_action=(
                f"Set the '{YAHOO_CLIENT_ID_ENV}' environment variable with a valid "
                "Yahoo Developer Network client ID, or place cached API responses in "
                f"{cache_dir}/."
            ),
        ))
    elif not allow_network and not cache_only and not has_cache:
        issue_seq += 1
        issues.append(HealthcareIngestionIssue(
            issue_id=f"hcs:network_not_enabled:{issue_seq}",
            severity="warning",
            issue_code="network_not_enabled",
            message=(
                "Yahoo Local Search API key is present but --allow-network was not passed. "
                "No live API call will be made. Use --allow-network to enable network access."
            ),
            study_area_id=sid,
            context={"env_var": YAHOO_CLIENT_ID_ENV},
            recommended_action="Pass --allow-network to enable live API calls.",
        ))
    else:
        source_available = True
        seen_ids: set[str] = set()

        for pref in target_prefectures:
            for query_template in [HOSPITAL_QUERY_TEMPLATE, CLINIC_QUERY_TEMPLATE]:
                query = query_template.format(prefecture=pref)
                for page in range(YAHOO_MAX_PAGES):
                    start = page * YAHOO_PAGE_SIZE + 1
                    data = _fetch_yahoo_local(
                        query, api_key or "", start, cache_dir,
                        allow_network=(allow_network and not cache_only),
                    )
                    if data is None:
                        issue_seq += 1
                        issues.append(HealthcareIngestionIssue(
                            issue_id=f"hcs:fetch_failed:{pref}:{page}:{issue_seq}",
                            severity="warning",
                            issue_code="healthcare_data_fetch_failed",
                            message=f"Failed to fetch Yahoo Local Search for query '{query}', page {page+1}.",
                            study_area_id=sid,
                            context={"prefecture": pref, "query": query, "page": page + 1},
                            recommended_action="Check API credentials and network access.",
                        ))
                        break

                    rhash = _response_hash(data)
                    features_list = data.get("Feature") or []
                    for feat in features_list:
                        rec = _normalize_yahoo_record(
                            feat, query, retrieved_at, rhash,
                            target_prefectures, known_munis,
                        )
                        if rec and rec.supply_record_id not in seen_ids:
                            seen_ids.add(rec.supply_record_id)
                            records.append(rec)

                    # Stop paging if fewer results than page size
                    if len(features_list) < YAHOO_PAGE_SIZE:
                        break

    # Check municipality coverage for records without municipality
    no_muni_count = sum(1 for r in records if r.municipality is None and r.prefecture is not None)
    if no_muni_count > 0:
        issue_seq += 1
        issues.append(HealthcareIngestionIssue(
            issue_id=f"hcs:municipality_parse_partial:{issue_seq}",
            severity="info",
            issue_code="municipality_parsing_partial",
            message=(
                f"{no_muni_count} healthcare supply records could not be assigned to a "
                "municipality via address parsing. They are excluded from municipality-level features."
            ),
            study_area_id=sid,
            context={"no_municipality_count": no_muni_count},
            recommended_action=(
                "Use geocoding or a municipality boundary lookup in a later phase "
                "to assign municipalities to unresolved records."
            ),
        ))

    features = _build_healthcare_features(
        records, all_municipalities, pop_by_muni, sid, pop_year
    )

    write_jsonl(records_path, records)
    write_jsonl(features_path, features)
    write_jsonl(issues_path, issues)

    sev_counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for iss in issues:
        sev_counts[iss.severity] = sev_counts.get(iss.severity, 0) + 1

    write_json(manifest_path, {
        "run_id": run_id,
        "generated_at": retrieved_at,
        "study_area_id": sid,
        "source_available": source_available,
        "output_files": {
            "healthcare_supply_records": str(records_path),
            "municipality_healthcare_supply_features": str(features_path),
        },
        "record_counts": {
            "healthcare_supply_records": len(records),
            "municipality_healthcare_supply_features": len(features),
            "issues": len(issues),
        },
        "issue_counts_by_severity": sev_counts,
    })

    _write_healthcare_report(
        report_json_path, report_md_path, sid,
        records, features, issues, source_available,
    )

    return HealthcareIngestionResult(
        run_id=run_id,
        study_area_id=sid,
        source_available=source_available,
        records_ingested=len(records),
        municipality_features_written=len(features),
        issue_count=len(issues),
        blocking_error_count=sev_counts.get("error", 0),
        output_paths={
            "healthcare_supply_records": str(records_path),
            "municipality_healthcare_supply_features": str(features_path),
            "healthcare_supply_ingestion_manifest": str(manifest_path),
            "healthcare_supply_ingestion_issues": str(issues_path),
            "healthcare_supply_ingestion_report_json": str(report_json_path),
            "healthcare_supply_ingestion_report_markdown": str(report_md_path),
        },
    )
