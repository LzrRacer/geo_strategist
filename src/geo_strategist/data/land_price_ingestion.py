"""Real land-price ingestion from MLIT Reinfolib API (Phase 6)."""

from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


# ---------------------------------------------------------------------------
# MLIT Reinfolib API constants
# ---------------------------------------------------------------------------

REINFOLIB_BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external/XIT001"
REINFOLIB_API_KEY_ENV = "REINFOLIB_API_KEY"

# Prefecture codes used by the MLIT Reinfolib API
PREFECTURE_CODES: dict[str, str] = {
    "北海道": "01", "青森県": "02", "岩手県": "03", "宮城県": "04",
    "秋田県": "05", "山形県": "06", "福島県": "07", "茨城県": "08",
    "栃木県": "09", "群馬県": "10", "埼玉県": "11", "千葉県": "12",
    "東京都": "13", "神奈川県": "14", "新潟県": "15", "富山県": "16",
    "石川県": "17", "福井県": "18", "山梨県": "19", "長野県": "20",
    "岐阜県": "21", "静岡県": "22", "愛知県": "23", "三重県": "24",
    "滋賀県": "25", "京都府": "26", "大阪府": "27", "兵庫県": "28",
    "奈良県": "29", "和歌山県": "30", "鳥取県": "31", "島根県": "32",
    "岡山県": "33", "広島県": "34", "山口県": "35", "徳島県": "36",
    "香川県": "37", "愛媛県": "38", "高知県": "39", "福岡県": "40",
    "佐賀県": "41", "長崎県": "42", "熊本県": "43", "大分県": "44",
    "宮崎県": "45", "鹿児島県": "46", "沖縄県": "47",
}

# Survey types: 01=地価公示 (official land price), 02=地価調査 (land price survey)
LAND_SURVEY_TYPES = ["01", "02"]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class LandPriceRecord(BaseModel):
    """Normalized land price record from MLIT Reinfolib."""

    model_config = ConfigDict(extra="forbid")

    land_record_id: str = Field(min_length=1)
    source_type: str = "reinfolib_api"
    source_name: str = "MLIT Real Estate Information Library (不動産情報ライブラリ)"
    source_url: str = REINFOLIB_BASE_URL
    retrieved_at: str
    prefecture: str
    municipality: str | None = None
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    price_value: float
    price_unit: str = "JPY/m2"
    price_year: int
    land_use: str | None = None
    survey_type: str | None = None
    source_record_id: str | None = None
    response_hash: str


class MunicipalityLandFeatureRecord(BaseModel):
    """Aggregated land-price features for one municipality."""

    model_config = ConfigDict(extra="forbid")

    feature_id: str = Field(min_length=1)
    study_area_id: str
    prefecture: str
    municipality: str
    land_price_record_count: int = 0
    land_price_median: float | None = None
    land_price_mean: float | None = None
    land_price_min: float | None = None
    land_price_max: float | None = None
    land_price_latest_year: int | None = None
    land_price_unit: str = "JPY/m2"
    land_price_coverage_status: str = "unavailable"


class LandIngestionIssue(BaseModel):
    """Issue from land price ingestion."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: str
    issue_code: str
    message: str
    study_area_id: str
    context: dict[str, str | int | float | None] = Field(default_factory=dict)
    recommended_action: str


class LandIngestionResult(BaseModel):
    """Summary of land price ingestion run."""

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
    return os.environ.get(REINFOLIB_API_KEY_ENV)


def _response_hash(data: Any) -> str:
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _fetch_reinfolib(
    prefecture_code: str,
    survey_type: str,
    year: int,
    api_key: str,
    cache_dir: Path,
    sleep_sec: float = 0.5,
    allow_network: bool = True,
) -> tuple[dict[str, Any] | None, bool]:
    """Fetch land price data; returns (data, from_cache)."""
    import requests

    cache_key = f"reinfolib_{prefecture_code}_{survey_type}_{year}.json"
    cache_path = cache_dir / cache_key

    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8")), True

    if not allow_network:
        return None, False

    try:
        resp = requests.get(
            REINFOLIB_BASE_URL,
            params={
                "response_format": "json",
                "year": str(year),
                "area": prefecture_code,
                "priceClassification": survey_type,
            },
            headers={
                "Ocp-Apim-Subscription-Key": api_key,
                "User-Agent": "geo-strategist/0.1 (research)",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        time.sleep(sleep_sec)
        return data, False
    except Exception:
        return None, False


def _parse_municipality_from_address(address: str | None, known_munis: list[str]) -> str | None:
    """Best-effort municipality extraction from a Japanese address string."""
    if not address:
        return None
    for muni in sorted(known_munis, key=len, reverse=True):
        if muni in address:
            return muni
    return None


_YEAR_RE = re.compile(r"(\d{4})年")


def _parse_year_from_period(period: str | None) -> int | None:
    """Extract year from Reinfolib Period field (e.g. '2023年第1四半期' -> 2023)."""
    if not period:
        return None
    m = _YEAR_RE.search(str(period))
    return int(m.group(1)) if m else None


def _normalize_reinfolib_record(
    raw: dict[str, Any],
    retrieved_at: str,
    response_hash: str,
    known_munis: list[str],
) -> LandPriceRecord | None:
    """Normalize one raw Reinfolib record to LandPriceRecord.

    Accepts the actual Reinfolib XIT001 JSON format (English keys) as well as
    legacy Japanese-key formats for robustness.
    Only keeps 宅地(土地) records with a valid UnitPrice per m2.
    """
    try:
        rec_type = raw.get("Type") or raw.get("種類", "")
        # Only process pure land (宅地(土地)) records — excludes buildings and condos
        if rec_type and "宅地(土地)" not in rec_type:
            # Accept records that have no Type field (unknown format) but not explicit non-land
            if rec_type:
                return None

        # Price per m2: prefer UnitPrice, fall back to computing TradePrice/Area
        unit_price_raw = raw.get("UnitPrice") or raw.get("価格") or raw.get("price") or raw.get("landPrice")
        trade_price_raw = raw.get("TradePrice")
        area_raw = raw.get("Area")

        price_float: float | None = None
        if unit_price_raw and str(unit_price_raw).strip():
            try:
                price_float = float(str(unit_price_raw).replace(",", ""))
            except (TypeError, ValueError):
                pass

        if price_float is None and trade_price_raw and area_raw:
            try:
                tp = float(str(trade_price_raw).replace(",", ""))
                ar = float(str(area_raw).replace(",", ""))
                if ar > 0:
                    price_float = tp / ar
            except (TypeError, ValueError):
                pass

        if price_float is None or price_float <= 0:
            return None

        # Year: prefer Period field over explicit year fields
        period = raw.get("Period")
        year_int = _parse_year_from_period(period)
        if year_int is None:
            year_raw = raw.get("調査年度") or raw.get("year") or raw.get("surveyYear")
            try:
                year_int = int(year_raw)
            except (TypeError, ValueError):
                return None

        # Prefecture and municipality: direct fields (Reinfolib returns them explicitly)
        pref = (
            raw.get("Prefecture")
            or raw.get("都道府県名")
            or raw.get("prefecture", "")
        )
        muni = (
            raw.get("Municipality")
            or raw.get("市区町村名")
            or raw.get("municipality")
        )
        # Fall back to address-based parsing only if municipality field is absent
        if not muni:
            address = raw.get("address") or raw.get("所在地")
            muni = _parse_municipality_from_address(address, known_munis) if address else None

        district = raw.get("DistrictName") or raw.get("地区名", "")
        address_full = f"{pref}{muni or ''}{district}" if district else None

        record_id = (
            raw.get("MunicipalityCode", "")
            + "_"
            + raw.get("DistrictCode", "")
            + "_"
            + str(year_int)
        )

        return LandPriceRecord(
            land_record_id=f"land:reinfolib:{pref}:{muni or 'unknown'}:{year_int}:{record_id}",
            retrieved_at=retrieved_at,
            prefecture=pref,
            municipality=muni,
            address=address_full,
            latitude=None,
            longitude=None,
            price_value=price_float,
            price_unit="JPY/m2",
            price_year=year_int,
            land_use=raw.get("Use") or raw.get("用途") or raw.get("landUse"),
            survey_type=raw.get("Type") or raw.get("PriceCategory") or raw.get("surveyType"),
            source_record_id=record_id,
            response_hash=response_hash,
        )
    except (TypeError, ValueError, Exception):
        return None


# ---------------------------------------------------------------------------
# Feature aggregation
# ---------------------------------------------------------------------------


def _build_land_features(
    records: list[LandPriceRecord],
    all_municipalities: list[tuple[str, str]],
    study_area_id: str,
) -> list[MunicipalityLandFeatureRecord]:
    by_muni: dict[tuple[str, str], list[LandPriceRecord]] = {}
    for pref, muni in all_municipalities:
        by_muni[(pref, muni)] = []

    for rec in records:
        if rec.municipality:
            key = (rec.prefecture, rec.municipality)
            if key in by_muni:
                by_muni[key].append(rec)

    results: list[MunicipalityLandFeatureRecord] = []
    for (pref, muni), recs in sorted(by_muni.items()):
        if not recs:
            results.append(MunicipalityLandFeatureRecord(
                feature_id=f"land_feat:{pref}:{muni}",
                study_area_id=study_area_id,
                prefecture=pref,
                municipality=muni,
                land_price_record_count=0,
                land_price_coverage_status="unavailable",
            ))
            continue

        prices = [r.price_value for r in recs]
        years = [r.price_year for r in recs]
        results.append(MunicipalityLandFeatureRecord(
            feature_id=f"land_feat:{pref}:{muni}",
            study_area_id=study_area_id,
            prefecture=pref,
            municipality=muni,
            land_price_record_count=len(recs),
            land_price_median=statistics.median(prices),
            land_price_mean=statistics.mean(prices),
            land_price_min=min(prices),
            land_price_max=max(prices),
            land_price_latest_year=max(years),
            land_price_unit="JPY/m2",
            land_price_coverage_status="available",
        ))

    return results


# ---------------------------------------------------------------------------
# Manifest / report helpers
# ---------------------------------------------------------------------------


def _write_land_report(
    json_path: Path,
    md_path: Path,
    study_area_id: str,
    records: list[LandPriceRecord],
    features: list[MunicipalityLandFeatureRecord],
    issues: list[LandIngestionIssue],
    source_available: bool,
) -> None:
    errors = sum(1 for i in issues if i.severity == "error")
    available_count = sum(1 for f in features if f.land_price_coverage_status == "available")

    report = {
        "study_area_id": study_area_id,
        "source_available": source_available,
        "land_price_record_count": len(records),
        "municipality_land_feature_count": len(features),
        "municipalities_with_land_data": available_count,
        "municipalities_without_land_data": len(features) - available_count,
        "issue_count": len(issues),
        "issue_counts_by_severity": {
            "error": errors,
            "warning": sum(1 for i in issues if i.severity == "warning"),
            "info": sum(1 for i in issues if i.severity == "info"),
        },
        "blocking_errors": errors,
        "land_score_available": available_count > 0,
        "land_score_unavailable_reason": (
            None if available_count > 0
            else "No real land-price records retrieved. Set REINFOLIB_API_KEY or provide cached responses."
        ),
    }
    write_json(json_path, report)

    lines = [
        f"# Land Price Ingestion Report — {study_area_id}",
        "",
        f"Source available: {source_available}",
        f"Land price records: {len(records)}",
        f"Municipality land features: {len(features)}",
        f"Municipalities with data: {available_count}",
        f"Issues: {len(issues)} (errors={errors})",
        "",
        "## Status",
        (
            "Land scoring available for municipalities with real data."
            if available_count > 0
            else (
                "Land score unavailable. "
                "Set `REINFOLIB_API_KEY` and re-run `scripts/ingest_land_prices.py` "
                "to enable real land-price features."
            )
        ),
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_land_price_ingestion(
    repo_root: Path = Path("."),
    config_path: str = "configs/study_area_tokyo_aichi_osaka.yaml",
    target_years: list[int] | None = None,
    allow_network: bool = False,
    prefecture_limit: int | None = None,
    cache_only: bool = False,
) -> LandIngestionResult:
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cfg_path = root / config_path

    if not cfg_path.exists():
        return LandIngestionResult(
            run_id=run_id, study_area_id="unknown", source_available=False
        )

    study_area, config = load_study_area_config(cfg_path)
    out = config["outputs"]
    sid = study_area.study_area_id
    target_prefectures = study_area.target_prefectures

    if target_years is None:
        target_years = [2023, 2022]

    records_path = root / out.get(
        "land_price_records",
        ".data/interim/study_area/tokyo_aichi_osaka/land_price_records.jsonl",
    )
    features_path = root / out.get(
        "municipality_land_features",
        ".data/interim/study_area/tokyo_aichi_osaka/municipality_land_features.jsonl",
    )
    manifest_path = root / out.get(
        "land_price_ingestion_manifest",
        ".cache/study_area/tokyo_aichi_osaka/land_price_ingestion_manifest.json",
    )
    issues_path = root / out.get(
        "land_price_ingestion_issues",
        ".cache/study_area/tokyo_aichi_osaka/land_price_ingestion_issues.jsonl",
    )
    report_json_path = root / out.get(
        "land_price_ingestion_report_json",
        ".cache/study_area/tokyo_aichi_osaka/land_price_ingestion_report.json",
    )
    report_md_path = root / out.get(
        "land_price_ingestion_report_markdown",
        ".cache/study_area/tokyo_aichi_osaka/land_price_ingestion_report.md",
    )

    # Load known municipalities from population features for address parsing
    pop_feat_path = root / out.get(
        "population_features",
        ".data/interim/study_area/tokyo_aichi_osaka/population_features.jsonl",
    )
    all_municipalities: list[tuple[str, str]] = []
    if pop_feat_path.exists():
        for line in pop_feat_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                all_municipalities.append((r["prefecture"], r["municipality"]))
    known_munis = [m for _, m in all_municipalities]

    issues: list[LandIngestionIssue] = []
    issue_seq = 0
    records: list[LandPriceRecord] = []
    retrieved_at = now_utc().isoformat()

    api_key = _get_api_key()
    cache_dir = root / ".data" / "api_raw" / "reinfolib"

    source_available = False

    # Apply prefecture limit for smoke runs
    if prefecture_limit is not None:
        target_prefectures = list(target_prefectures)[:prefecture_limit]

    has_cache = any(cache_dir.glob("*.json"))

    if not api_key and not has_cache:
        # No credentials and no cache
        issue_seq += 1
        issues.append(LandIngestionIssue(
            issue_id=f"land:source_credentials_unavailable:{issue_seq}",
            severity="warning",
            issue_code="source_credentials_unavailable",
            message=(
                f"MLIT Reinfolib API key not found in environment variable "
                f"'{REINFOLIB_API_KEY_ENV}' and no cached responses found at "
                f"{cache_dir}. Land price data unavailable for this run."
            ),
            study_area_id=sid,
            context={"env_var": REINFOLIB_API_KEY_ENV, "cache_dir": str(cache_dir)},
            recommended_action=(
                f"Set the '{REINFOLIB_API_KEY_ENV}' environment variable with a valid "
                "MLIT Reinfolib API key, or place cached API responses in "
                f"{cache_dir}/<prefecture_code>_<survey_type>_<year>.json."
            ),
        ))
    elif not allow_network and not cache_only and not has_cache:
        # Credentials present but --allow-network not passed and no cache
        issue_seq += 1
        issues.append(LandIngestionIssue(
            issue_id=f"land:network_not_enabled:{issue_seq}",
            severity="warning",
            issue_code="network_not_enabled",
            message=(
                "MLIT Reinfolib API key is present but --allow-network was not passed. "
                "No live API call will be made. Use --allow-network to enable network access."
            ),
            study_area_id=sid,
            context={"env_var": REINFOLIB_API_KEY_ENV},
            recommended_action="Pass --allow-network to enable live API calls.",
        ))
    else:
        # Try to fetch or load cached data
        source_available = True
        for pref in target_prefectures:
            pref_code = PREFECTURE_CODES.get(pref)
            if not pref_code:
                issue_seq += 1
                issues.append(LandIngestionIssue(
                    issue_id=f"land:unknown_prefecture_code:{pref}:{issue_seq}",
                    severity="warning",
                    issue_code="unknown_prefecture_code",
                    message=f"No prefecture code found for '{pref}'.",
                    study_area_id=sid,
                    context={"prefecture": pref},
                    recommended_action="Add the prefecture code to PREFECTURE_CODES.",
                ))
                continue

            for survey_type in LAND_SURVEY_TYPES:
                for year in target_years:
                    data, from_cache = _fetch_reinfolib(
                        pref_code, survey_type, year,
                        api_key or "", cache_dir,
                        sleep_sec=0.5,
                        allow_network=(allow_network and not cache_only),
                    )
                    if data is None:
                        issue_seq += 1
                        issues.append(LandIngestionIssue(
                            issue_id=f"land:fetch_failed:{pref}:{survey_type}:{year}:{issue_seq}",
                            severity="warning",
                            issue_code="land_data_fetch_failed",
                            message=f"Failed to fetch land data for {pref} (code {pref_code}), survey_type={survey_type}, year={year}.",
                            study_area_id=sid,
                            context={"prefecture": pref, "survey_type": survey_type, "year": year},
                            recommended_action="Check API credentials and network access.",
                        ))
                        continue

                    rhash = _response_hash(data)
                    raw_records = data if isinstance(data, list) else data.get("data") or data.get("records") or []
                    for raw in raw_records:
                        rec = _normalize_reinfolib_record(raw, retrieved_at, rhash, known_munis)
                        if rec:
                            records.append(rec)

    # Build municipality features for all known municipalities
    features = _build_land_features(records, all_municipalities, sid)

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
        "input_files": {},
        "output_files": {
            "land_price_records": str(records_path),
            "municipality_land_features": str(features_path),
        },
        "record_counts": {
            "land_price_records": len(records),
            "municipality_land_features": len(features),
            "issues": len(issues),
        },
        "issue_counts_by_severity": sev_counts,
    })

    _write_land_report(
        report_json_path, report_md_path, sid,
        records, features, issues, source_available,
    )

    return LandIngestionResult(
        run_id=run_id,
        study_area_id=sid,
        source_available=source_available,
        records_ingested=len(records),
        municipality_features_written=len(features),
        issue_count=len(issues),
        blocking_error_count=sev_counts.get("error", 0),
        output_paths={
            "land_price_records": str(records_path),
            "municipality_land_features": str(features_path),
            "land_price_ingestion_manifest": str(manifest_path),
            "land_price_ingestion_issues": str(issues_path),
            "land_price_ingestion_report_json": str(report_json_path),
            "land_price_ingestion_report_markdown": str(report_md_path),
        },
    )
