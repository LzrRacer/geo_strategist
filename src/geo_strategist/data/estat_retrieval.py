"""Controlled e-Stat API retrieval adapter for LLM validation experiments (Phase 7+).

Purpose: future LLM validation experiments (E3+) where a model may request
official statistics through a structured adapter. No LLMs are run here.

Supported operations:
  - estat_stats_list_search  (wraps getStatsList — table metadata only)
  - estat_stats_data_fetch   (wraps getStatsData — actual cell values)

Network is disabled by default. Pass allow_network=True to enable live calls.
Responses are cached; no secret values are logged.
Credentials come from ESTAT_APP_ID environment variable only.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.settings import load_settings


ESTAT_API_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"
ESTAT_CACHE_ROOT = Path(".data/api_raw/estat")
ESTAT_LOG_ROOT = Path(".cache/study_area/tokyo_aichi_osaka")

_DEFAULT_TIMEOUT = 30
_DEFAULT_RETRY_LIMIT = 2
_RETRY_BACKOFF = 2.0


class EStatRetrievalIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_id: str
    issue_code: str
    severity: str
    detail: str
    query_params: dict[str, str] = Field(default_factory=dict)


class EStatRetrievalLogRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_id: str
    retrieved_at: str
    query_params: dict[str, str]
    endpoint: str
    cache_hit: bool
    cache_key: str
    response_status: int | None = None
    result_count: int | None = None
    issue_code: str | None = None


class EStatRetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    source_available: bool = False
    allow_network: bool = False
    queries_attempted: int = 0
    cache_hits: int = 0
    live_fetches: int = 0
    issue_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


def _cache_key(params: dict[str, str]) -> str:
    canonical = json.dumps(dict(sorted(params.items())), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _cache_path(cache_root: Path, cache_key: str, endpoint: str) -> Path:
    safe_ep = endpoint.replace("/", "_").strip("_")
    return cache_root / f"estat_{safe_ep}_{cache_key}.json"


def _read_cache(path: Path) -> dict[str, Any] | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _write_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_estat(
    endpoint: str,
    params: dict[str, str],
    allow_network: bool = False,
    cache_only: bool = False,
    cache_root: Path = ESTAT_CACHE_ROOT,
    timeout: int = _DEFAULT_TIMEOUT,
) -> tuple[dict[str, Any] | None, bool, EStatRetrievalIssue | None]:
    """Fetch from e-Stat API with cache-first, network-gated behavior.

    Returns (response_data, cache_hit, issue_or_None).
    Credentials must be in ESTAT_APP_ID env var.
    """
    app_id = load_settings().estat_app_id
    key = _cache_key(params)
    cache_file = _cache_path(cache_root, key, endpoint)

    cached = _read_cache(cache_file)
    if cached is not None:
        return cached, True, None

    if not app_id:
        iss = EStatRetrievalIssue(
            issue_id=str(uuid.uuid4()),
            issue_code="source_credentials_unavailable",
            severity="warning",
            detail="ESTAT_APP_ID not set. Set it and re-run with --allow-network.",
            query_params={k: v for k, v in params.items() if k != "appId"},
        )
        return None, False, iss

    if cache_only or not allow_network:
        iss = EStatRetrievalIssue(
            issue_id=str(uuid.uuid4()),
            issue_code="network_not_enabled",
            severity="warning",
            detail="Cache miss and network not enabled. Pass --allow-network to fetch live data.",
            query_params={k: v for k, v in params.items() if k != "appId"},
        )
        return None, False, iss

    # Live fetch
    try:
        import requests
    except ImportError:
        iss = EStatRetrievalIssue(
            issue_id=str(uuid.uuid4()),
            issue_code="network_not_enabled",
            severity="error",
            detail="requests library not available.",
            query_params={},
        )
        return None, False, iss

    full_params = {"appId": app_id, **params}
    url = f"{ESTAT_API_BASE}/{endpoint}"

    last_exc: Exception | None = None
    for attempt in range(_DEFAULT_RETRY_LIMIT):
        try:
            resp = requests.get(
                url,
                params=full_params,
                timeout=timeout,
                headers={"User-Agent": "geo-strategist/0.1 (research)"},
            )
            if resp.status_code == 200:
                data = resp.json()
                _write_cache(cache_file, data)
                return data, False, None
            iss = EStatRetrievalIssue(
                issue_id=str(uuid.uuid4()),
                issue_code="network_not_enabled",
                severity="error",
                detail=f"e-Stat API returned HTTP {resp.status_code}.",
                query_params={k: v for k, v in params.items() if k != "appId"},
            )
            return None, False, iss
        except Exception as exc:
            last_exc = exc
            if attempt < _DEFAULT_RETRY_LIMIT - 1:
                time.sleep(_RETRY_BACKOFF)

    iss = EStatRetrievalIssue(
        issue_id=str(uuid.uuid4()),
        issue_code="network_not_enabled",
        severity="error",
        detail=f"e-Stat API request failed: {last_exc}",
        query_params={k: v for k, v in params.items() if k != "appId"},
    )
    return None, False, iss


class EStatValueRecord(BaseModel):
    """One normalized cell value from a getStatsData response."""
    model_config = ConfigDict(extra="forbid")

    record_id: str
    retrieval_id: str
    source_request_id: str | None = None
    stats_data_id: str
    table_title: str | None = None
    survey_date: str | None = None
    area_code: str | None = None
    area_name: str | None = None
    cat01_code: str | None = None
    cat01_name: str | None = None
    cat02_code: str | None = None
    cat02_name: str | None = None
    time_code: str | None = None
    time_name: str | None = None
    unit: str | None = None
    raw_value: str | None = None
    numeric_value: float | None = None
    numeric_parse_ok: bool = False
    provenance: str = "estat_getStatsData"
    cache_path: str | None = None
    retrieval_status: str = "ok"


class EStatStatsDataResult(BaseModel):
    """Result of a fetch_estat_stats_data call."""
    model_config = ConfigDict(extra="forbid")

    retrieval_id: str
    source_request_id: str | None = None
    stats_data_id: str
    status: str  # ok | no_data | no_credential | network_disabled | cache_miss | parse_error
    cache_hit: bool = False
    live_fetch: bool = False
    record_count: int = 0
    records: list[EStatValueRecord] = Field(default_factory=list)
    issue: EStatRetrievalIssue | None = None
    retrieved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cache_path: str | None = None
    error_message: str | None = None


def _parse_stats_data_records(
    data: dict[str, Any],
    stats_data_id: str,
    retrieval_id: str,
    source_request_id: str | None,
    cache_path: str | None,
) -> list[EStatValueRecord]:
    """Parse getStatsData JSON into EStatValueRecord list.

    e-Stat getStatsData structure:
      GET_STATS_DATA > STATISTICAL_DATA > CLASS_INF > CLASS_OBJ[]
      GET_STATS_DATA > STATISTICAL_DATA > DATA_INF > VALUE[]
    """
    records: list[EStatValueRecord] = []

    stat_data = data.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
    table_inf = stat_data.get("TABLE_INF", {})
    table_title: str | None = None
    survey_date: str | None = None
    if isinstance(table_inf, dict):
        title = table_inf.get("TITLE")
        if isinstance(title, dict):
            table_title = title.get("$")
        elif isinstance(title, str):
            table_title = title
        sd = table_inf.get("SURVEY_DATE")
        if sd is not None:
            survey_date = str(sd)

    # Build lookup: category code → name, per CLASS_OBJ
    class_inf = stat_data.get("CLASS_INF", {})
    class_objs = class_inf.get("CLASS_OBJ", [])
    if isinstance(class_objs, dict):
        class_objs = [class_objs]
    class_lookup: dict[str, dict[str, str]] = {}
    for obj in class_objs:
        obj_id = obj.get("@id", "")
        classes = obj.get("CLASS", [])
        if isinstance(classes, dict):
            classes = [classes]
        mapping: dict[str, str] = {}
        for cls in classes:
            code = cls.get("@code", "")
            name = cls.get("@name", "")
            mapping[code] = name
        class_lookup[obj_id] = mapping

    data_inf = stat_data.get("DATA_INF", {})
    values = data_inf.get("VALUE", [])
    if isinstance(values, dict):
        values = [values]

    for v in values:
        if not isinstance(v, dict):
            continue
        raw = v.get("$")
        numeric: float | None = None
        parse_ok = False
        if raw is not None:
            try:
                numeric = float(raw)
                parse_ok = True
            except (ValueError, TypeError):
                pass

        area_code = v.get("@area")
        area_name: str | None = None
        area_map = class_lookup.get("area", class_lookup.get("地域", {}))
        if area_code and area_code in area_map:
            area_name = area_map[area_code]

        cat01_code = v.get("@cat01")
        cat01_name = class_lookup.get("cat01", {}).get(cat01_code or "", None)
        cat02_code = v.get("@cat02")
        cat02_name = class_lookup.get("cat02", {}).get(cat02_code or "", None)
        time_code = v.get("@time")
        time_name = class_lookup.get("time", class_lookup.get("時間軸", {})).get(time_code or "", None)

        records.append(EStatValueRecord(
            record_id=str(uuid.uuid4()),
            retrieval_id=retrieval_id,
            source_request_id=source_request_id,
            stats_data_id=stats_data_id,
            table_title=table_title,
            survey_date=survey_date,
            area_code=area_code,
            area_name=area_name,
            cat01_code=cat01_code,
            cat01_name=cat01_name,
            cat02_code=cat02_code,
            cat02_name=cat02_name,
            time_code=time_code,
            time_name=time_name,
            unit=v.get("@unit"),
            raw_value=raw,
            numeric_value=numeric,
            numeric_parse_ok=parse_ok,
            provenance="estat_getStatsData",
            cache_path=cache_path,
            retrieval_status="ok",
        ))
    return records


def fetch_estat_stats_data(
    *,
    stats_data_id: str,
    cd_area: str | None = None,
    cd_cat01: str | None = None,
    cd_cat02: str | None = None,
    cd_time: str | None = None,
    lang: str = "J",
    retrieval_id: str | None = None,
    source_request_id: str | None = None,
    cache_only: bool = True,
    allow_network: bool = False,
    cache_root: Path = ESTAT_CACHE_ROOT,
) -> EStatStatsDataResult:
    """Fetch cell values from e-Stat getStatsData.

    Network is gated behind allow_network=True.
    Credentials come from ESTAT_APP_ID only; never logged.
    Results are cached under cache_root; cache is checked before any network call.
    Actual domain values are not fabricated — if no data is available, status is 'no_data'.
    """
    rid = retrieval_id or str(uuid.uuid4())
    params: dict[str, str] = {"statsDataId": stats_data_id, "lang": lang}
    if cd_area:
        params["cdArea"] = cd_area
    if cd_cat01:
        params["cdCat01"] = cd_cat01
    if cd_cat02:
        params["cdCat02"] = cd_cat02
    if cd_time:
        params["cdTime"] = cd_time

    data, cache_hit, issue = fetch_estat(
        endpoint="getStatsData",
        params=params,
        allow_network=allow_network,
        cache_only=cache_only,
        cache_root=cache_root,
    )

    cache_key = _cache_key(params)
    cp = str(_cache_path(cache_root, cache_key, "getStatsData"))

    if issue is not None:
        status_map = {
            "source_credentials_unavailable": "no_credential",
            "network_not_enabled": "network_disabled",
        }
        status = status_map.get(issue.issue_code, "no_data")
        err_code = issue.issue_code
        if err_code == "source_credentials_unavailable":
            err_msg = "ESTAT_APP_ID not set. Set it and re-run with --allow-network."
        else:
            err_msg = issue.detail
        return EStatStatsDataResult(
            retrieval_id=rid,
            source_request_id=source_request_id,
            stats_data_id=stats_data_id,
            status=status,
            cache_hit=False,
            live_fetch=False,
            record_count=0,
            issue=issue,
            cache_path=cp,
            error_message=err_msg,
        )

    if data is None:
        return EStatStatsDataResult(
            retrieval_id=rid,
            source_request_id=source_request_id,
            stats_data_id=stats_data_id,
            status="no_data",
            cache_hit=cache_hit,
            live_fetch=not cache_hit,
            record_count=0,
            cache_path=cp,
            error_message="No data returned by API.",
        )

    # Check API-level error
    result_status = (
        data.get("GET_STATS_DATA", {}).get("RESULT", {}).get("STATUS", -1)
    )
    if result_status != 0:
        err_msg_api = (
            data.get("GET_STATS_DATA", {}).get("RESULT", {}).get("ERROR_MSG", "API error")
        )
        return EStatStatsDataResult(
            retrieval_id=rid,
            source_request_id=source_request_id,
            stats_data_id=stats_data_id,
            status="no_data",
            cache_hit=cache_hit,
            live_fetch=not cache_hit,
            record_count=0,
            cache_path=cp,
            error_message=f"e-Stat API error: {err_msg_api}",
        )

    try:
        records = _parse_stats_data_records(
            data=data,
            stats_data_id=stats_data_id,
            retrieval_id=rid,
            source_request_id=source_request_id,
            cache_path=cp,
        )
    except Exception as exc:
        iss = EStatRetrievalIssue(
            issue_id=str(uuid.uuid4()),
            issue_code="parse_error",
            severity="error",
            detail=f"Failed to parse getStatsData response: {exc}",
            query_params=params,
        )
        return EStatStatsDataResult(
            retrieval_id=rid,
            source_request_id=source_request_id,
            stats_data_id=stats_data_id,
            status="parse_error",
            cache_hit=cache_hit,
            live_fetch=not cache_hit,
            record_count=0,
            issue=iss,
            cache_path=cp,
            error_message=str(exc),
        )

    return EStatStatsDataResult(
        retrieval_id=rid,
        source_request_id=source_request_id,
        stats_data_id=stats_data_id,
        status="ok" if records else "no_data",
        cache_hit=cache_hit,
        live_fetch=not cache_hit,
        record_count=len(records),
        records=records,
        cache_path=cp,
    )


def run_estat_retrieval_smoke(
    allow_network: bool = False,
    cache_only: bool = False,
    limit: int = 1,
    repo_root: Path = Path("."),
) -> EStatRetrievalResult:
    """Smoke test for e-Stat retrieval readiness. Does not run any LLM."""
    run_id = str(uuid.uuid4())
    root = repo_root.resolve()
    cache_root = root / ".data" / "api_raw" / "estat"
    log_root = root / ESTAT_LOG_ROOT

    app_id_present = bool(load_settings().estat_app_id)

    # Smoke query: list stat categories (getStatsList with minimal params)
    smoke_queries = [
        {
            "endpoint": "getStatsList",
            "params": {"searchWord": "人口", "limit": str(min(limit, 5))},
        }
    ][:limit]

    issues: list[EStatRetrievalIssue] = []
    log_records: list[EStatRetrievalLogRecord] = []
    cache_hits = 0
    live_fetches = 0

    from geo_strategist.data.normalization import now_utc

    for query in smoke_queries:
        endpoint = query["endpoint"]
        params = query["params"]
        cache_key = _cache_key(params)
        retrieved_at = now_utc().isoformat()

        data, hit, iss = fetch_estat(
            endpoint=endpoint,
            params=params,
            allow_network=allow_network,
            cache_only=cache_only,
            cache_root=cache_root,
        )

        if hit:
            cache_hits += 1
        elif data is not None:
            live_fetches += 1

        result_count: int | None = None
        if data:
            try:
                result_count = data.get("GET_STATS_LIST", {}).get("DATALIST_INF", {}).get(
                    "NUMBER", None
                )
                if result_count is not None:
                    result_count = int(result_count)
            except Exception:
                pass

        log_records.append(EStatRetrievalLogRecord(
            log_id=str(uuid.uuid4()),
            retrieved_at=retrieved_at,
            query_params=params,
            endpoint=endpoint,
            cache_hit=hit,
            cache_key=cache_key,
            response_status=200 if data is not None else None,
            result_count=result_count,
            issue_code=iss.issue_code if iss else None,
        ))

        if iss:
            issues.append(iss)

    # Write outputs
    manifest_path = log_root / "estat_retrieval_manifest.json"
    issues_path = log_root / "estat_retrieval_issues.jsonl"
    log_path = log_root / "estat_retrieval_log.jsonl"
    log_root.mkdir(parents=True, exist_ok=True)

    from geo_strategist.data.views.common import write_json, write_jsonl
    from geo_strategist.data.normalization import now_utc as _now

    write_jsonl(log_path, log_records)
    write_jsonl(issues_path, issues)
    write_json(manifest_path, {
        "run_id": run_id,
        "generated_at": _now().isoformat(),
        "source_available": app_id_present,
        "allow_network": allow_network,
        "queries_attempted": len(smoke_queries),
        "cache_hits": cache_hits,
        "live_fetches": live_fetches,
        "issue_count": len(issues),
        "issue_codes": [i.issue_code for i in issues],
    })

    return EStatRetrievalResult(
        run_id=run_id,
        source_available=app_id_present,
        allow_network=allow_network,
        queries_attempted=len(smoke_queries),
        cache_hits=cache_hits,
        live_fetches=live_fetches,
        issue_count=len(issues),
        output_paths={
            "estat_retrieval_manifest": str(manifest_path),
            "estat_retrieval_issues": str(issues_path),
            "estat_retrieval_log": str(log_path),
        },
    )
