"""Deterministic e-Stat CLASS_INF inspection for municipality retrieval.

This module inspects cached getStatsData responses. It does not call e-Stat,
does not run LLMs, and does not infer missing area/category codes.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


READINESS_MUNICIPALITY_FILTER_READY = "municipality_filter_ready"
READINESS_NOT_MUNICIPALITY_GRAIN = "not_municipality_grain"
READINESS_NEEDS_MANUAL_DIMENSION_REVIEW = "needs_manual_dimension_review"
READINESS_UNUSABLE = "unusable_for_e3_municipality_validation"

DEFAULT_TARGET_MUNICIPALITIES = ("13102", "13113", "27128", "27102", "23204")
FACILITY_CATEGORY_KEYWORDS = (
    "施設数（総数）",
    "施設数（病院）",
    "施設数（一般診療所）",
    "病院数",
    "病床数",
    "一般診療所",
    "施設数",
)


@dataclass(frozen=True)
class DimensionInspectionIssue:
    issue_id: str
    severity: str
    issue_code: str
    message: str
    stats_data_id: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EStatClassEntry:
    stats_data_id: str
    dimension_id: str
    dimension_name: str | None
    class_code: str
    class_name: str | None
    level: str | None = None
    parent_code: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class EStatDimension:
    stats_data_id: str
    dimension_id: str
    dimension_name: str | None
    class_count: int
    class_codes: tuple[str, ...]
    sample_class_names: tuple[str, ...]
    levels: tuple[str, ...]
    parent_code_count: int
    unit_count: int
    is_area_dimension: bool
    contains_target_municipality_codes: bool
    target_municipality_codes_present: tuple[str, ...]
    target_municipality_codes_missing: tuple[str, ...]


@dataclass(frozen=True)
class EStatTableInspection:
    stats_data_id: str
    table_title: str | None
    stat_name: str | None
    survey_date: str | None
    cache_path: str | None
    status: str
    dimension_count: int
    class_count: int
    has_area_dimension: bool
    area_dimension_id: str | None
    area_dimension_name: str | None
    area_class_count: int
    target_municipality_codes_present: tuple[str, ...]
    target_municipality_codes_missing: tuple[str, ...]
    hospital_facility_category_matches: tuple[dict[str, str | None], ...]
    hospital_facility_category_match_count: int
    readiness_classification: str
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class LoadedGetStatsDataResponse:
    stats_data_id: str
    cache_path: Path
    data: dict[str, Any] | None
    issue: DimensionInspectionIssue | None = None


@dataclass(frozen=True)
class EStatDimensionInspectionReport:
    generated_at: str
    target_municipality_codes: tuple[str, ...]
    table_count: int
    dimension_count: int
    class_count: int
    municipality_filter_ready_count: int
    not_municipality_grain_count: int
    needs_manual_dimension_review_count: int
    unusable_for_e3_municipality_validation_count: int
    suitable_for_e3_rerun_with_municipality_values: bool
    issue_counts_by_severity: dict[str, int]
    issue_counts_by_code: dict[str, int]
    recommended_followup_queries: tuple[str, ...]
    tables: tuple[EStatTableInspection, ...]
    issues: tuple[DimensionInspectionIssue, ...]


@dataclass(frozen=True)
class MunicipalityRetrievalPlanItem:
    stats_data_id: str
    table_title: str | None
    target_municipality_code: str
    area_dimension_id: str
    source_dimension_inspection_cache_path: str | None
    planned_params: dict[str, str]


@dataclass(frozen=True)
class MunicipalityRetrievalPlan:
    generated_at: str
    target_municipality_codes: tuple[str, ...]
    table_count: int
    municipality_filter_ready_table_count: int
    retrieval_count: int
    items: tuple[MunicipalityRetrievalPlanItem, ...]
    issues: tuple[DimensionInspectionIssue, ...]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def issue_id_for(issue_code: str, context: dict[str, Any] | None = None) -> str:
    payload = json.dumps(
        {"issue_code": issue_code, "context": context or {}},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _issue(
    severity: str,
    issue_code: str,
    message: str,
    stats_data_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> DimensionInspectionIssue:
    full_context = dict(context or {})
    if stats_data_id is not None:
        full_context.setdefault("stats_data_id", stats_data_id)
    return DimensionInspectionIssue(
        issue_id=issue_id_for(issue_code, full_context),
        severity=severity,
        issue_code=issue_code,
        message=message,
        stats_data_id=stats_data_id,
        context=full_context,
    )


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        inner = value.get("$")
        return str(inner) if inner is not None else None
    return str(value)


def _response_stats_data_id(data: dict[str, Any], fallback: str = "") -> str:
    parameter = data.get("GET_STATS_DATA", {}).get("PARAMETER", {})
    for key in ("STATS_DATA_ID", "statsDataId", "stats_data_id"):
        value = parameter.get(key)
        if value is not None:
            return str(value)
    table_inf = (
        data.get("GET_STATS_DATA", {})
        .get("STATISTICAL_DATA", {})
        .get("TABLE_INF", {})
    )
    if isinstance(table_inf, dict) and table_inf.get("@id") is not None:
        return str(table_inf.get("@id"))
    return fallback


def load_cached_getstatsdata_responses(
    cache_root: Path,
    stats_data_ids: list[str] | None = None,
) -> list[LoadedGetStatsDataResponse]:
    """Load cached getStatsData response files.

    When stats_data_ids is provided, only unfiltered getStatsData cache files
    matching those ids are considered, using the repository adapter's cache key.
    """
    cache_root = Path(cache_root)
    candidates: list[tuple[str, Path]] = []
    if stats_data_ids:
        from geo_strategist.data.estat_retrieval import _cache_key, _cache_path

        for stats_data_id in stats_data_ids:
            params = {"statsDataId": stats_data_id, "lang": "J"}
            candidates.append((
                stats_data_id,
                _cache_path(cache_root, _cache_key(params), "getStatsData"),
            ))
    elif cache_root.exists():
        candidates = [("", p) for p in sorted(cache_root.glob("estat_getStatsData_*.json"))]

    loaded: list[LoadedGetStatsDataResponse] = []
    for fallback_id, path in candidates:
        if not path.exists():
            loaded.append(LoadedGetStatsDataResponse(
                stats_data_id=fallback_id,
                cache_path=path,
                data=None,
                issue=_issue(
                    "error",
                    "missing_cached_getstatsdata_response",
                    "Required cached getStatsData response was not found.",
                    fallback_id or None,
                    {"cache_path": str(path)},
                ),
            ))
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            loaded.append(LoadedGetStatsDataResponse(
                stats_data_id=fallback_id,
                cache_path=path,
                data=None,
                issue=_issue(
                    "error",
                    "invalid_json",
                    f"Cached getStatsData response is not valid JSON: {exc}",
                    fallback_id or None,
                    {"cache_path": str(path)},
                ),
            ))
            continue
        stats_data_id = _response_stats_data_id(data, fallback_id)
        loaded.append(LoadedGetStatsDataResponse(
            stats_data_id=stats_data_id,
            cache_path=path,
            data=data,
        ))
    return loaded


def extract_class_dimensions(
    response: dict[str, Any],
    cache_path: Path | None = None,
    target_municipality_codes: tuple[str, ...] = DEFAULT_TARGET_MUNICIPALITIES,
) -> tuple[list[EStatDimension], list[EStatClassEntry], list[DimensionInspectionIssue]]:
    """Extract CLASS_INF dimensions and classes from one getStatsData response."""
    issues: list[DimensionInspectionIssue] = []
    stats_data_id = _response_stats_data_id(response)
    root = response.get("GET_STATS_DATA")
    if not isinstance(root, dict):
        return [], [], [_issue(
            "error",
            "missing_get_stats_data",
            "Response is missing GET_STATS_DATA.",
            stats_data_id or None,
            {"cache_path": str(cache_path) if cache_path else None},
        )]
    stat_data = root.get("STATISTICAL_DATA")
    if not isinstance(stat_data, dict):
        return [], [], [_issue(
            "error",
            "missing_statistical_data",
            "Response is missing GET_STATS_DATA.STATISTICAL_DATA.",
            stats_data_id or None,
            {"cache_path": str(cache_path) if cache_path else None},
        )]
    class_inf = stat_data.get("CLASS_INF")
    if not isinstance(class_inf, dict):
        return [], [], [_issue(
            "error",
            "missing_class_inf",
            "Response is missing CLASS_INF dimension metadata.",
            stats_data_id or None,
            {"cache_path": str(cache_path) if cache_path else None},
        )]

    dimensions: list[EStatDimension] = []
    entries: list[EStatClassEntry] = []
    class_objs = _as_list(class_inf.get("CLASS_OBJ"))
    if not class_objs:
        issues.append(_issue(
            "error",
            "missing_class_obj",
            "CLASS_INF does not contain CLASS_OBJ entries.",
            stats_data_id or None,
            {"cache_path": str(cache_path) if cache_path else None},
        ))
        return [], [], issues

    for obj in class_objs:
        if not isinstance(obj, dict):
            issues.append(_issue(
                "error",
                "malformed_class_obj",
                "CLASS_OBJ entry is not an object.",
                stats_data_id or None,
                {"cache_path": str(cache_path) if cache_path else None},
            ))
            continue
        dim_id = str(obj.get("@id", ""))
        dim_name = _text(obj.get("@name"))
        if not dim_id:
            issues.append(_issue(
                "error",
                "malformed_class_obj",
                "CLASS_OBJ entry is missing @id.",
                stats_data_id or None,
                {"dimension_name": dim_name, "cache_path": str(cache_path) if cache_path else None},
            ))
            continue
        classes = _as_list(obj.get("CLASS"))
        dim_entries: list[EStatClassEntry] = []
        for cls in classes:
            if not isinstance(cls, dict):
                issues.append(_issue(
                    "error",
                    "malformed_class",
                    "CLASS entry is not an object.",
                    stats_data_id or None,
                    {"dimension_id": dim_id},
                ))
                continue
            code = cls.get("@code")
            if code is None:
                issues.append(_issue(
                    "error",
                    "malformed_class",
                    "CLASS entry is missing @code.",
                    stats_data_id or None,
                    {"dimension_id": dim_id},
                ))
                continue
            entry = EStatClassEntry(
                stats_data_id=stats_data_id,
                dimension_id=dim_id,
                dimension_name=dim_name,
                class_code=str(code),
                class_name=_text(cls.get("@name")),
                level=_text(cls.get("@level")),
                parent_code=_text(cls.get("@parentCode") or cls.get("@parent_code")),
                unit=_text(cls.get("@unit")),
            )
            dim_entries.append(entry)
            entries.append(entry)

        codes = tuple(e.class_code for e in dim_entries)
        levels = tuple(sorted({e.level for e in dim_entries if e.level is not None}))
        samples = tuple(e.class_name for e in dim_entries[:5] if e.class_name is not None)
        lower_id = dim_id.lower()
        is_area = lower_id == "area" or "area" in lower_id or "地域" in (dim_name or "") or "市区町村" in (dim_name or "")
        present = tuple(code for code in target_municipality_codes if code in codes)
        missing = tuple(code for code in target_municipality_codes if code not in codes)
        dimensions.append(EStatDimension(
            stats_data_id=stats_data_id,
            dimension_id=dim_id,
            dimension_name=dim_name,
            class_count=len(dim_entries),
            class_codes=codes,
            sample_class_names=samples,
            levels=levels,
            parent_code_count=sum(1 for e in dim_entries if e.parent_code),
            unit_count=sum(1 for e in dim_entries if e.unit),
            is_area_dimension=is_area,
            contains_target_municipality_codes=bool(present),
            target_municipality_codes_present=present,
            target_municipality_codes_missing=missing,
        ))

    return dimensions, entries, issues


def _table_metadata(response: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    stat_data = response.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
    table_inf = stat_data.get("TABLE_INF", {})
    if not isinstance(table_inf, dict):
        return None, None, None
    title = _text(table_inf.get("TITLE"))
    stat_name = _text(table_inf.get("STAT_NAME"))
    survey_date = _text(table_inf.get("SURVEY_DATE"))
    return title, stat_name, survey_date


def _category_matches(entries: list[EStatClassEntry]) -> tuple[dict[str, str | None], ...]:
    matches: list[dict[str, str | None]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        if entry.dimension_id.lower() == "area":
            continue
        name = entry.class_name or ""
        if any(keyword in name for keyword in FACILITY_CATEGORY_KEYWORDS):
            key = (entry.dimension_id, entry.class_code)
            if key in seen:
                continue
            seen.add(key)
            matches.append({
                "dimension_id": entry.dimension_id,
                "dimension_name": entry.dimension_name,
                "class_code": entry.class_code,
                "class_name": entry.class_name,
            })
    return tuple(matches)


def classify_table_municipality_readiness(
    *,
    stats_data_id: str,
    dimensions: list[EStatDimension],
    classes: list[EStatClassEntry],
    extraction_issues: list[DimensionInspectionIssue],
) -> tuple[str, list[DimensionInspectionIssue]]:
    """Classify whether a table can safely be filtered by target cdArea codes."""
    issues = list(extraction_issues)
    if any(i.severity == "error" for i in extraction_issues):
        return READINESS_UNUSABLE, issues

    area_dimensions = [d for d in dimensions if d.is_area_dimension]
    if not area_dimensions:
        issues.append(_issue(
            "warning",
            "table_lacks_area_dimension",
            "No area dimension is present in CLASS_INF; table is not municipality-grain for cdArea retrieval.",
            stats_data_id,
        ))
        return READINESS_NOT_MUNICIPALITY_GRAIN, issues

    area_with_targets = [d for d in area_dimensions if d.contains_target_municipality_codes]
    category_matches = _category_matches(classes)
    if not category_matches:
        issues.append(_issue(
            "warning",
            "no_hospital_facility_category_match",
            "No hospital/facility count category match was found in non-area dimensions.",
            stats_data_id,
        ))

    if not area_with_targets:
        issues.append(_issue(
            "warning",
            "target_municipality_codes_absent",
            "Area dimension exists, but none of the target municipality JIS codes are present.",
            stats_data_id,
            {"target_municipality_codes": list(DEFAULT_TARGET_MUNICIPALITIES)},
        ))
        return READINESS_NOT_MUNICIPALITY_GRAIN, issues

    if not category_matches:
        return READINESS_NEEDS_MANUAL_DIMENSION_REVIEW, issues

    return READINESS_MUNICIPALITY_FILTER_READY, issues


def inspect_estat_table_dimensions(
    response: dict[str, Any],
    cache_path: Path | None = None,
    target_municipality_codes: tuple[str, ...] = DEFAULT_TARGET_MUNICIPALITIES,
) -> tuple[EStatTableInspection, list[EStatDimension], list[EStatClassEntry], list[DimensionInspectionIssue]]:
    stats_data_id = _response_stats_data_id(response)
    title, stat_name, survey_date = _table_metadata(response)
    dimensions, classes, extraction_issues = extract_class_dimensions(
        response,
        cache_path=cache_path,
        target_municipality_codes=target_municipality_codes,
    )
    classification, issues = classify_table_municipality_readiness(
        stats_data_id=stats_data_id,
        dimensions=dimensions,
        classes=classes,
        extraction_issues=extraction_issues,
    )

    area_dimensions = [d for d in dimensions if d.is_area_dimension]
    area_dimension = area_dimensions[0] if area_dimensions else None
    present: list[str] = []
    missing = list(target_municipality_codes)
    if area_dimension is not None:
        present = list(area_dimension.target_municipality_codes_present)
        missing = list(area_dimension.target_municipality_codes_missing)
    matches = _category_matches(classes)
    table = EStatTableInspection(
        stats_data_id=stats_data_id,
        table_title=title,
        stat_name=stat_name,
        survey_date=survey_date,
        cache_path=str(cache_path) if cache_path else None,
        status="ok" if not any(i.severity == "error" for i in issues) else "error",
        dimension_count=len(dimensions),
        class_count=len(classes),
        has_area_dimension=area_dimension is not None,
        area_dimension_id=area_dimension.dimension_id if area_dimension else None,
        area_dimension_name=area_dimension.dimension_name if area_dimension else None,
        area_class_count=area_dimension.class_count if area_dimension else 0,
        target_municipality_codes_present=tuple(present),
        target_municipality_codes_missing=tuple(missing),
        hospital_facility_category_matches=matches,
        hospital_facility_category_match_count=len(matches),
        readiness_classification=classification,
        issue_codes=tuple(i.issue_code for i in issues),
    )
    return table, dimensions, classes, issues


def build_estat_dimension_inspection_report(
    loaded_responses: list[LoadedGetStatsDataResponse],
    target_municipality_codes: tuple[str, ...] = DEFAULT_TARGET_MUNICIPALITIES,
    generated_at: str | None = None,
) -> tuple[EStatDimensionInspectionReport, list[EStatDimension], list[EStatClassEntry]]:
    tables: list[EStatTableInspection] = []
    dimensions: list[EStatDimension] = []
    classes: list[EStatClassEntry] = []
    issues: list[DimensionInspectionIssue] = []

    for loaded in sorted(loaded_responses, key=lambda r: (r.stats_data_id, str(r.cache_path))):
        if loaded.issue is not None:
            issues.append(loaded.issue)
            tables.append(EStatTableInspection(
                stats_data_id=loaded.stats_data_id,
                table_title=None,
                stat_name=None,
                survey_date=None,
                cache_path=str(loaded.cache_path),
                status="error",
                dimension_count=0,
                class_count=0,
                has_area_dimension=False,
                area_dimension_id=None,
                area_dimension_name=None,
                area_class_count=0,
                target_municipality_codes_present=(),
                target_municipality_codes_missing=target_municipality_codes,
                hospital_facility_category_matches=(),
                hospital_facility_category_match_count=0,
                readiness_classification=READINESS_UNUSABLE,
                issue_codes=(loaded.issue.issue_code,),
            ))
            continue
        assert loaded.data is not None
        table, dims, cls, table_issues = inspect_estat_table_dimensions(
            loaded.data,
            cache_path=loaded.cache_path,
            target_municipality_codes=target_municipality_codes,
        )
        tables.append(table)
        dimensions.extend(dims)
        classes.extend(cls)
        issues.extend(table_issues)

    if tables and not any(t.readiness_classification == READINESS_MUNICIPALITY_FILTER_READY for t in tables):
        issues.append(_issue(
            "warning",
            "no_tables_municipality_filter_ready",
            "No inspected table has an area dimension containing the target municipality JIS codes.",
            context={"table_count": len(tables), "target_municipality_codes": list(target_municipality_codes)},
        ))

    severity_counts = dict(sorted(Counter(i.severity for i in issues).items()))
    code_counts = dict(sorted(Counter(i.issue_code for i in issues).items()))
    classification_counts = Counter(t.readiness_classification for t in tables)
    report = EStatDimensionInspectionReport(
        generated_at=generated_at or now_utc_iso(),
        target_municipality_codes=target_municipality_codes,
        table_count=len(tables),
        dimension_count=len(dimensions),
        class_count=len(classes),
        municipality_filter_ready_count=classification_counts.get(READINESS_MUNICIPALITY_FILTER_READY, 0),
        not_municipality_grain_count=classification_counts.get(READINESS_NOT_MUNICIPALITY_GRAIN, 0),
        needs_manual_dimension_review_count=classification_counts.get(READINESS_NEEDS_MANUAL_DIMENSION_REVIEW, 0),
        unusable_for_e3_municipality_validation_count=classification_counts.get(READINESS_UNUSABLE, 0),
        suitable_for_e3_rerun_with_municipality_values=False,
        issue_counts_by_severity=severity_counts,
        issue_counts_by_code=code_counts,
        recommended_followup_queries=(
            "医療施設調査 市区町村 病院数",
            "医療施設調査 都道府県 指定都市 特別区 中核市 病院数",
            "病院報告 市区町村 病院数",
        ),
        tables=tuple(tables),
        issues=tuple(issues),
    )
    return report, dimensions, classes


def plan_municipality_filtered_retrievals(
    report: EStatDimensionInspectionReport,
) -> MunicipalityRetrievalPlan:
    items: list[MunicipalityRetrievalPlanItem] = []
    issues: list[DimensionInspectionIssue] = []
    for table in report.tables:
        if table.readiness_classification != READINESS_MUNICIPALITY_FILTER_READY:
            continue
        if not table.area_dimension_id:
            issues.append(_issue(
                "error",
                "verified_area_dimension_missing_for_planned_retrieval",
                "Table was classified ready but has no area dimension id.",
                table.stats_data_id,
            ))
            continue
        for code in table.target_municipality_codes_present:
            items.append(MunicipalityRetrievalPlanItem(
                stats_data_id=table.stats_data_id,
                table_title=table.table_title,
                target_municipality_code=code,
                area_dimension_id=table.area_dimension_id,
                source_dimension_inspection_cache_path=table.cache_path,
                planned_params={"statsDataId": table.stats_data_id, "lang": "J", "cdArea": code},
            ))
    if not items:
        issues.append(_issue(
            "warning",
            "no_municipality_filtered_retrievals_planned",
            "No cdArea-filtered retrievals were planned because no table was municipality_filter_ready.",
            context={"table_count": report.table_count},
        ))
    return MunicipalityRetrievalPlan(
        generated_at=now_utc_iso(),
        target_municipality_codes=report.target_municipality_codes,
        table_count=report.table_count,
        municipality_filter_ready_table_count=report.municipality_filter_ready_count,
        retrieval_count=len(items),
        items=tuple(items),
        issues=tuple(issues),
    )


def dataclass_to_dict(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataclass_to_dict(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Any] | tuple[Any, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dataclass_to_dict(row), ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def report_to_markdown(report: EStatDimensionInspectionReport) -> str:
    lines = [
        "# e-Stat Dimension Inspection Report",
        "",
        f"**Generated:** {report.generated_at}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Tables inspected | {report.table_count} |",
        f"| Dimensions found | {report.dimension_count} |",
        f"| Classes found | {report.class_count} |",
        f"| Municipality filter ready | {report.municipality_filter_ready_count} |",
        f"| Not municipality grain | {report.not_municipality_grain_count} |",
        f"| Needs manual dimension review | {report.needs_manual_dimension_review_count} |",
        f"| Unusable for E3 municipality validation | {report.unusable_for_e3_municipality_validation_count} |",
        f"| Suitable for E3 rerun with municipality values | {report.suitable_for_e3_rerun_with_municipality_values} |",
        "",
        "## Tables",
        "",
        "| statsDataId | Classification | Area Dim | Target Codes Present | Facility Category Matches |",
        "|-------------|----------------|----------|----------------------|---------------------------|",
    ]
    for table in report.tables:
        present = ", ".join(table.target_municipality_codes_present) or "-"
        area = table.area_dimension_id or "-"
        lines.append(
            f"| `{table.stats_data_id}` | `{table.readiness_classification}` | {area} | {present} | {table.hospital_facility_category_match_count} |"
        )
    lines += ["", "## Issues", ""]
    if report.issues:
        for issue in report.issues:
            sid = f" `{issue.stats_data_id}`" if issue.stats_data_id else ""
            lines.append(f"- **[{issue.severity}]** `{issue.issue_code}`{sid}: {issue.message}")
    else:
        lines.append("_No issues._")
    lines += [
        "",
        "## Follow-Up Queries",
        "",
    ]
    for query in report.recommended_followup_queries:
        lines.append(f"- {query}")
    lines += [
        "",
        "## Limitations",
        "",
        "- This inspection only validates cached getStatsData CLASS_INF metadata.",
        "- cdArea retrieval is planned only when target municipality codes are explicitly present.",
        "- No proposals, reviewer agents, tree search, cash-flow modeling, site selection, or final recommendations are generated.",
        "",
    ]
    return "\n".join(lines)


def municipality_plan_to_markdown(
    manifest: dict[str, Any],
    plan: MunicipalityRetrievalPlan,
    results: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> str:
    lines = [
        "# e-Stat Municipality Value Retrieval Report",
        "",
        f"**Run ID:** `{manifest['run_id']}`",
        f"**Generated:** {manifest['generated_at']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Allow network | {manifest['allow_network']} |",
        f"| ESTAT_APP_ID present | {manifest['estat_app_id_present']} |",
        f"| Tables in inspection | {plan.table_count} |",
        f"| Municipality-ready tables | {plan.municipality_filter_ready_table_count} |",
        f"| Retrievals planned | {plan.retrieval_count} |",
        f"| Retrievals attempted | {manifest['retrieval_attempts']} |",
        f"| Cache hits | {manifest['cache_hits']} |",
        f"| Live fetches | {manifest['live_fetches']} |",
        f"| Records normalized | {manifest['records_normalized']} |",
        f"| Suitable for E3 rerun with municipality values | {manifest['suitable_for_e3_rerun_with_municipality_values']} |",
        "",
    ]
    if results:
        lines += [
            "## Retrieval Results",
            "",
            "| statsDataId | cdArea | Status | Records | Cache Hit |",
            "|-------------|--------|--------|---------|-----------|",
        ]
        for result in results:
            lines.append(
                f"| `{result['stats_data_id']}` | `{result['target_municipality_code']}` | {result['status']} | {result['record_count']} | {result['cache_hit']} |"
            )
        lines.append("")
    lines += ["## Issues", ""]
    if issues:
        for issue in issues:
            lines.append(f"- **[{issue.get('severity', 'info')}]** `{issue.get('issue_code', '')}`: {issue.get('message', '')}")
    else:
        lines.append("_No issues._")
    lines += [
        "",
        "## Limitations",
        "",
        "- cdArea calls are attempted only for verified target municipality codes.",
        "- Cache-only mode does not perform live e-Stat calls.",
        "- No proposals, reviewer agents, tree search, cash-flow modeling, site selection, or final recommendations are generated.",
        "",
    ]
    return "\n".join(lines)
