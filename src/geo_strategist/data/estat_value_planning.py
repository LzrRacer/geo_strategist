"""e-Stat value retrieval planner (Phase 10/11).

Given E3 retrieval requests and available getStatsList metadata (from cache or
metadata search results), propose which statsDataId values are candidates for
getStatsData calls.

Rules:
  - Only real metadata or existing cached metadata is used.
  - Table IDs are never invented.
  - Ambiguous cases are explicitly classified.
  - Does not execute any value retrieval itself.
  - Does not call any LLM.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ESTAT_CACHE_ROOT = Path(".data/api_raw/estat")
MEDICAL_KEYWORDS = frozenset({
    "医療施設", "病院", "診療所", "病床", "beds", "hospital",
    "医師", "看護", "医療機関", "施設調査",
})
POPULATION_KEYWORDS = frozenset({
    "人口", "population", "住民基本", "国勢調査",
})

# Candidate classification values
CLS_UNAMBIGUOUS = "unambiguous_candidate"
CLS_AMBIGUOUS = "ambiguous_candidates"
CLS_NO_CANDIDATE = "no_candidate_found"
CLS_NEEDS_DIMENSION = "metadata_only_needs_dimension_inspection"


@dataclass
class CandidateTable:
    stats_data_id: str
    title: str
    stat_name: str
    gov_org: str
    survey_date: str | None
    overall_total_number: int
    relevance_score: float  # 0.0–1.0
    relevance_reasons: list[str]
    classification: str = CLS_AMBIGUOUS  # set by build_value_retrieval_plan
    ambiguous: bool = False
    ambiguity_reasons: list[str] = field(default_factory=list)
    selected: bool = False
    selection_reason: str = ""
    source: str = "getStatsList_cache"  # getStatsList_cache | metadata_search


@dataclass
class PlanningIssue:
    issue_id: str
    severity: str  # info | warning | error
    issue_code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValueRetrievalPlan:
    plan_id: str
    generated_at: str
    e3_request_count: int
    cached_metadata_tables_scanned: int
    metadata_search_tables_loaded: int
    candidate_count: int
    unambiguous_count: int
    ambiguous_count: int
    selected_count: int
    overall_classification: str  # one of CLS_* values
    candidates: list[CandidateTable]
    issues: list[PlanningIssue]
    no_unambiguous_note: str | None = None
    recommendation: str = ""


def _load_cached_tables(cache_root: Path) -> list[dict[str, Any]]:
    """Load TABLE_INF entries from cached getStatsList raw response files."""
    tables: list[dict[str, Any]] = []
    if not cache_root.exists():
        return tables
    for f in sorted(cache_root.glob("estat_getStatsList_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            tbl = (
                data.get("GET_STATS_LIST", {})
                    .get("DATALIST_INF", {})
                    .get("TABLE_INF", [])
            )
            if isinstance(tbl, dict):
                tbl = [tbl]
            tables.extend(tbl)
        except Exception:
            pass
    return tables


def _load_metadata_search_tables(search_results_path: Path) -> list[dict[str, Any]]:
    """Load TABLE_INF-compatible entries from a metadata_search_results.jsonl file.

    Each JSONL row has {query, status, tables: [{stats_data_id, stat_name, ...}]}.
    Converts to TABLE_INF format for uniform scoring.
    """
    tables: list[dict[str, Any]] = []
    if not search_results_path.exists():
        return tables
    seen_ids: set[str] = set()
    for line in search_results_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        for t in row.get("tables", []):
            sid = t.get("stats_data_id", "")
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)
            # Re-normalize to TABLE_INF-like dict
            tables.append({
                "@id": sid,
                "STAT_NAME": {"$": t.get("stat_name", "")},
                "GOV_ORG": {"$": t.get("gov_org", "")},
                "STATISTICS_NAME": t.get("statistics_name", t.get("stat_name", "")),
                "TITLE": {"$": t.get("title", "")},
                "SURVEY_DATE": t.get("survey_date"),
                "OVERALL_TOTAL_NUMBER": t.get("overall_total_number", 0),
                "_source": "metadata_search",
            })
    return tables


def _title_str(tbl: dict[str, Any]) -> str:
    title = tbl.get("TITLE", "")
    if isinstance(title, dict):
        return title.get("$", "")
    return str(title)


def _stat_name_str(tbl: dict[str, Any]) -> str:
    sn = tbl.get("STAT_NAME", "")
    if isinstance(sn, dict):
        return sn.get("$", "")
    return str(sn)


def _gov_org_str(tbl: dict[str, Any]) -> str:
    go = tbl.get("GOV_ORG", "")
    if isinstance(go, dict):
        return go.get("$", "")
    return str(go)


def _score_table(tbl: dict[str, Any], request_keywords: set[str]) -> tuple[float, list[str]]:
    """Score a table's relevance to the given request keywords (0.0–1.0)."""
    score = 0.0
    reasons: list[str] = []
    title = _title_str(tbl).lower()
    stat_name = _stat_name_str(tbl).lower()
    stats_name_full = tbl.get("STATISTICS_NAME", "").lower()

    for kw in request_keywords:
        kw_l = kw.lower()
        if kw_l in title:
            score += 0.4
            reasons.append(f"keyword '{kw}' in title")
        elif kw_l in stat_name or kw_l in stats_name_full:
            score += 0.3
            reasons.append(f"keyword '{kw}' in stat_name/statistics_name")

    # Bonus for exact survey name
    for survey_kw in ("医療施設調査", "医療施設"):
        if survey_kw in tbl.get("STATISTICS_NAME", ""):
            score += 0.35
            reasons.append(f"exact survey keyword '{survey_kw}' in STATISTICS_NAME")
        if survey_kw in stat_name:
            score += 0.25
            reasons.append(f"exact survey keyword '{survey_kw}' in stat_name")

    # Bonus: publisher is 厚生労働省 (MHLW — publishes 医療施設調査)
    gov_org = _gov_org_str(tbl)
    if "厚生労働省" in gov_org or "00450" in gov_org:
        score += 0.2
        reasons.append("publisher is 厚生労働省 (MHLW)")

    return min(score, 1.0), reasons


def _extract_request_keywords(request: dict[str, Any]) -> set[str]:
    """Extract search keywords from an E3 retrieval request."""
    query = request.get("query", "")
    keywords: set[str] = set()
    for kw in MEDICAL_KEYWORDS | POPULATION_KEYWORDS:
        if kw in query:
            keywords.add(kw)
    for token in query.split():
        if len(token) >= 2:
            keywords.add(token)
    return keywords


def _find_latest_metadata_search() -> Path | None:
    """Find latest metadata search results file in .runs."""
    search_root = Path(".runs") / "experiments" / "estat_metadata_search"
    if not search_root.exists():
        return None
    runs = [p / "metadata_search_results.jsonl"
            for p in search_root.iterdir()
            if p.is_dir() and (p / "metadata_search_results.jsonl").exists()]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0] if runs else None


def build_value_retrieval_plan(
    e3_requests: list[dict[str, Any]],
    cache_root: Path = ESTAT_CACHE_ROOT,
    metadata_search_results_path: Path | None = None,
    relevance_threshold: float = 0.3,
    auto_load_metadata_search: bool = True,
) -> ValueRetrievalPlan:
    """Build a value retrieval plan from E3 requests and available metadata.

    Sources (in priority order):
    1. metadata_search_results_path (if provided or auto-discovered)
    2. Cached getStatsList raw JSON files under cache_root

    Table IDs are taken only from real cached metadata — never invented.
    """
    plan_id = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc).isoformat()
    issues: list[PlanningIssue] = []

    # Load tables from all sources
    cached_tables = _load_cached_tables(cache_root)
    search_tables: list[dict[str, Any]] = []

    if metadata_search_results_path is not None:
        search_tables = _load_metadata_search_tables(metadata_search_results_path)
    elif auto_load_metadata_search:
        latest_search = _find_latest_metadata_search()
        if latest_search is not None:
            search_tables = _load_metadata_search_tables(latest_search)

    # Tag sources
    for t in search_tables:
        t.setdefault("_source", "metadata_search")
    for t in cached_tables:
        t.setdefault("_source", "getStatsList_cache")

    scanned = len(cached_tables)
    search_loaded = len(search_tables)

    # Merge, dedup by @id (search tables take priority — more targeted)
    seen_ids: set[str] = set()
    all_tables: list[dict[str, Any]] = []
    for t in search_tables + cached_tables:
        sid = t.get("@id", "")
        if sid and sid not in seen_ids:
            seen_ids.add(sid)
            all_tables.append(t)

    if not all_tables:
        issues.append(PlanningIssue(
            issue_id=str(uuid.uuid4()),
            severity="warning",
            issue_code="no_cached_metadata",
            message=(
                "No cached getStatsList metadata found. "
                "Run scripts/run_estat_metadata_search.py --allow-network first, "
                "then re-run planning."
            ),
        ))
        return ValueRetrievalPlan(
            plan_id=plan_id,
            generated_at=generated_at,
            e3_request_count=len(e3_requests),
            cached_metadata_tables_scanned=scanned,
            metadata_search_tables_loaded=search_loaded,
            candidate_count=0,
            unambiguous_count=0,
            ambiguous_count=0,
            selected_count=0,
            overall_classification=CLS_NO_CANDIDATE,
            candidates=[],
            issues=issues,
            no_unambiguous_note=(
                "No unambiguous statsDataId found. "
                "No cached metadata is available to identify candidate tables."
            ),
            recommendation=(
                "Run scripts/run_estat_metadata_search.py --allow-network "
                "--queries 医療施設調査 医療施設 病院 一般診療所, "
                "then re-run value planning."
            ),
        )

    # Collect all request keywords
    all_keywords: set[str] = set()
    for req in e3_requests:
        all_keywords |= _extract_request_keywords(req)

    # Score and rank
    candidates: list[CandidateTable] = []

    for tbl in all_tables:
        stats_data_id = tbl.get("@id", "")
        if not stats_data_id:
            continue

        score, reasons = _score_table(tbl, all_keywords)
        if score < relevance_threshold:
            continue

        total = tbl.get("OVERALL_TOTAL_NUMBER", 0)
        try:
            total = int(total)
        except (TypeError, ValueError):
            total = 0

        sd = tbl.get("SURVEY_DATE")
        survey_date = str(sd) if sd is not None else None

        medical_match = any(
            "医療" in r or "病院" in r or "診療" in r or "厚生" in r
            for r in reasons
        )
        ambiguous = score < 0.6 or not medical_match
        ambiguity_reasons: list[str] = []
        if score < 0.6:
            ambiguity_reasons.append(f"relevance score {score:.2f} < 0.6 threshold")
        if not medical_match:
            ambiguity_reasons.append("no direct medical facility keyword matched")

        source = tbl.get("_source", "getStatsList_cache")

        candidates.append(CandidateTable(
            stats_data_id=stats_data_id,
            title=_title_str(tbl),
            stat_name=_stat_name_str(tbl),
            gov_org=_gov_org_str(tbl),
            survey_date=survey_date,
            overall_total_number=total,
            relevance_score=score,
            relevance_reasons=reasons,
            ambiguous=ambiguous,
            ambiguity_reasons=ambiguity_reasons,
            source=source,
        ))

    candidates.sort(key=lambda c: c.relevance_score, reverse=True)

    unambiguous = [c for c in candidates if not c.ambiguous]
    ambiguous_only = [c for c in candidates if c.ambiguous]
    unambiguous_count = len(unambiguous)
    ambiguous_count = len(ambiguous_only)

    # Classify candidates and select
    selected_count = 0
    for c in candidates:
        # Per-candidate classification
        if not c.ambiguous:
            # Check if this is the only strong candidate for the survey
            same_survey = [
                x for x in unambiguous
                if _stat_name_str({"STAT_NAME": x.stat_name}) ==
                   _stat_name_str({"STAT_NAME": c.stat_name})
                or x.stat_name == c.stat_name
            ]
            if len(same_survey) == 1:
                c.classification = CLS_UNAMBIGUOUS
            else:
                c.classification = CLS_AMBIGUOUS
        else:
            c.classification = CLS_AMBIGUOUS

        # Mark as needing dimension inspection if no SURVEY_DATE implies dimensions unknown
        if c.survey_date is None or c.overall_total_number == 0:
            c.classification = CLS_NEEDS_DIMENSION

    for c in unambiguous[:3]:
        c.selected = True
        c.selection_reason = (
            f"Unambiguous match (score={c.relevance_score:.2f}); "
            f"reasons: {', '.join(c.relevance_reasons)}"
        )
        selected_count += 1

    # Overall classification
    if unambiguous_count == 1:
        overall_classification = CLS_UNAMBIGUOUS
    elif unambiguous_count > 1:
        overall_classification = CLS_AMBIGUOUS
    elif ambiguous_count > 0:
        overall_classification = CLS_AMBIGUOUS
    else:
        overall_classification = CLS_NO_CANDIDATE

    no_unambiguous_note: str | None = None
    recommendation: str

    if overall_classification == CLS_NO_CANDIDATE:
        no_unambiguous_note = (
            "No unambiguous statsDataId found. "
            f"Scanned {len(all_tables)} table(s) from all sources, "
            "none scored >= 0.6 with a medical facility keyword match."
        )
        recommendation = (
            "Run scripts/run_estat_metadata_search.py --allow-network "
            "--queries 医療施設調査 医療施設 病院 to populate medical facility "
            "survey metadata, then re-run value planning."
        )
        issues.append(PlanningIssue(
            issue_id=str(uuid.uuid4()),
            severity="info",
            issue_code="no_unambiguous_table_id",
            message=no_unambiguous_note,
            context={"candidate_count": len(candidates), "tables_scanned": len(all_tables)},
        ))
    elif overall_classification == CLS_AMBIGUOUS:
        if unambiguous_count > 1:
            no_unambiguous_note = (
                f"{unambiguous_count} individually-scoring candidates found, "
                "but they are ambiguous by plurality — multiple medical survey tables "
                "exist and it is unclear which is the best fit for the E3 requests. "
                "Dimension inspection recommended before filtering by area/category."
            )
        else:
            no_unambiguous_note = (
                f"{len(candidates)} candidate table(s) found but all are ambiguous "
                "(score < 0.6 or no medical keyword match). "
                "Multiple plausible tables exist; dimension inspection recommended."
            )
        recommendation = (
            f"Inspect the {unambiguous_count} strongly-scoring and {ambiguous_count} ambiguous "
            "candidate tables. For each, call getStatsData with the statsDataId to inspect "
            "available dimensions before filtering by area/category. "
            "Consider running with a more specific query like '医療施設調査 病院 病床数' "
            "or '医療施設調査 市区町村' to narrow to municipality-grain tables."
        )
        issues.append(PlanningIssue(
            issue_id=str(uuid.uuid4()),
            severity="info",
            issue_code="ambiguous_candidates_only",
            message=no_unambiguous_note,
            context={"candidate_ids": [c.stats_data_id for c in candidates[:5]]},
        ))
    else:
        recommendation = (
            f"Found {unambiguous_count} unambiguous candidate table(s). "
            f"Selected {selected_count} for getStatsData retrieval. "
            "Verify area (cdArea) and category (cdCat01) filter codes against "
            "actual table dimension metadata before filtering. "
            "Note: area codes require JIS municipality codes (e.g., 東京都中央区=13102)."
        )
        if selected_count > 0 and any(
            c.overall_total_number == 0 for c in unambiguous[:3]
        ):
            issues.append(PlanningIssue(
                issue_id=str(uuid.uuid4()),
                severity="info",
                issue_code="metadata_only_needs_dimension_inspection",
                message=(
                    "Selected table(s) have no OVERALL_TOTAL_NUMBER or dimension info. "
                    "Run getStatsData without filters first to inspect available dimensions."
                ),
                context={"selected_ids": [c.stats_data_id for c in unambiguous[:3]]},
            ))

    return ValueRetrievalPlan(
        plan_id=plan_id,
        generated_at=generated_at,
        e3_request_count=len(e3_requests),
        cached_metadata_tables_scanned=scanned,
        metadata_search_tables_loaded=search_loaded,
        candidate_count=len(candidates),
        unambiguous_count=unambiguous_count,
        ambiguous_count=ambiguous_count,
        selected_count=selected_count,
        overall_classification=overall_classification,
        candidates=candidates,
        issues=issues,
        no_unambiguous_note=no_unambiguous_note,
        recommendation=recommendation,
    )


def plan_to_dict(plan: ValueRetrievalPlan) -> dict[str, Any]:
    return {
        "plan_id": plan.plan_id,
        "generated_at": plan.generated_at,
        "e3_request_count": plan.e3_request_count,
        "cached_metadata_tables_scanned": plan.cached_metadata_tables_scanned,
        "metadata_search_tables_loaded": plan.metadata_search_tables_loaded,
        "candidate_count": plan.candidate_count,
        "unambiguous_count": plan.unambiguous_count,
        "ambiguous_count": plan.ambiguous_count,
        "selected_count": plan.selected_count,
        "overall_classification": plan.overall_classification,
        "no_unambiguous_note": plan.no_unambiguous_note,
        "recommendation": plan.recommendation,
        "candidates": [
            {
                "stats_data_id": c.stats_data_id,
                "title": c.title,
                "stat_name": c.stat_name,
                "gov_org": c.gov_org,
                "survey_date": c.survey_date,
                "overall_total_number": c.overall_total_number,
                "relevance_score": round(c.relevance_score, 4),
                "relevance_reasons": c.relevance_reasons,
                "classification": c.classification,
                "ambiguous": c.ambiguous,
                "ambiguity_reasons": c.ambiguity_reasons,
                "selected": c.selected,
                "selection_reason": c.selection_reason,
                "source": c.source,
            }
            for c in plan.candidates
        ],
        "issues": [
            {
                "issue_id": i.issue_id,
                "severity": i.severity,
                "issue_code": i.issue_code,
                "message": i.message,
                "context": i.context,
            }
            for i in plan.issues
        ],
    }


def plan_to_md(plan: ValueRetrievalPlan) -> str:
    lines = [
        "# e-Stat Value Retrieval Plan",
        "",
        f"**Plan ID:** `{plan.plan_id}`",
        f"**Generated:** {plan.generated_at}",
        f"**Classification:** `{plan.overall_classification}`",
        "",
        "## Summary",
        "",
        "| Item | Count |",
        "|------|-------|",
        f"| E3 retrieval requests analyzed | {plan.e3_request_count} |",
        f"| Cached metadata tables scanned | {plan.cached_metadata_tables_scanned} |",
        f"| Metadata search tables loaded | {plan.metadata_search_tables_loaded} |",
        f"| Candidate tables (score ≥ threshold) | {plan.candidate_count} |",
        f"| Unambiguous candidates | {plan.unambiguous_count} |",
        f"| Ambiguous candidates | {plan.ambiguous_count} |",
        f"| Selected for getStatsData | {plan.selected_count} |",
        "",
    ]

    if plan.no_unambiguous_note:
        lines += [f"> **{plan.no_unambiguous_note}**", ""]

    lines += [f"**Recommendation:** {plan.recommendation}", ""]

    if plan.candidates:
        lines += ["## Candidate Tables", ""]
        for c in plan.candidates:
            lines += [
                f"### `{c.stats_data_id}` — {c.stat_name}",
                "",
                f"- **Classification:** `{c.classification}`",
                f"- **Source:** {c.source}",
                f"- **Title:** {c.title}",
                f"- **Gov org:** {c.gov_org}",
                f"- **Survey date:** {c.survey_date}",
                f"- **Relevance score:** {c.relevance_score:.2f}",
                f"- **Reasons:** {', '.join(c.relevance_reasons)}",
            ]
            if c.ambiguity_reasons:
                lines.append(f"- **Ambiguity reasons:** {'; '.join(c.ambiguity_reasons)}")
            if c.selected:
                lines.append(f"- **Selected:** Yes — {c.selection_reason}")
            lines.append("")
    else:
        lines += ["## Candidate Tables", "", "_No candidate tables found._", ""]

    if plan.issues:
        lines += ["## Issues", ""]
        for iss in plan.issues:
            lines.append(f"- **[{iss.severity}]** `{iss.issue_code}`: {iss.message}")
        lines.append("")

    return "\n".join(lines)
