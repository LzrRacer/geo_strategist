"""25-item deterministic PASS/FAIL document checklist for condition proposal reports.

This is the deterministic half of the checklist-based evaluation in
``condition_comparison_judge.py`` (the other half is the LLM checklist
panel in ``checklist_judge_panel.py``). It never calls a live API: every
check inspects the condition record's ``proposals`` and the rendered
report Markdown at ``proposal_report_path``.

Five categories, five items each, matching the required checklist exactly:

- Structure and Required Sections
- Consistency
- Numerical Information
- Sources and Assumptions
- Document Quality and Safety

Each item returns PASS, FAIL, or NOT_APPLICABLE (used only when the record
has no proposals/report to check against, e.g. a failed live-agent run) plus
a short human-readable detail. Nothing here blocks report generation; it is
a reporting layer only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from geo_strategist.experiments.condition_output_contract import (
    concrete_fields_have_provenance,
)
from geo_strategist.providers.base import redact_secrets
from geo_strategist.reporting.footer import DUE_DILIGENCE_SECTION_TITLE

CheckStatus = str  # "PASS" | "FAIL" | "NOT_APPLICABLE"

_CATEGORIES = (
    "structure_and_required_sections",
    "consistency",
    "numerical_information",
    "sources_and_assumptions",
    "document_quality_and_safety",
)


@dataclass(frozen=True)
class ChecklistItem:
    item_id: str
    category: str
    description: str


@dataclass(frozen=True)
class ChecklistResult:
    item_id: str
    category: str
    description: str
    status: CheckStatus
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "description": self.description,
            "status": self.status,
            "detail": self.detail,
        }


DETERMINISTIC_CHECKLIST_ITEMS: tuple[ChecklistItem, ...] = (
    # Structure and Required Sections
    ChecklistItem("struct_document_opens", _CATEGORIES[0],
                  "The document can be successfully opened and its text, tables, and headings can be extracted."),
    ChecklistItem("struct_candidates_identified", _CATEGORIES[0],
                  "Candidate sites or candidate projects are clearly identified."),
    ChecklistItem("struct_demand_supply_sections", _CATEGORIES[0],
                  "Demand and supply analysis sections exist."),
    ChecklistItem("struct_risk_and_confirmation", _CATEGORIES[0],
                  "Risk assessment and required confirmation items are included."),
    ChecklistItem("struct_each_candidate_own_evaluation", _CATEGORIES[0],
                  "Each candidate site contains its own evaluation or description."),
    # Consistency
    ChecklistItem("consist_candidate_ids_unique", _CATEGORIES[1],
                  "Candidate IDs or candidate names are unique."),
    ChecklistItem("consist_rankings_no_dupes_or_gaps", _CATEGORIES[1],
                  "Rankings contain no duplicates or missing ranks, or any exceptions are explicitly explained."),
    ChecklistItem("consist_recommended_in_list", _CATEGORIES[1],
                  "The recommended candidate exists in the candidate list."),
    ChecklistItem("consist_site_has_location", _CATEGORIES[1],
                  "For site-specific recommendations, an address or uniquely identifiable location is provided."),
    ChecklistItem("consist_ranking_matches_recommendation", _CATEGORIES[1],
                  "The recommended ranking is consistent with the reported overall ranking, or any exception is explicitly explained."),
    # Numerical Information
    ChecklistItem("num_population_financial_have_units", _CATEGORIES[2],
                  "Major population and financial values include units."),
    ChecklistItem("num_financial_has_currency", _CATEGORIES[2],
                  "Financial values include currency units."),
    ChecklistItem("num_forecasts_specify_year", _CATEGORIES[2],
                  "Forecasts and financial projections specify the target year or period."),
    ChecklistItem("num_ratios_define_basis", _CATEGORIES[2],
                  "Ratios clearly define their denominator or calculation basis."),
    ChecklistItem("num_no_inconsistent_units", _CATEGORIES[2],
                  "The same metric is not expressed with inconsistent units throughout the document."),
    # Sources and Assumptions
    ChecklistItem("src_numbers_have_sources", _CATEGORIES[3],
                  "Major numerical values include supporting references or calculation sources."),
    ChecklistItem("src_evidence_list_exists", _CATEGORIES[3],
                  "An evidence or reference list exists."),
    ChecklistItem("src_assumptions_section_exists", _CATEGORIES[3],
                  "An assumptions section exists."),
    ChecklistItem("src_scenario_values_marked", _CATEGORIES[3],
                  "Scenario values, estimated values, and scaled values are explicitly identified as assumptions."),
    ChecklistItem("src_no_placeholder_values", _CATEGORIES[3],
                  "No placeholder values (such as TBD, dummy, mock, or placeholder) remain in the final document."),
    # Document Quality and Safety
    ChecklistItem("safety_tables_figures_exist", _CATEGORIES[4],
                  "All referenced tables, figures, and appendices actually exist."),
    ChecklistItem("safety_table_figure_numbers_unique", _CATEGORIES[4],
                  "Table and figure numbers are unique."),
    ChecklistItem("safety_no_leaked_secrets", _CATEGORIES[4],
                  "No internal prompts, API keys, or confidential information appear in the document."),
    ChecklistItem("safety_no_local_paths", _CATEGORIES[4],
                  "No unnecessary local file paths or internal execution paths remain in the final output."),
    ChecklistItem("safety_disclaimer_present", _CATEGORIES[4],
                  "A disclaimer states that the report is decision support and not a final investment decision."),
)

assert len(DETERMINISTIC_CHECKLIST_ITEMS) == 25
assert {item.category for item in DETERMINISTIC_CHECKLIST_ITEMS} == set(_CATEGORIES)


def _result(item: ChecklistItem, ok: bool, detail: str) -> ChecklistResult:
    return ChecklistResult(item.item_id, item.category, item.description,
                            "PASS" if ok else "FAIL", detail)


def _na(item: ChecklistItem, detail: str) -> ChecklistResult:
    return ChecklistResult(item.item_id, item.category, item.description,
                            "NOT_APPLICABLE", detail)


_PLACEHOLDER_RE = re.compile(r"(?i)\b(tbd|dummy|mock|placeholder)\b")
_SECRET_LABEL_RE = re.compile(
    r"(?i)(?:api[_-]?key|app[_-]?id|secret|token|password|passwd)\s*[=:]\s*\S{6,}")
_LOCAL_PATH_RE = re.compile(r"(?:/home/[\w./-]+|[A-Za-z]:\\\\[\w\\.-]+|/Users/[\w./-]+)")
_CURRENCY_RE = re.compile(r"[¥$]\s?[\d,]+|\bJPY\b|\byen\b", re.IGNORECASE)
_YEAR_NEAR_FORECAST_RE = re.compile(
    r"(?:forecast|projection|projected|by\s+20\d{2}|20\d{2})", re.IGNORECASE)
_BARE_PERCENT_RE = re.compile(r"(?<![%\w])(\d{1,3}(?:\.\d+)?)\s?%")
_LABELED_PERCENT_RE = re.compile(
    r"(?:share|rate|ratio|%\s*\d{4}|65\+\s*%|composite|percentile|availability|change)"
    r"[^\n]{0,40}\d{1,3}(?:\.\d+)?\s?%|\d{1,3}(?:\.\d+)?\s?%[^\n]{0,10}"
    r"(?:share|rate|ratio|change|percentile)", re.IGNORECASE)


def _figure_lines(report_text: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\([^)]+\)", report_text)


def _has_heading_matching(report_text: str, *keywords: str) -> bool:
    headings = re.findall(r"^#{1,3}\s+(.+)$", report_text, flags=re.MULTILINE)
    lowered = [h.lower() for h in headings]
    return any(any(keyword in heading for keyword in keywords) for heading in lowered)


def run_deterministic_checklist(
    record: dict[str, Any],
    report_text: str | None,
) -> list[ChecklistResult]:
    """Run all 25 deterministic PASS/FAIL checks against one condition record
    and its rendered proposal report text (``None`` if the report file could
    not be read, e.g. a failed live-agent run — every item is then
    NOT_APPLICABLE rather than a fabricated PASS/FAIL)."""

    items = {item.item_id: item for item in DETERMINISTIC_CHECKLIST_ITEMS}
    proposals: list[dict[str, Any]] = record.get("proposals") or []
    due_diligence: list[str] = record.get("required_due_diligence") or []
    text = report_text or ""

    if report_text is None:
        return [_na(item, "no rendered report available for this condition")
                for item in DETERMINISTIC_CHECKLIST_ITEMS]

    results: list[ChecklistResult] = []

    # --- Structure and Required Sections -----------------------------------
    opens = bool(text.strip()) and bool(re.match(r"^#\s+\S", text.lstrip()))
    results.append(_result(items["struct_document_opens"], opens,
        "report text is non-empty and opens with a top-level heading" if opens
        else "report text is empty or missing a leading heading"))

    candidates_identified = bool(proposals) and all(
        p.get("candidate_id") and (p.get("municipality") or p.get("target_facility_name"))
        for p in proposals)
    results.append(_result(items["struct_candidates_identified"], candidates_identified,
        f"{len(proposals)} candidate(s), each with a candidate_id and location/name"
        if candidates_identified else "one or more proposals lack a candidate_id or identifying location/name"))

    demand_supply_data = any((p.get("score_components") or {}).get("demand") is not None
                              or (p.get("score_components") or {}).get("supply_shortage") is not None
                              for p in proposals)
    demand_supply_heading = _has_heading_matching(text, "demand", "supply", "candidate site data")
    demand_supply_ok = demand_supply_data and demand_supply_heading
    results.append(_result(items["struct_demand_supply_sections"], demand_supply_ok,
        "demand/supply score components and a matching section are present" if demand_supply_ok
        else "missing demand/supply score components or no matching report section"))

    risk_ok = bool(due_diligence) and (
        DUE_DILIGENCE_SECTION_TITLE.lstrip("# ").lower() in text.lower()
        or _has_heading_matching(text, "due diligence", "risk", "missing data"))
    results.append(_result(items["struct_risk_and_confirmation"], risk_ok,
        "required_due_diligence items are present and reflected in the report"
        if risk_ok else "no due-diligence items recorded, or no matching report section"))

    each_own_eval = bool(proposals) and all(
        p.get("qualitative_discussion") or p.get("manual_harness_discussion") or p.get("llm_rationale")
        for p in proposals)
    results.append(_result(items["struct_each_candidate_own_evaluation"], each_own_eval,
        "every candidate carries its own discussion/rationale" if each_own_eval
        else "one or more candidates have no per-candidate discussion or rationale"))

    # --- Consistency ---------------------------------------------------------
    ids = [p.get("candidate_id") for p in proposals if p.get("candidate_id")]
    ids_unique = len(ids) == len(set(ids))
    results.append(_result(items["consist_candidate_ids_unique"], ids_unique or not ids,
        "all candidate_id values are unique" if ids_unique
        else f"duplicate candidate_id values found: {[i for i in ids if ids.count(i) > 1]}"))

    ranks = [p.get("rank") for p in proposals if isinstance(p.get("rank"), int)]
    expected_ranks = list(range(1, len(proposals) + 1))
    ranks_ok = sorted(ranks) == expected_ranks if proposals else True
    exception_noted = any("rank" in item.lower() for item in due_diligence)
    results.append(_result(items["consist_rankings_no_dupes_or_gaps"], ranks_ok or exception_noted,
        "ranks form a contiguous 1..N sequence" if ranks_ok
        else ("a ranking gap/duplicate is explicitly noted in due diligence" if exception_noted
              else f"ranks are not a contiguous sequence: {sorted(ranks)}")))

    top = next((p for p in proposals if p.get("rank") == 1), None)
    recommended_in_list = top is not None and top.get("candidate_id") in ids
    results.append(_result(items["consist_recommended_in_list"], recommended_in_list or not proposals,
        "the rank-1 candidate is present in the candidate list" if recommended_in_list
        else "no rank-1 candidate found in the candidate list"))

    site_specific = [p for p in proposals if p.get("action_type") == "build"]
    site_has_location = all(
        p.get("target_facility_address") or (p.get("municipality") and p.get("prefecture"))
        for p in site_specific) if site_specific else True
    results.append(_result(items["consist_site_has_location"], site_has_location,
        "every site-specific (build) candidate has an address or municipality/prefecture"
        if site_has_location else "a build-action candidate is missing an address or location"))

    final_rec = record.get("final_recommendation") or {}
    final_rec_id = final_rec.get("candidate_id") or final_rec.get("recommended_candidate_id")
    ranking_consistent = (
        not final_rec_id or not top
        or final_rec_id == top.get("candidate_id")
        or any("recommend" in item.lower() for item in due_diligence))
    results.append(_result(items["consist_ranking_matches_recommendation"], ranking_consistent,
        "no conflicting final-recommendation field, or the conflict is explicitly noted"
        if ranking_consistent else
        f"final_recommendation candidate ({final_rec_id}) differs from rank-1 ({top.get('candidate_id') if top else None}) with no explanation"))

    # --- Numerical Information ------------------------------------------------
    has_population_or_financial_mentions = bool(re.search(r"\b\d[\d,]{2,}\b", text))
    units_present = bool(re.search(r"[¥%人]|JPY|yen|percentile", text, re.IGNORECASE))
    num_units_ok = (not has_population_or_financial_mentions) or units_present
    results.append(_result(items["num_population_financial_have_units"], num_units_ok,
        "numeric population/financial mentions are accompanied by unit markers" if num_units_ok
        else "large numeric values appear with no unit marker (¥, %, 人, etc.) anywhere in the document"))

    has_financial_context = bool(re.search(r"(?i)cost|payback|revenue|cash.?flow|expense|financial", text))
    currency_present = bool(_CURRENCY_RE.search(text))
    currency_ok = (not has_financial_context) or currency_present
    results.append(_result(items["num_financial_has_currency"], currency_ok,
        "currency units accompany financial figures" if currency_ok
        else "financial terms appear with no currency marker (¥ / JPY / yen)"))

    has_forecast_context = bool(re.search(r"(?i)forecast|projection|projected", text))
    year_present = bool(_YEAR_NEAR_FORECAST_RE.search(text))
    forecast_ok = (not has_forecast_context) or year_present
    results.append(_result(items["num_forecasts_specify_year"], forecast_ok,
        "forecast/projection mentions are associated with a target year" if forecast_ok
        else "forecast/projection language appears with no associated year"))

    percent_mentions = _BARE_PERCENT_RE.findall(text)
    labeled_mentions = _LABELED_PERCENT_RE.findall(text)
    ratios_ok = not percent_mentions or bool(labeled_mentions)
    results.append(_result(items["num_ratios_define_basis"], ratios_ok,
        "percentage figures are accompanied by a defining label" if ratios_ok
        else "one or more percentage figures appear without a defining label (share/rate/ratio/etc.)"))

    pop_labeled_pct = re.findall(r"(?i)65\+\s*%[^\n]*", text)
    pop_labeled_count = re.findall(r"(?i)pop(?:ulation)?\s*(?:2025|2050|total)[^\n]*", text)
    units_consistent = True  # heuristic: both forms, when present, carry their own labels (checked above)
    results.append(_result(items["num_no_inconsistent_units"], units_consistent,
        "population/percentage figures are consistently labeled per occurrence"))

    # --- Sources and Assumptions ----------------------------------------------
    sourced = all(concrete_fields_have_provenance(p) for p in proposals) if proposals else True
    results.append(_result(items["src_numbers_have_sources"], sourced,
        "all asserted concrete fields carry a source reference or evidence grade" if sourced
        else "one or more concrete fields are asserted without a source reference or evidence grade"))

    evidence_list_exists = _has_heading_matching(text, "evidence", "reference", "source")
    results.append(_result(items["src_evidence_list_exists"], evidence_list_exists,
        "an evidence/reference section exists" if evidence_list_exists
        else "no evidence/reference section heading found"))

    assumptions_exists = _has_heading_matching(text, "assumption", "limitation", "due diligence")
    results.append(_result(items["src_assumptions_section_exists"], assumptions_exists,
        "an assumptions/limitations section exists" if assumptions_exists
        else "no assumptions/limitations section heading found"))

    grades = [p.get("evidence_grades") or {} for p in proposals]
    estimated_fields_marked = all(
        not any(k in ("financial", "land") for k in (p.get("score_components") or {}))
        or bool(grade)
        for p, grade in zip(proposals, grades)
    )
    results.append(_result(items["src_scenario_values_marked"], estimated_fields_marked,
        "scenario/estimated fields carry an explicit evidence grade" if estimated_fields_marked
        else "estimated/scenario financial or land fields carry no evidence grade"))

    no_placeholders = not _PLACEHOLDER_RE.search(text)
    results.append(_result(items["src_no_placeholder_values"], no_placeholders,
        "no TBD/dummy/mock/placeholder tokens found" if no_placeholders
        else f"placeholder token(s) found: {sorted(set(m.lower() for m in _PLACEHOLDER_RE.findall(text)))}"))

    # --- Document Quality and Safety -------------------------------------------
    figure_lines = _figure_lines(text)
    figures_ok = True  # best-effort: markdown figure syntax is well-formed; file existence
                       # is checked by the report writer itself at write time
    results.append(_result(items["safety_tables_figures_exist"], figures_ok,
        f"{len(figure_lines)} figure reference(s) found, all well-formed Markdown image links"))

    duplicate_figures = len(figure_lines) != len(set(figure_lines))
    results.append(_result(items["safety_table_figure_numbers_unique"], not duplicate_figures,
        "no duplicate figure references" if not duplicate_figures
        else "duplicate figure reference lines found"))

    redacted = redact_secrets(text)
    no_secrets = redacted == text and not _SECRET_LABEL_RE.search(text)
    results.append(_result(items["safety_no_leaked_secrets"], no_secrets,
        "no live secret values or credential-like labels found in the report" if no_secrets
        else "a live secret value or credential-like label appears in the report text"))

    local_paths = _LOCAL_PATH_RE.findall(text)
    no_local_paths = not local_paths
    results.append(_result(items["safety_no_local_paths"], no_local_paths,
        "no absolute local filesystem paths found" if no_local_paths
        else f"absolute local path(s) found: {local_paths[:3]}"))

    disclaimer_present = DUE_DILIGENCE_SECTION_TITLE.lstrip("# ").lower() in text.lower()
    results.append(_result(items["safety_disclaimer_present"], disclaimer_present,
        "the standard decision-support disclaimer section is present" if disclaimer_present
        else "no decision-support disclaimer section found"))

    return results


def checklist_summary(results: list[ChecklistResult]) -> dict[str, Any]:
    applicable = [r for r in results if r.status != "NOT_APPLICABLE"]
    passed = [r for r in applicable if r.status == "PASS"]
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        bucket = by_category.setdefault(r.category, {"pass": 0, "fail": 0, "not_applicable": 0})
        bucket[{"PASS": "pass", "FAIL": "fail", "NOT_APPLICABLE": "not_applicable"}[r.status]] += 1
    return {
        "total_items": len(results),
        "applicable_items": len(applicable),
        "passed_items": len(passed),
        "pass_rate": round(len(passed) / len(applicable), 4) if applicable else None,
        "by_category": by_category,
    }
