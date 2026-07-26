"""Per-condition proposal report writer (Markdown + figures).

Every proposal-producing condition emits the same report skeleton so the
comparison judge and human reviewers can compare like for like:

- executive summary and model-reported method,
- **evaluation criteria and selection method** (candidate universe, component
  definitions and weights, branch objectives, and a selection-justification
  table placing each chosen candidate against the whole district),
- ranked candidates, branch-by-branch winners,
- **candidate site data** (real census population figures, MLIT land prices,
  Yahoo facility counts) and an **indicative cost basis** table (cash-flow
  workbook model estimates, clearly graded),
- evidence grades, missing data, generated-code execution, model calls,
- a location map panel (facility geocodes; schematic, not cartographic),
- the candidate-level **Qualitative Site Discussion**, and
- the standard limitations/due-diligence footer.

Reports are written as ``<output_dir>/reports/<CNN_slug>.md`` with figures
under ``<output_dir>/figures/``.

Vanilla LLM conditions (``spec.algorithm == "vanilla_llm"``, C1-C4) are true
single-pass baselines: the shared post-hoc candidate-review/author-response
augmentation never runs for them, and this writer must not render it, its
figures, or any "review was not run/disabled" boilerplate in their reports
either (``include_candidate_review`` below) — only the augmentation itself is
suppressed; candidate-specific evidence interpretation, model rationale,
evidence gaps, and due-diligence text are unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from geo_strategist.experiments.branch_objectives import BRANCH_OBJECTIVES
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DEFAULT_WEIGHTS,
    DataBundle,
    rank_candidates,
)
from geo_strategist.experiments.live_common import LiveConditionResult
from geo_strategist.experiments.decision_analysis import DecisionAnalysisBundle
from geo_strategist.experiments.location_costing import location_cost_model
from geo_strategist.experiments.qualitative_discussion import (
    build_qualitative_site_discussions,
    district_data_quality_note,
    render_qualitative_site_discussion_section,
)
from geo_strategist.reporting import markdown_table, required_due_diligence_section
from geo_strategist.reporting.figures import (
    call_distribution_figure,
    candidate_map_figure,
    candidate_scores_figure,
    component_breakdown_figure,
    grouped_bar_figure,
    metric_map_figure,
    success_failure_figure,
    write_figure_data,
)

BUSINESS_REPORT_CONTRACT_VERSION = "business_decision_report_v1"

# Metric-overlay maps: (facts key, figure suffix, metric label, title suffix).
_METRIC_MAPS: tuple[tuple[str, str, str, str], ...] = (
    ("population_total_2025", "map_population",
     "2025 total population (census projection)",
     "candidates vs total population"),
    ("share_65_plus_2025_pct", "map_aging",
     "population share aged 65+ in 2025, % (census projection)",
     "candidates vs elderly share"),
    ("supply_density_per_100k", "map_hospital_density",
     "hospitals per 100k residents (Yahoo Local Search records)",
     "candidates vs hospital density"),
)

_COMPONENT_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("demand", "healthcare demand pressure (0-1)", "census projections"),
    ("aging", "population aging pressure (0-1)", "census projections"),
    ("supply_shortage", "supply gap for build / facility density for reorganize (0-1)",
     "Yahoo Local Search facility records"),
    ("financial", "payback plausibility, prefecture median (0-1)",
     "hospital cash-flow workbook (model estimate)"),
    ("land", "land-price availability score (0-1)", "MLIT land prices"),
    ("demographic_risk", "population stability 2020-2050 (1=stable)", "census projections"),
    ("evidence_completeness", "data completeness for the municipality (0-1)",
     "pipeline coverage flags"),
)


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value is None:
        return "not_available"
    return str(value)


def _fmt_count(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{int(round(value)):,}"
    return "not_available"


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

_OBJECTIVE_TABLE_REPORT_PROFILES: frozenset[str] = frozenset({"skills_objectives", "ai_scientist"})


def _shared_criteria_section(result: LiveConditionResult, data: DataBundle | None) -> list[str]:
    """The comparability-supporting half of the old criteria section: the
    fixed evaluation-model components and where each selected candidate
    sits against the whole district under those shared, condition-
    independent default weights. Identical in kind across every condition —
    this is what makes the candidates comparable at all."""

    lines = [
        "## Evaluation criteria and selection method",
        "",
        "Candidates are never invented: the candidate universe is "
        f"`{len(data.candidates) if data else 'n/a'}` build/reorganize/consolidate "
        "actions generated upstream from the shared evidence base "
        "(`candidate_actions.jsonl`), covering the Tokyo / Aichi / Osaka study "
        "area. Every ranking — whether produced by the deterministic baseline, "
        "an LLM, or generated code — is validated against this universe and "
        "scored with the same evaluation-model components:",
        "",
        markdown_table(
            ["component", "definition", "data source", "default weight"],
            [[name, definition, source, DEFAULT_WEIGHTS.get(name)]
             for name, definition, source in _COMPONENT_DEFINITIONS],
        ),
        "",
    ]
    if data and result.proposals:
        ranked_all = data.runtime_cache.get("default_ranking")
        if ranked_all is None:
            ranked_all = rank_candidates(data, DEFAULT_WEIGHTS)
            data.runtime_cache["default_ranking"] = ranked_all
        total = len(ranked_all)
        composite_by_id = {row["candidate_id"]: row for row in ranked_all}
        district_median = median(row["composite_score"] for row in ranked_all) if ranked_all else 0.0
        slate_ids = {p.get("candidate_id") for p in result.proposals}
        best_rejected = next((row for row in ranked_all if row["candidate_id"] not in slate_ids), None)
        rows = []
        for proposal in result.proposals:
            row = composite_by_id.get(proposal.get("candidate_id"))
            if row is None:
                continue
            percentile = 100.0 * (total - row["rank"] + 1) / total
            rows.append([
                proposal.get("rank"), proposal.get("municipality"),
                row["composite_score"], f"{percentile:.0f}th",
                row["component_availability"],
                f"{row['composite_score'] - district_median:+.3f}",
            ])
        lines.extend([
            "**Why these sites and not others:** the table places each selected "
            f"candidate against all {total} candidates in the district under the "
            "default weights (higher percentile = stronger on the shared criteria). "
            f"The district median composite is {district_median:.3f}"
            + (f"; the strongest candidate *not* selected is "
               f"{best_rejected['municipality']} ({best_rejected['action_type']}, "
               f"composite {best_rejected['composite_score']:.3f}), which readers can "
               "use to critique the condition's choice." if best_rejected else "."),
            "",
            markdown_table(
                ["rank", "municipality", "composite", "district percentile",
                 "data availability", "vs district median"],
                rows,
            ),
            "",
        ])
    return lines


def _condition_specific_approach_section(spec: ConditionSpec) -> list[str]:
    """The condition-discriminating half of the old criteria section: the
    five-objective table (only rendered for report profiles that actually
    commit to covering all five as part of their contract — skills_objectives
    and ai_scientist; an open-ended native-agent or vanilla condition is not
    held to that contract and must not imply it was) and a one-line summary
    of how this specific runner produced its slate."""

    lines: list[str] = []
    if spec.report_profile_id in _OBJECTIVE_TABLE_REPORT_PROFILES:
        lines.extend([
            "## Branch objectives explored",
            "",
            "This condition's Skills/AI-Scientist contract requires covering all "
            "five shared objectives; a slate is scored per objective by "
            "re-weighting the components with the objective's emphasis "
            "multipliers (availability-weighted, so data-poor candidates cannot "
            "win):",
            "",
            markdown_table(
                ["objective", "definition", "component emphasis"],
                [[o.label, o.description,
                  ", ".join(f"{k}×{v}" for k, v in o.component_emphasis.items())]
                 for o in BRANCH_OBJECTIVES],
            ),
            "",
        ])
    method = {
        "c0_deterministic": "Fixed default weights over the components; the ranking is the weighted composite, fully reproducible.",
        "vanilla_llm": "One direct LLM call over the full candidate table; the returned slate is validated and re-scored with the shared components.",
        "manual_harness": (
            "A coding-agent harness session walks the Skills-unified contract; its returned slate is validated and re-scored with the shared components."
            if spec.skills_unified else
            "A pure interactive coding-agent harness session (no Skills contract) solves the task with the harness's native abilities; its returned slate is validated and re-scored with the shared components."),
        "c14_ai_scientist_deepseek": "AI-Scientist-style search: LLM-written scoring scripts are executed in a sandbox per objective; branch winners are chosen by the external objective metric, and the final slate is synthesized from executed branch-winner rankings.",
        "c13_ai_scientist_gemini": "Same AI-Scientist-style algorithm as C14 with the same aligned budget envelope, run on Gemini.",
    }.get(spec.runner, "See the method summary above.")
    lines.extend([
        "## Condition-specific selection approach", "",
        f"**How this condition selected its slate:** {method}", "",
    ])
    return lines


def _site_data_section(result: LiveConditionResult, data: DataBundle | None) -> list[str]:
    if not data or not result.proposals:
        return []
    rows = []
    for proposal in result.proposals:
        facts = data.municipal_facts_by_key.get(
            (proposal.get("prefecture"), proposal.get("municipality")), {})
        rows.append([
            proposal.get("rank"), proposal.get("municipality"),
            _fmt_count(facts.get("population_total_2025")),
            _fmt_count(facts.get("population_65_plus_2025")),
            _fmt(facts.get("share_65_plus_2025_pct"), 1),
            _fmt(facts.get("share_65_plus_2050_pct"), 1),
            (f"{facts['population_pct_change_2020_2050'] * 100:+.1f}%"
             if isinstance(facts.get("population_pct_change_2020_2050"), (int, float))
             else "not_available"),
            (f"¥{_fmt_count(facts.get('land_price_median_jpy_per_sqm'))}"
             if facts.get("land_price_median_jpy_per_sqm") is not None else "not_available"),
            _fmt_count(facts.get("hospital_count")),
        ])
    return [
        "## Candidate site data (real sourced figures)",
        "",
        "Population figures are census projections (2025 values; 65+ share also "
        "shown for 2050); land prices are municipal medians from MLIT records; "
        "hospital counts are Yahoo Local Search facility records. All values are "
        "`verified_source`; blanks are stated, never imputed.",
        "",
        markdown_table(
            ["rank", "municipality", "pop 2025", "pop 65+ 2025", "65+ % 2025",
             "65+ % 2050", "pop Δ 2020-50", "land median /m²", "hospitals"],
            rows,
        ),
        "",
    ]


def _cost_basis_section(result: LiveConditionResult, data: DataBundle | None) -> list[str]:
    if not data or not result.proposals:
        return []
    rows = []
    for proposal in result.proposals:
        key = (proposal.get("prefecture"), proposal.get("municipality"))
        facts = data.municipal_facts_by_key.get(key, {})
        model = data.cost_model_by_prefecture.get(proposal.get("prefecture"), {})
        scenario = location_cost_model(facts, model)
        rows.append([
            proposal.get("rank"),
            proposal.get("municipality"),
            proposal.get("prefecture"),
            (f"¥{_fmt_count(scenario.get('municipality_land_price_jpy_per_sqm'))}/m²"
             if scenario.get("municipality_land_price_jpy_per_sqm") is not None else "not_available"),
            (_fmt(scenario.get("location_cost_factor"), 2)
             if scenario.get("location_cost_factor") is not None else "not_available"),
            (f"¥{scenario['estimated_construction_cost_per_bed_jpy_mm']:,.0f}M/bed"
             if scenario.get("estimated_construction_cost_per_bed_jpy_mm") is not None else "not_available"),
            (f"¥{scenario['estimated_initial_investment_per_bed_jpy_mm']:,.0f}M/bed"
             if scenario.get("estimated_initial_investment_per_bed_jpy_mm") is not None else "not_available"),
            (f"{scenario['estimated_payback_years']:.0f}y"
             if scenario.get("estimated_payback_years") is not None else "not_available"),
            scenario.get("evidence_grade"),
        ])
    return [
        "## Candidate-specific indicative cost basis (scenario estimates)",
        "",
        "Municipal land price comes from MLIT Reinfolib records. Construction cost, "
        "initial investment, and payback are scenario estimates: the hospital "
        "cash-flow workbook's prefecture medians (`data_basis: モデル推計` — "
        "**model_estimate**) are multiplied by the candidate municipality's "
        "MLIT land median divided by the workbook prefecture land median. These "
        "values are **scenario_assumption**, not verified acquisition, "
        "construction, renovation, or operating costings.",
        "",
        markdown_table(
            ["rank", "municipality", "prefecture", "MLIT land median",
             "land cost factor", "scaled construction cost",
             "scaled initial investment", "scaled payback", "evidence grade"],
            rows,
        ),
        "",
    ]


def _map_points(result: LiveConditionResult, data: DataBundle | None):
    if not data or not result.proposals:
        return {}, []
    background: dict[str, list[tuple[float, float]]] = {}
    for (prefecture, _municipality), records in data.facilities_by_key.items():
        background.setdefault(prefecture, []).extend(
            (r.longitude, r.latitude) for r in records
            if isinstance(r.latitude, (int, float)) and isinstance(r.longitude, (int, float)))
    centroids = _municipality_anchors(data)
    candidates = []
    for proposal in result.proposals:
        centroid = centroids.get((proposal.get("prefecture"), proposal.get("municipality")))
        if centroid is None:
            continue
        candidates.append({
            "prefecture": proposal.get("prefecture"),
            "municipality": proposal.get("municipality"),
            "rank": proposal.get("rank"),
            "lon": centroid[0], "lat": centroid[1],
        })
    return background, candidates


def _municipality_anchors(data: DataBundle) -> dict[tuple[str, str], tuple[float, float]]:
    """Municipality anchor points: mean of that municipality's facility
    geocodes. Memoized per bundle — every map in a run shares the anchors."""

    cached = data.runtime_cache.get("municipality_anchors")
    if cached is not None:
        return cached
    anchors: dict[tuple[str, str], tuple[float, float]] = {}
    for key, records in data.facilities_by_key.items():
        coords = [(r.longitude, r.latitude) for r in records
                  if isinstance(r.latitude, (int, float)) and isinstance(r.longitude, (int, float))]
        if coords:
            anchors[key] = (mean(lon for lon, _ in coords), mean(lat for _, lat in coords))
    data.runtime_cache["municipality_anchors"] = anchors
    return anchors


def _metric_map_points(data: DataBundle, metric_key: str) -> dict[str, list[dict]]:
    """All study-area municipalities with a geocode anchor, carrying one metric
    value (None when the source has no figure for that municipality)."""

    anchors = _municipality_anchors(data)
    points: dict[str, list[dict]] = {}
    for (prefecture, municipality), (lon, lat) in anchors.items():
        facts = data.municipal_facts_by_key.get((prefecture, municipality), {})
        value = facts.get(metric_key)
        points.setdefault(prefecture, []).append({
            "lon": lon, "lat": lat, "municipality": municipality,
            "value": float(value) if isinstance(value, (int, float)) else None,
        })
    return points


def _selection_funnel_section(result: LiveConditionResult, data: DataBundle | None) -> list[str]:
    """Table tracing the real selection funnel from the nationwide census view
    down to this condition's final slate; every count comes from a local
    validated artifact."""

    if not data or not result.proposals:
        return []
    national = data.national_coverage or {}
    study_prefectures = sorted({p for p, _m in data.scores_by_key})
    action_counts: dict[str, int] = {}
    for candidate in data.candidates:
        action = str(candidate.get("candidate_action"))
        action_counts[action] = action_counts.get(action, 0) + 1
    rows = []
    if national:
        rows.append([
            "1. National census projection coverage",
            f"{national['prefecture_count']} prefectures / "
            f"{national['municipality_count']:,} municipalities",
            "normalized e-Stat census projection workbook (2020-2050)",
            "`.data/interim/views/population_long.jsonl`",
        ])
    rows.extend([
        [
            "2. Study-area filter",
            f"{len(study_prefectures)} prefectures / "
            f"{len(data.scores_by_key)} municipalities",
            "configured study area: " + ", ".join(study_prefectures),
            "`municipality_master_records.jsonl`",
        ],
        [
            "3. Municipalities scored on shared components",
            f"{len(data.scores_by_key)} municipalities",
            "demand / aging / supply / land / financial / risk / completeness",
            "`municipality_scores_enriched.jsonl`",
        ],
        [
            "4. Candidate actions generated",
            f"{len(data.candidates)} candidates ("
            + ", ".join(f"{action} {count}" for action, count in sorted(action_counts.items()))
            + ")",
            "action-generation rules over the score layer (no invented sites)",
            "`candidate_actions.jsonl`",
        ],
        [
            "5. Final slate for this condition",
            f"{len(result.proposals)} proposals",
            "condition-specific ranking, validated against the candidate universe",
            "this report",
        ],
    ])
    return [
        "## Selection funnel (nationwide → final slate)",
        "",
        "Every stage below is a real, source-traceable count — the funnel shows "
        "how the nationwide e-Stat census universe narrows to this condition's "
        "final slate. No stage invents candidates.",
        "",
        markdown_table(["stage", "count", "criterion", "source artifact"], rows),
        "",
    ]


def _evidence_grade_table(proposals: list[dict[str, Any]]) -> str:
    rows = []
    for proposal in proposals:
        grades = proposal.get("evidence_grades") or {}
        rows.append([
            proposal.get("rank"), proposal.get("municipality"),
            proposal.get("target_facility_name") or "not_available",
            grades.get("target_facility_name", "not_available"),
            grades.get("financial", "not_available"),
            grades.get("land", "not_available"),
            len(proposal.get("source_evidence_refs") or []),
        ])
    return markdown_table(
        ["rank", "municipality", "facility target", "facility grade",
         "financial grade", "land grade", "evidence refs"],
        rows,
    )


def _missing_data_table(proposals: list[dict[str, Any]]) -> str:
    rows = []
    for proposal in proposals:
        gaps = proposal.get("evidence_gaps") or []
        rows.append([
            proposal.get("rank"), proposal.get("municipality"),
            ", ".join(gaps[:4]) or "none recorded",
            len(proposal.get("required_due_diligence") or []),
        ])
    return markdown_table(
        ["rank", "municipality", "evidence gaps", "due-diligence items"], rows,
    )


def _generated_code_table(stats: dict[str, Any]) -> str:
    if not stats:
        return "No generated-code execution in this condition."
    return markdown_table(
        ["metric", "value"],
        [[key, value] for key, value in sorted(stats.items())],
    )


def _model_call_table(summary: dict[str, Any]) -> str:
    rows = summary.get("by_model") or []
    if not rows:
        return "No live model calls in this condition."
    return markdown_table(
        ["provider", "model", "purpose", "requests", "errors",
         "prompt tokens", "completion tokens", "reasoning tokens"],
        [[row.get("provider"), row.get("model"), row.get("purpose"),
          row.get("request_count"), row.get("error_count"),
          row.get("prompt_tokens"), row.get("completion_tokens"),
          row.get("reasoning_tokens")] for row in rows],
    )


def _branch_table(branch_results: list[dict[str, Any]]) -> str:
    return markdown_table(
        ["objective", "nodes evaluated", "nodes succeeded", "winner metric",
         "winner top candidates"],
        [[row.get("objective_label") or row.get("objective"),
          row.get("nodes_evaluated", "n/a"), row.get("nodes_succeeded", "n/a"),
          row.get("winner_external_metric"),
          ", ".join((row.get("winner_top_candidates") or [])[:3]) or "none"]
         for row in branch_results],
    )


_MAX_REPRESENTATIVE_ALTERNATIVES = 6


def _select_representative_alternatives(
    bundle: DecisionAnalysisBundle, run_dir: Path | None,
) -> tuple[list[Any], int]:
    """Bound the alternatives actually rendered in the main report to a
    small, deterministic, representative set (never the full search
    universe), while keeping the complete list available as an artifact.

    Priority order, first-fill, capped at ``_MAX_REPRESENTATIVE_ALTERNATIVES``:
    (1) every alternative actually adopted into a final candidate (via
    ``candidate_to_alternative_provenance``); (2) the best-scored non-adopted
    alternative (highest best candidate-outcome score, or lowest best rank
    when no score is recorded); (3) the alternative most materially
    different from the adopted set (largest symmetric difference of
    top_candidates); (4) one rejected/failed alternative with a recorded
    reason; (5) one alternative referenced by a contingency portfolio.
    Ties broken by ``alternative_id`` for determinism. Returns the
    representative subset and the count of alternatives NOT shown.
    """

    alternatives = bundle.decision_alternatives
    if len(alternatives) <= _MAX_REPRESENTATIVE_ALTERNATIVES:
        return alternatives, 0

    by_id = {a.alternative_id: a for a in alternatives}
    adopted_ids: set[str] = set()
    for ids in bundle.candidate_to_alternative_provenance.values():
        adopted_ids.update(ids)
    adopted = sorted((by_id[i] for i in adopted_ids if i in by_id), key=lambda a: a.alternative_id)

    selected: list[Any] = []
    selected_ids: set[str] = set()

    def add(candidates: list[Any]) -> None:
        for alt in candidates:
            if len(selected) >= _MAX_REPRESENTATIVE_ALTERNATIVES:
                return
            if alt.alternative_id in selected_ids:
                continue
            selected.append(alt)
            selected_ids.add(alt.alternative_id)

    add(adopted)

    def best_score(alt: Any) -> tuple[float, int]:
        scores = [o.score for o in alt.candidate_outcomes if o.score is not None]
        if scores:
            return (max(scores), 0)
        ranks = [o.rank for o in alt.candidate_outcomes if o.rank is not None]
        return (0.0, -min(ranks)) if ranks else (float("-inf"), 0)

    non_adopted = sorted(
        (a for a in alternatives if a.alternative_id not in selected_ids),
        key=lambda a: (best_score(a), a.alternative_id), reverse=True)
    add(non_adopted[:1])

    adopted_candidate_union = {c for a in adopted for c in a.top_candidates}
    if adopted_candidate_union:
        def symmetric_difference(alt: Any) -> int:
            return len(set(alt.top_candidates) ^ adopted_candidate_union)
        most_different = sorted(
            (a for a in alternatives if a.alternative_id not in selected_ids),
            key=lambda a: (-symmetric_difference(a), a.alternative_id))
        add(most_different[:1])

    rejected = sorted(
        (a for a in alternatives if a.alternative_id not in selected_ids
         and (a.rejection_reason or a.execution_status == "failed")),
        key=lambda a: a.alternative_id)
    add(rejected[:1])

    contingency_alt_ids: set[str] = set()
    for portfolio in bundle.contingency_portfolios:
        contingency_alt_ids.update(portfolio.source_alternative_ids)
    contingency_linked = sorted(
        (a for a in alternatives if a.alternative_id not in selected_ids
         and a.alternative_id in contingency_alt_ids),
        key=lambda a: a.alternative_id)
    add(contingency_linked[:1])

    remainder = len(alternatives) - len(selected)
    if run_dir is not None:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "decision_alternatives_full.json").write_text(
                json.dumps([a.model_dump(mode="json") for a in alternatives],
                           ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError:
            pass
    # Preserve original registry order among the selected subset so the
    # report reads in the same relative order as a small alternatives list
    # always did, rather than the priority-fill order.
    order = {a.alternative_id: i for i, a in enumerate(alternatives)}
    selected.sort(key=lambda a: order[a.alternative_id])
    return selected, remainder


def _decision_analysis_sections(
    bundle: DecisionAnalysisBundle | None,
    proposals: list[dict[str, Any]],
    run_dir: Path | None = None,
) -> list[str]:
    """Compile only decision analysis directly supported by bundle artifacts."""

    if bundle is None:
        return []
    if bundle.reporting_contract is not None:
        return _reporting_v2_sections(bundle, proposals)
    lines: list[str] = []
    representative_alternatives, hidden_count = _select_representative_alternatives(bundle, run_dir)
    if representative_alternatives:
        lines.extend(["## Strategic alternatives evaluated", ""])
        if hidden_count:
            lines.extend([
                f"Showing {len(representative_alternatives)} representative alternatives out of "
                f"{len(bundle.decision_alternatives)} actually executed (adopted, best-"
                "scored, most materially different, one rejected, one contingency-"
                f"linked). The full list is in `decision_alternatives_full.json`.",
                "",
            ])
        for alternative in representative_alternatives:
            lines.extend([f"### {alternative.alternative_id}", ""])
            fields = [
                ("Hypothesis", alternative.hypothesis),
                ("Objective", alternative.objective),
                ("Decision regime", alternative.decision_regime),
                ("Assumptions", "; ".join(alternative.assumptions)),
                ("Eligibility constraints", "; ".join(alternative.eligibility_rules)),
                ("Evaluation logic", alternative.scoring_rule),
                ("Risk tolerance", alternative.risk_tolerance),
                ("Portfolio rule", alternative.portfolio_rule),
                ("Top candidates", ", ".join(alternative.top_candidates)),
                ("Strengths / interpretation", alternative.interpretation),
                ("Adoption rationale", alternative.selection_reason),
                ("Rejection rationale", alternative.rejection_reason),
            ]
            for label, value in fields:
                if value not in (None, "", [], {}):
                    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else value
                    lines.append(f"- **{label}:** {rendered}")
            lines.append("")

    if len(representative_alternatives) >= 2:
        candidate_ids = sorted({outcome.candidate_id for alternative in representative_alternatives
                                for outcome in alternative.candidate_outcomes})
        if candidate_ids:
            matrix_rows = []
            by_alternative = {
                alternative.alternative_id: {row.candidate_id: row for row in alternative.candidate_outcomes}
                for alternative in representative_alternatives
            }
            for candidate_id in candidate_ids:
                row = [candidate_id]
                for alternative in representative_alternatives:
                    outcome = by_alternative[alternative.alternative_id].get(candidate_id)
                    if outcome is None:
                        row.append("not evaluated")
                    elif outcome.status:
                        row.append(f"{outcome.status} (rank {outcome.rank})" if outcome.rank else outcome.status)
                    elif outcome.score is not None:
                        row.append(f"rank {outcome.rank}; score {outcome.score}" if outcome.rank else str(outcome.score))
                    else:
                        row.append(f"rank {outcome.rank}" if outcome.rank else "evaluated")
                matrix_rows.append(row)
            lines.extend([
                "## Candidate × strategy outcome matrix", "",
                markdown_table(
                    ["candidate_id", *[row.alternative_id for row in representative_alternatives]], matrix_rows),
                "",
            ])

    diagnostics = bundle.search_diagnostics
    if diagnostics.classification != "not_available":
        lines.extend([
            "## Search-derived findings", "",
            f"**Search classification:** `{diagnostics.classification}`. {diagnostics.explanation}", "",
            f"The alternatives covered {diagnostics.union_candidate_count} unique candidates; "
            f"{diagnostics.branch_exclusive_candidate_count} appeared in only one alternative. "
            f"There were {diagnostics.materially_distinct_pair_count} materially distinct alternative pairs.",
            "",
        ])
        if diagnostics.shortlist_only_permutation:
            lines.extend([
                "The executed alternatives only permuted a preselected shortlist rather than "
                "evaluating the full candidate universe. This is a negative finding: the run "
                "does not establish that the recorded frontier dominates excluded candidates.", "",
            ])
        if bundle.stable_candidates:
            lines.append("**Stable candidates:** " + ", ".join(bundle.stable_candidates))
        if bundle.sensitive_candidates:
            lines.append("**Sensitive candidates:** " + ", ".join(bundle.sensitive_candidates))
        if bundle.stable_candidates or bundle.sensitive_candidates:
            lines.append("")

    performed_tests = [row for row in bundle.robustness_analysis if row.status != "not_performed"]
    if performed_tests:
        lines.extend([
            "## Validation, ablation, and stress tests", "",
            markdown_table(
                ["test", "type", "status", "specification", "conclusion"],
                [[row.test_id, row.test_type, row.status,
                  json.dumps(row.specification, ensure_ascii=False) if isinstance(row.specification, dict)
                  else row.specification or "not_available",
                  row.conclusion or "not_available"] for row in performed_tests]),
            "",
        ])

    # Failure visibility is computed from the FULL bundle, never the bounded
    # representative subset above -- a failure must never be hidden by the
    # alternatives cap.
    rejected = [row for row in bundle.decision_alternatives
                if row.rejection_reason or row.execution_status == "failed"]
    failed_tests = [row for row in bundle.robustness_analysis if row.status == "failed"]
    if rejected or failed_tests or diagnostics.classification == "degenerate_search":
        lines.extend(["## Failed, rejected, or inconclusive analyses", ""])
        for alternative in rejected:
            lesson = alternative.rejection_reason or "Execution failed; this strategy was not selected."
            lines.append(f"- **{alternative.alternative_id}:** {lesson}")
        for test in failed_tests:
            lines.append(f"- **{test.test_id} failed:** {test.conclusion or 'no conclusion recorded'}")
        if diagnostics.classification == "degenerate_search":
            lines.append(f"- **Degenerate search:** {diagnostics.explanation}")
        lines.append("")

    if bundle.critical_issues:
        lines.extend([
            "## Critical decision issues and resulting adjustments", "",
            markdown_table(
                ["issue", "candidate", "evidence", "severity", "resolution status",
                 "decision effect", "residual risk"],
                [[row.issue, row.candidate_id or "portfolio",
                  ", ".join(row.evidence_refs) or "not_available", row.severity,
                  row.resolution_status, row.decision_effect or "no recorded change",
                  row.residual_risk or "not_available"] for row in bundle.critical_issues]),
            "",
        ])
        if bundle.revision_effects:
            lines.extend([
                markdown_table(
                    ["candidate", "issue", "change", "before", "after", "reason", "decision effect"],
                    [[row.candidate_id, row.issue_id, row.change_type, row.before, row.after,
                      row.reason, row.decision_effect or "not recorded"]
                     for row in bundle.revision_effects]),
                "",
            ])

    portfolios = ([bundle.primary_portfolio] if bundle.primary_portfolio else []) + bundle.contingency_portfolios
    if portfolios:
        lines.extend([
            "## Primary and contingency portfolios", "",
            markdown_table(
                ["portfolio", "role", "candidates", "source alternatives", "rationale"],
                [[row.label, "primary" if row is bundle.primary_portfolio else "contingency",
                  ", ".join(row.candidate_ids), ", ".join(row.source_alternative_ids),
                  row.rationale or "not_available"] for row in portfolios]),
            "",
        ])

    if bundle.reversal_conditions:
        lines.extend([
            "## Reversal conditions", "",
            markdown_table(
                ["candidate", "current decision", "trigger", "threshold / scenario",
                 "new decision", "replacement"],
                [[row.candidate_id, row.current_decision, row.triggering_variable_or_finding,
                  row.threshold_or_scenario, row.new_decision,
                  row.replacement_candidate_id or "not specified"]
                 for row in bundle.reversal_conditions]),
            "",
        ])

    roadmap_rows = []
    for decision in bundle.final_decision_rows:
        for next_step in decision.required_next_steps:
            roadmap_rows.append([decision.candidate_id, decision.decision_status, next_step,
                                 "; ".join(decision.blocking_conditions) or "none recorded"])
    if roadmap_rows:
        lines.extend([
            "## Implementation and due-diligence roadmap", "",
            markdown_table(["candidate", "status", "next step", "decision gate"], roadmap_rows),
            "",
        ])
    return lines


def _reporting_v2_sections(
    bundle: DecisionAnalysisBundle,
    proposals: list[dict[str, Any]],
) -> list[str]:
    """Render every reporting-v2 category without upgrading its status."""

    report = bundle.reporting_contract
    assert report is not None
    proposal_by_id = {str(row.get("candidate_id")): row for row in proposals}
    rank_by_id = {
        str(row.get("candidate_id")): int(row.get("rank") or index + 1)
        for index, row in enumerate(proposals)
    }
    vanilla = bundle.condition_group in {"C1", "C2", "C3", "C4"}
    lines = [
        "## Analytical approach and permitted execution mode", "",
        ("Single-pass vanilla reasoning only: no tools, code execution, empirical validation, "
         "branch-search execution, iterative review, or additional model calls were permitted."
         if vanilla else
         "Native coding-agent analysis without project Skills: repository inspection, code execution, "
         "validation, review, and revision were permitted but were not mandatory."),
        "",
        f"**Branch-search status:** `{bundle.branch_search_status}`. Per-alternative statuses below "
        "are authoritative; `reasoned_only` comparisons are not executed branch search.", "",
        "## Decision alternatives", "",
        markdown_table(
            ["alternative", "status", "objective", "selection reason", "rejection reason", "evidence"],
            [[row.alternative_id, row.execution_status, row.objective,
              row.selection_reason or "none", row.rejection_reason or "none",
              ", ".join(row.evidence_refs) or "none"]
             for row in report.decision_alternatives]),
        "",
        "## Candidate outcomes by alternative", "",
        markdown_table(
            ["alternative", "status", "candidate", "rank", "outcome", "summary", "evidence"],
            [[alternative.alternative_id, alternative.execution_status, outcome.candidate_id,
              outcome.rank, outcome.outcome, outcome.summary,
              ", ".join(outcome.evidence_refs) or "none"]
             for alternative in report.decision_alternatives
             for outcome in alternative.candidate_outcomes]),
        "",
        "## Validation and robustness tests", "",
        markdown_table(
            ["test", "status", "objective", "result", "decision effect", "evidence"],
            [[row.test_id, row.status, row.objective, row.result, row.decision_effect,
              ", ".join(row.evidence_refs) or "none"] for row in report.validation_tests]),
        "",
        "## Negative, failed, or inconclusive findings", "",
    ]
    negative_rows: list[str] = []
    for row in report.decision_alternatives:
        if row.rejection_reason or row.execution_status == "not_performed":
            negative_rows.append(
                f"- **{row.alternative_id}** (`{row.execution_status}`): "
                f"{row.rejection_reason or 'not selected'}")
    for row in report.validation_tests:
        result_lower = row.result.lower()
        if (row.status == "not_performed"
                or any(word in result_lower for word in ("fail", "inconclusive", "negative", "contradict"))):
            negative_rows.append(f"- **{row.test_id}** (`{row.status}`): {row.result}")
    lines.extend(negative_rows or ["- No negative, failed, or inconclusive result was reported."])
    lines.extend([
        "",
        "## Synthesis rule and tradeoffs", "",
        f"**Rule:** {report.synthesis.rule}", "",
        "**Source alternatives:** "
        + (", ".join(report.synthesis.source_alternative_ids) or "none"), "",
        *([f"- {tradeoff}" for tradeoff in report.synthesis.tradeoffs]
          or ["- No tradeoff was reported."]), "",
        "## Excluded candidates and exclusion reasons", "",
    ])
    if report.synthesis.excluded_candidates:
        lines.extend([
            markdown_table(
                ["candidate", "reason", "source alternatives"],
                [[row.candidate_id, row.reason,
                  ", ".join(row.source_alternative_ids) or "none"]
                 for row in report.synthesis.excluded_candidates]), "",
        ])
    else:
        lines.extend(["No competitive excluded candidate was reported.", ""])
    lines.extend(["## Critical issues and decision effects", ""])
    if report.critical_issues:
        lines.extend([
            markdown_table(
                ["candidate", "issue", "decision effect", "evidence"],
                [[row.candidate_id or "slate", row.issue, row.decision_effect,
                  ", ".join(row.evidence_refs) or "none"] for row in report.critical_issues]), "",
        ])
    else:
        lines.extend(["No critical issue was reported.", ""])
    lines.extend(["## Reversal conditions", ""])
    lines.extend([
        markdown_table(
            ["candidate", "condition", "decision change", "evidence"],
            [[row.candidate_id or "slate", row.condition, row.decision_change,
              ", ".join(row.evidence_refs) or "none"] for row in report.reversal_conditions]),
        "",
        "## Final candidate decisions", "",
        markdown_table(
            ["rank", "candidate", "preferred action", "decision status", "status reason",
             "blocking conditions", "prioritized next steps"],
            [[rank_by_id.get(row.candidate_id, "not_available"), row.candidate_id,
              proposal_by_id.get(row.candidate_id, {}).get("action_type", "not_available"),
              row.decision_status, row.status_reason,
              "; ".join(row.blocking_conditions) or "none",
              "; ".join(f"{index + 1}. {step}" for index, step in enumerate(row.required_next_steps))]
             for row in sorted(report.final_decisions,
                               key=lambda item: rank_by_id.get(item.candidate_id, 10**9))]),
        "",
        "## Evidence and provenance references", "",
    ])
    evidence_refs = list(dict.fromkeys(
        [ref for row in report.decision_alternatives for ref in row.evidence_refs]
        + [ref for row in report.decision_alternatives for outcome in row.candidate_outcomes
           for ref in outcome.evidence_refs]
        + [ref for row in report.validation_tests for ref in row.evidence_refs]
        + [ref for row in report.critical_issues for ref in row.evidence_refs]
        + [ref for row in report.reversal_conditions for ref in row.evidence_refs]
    ))
    lines.extend([f"- `{ref}`" for ref in evidence_refs] or [
        "- No execution artifact was cited; all reported work is reasoned-only or not performed."])
    not_performed = [
        f"alternative `{row.alternative_id}`: {row.rejection_reason}"
        for row in report.decision_alternatives if row.execution_status == "not_performed"
    ] + [
        f"validation `{row.test_id}`: {row.result}"
        for row in report.validation_tests if row.status == "not_performed"
    ]
    lines.extend([
        "", "## Explicitly not-performed analyses", "",
        *([f"- {item}" for item in not_performed]
          or ["- No analysis was labeled `not_performed`."]), "",
    ])
    reported_discussions = [
        (str(row.get("candidate_id")), row.get("model_reported_qualitative_discussion"))
        for row in proposals if isinstance(row.get("model_reported_qualitative_discussion"), dict)
    ]
    if reported_discussions:
        dimensions = (
            "regional", "population", "demand_supply", "access", "cost_financial",
            "preferred_action", "review_comments",
        )
        lines.extend([
            "## Model-reported qualitative discussion", "",
            "These concise statements are preserved from the condition response and are graded "
            "`model_estimate`; shared sourced site facts are rendered separately above.", "",
            markdown_table(
                ["candidate", *dimensions],
                [[candidate_id, *[discussion.get(name, "not_available") for name in dimensions]]
                 for candidate_id, discussion in reported_discussions]), "",
        ])
    return lines


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_condition_report(
    spec: ConditionSpec,
    result: LiveConditionResult,
    output_dir: Path,
    *,
    data: DataBundle | None = None,
    figures_dir: Path | None = None,
) -> Path:
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = figures_dir or output_dir / "figures"
    figure_lines: list[str] = []

    def add_figure(path: Path | None, caption: str) -> None:
        if path is not None:
            try:
                rel = Path("..") / path.relative_to(output_dir)
            except ValueError:
                rel = path
            figure_lines.extend([f"![{caption}]({rel})", ""])

    proposals = result.proposals
    # Vanilla LLM conditions (C1-C4) are true single-pass baselines: the
    # shared post-hoc candidate-review/author-response augmentation never
    # runs for them (candidate_deliberation_runtime.py), so their reports
    # must not render it or any "review was not run/disabled" boilerplate
    # either — checked by spec.algorithm, matching
    # condition_supports_candidate_deliberation's semantic gate, not a
    # hard-coded C1-C4 list.
    include_candidate_review = spec.algorithm != "vanilla_llm"
    review_packet_by_candidate_id = {
        str(packet.get("candidate_id")): packet
        for packet in result.candidate_review_packets
    }
    review_threads_by_candidate_id: dict[str, list[dict[str, object]]] = {}
    for thread in result.candidate_review_threads:
        review_threads_by_candidate_id.setdefault(
            str(thread.get("candidate_id")), []).append(thread)
    if result.candidate_review_threads or result.candidate_review_packets:
        candidate_deliberation_state = "attempted"
    elif spec.algorithm == "deterministic_baseline":
        candidate_deliberation_state = "skipped_deterministic"
    else:
        candidate_deliberation_state = "not_run"
    discussion_entries = (
        build_qualitative_site_discussions(
            proposals, data, narrative_sections=result.narrative_sections,
            review_packet_by_candidate_id=review_packet_by_candidate_id,
            review_threads_by_candidate_id=review_threads_by_candidate_id,
            candidate_deliberation_state=candidate_deliberation_state,
            include_candidate_review=include_candidate_review)
        if proposals and data is not None else []
    )

    slug = spec.report_slug
    if proposals:
        candidate_scores_path = figures_dir / f"{slug}_candidate_scores.png"
        write_figure_data(candidate_scores_path, {
            "figure": "candidate_scores",
            "condition_group": spec.group,
            "rows": [
                {
                    "rank": p.get("rank"),
                    "candidate_id": p.get("candidate_id"),
                    "prefecture": p.get("prefecture"),
                    "municipality": p.get("municipality"),
                    "action_type": p.get("action_type"),
                    "composite_score": p.get("composite_score"),
                }
                for p in proposals
            ],
        })
        add_figure(candidate_scores_figure(
            proposals, candidate_scores_path,
            title=f"{spec.group}: top candidate composite scores"),
            "Top candidate composite scores")
        component_breakdown_path = figures_dir / f"{slug}_component_breakdown.png"
        write_figure_data(component_breakdown_path, {
            "figure": "component_breakdown",
            "condition_group": spec.group,
            "weights": DEFAULT_WEIGHTS,
            "rows": [
                {
                    "rank": p.get("rank"),
                    "candidate_id": p.get("candidate_id"),
                    "prefecture": p.get("prefecture"),
                    "municipality": p.get("municipality"),
                    "action_type": p.get("action_type"),
                    "score_components": p.get("score_components") or {},
                }
                for p in proposals
            ],
        })
        add_figure(component_breakdown_figure(
            proposals, DEFAULT_WEIGHTS, component_breakdown_path,
            title=f"{spec.group}: weighted component contributions"),
            "Weighted component contributions")
        background, candidate_points = _map_points(result, data)
        candidate_map_path = figures_dir / f"{slug}_map.png"
        write_figure_data(candidate_map_path, {
            "figure": "candidate_map",
            "condition_group": spec.group,
            "background_points": background,
            "candidate_points": candidate_points,
            "note": "Schematic lon/lat scatter from Yahoo facility geocodes; not administrative boundaries.",
        })
        add_figure(candidate_map_figure(
            background, candidate_points, candidate_map_path,
            title=f"{spec.group}: candidate locations (schematic, facility geocodes)"),
            "Candidate locations (schematic panels from Yahoo facility geocodes)")
        if data is not None:
            for metric_key, suffix, metric_label, title_suffix in _METRIC_MAPS:
                metric_points = _metric_map_points(data, metric_key)
                metric_path = figures_dir / f"{slug}_{suffix}.png"
                write_figure_data(metric_path, {
                    "figure": "metric_map",
                    "condition_group": spec.group,
                    "metric_key": metric_key,
                    "metric_label": metric_label,
                    "metric_points": metric_points,
                    "candidate_points": candidate_points,
                    "note": "Schematic municipality anchors from facility geocodes; hollow values are missing data.",
                })
                add_figure(metric_map_figure(
                    metric_points, candidate_points, metric_path,
                    title=f"{spec.group}: {title_suffix} (study-area municipalities)",
                    metric_label=metric_label),
                    f"Selected candidates (amber rings, ranked) over {metric_label}; "
                    "municipality anchors are facility-geocode centroids "
                    "(schematic, not administrative boundaries); hollow gray = no data")
    if result.branch_results:
        branch_path = figures_dir / f"{slug}_branch_objectives.png"
        write_figure_data(branch_path, {
            "figure": "branch_objectives",
            "condition_group": spec.group,
            "rows": result.branch_results,
        })
        add_figure(grouped_bar_figure(
            [str(row.get("objective")) for row in result.branch_results],
            {"winner external metric": [row.get("winner_external_metric") or 0.0
                                        for row in result.branch_results]},
            branch_path,
            title=f"{spec.group}: branch-objective winner metrics",
            ylabel="external objective metric"),
            "Branch-objective winner metrics")
    if proposals:
        evidence_path = figures_dir / f"{slug}_evidence_completeness.png"
        write_figure_data(evidence_path, {
            "figure": "evidence_completeness",
            "condition_group": spec.group,
            "rows": [
                {
                    "candidate_id": p.get("candidate_id"),
                    "municipality": p.get("municipality"),
                    "evidence_completeness": (
                        (p.get("score_components") or {}).get("evidence_completeness")
                    ),
                }
                for p in proposals
            ],
        })
        add_figure(grouped_bar_figure(
            [p.get("municipality") or "?" for p in proposals],
            {"evidence completeness": [
                (p.get("score_components") or {}).get("evidence_completeness") or 0.0
                for p in proposals]},
            evidence_path,
            title=f"{spec.group}: evidence completeness of the slate",
            ylabel="completeness score"),
            "Evidence completeness of the slate")
    stats = result.generated_code_stats
    if stats and (stats.get("nodes") or stats.get("generated")):
        succeeded = int(stats.get("nodes_succeeded") or stats.get("executed_ok") or 0)
        total = int(stats.get("nodes") or stats.get("generated") or 0)
        code_success_path = figures_dir / f"{slug}_code_success.png"
        write_figure_data(code_success_path, {
            "figure": "code_success",
            "condition_group": spec.group,
            "generated_code_stats": stats,
            "succeeded": succeeded,
            "failed": max(total - succeeded, 0),
        })
        add_figure(success_failure_figure(
            ["generated code"], [succeeded], [max(total - succeeded, 0)],
            code_success_path,
            title=f"{spec.group}: generated-code execution outcomes"),
            "Generated-code execution outcomes")
    # by_model rows are per (provider, model, purpose): sum per model.
    model_counts: dict[str, int] = {}
    for row in result.model_call_summary.get("by_model") or []:
        model_counts[row["model"]] = (
            model_counts.get(row["model"], 0) + int(row.get("request_count") or 0))
    if len(model_counts) > 1:
        call_distribution_path = figures_dir / f"{slug}_call_distribution.png"
        write_figure_data(call_distribution_path, {
            "figure": "call_distribution",
            "condition_group": spec.group,
            "model_counts": model_counts,
            "by_model": result.model_call_summary.get("by_model") or [],
        })
        add_figure(call_distribution_figure(
            model_counts, call_distribution_path,
            title=f"{spec.group}: model-call distribution"),
            "Model-call distribution")
    # Guarded defensively (not only via Part 1's empty-fields invariant): a
    # rewrite from an older record could still carry stale candidate-review
    # artifact references, and this must never surface them for Vanilla.
    if include_candidate_review:
        _review_figure_captions = (
            (f"{slug}_candidate_review_severity", "Reviewer finding severity per candidate"),
            (f"{slug}_candidate_review_coverage", "Reviewer coverage per candidate"),
            (f"{slug}_candidate_author_response_status", "Author response status per candidate"),
            (f"{slug}_candidate_residual_risk", "Unresolved major/blocking risk per candidate"),
        )
        for artifact_key, caption in _review_figure_captions:
            path = result.artifacts.get(artifact_key)
            if path:
                add_figure(Path(path), caption)

    top = proposals[0] if proposals else None
    narrative = result.narrative_sections
    lines = [
        f"# {spec.group} — {spec.label}: Hospital Location / Reorganization Proposal",
        "",
        f"- Condition: `{spec.condition_id}` — algorithm `{spec.algorithm}`",
        f"- Provider/model/harness: `{spec.provider}` / `{spec.model or 'none'}` / `{spec.harness}`",
        f"- Execution mode: `{result.execution_mode}`"
        + ("" if result.comparable_for_e13 else " — **not comparable for the judge**"),
        *( [f"- Exclusion reason: {result.exclusion_reason}"] if result.exclusion_reason else [] ),
        f"- Live steps/calls used: {result.steps_run}",
        "",
    ]

    # ---- Layer 1: shared evidence and candidate context --------------------
    # Generated consistently for every condition: the candidate universe,
    # shared evaluation-model components, real sourced site/cost figures,
    # and evidence/missing-data notes. This is what makes conditions
    # comparable at all, never condition-specific analysis.
    lines.extend(["# Shared evidence and candidate context", ""])
    lines.extend(_shared_criteria_section(result, data))
    lines.extend(_site_data_section(result, data))
    lines.extend(_cost_basis_section(result, data))
    if proposals:
        lines.extend(["## Evidence grades", "", _evidence_grade_table(proposals), ""])
        lines.extend(["## Missing data / due-diligence overview", "",
                      _missing_data_table(proposals), ""])
    if data is not None and proposals:
        note = district_data_quality_note(data)
        if note:
            lines.extend(["## Study-area data quality note", "", note, ""])

    # ---- Layer 2: condition-specific decision analysis ---------------------
    # Only analysis this condition actually produced: its own narrative,
    # selection approach, executed strategies/branches, search-derived
    # findings, validation/robustness tests, rejected strategies, critique
    # and revision effects, synthesis, portfolios, reversal conditions, the
    # final slate, and qualitative discussion.
    lines.extend(["# Condition-specific decision analysis", "", "## Executive decision brief", ""])
    if narrative.get("executive_summary"):
        lines.extend([narrative["executive_summary"], ""])
        if narrative.get("executive_summary_flags"):
            lines.extend([
                "> Fabrication-guard flags on the model narrative (verify before use): "
                + narrative["executive_summary_flags"], "",
            ])
    elif top:
        lines.extend([
            f"Top candidate: **{top.get('municipality')} ({top.get('prefecture')})** — "
            f"`{top.get('action_type')}` action, composite score {top.get('composite_score')}. "
            f"{len(proposals)} ranked candidates follow.",
            "",
        ])
    else:
        lines.extend([
            "This condition did not produce a comparable proposal slate; see the "
            "execution notes below.",
            "",
        ])
    if narrative.get("method_summary"):
        lines.extend(["## Method (model-reported)", "", narrative["method_summary"], ""])
    if result.failure_notes:
        lines.extend(["## Execution notes", "",
                      *[f"- {note}" for note in result.failure_notes], ""])

    lines.extend(_condition_specific_approach_section(spec))
    lines.extend(_selection_funnel_section(result, data))
    lines.extend(_decision_analysis_sections(
        result.analysis_bundle, proposals, run_dir=output_dir / "runs" / spec.padded_id))

    if proposals:
        provenance = (result.analysis_bundle.candidate_to_alternative_provenance
                      if result.analysis_bundle is not None else {})
        # Only add the provenance column when the bundle actually carries it
        # for at least one candidate -- never a fabricated/always-empty column.
        show_provenance = any(provenance.get(p.get("candidate_id")) for p in proposals)
        headers = ["rank", "prefecture", "municipality", "action", "composite",
                   "facility target", "candidate_id"]
        rows = [[p.get("rank"), p.get("prefecture"), p.get("municipality"),
                 p.get("action_type"), p.get("composite_score"),
                 p.get("target_facility_name") or "not_available",
                 p.get("candidate_id")] for p in proposals]
        if show_provenance:
            headers.append("source alternatives")
            for row, p in zip(rows, proposals):
                row.append(", ".join(provenance.get(p.get("candidate_id")) or []) or "not_available")
        lines.extend([
            "## Ranked candidates",
            "",
            markdown_table(headers, rows),
            "",
        ])
    if discussion_entries:
        lines.extend([render_qualitative_site_discussion_section(discussion_entries)])
    if figure_lines:
        lines.extend(["## Figures", "", *figure_lines])
    lines.append(required_due_diligence_section(result.due_diligence))

    report_path = reports_dir / f"{slug}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
