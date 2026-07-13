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
"""

from __future__ import annotations

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
from geo_strategist.experiments.location_costing import location_cost_model
from geo_strategist.experiments.qualitative_discussion import (
    build_qualitative_site_discussions,
    district_data_quality_note,
    render_candidate_deliberation_section,
    render_candidate_deliberation_summary,
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

def _criteria_section(spec: ConditionSpec, result: LiveConditionResult,
                      data: DataBundle | None) -> list[str]:
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
    if spec.branch_search:
        lines.extend([
            "Branch/search conditions explore exactly five shared objectives; a "
            "slate is scored per objective by re-weighting the components with "
            "the objective's emphasis multipliers (availability-weighted, so "
            "data-poor candidates cannot win):",
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
        "c10_ai_scientist_deepseek": "AI-Scientist-style search: LLM-written scoring scripts are executed in a sandbox per objective; branch winners are chosen by the external objective metric, and the final slate is synthesized from executed branch-winner rankings.",
        "c9_ai_scientist_gemini": "Same AI-Scientist-style algorithm as C10 with the same aligned budget envelope, run on Gemini.",
        "c11_evolution": "Evolutionary search over scoring strategies; fitness is the mean of the five external objective metrics; novelty is enforced against the C9 baseline slate.",
        "c12_ab_mcts": "Adaptive branching over proposal programs (generated code executed per node); width-vs-depth decided by externally measured improvement.",
        "c13_router": "A router model assigns each pipeline task to a model role; the synthesized slate is confirmed by a bounded judge.",
    }.get(spec.runner, "See the method summary above.")
    lines.extend([f"**How this condition selected its slate:** {method}", ""])

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
            candidate_deliberation_state=candidate_deliberation_state)
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
        "## Executive summary",
        "",
    ]
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

    lines.extend(_criteria_section(spec, result, data))
    lines.extend(_selection_funnel_section(result, data))

    if proposals:
        lines.extend([
            "## Ranked candidates",
            "",
            markdown_table(
                ["rank", "prefecture", "municipality", "action", "composite",
                 "facility target", "candidate_id"],
                [[p.get("rank"), p.get("prefecture"), p.get("municipality"),
                  p.get("action_type"), p.get("composite_score"),
                  p.get("target_facility_name") or "not_available",
                  p.get("candidate_id")] for p in proposals],
            ),
            "",
        ])
    lines.extend(_site_data_section(result, data))
    lines.extend(_cost_basis_section(result, data))
    if result.branch_results:
        lines.extend(["## Branch-by-branch top candidates", "",
                      _branch_table(result.branch_results), ""])
    if proposals:
        lines.extend(["## Evidence grades", "", _evidence_grade_table(proposals), ""])
        lines.extend(["## Missing data / due-diligence overview", "",
                      _missing_data_table(proposals), ""])
    if data is not None and proposals:
        note = district_data_quality_note(data)
        if note:
            lines.extend(["## Study-area data quality note", "", note, ""])
    if discussion_entries:
        lines.extend([render_qualitative_site_discussion_section(discussion_entries)])
    if result.candidate_qualitative_assessments:
        assessment_by_candidate_id = {
            str(row.get("candidate_id")): row
            for row in result.candidate_qualitative_assessments
        }
        lines.extend([render_candidate_deliberation_section(
            proposals, assessment_by_candidate_id, review_packet_by_candidate_id)])
        lines.extend([render_candidate_deliberation_summary(
            result.candidate_deliberation_summary)])
    lines.extend(["## Generated-code execution", "",
                  _generated_code_table(result.generated_code_stats), ""])
    lines.extend(["## Model-call summary", "",
                  _model_call_table(result.model_call_summary), ""])
    if figure_lines:
        lines.extend(["## Figures", "", *figure_lines])
    if result.artifacts:
        lines.extend([
            "## Run artifacts", "",
            *[f"- `{name}`: `{path}`" for name, path in sorted(result.artifacts.items())],
            "",
        ])
    lines.append(required_due_diligence_section(result.due_diligence))

    report_path = reports_dir / f"{slug}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
