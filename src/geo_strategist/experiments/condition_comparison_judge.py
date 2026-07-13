"""Condition comparison judge: cross-condition proposal + agentic-process
scoring for C0-C13 (historically labeled "E13").

Scores every condition record on 18 dimensions covering proposal quality
(usefulness, site specificity, evidence traceability, demand/supply
reasoning, financial reasoning, risk handling, actionability, uncertainty
transparency, **qualitative site reasoning**), method quality (model
innovation, reproducibility, agentic search, generated-code quality,
review/revision quality, branch diversity, novelty vs the C9 baseline,
model-call efficiency), and harness/orchestration clarity.

Scoring is two-layered:

- a deterministic structural judge computes every dimension from measurable
  artifact properties (always runs), including a rubric over each proposal's
  structured ``qualitative_discussion`` entry;
- an optional live LLM judge (primary: Gemini per ``E13_JUDGE_PROVIDER``;
  secondary: OpenRouter when enabled) scores anonymized records — including
  the extracted qualitative site discussions — and is averaged into the
  structural scores. Judge failures degrade to structural-only and are
  reported, never silently ignored.

Artifacts (first-reader-friendly names): ``reports/condition_comparison_report.md``,
``condition_judge_scores.jsonl``, ``condition_judge_manifest.json``,
``condition_judge_anonymized_inputs.jsonl``. The judge explicitly separates
``comparable_live_agent_results`` from ``non_comparable_fallback_results``
and ``waiting_for_manual_harness_results`` and includes four strict
same-provider/model comparisons — C1-vs-C9 (Gemini), C2-vs-C6 (Codex),
C3-vs-C7 (Claude Code), and the C4/C8/C10 trio (OpenCode Go/DeepSeek) — plus
the C9-vs-C10 model/provider comparison, the C1-C4 vanilla-baseline table,
the C5-C8 Skills-in-harness table, and the C11-C13 orchestration comparison.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from geo_strategist.experiments.condition_output_contract import (
    JUDGMENT_SCOPE,
    anonymize_condition_record_for_judge,
)
from geo_strategist.experiments.condition_registry import (
    build_condition_registry,
    c1_c9_strict_comparison_status,
    c2_c6_strict_comparison_status,
    c3_c7_strict_comparison_status,
    c4_c8_c10_strict_comparison_status,
)
from geo_strategist.experiments.condition_utils import (
    _read_jsonl,
    _rel,
    _stable_id,
    _write_json,
    _write_jsonl,
)
from geo_strategist.reporting import markdown_table, required_due_diligence_section
from geo_strategist.reporting.figures import (
    condition_comparison_figure,
    grouped_bar_figure,
    radar_figure,
)

DEFAULT_PROPOSALS_DIR = Path("outputs/condition_proposals/live")

SCORE_DIMENSIONS: tuple[str, ...] = (
    "proposal_usefulness",
    "site_specificity",
    "evidence_traceability",
    "demand_supply_reasoning",
    "financial_reasoning",
    "risk_handling",
    "actionability",
    "uncertainty_transparency",
    "qualitative_site_reasoning",
    "model_innovation",
    "reproducibility",
    "agentic_search_quality",
    "generated_code_quality",
    "review_revision_quality",
    "branch_diversity",
    "novelty_vs_c9",
    "model_call_efficiency",
    "harness_orchestration_clarity",
)

# Dimensions the live LLM judge scores from anonymized proposal content.
LIVE_JUDGE_DIMENSIONS: tuple[str, ...] = (
    "proposal_usefulness",
    "demand_supply_reasoning",
    "financial_reasoning",
    "risk_handling",
    "actionability",
    "uncertainty_transparency",
    "qualitative_site_reasoning",
)

_QUALITATIVE_CRITERIA: tuple[str, ...] = (
    "candidate-level differences are explained (not generic boilerplate)",
    "regional characteristics and population composition are discussed",
    "elderly demand and medical supply are discussed",
    "nearby hospital/facility mentions are source-traceable",
    "real-estate and cost issues are not asserted without verification",
    "recommended actions are consistent with the score components",
    "reviewer/judge/due-diligence comments are reflected",
    "no fabrication is present",
)

_INNOVATION_BY_ALGORITHM = {
    "deterministic_baseline": 1,
    "vanilla_llm": 2,
    "skills_branch_search": 4,
    "ai_scientist_style": 4,
    "ai_scientist_style_large_scale": 5,
    "multi_model_evolution": 5,
    "adaptive_branching_mcts": 5,
    "dynamic_multi_agent_orchestrator": 5,
}


@dataclass(frozen=True)
class ConditionJudgeResult:
    run_id: str
    output_dir: Path
    scored_condition_count: int
    eligible_condition_count: int
    live_judge_enabled: bool
    live_judge_status: str
    comparison_report_path: str | None
    output_paths: dict[str, str]




def _qualitative_site_reasoning_score(record: dict[str, Any]) -> int:
    """Rubric (1-5) over structured qualitative_discussion entries.

    5: every candidate covers all seven dimensions with substantive,
       differentiated text and no fabrication flags; 4: most candidates
       covered; 3: present but generic/undifferentiated; 2: short/missing
       dimensions; 1: missing entirely or fabrication flags present.
    """

    from geo_strategist.experiments.qualitative_discussion import (
        discussion_dimension_coverage,
        unsourced_currency_flags,
    )

    proposals = record.get("proposals") or []
    entries = [p.get("qualitative_discussion") for p in proposals]
    entries = [e for e in entries if isinstance(e, dict) and e.get("paragraphs")]
    if not proposals or not entries:
        return 1
    all_text = " ".join(
        str(v) for e in entries for v in (e.get("paragraphs") or {}).values())
    sourced = [a for e in entries for a in (e.get("sourced_amounts") or [])]
    if unsourced_currency_flags(all_text, sourced):
        return 1
    # Facility mentions must be consistent with source-traceable targets.
    for proposal in proposals:
        entry = proposal.get("qualitative_discussion") or {}
        text = " ".join(str(v) for v in (entry.get("paragraphs") or {}).values())
        name = proposal.get("target_facility_name")
        if name and name in text and not proposal.get("source_evidence_refs"):
            return 1
    coverage_ratios = []
    for entry in entries:
        coverage = discussion_dimension_coverage(entry)
        coverage_ratios.append(sum(coverage.values()) / max(len(coverage), 1))
    full_coverage = sum(1 for ratio in coverage_ratios if ratio >= 0.99)
    # Differentiation: distinct demand/population prose across candidates.
    distinct = len({
        str((e.get("paragraphs") or {}).get("population", ""))[:160] for e in entries
    })
    differentiated = distinct >= max(2, int(0.8 * len(entries)))
    if len(entries) == len(proposals) and full_coverage == len(entries) and differentiated:
        return 5
    if full_coverage >= max(1, int(0.7 * len(proposals))):
        return 4 if differentiated else 3
    if mean(coverage_ratios) >= 0.5:
        return 3 if differentiated else 2
    return 2


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ratio_score(part: int, whole: int) -> int:
    if whole == 0:
        return 1
    share = part / whole
    return 5 if share >= 0.8 else 4 if share >= 0.5 else 3 if share > 0 else 2


def _dimension_scores(
    record: dict[str, Any],
    c9_slate: list[str],
) -> dict[str, int]:
    proposals = record.get("proposals") or []
    proposal_count = len(proposals)
    with_name = sum(1 for p in proposals if p.get("target_facility_name") not in (None, "", "not_available"))
    with_refs = sum(1 for p in proposals if p.get("source_evidence_refs"))
    with_grades = sum(1 for p in proposals if p.get("evidence_grades"))
    with_components = [p for p in proposals if p.get("score_components")]
    with_financial = sum(1 for p in with_components
                         if (p.get("score_components") or {}).get("financial") is not None)
    with_risk = sum(1 for p in with_components
                    if (p.get("score_components") or {}).get("demographic_risk") is not None)
    gaps_disclosed = sum(1 for p in proposals if p.get("evidence_gaps") or p.get("required_due_diligence"))
    revised = sum(1 for p in proposals if p.get("proposal_status") == "revised_proposal")
    reviewed = sum(1 for p in proposals if p.get("reviewer_scores"))
    due_diligence = record.get("required_due_diligence") or []

    branch_results = record.get("branch_results") or []
    branch_winner_sets = [set(row.get("winner_top_candidates") or []) for row in branch_results
                          if row.get("winner_top_candidates")]
    distinct_winners = len(set().union(*branch_winner_sets)) if branch_winner_sets else 0

    code_stats = record.get("generated_code_stats") or {}
    code_total = int(code_stats.get("nodes") or code_stats.get("generated") or 0)
    code_ok = int(code_stats.get("nodes_succeeded") or code_stats.get("executed_ok") or 0)

    call_summary = record.get("model_call_summary") or {}
    total_requests = int(call_summary.get("total_requests") or 0)
    total_errors = int(call_summary.get("total_errors") or 0)

    slate_ids = [p.get("candidate_id") for p in proposals if p.get("candidate_id")]
    novelty = None
    if c9_slate and slate_ids and record.get("condition_group") != "C9":
        overlap = len(set(slate_ids) & set(c9_slate)) / max(len(set(slate_ids) | set(c9_slate)), 1)
        novelty = 1.0 - overlap

    live = record.get("execution_mode") in ("live", "live_manual_harness")

    scores = {
        "proposal_usefulness": 5 if proposal_count >= 3 and record.get("proposal_report_path") else 3 if proposal_count else 1,
        "site_specificity": _ratio_score(with_name, proposal_count),
        "evidence_traceability": min(_ratio_score(with_refs, proposal_count),
                                     _ratio_score(with_grades, proposal_count)) if proposal_count else 1,
        "demand_supply_reasoning": _ratio_score(len(with_components), proposal_count),
        "financial_reasoning": _ratio_score(with_financial, proposal_count),
        "risk_handling": _ratio_score(with_risk, proposal_count),
        "actionability": 5 if proposal_count and len({p.get("action_type") for p in proposals}) >= 2 and due_diligence else 3 if proposal_count else 1,
        "uncertainty_transparency": 5 if proposal_count and gaps_disclosed == proposal_count and due_diligence else _ratio_score(gaps_disclosed, proposal_count),
        "model_innovation": _INNOVATION_BY_ALGORITHM.get(str(record.get("algorithm") or record.get("workflow_type")), 3),
        "reproducibility": (
            5 if record.get("provider") in (None, "none") and record.get("source_artifacts")
            else 4 if record.get("source_artifacts") and record.get("model_call_summary")
            else 3 if record.get("source_artifacts") else 2),
        "agentic_search_quality": (
            5 if live and branch_winner_sets and int(record.get("steps_run") or 0) >= 10
            else 4 if live and branch_winner_sets
            else 3 if live else 2 if record.get("execution_mode") == "deterministic_baseline" else 1),
        "generated_code_quality": (
            5 if code_total and code_ok == code_total
            else 4 if code_total and code_ok >= max(code_total // 2, 1)
            else 3 if code_ok else 2 if code_total else 1),
        "review_revision_quality": 5 if revised == proposal_count and proposal_count else 4 if reviewed else 2,
        "branch_diversity": (
            5 if distinct_winners >= 12 else 4 if distinct_winners >= 8
            else 3 if distinct_winners >= 5 else 2 if branch_winner_sets else 1),
        "novelty_vs_c9": (
            3 if novelty is None
            else 5 if novelty >= 0.6 else 4 if novelty >= 0.3 else 2 if novelty > 0 else 1),
        "model_call_efficiency": (
            3 if total_requests == 0
            else 5 if total_errors == 0 and total_requests <= 60
            else 4 if total_errors <= max(total_requests // 10, 1)
            else 2),
        "harness_orchestration_clarity": (
            5 if record.get("model_call_summary") and (record.get("branch_results") or record.get("steps_run"))
            else 4 if record.get("model_call_summary")
            else 3 if record.get("execution_mode") == "deterministic_baseline" else 2),
        "qualitative_site_reasoning": _qualitative_site_reasoning_score(record),
    }
    return scores


def _compact_judge_view(record: dict[str, Any], clip_scale: float = 1.0) -> dict[str, Any]:
    """Per-method summary small enough that every method fits in one judge
    prompt (the full anonymized records run to hundreds of KB). ``clip_scale``
    lets the caller shrink text clips uniformly until the payload fits the
    prompt budget by construction — no mid-JSON byte slicing."""

    def _clip(value: Any, limit: int) -> str:
        return str(value or "")[:max(int(limit * clip_scale), 40)]

    return {
        "method": record.get("method_alias") or record.get("judge_alias"),
        "execution_mode": record.get("execution_mode"),
        "workflow_type": record.get("workflow_type"),
        "steps_run": record.get("steps_run"),
        "required_due_diligence": [
            _clip(item, 200) for item in (record.get("required_due_diligence") or [])[:8]],
        "proposals": [
            {
                "rank": p.get("rank"),
                "action_type": p.get("action_type"),
                "composite_score": p.get("composite_score"),
                "score_components": p.get("score_components"),
                "evidence_grades": p.get("evidence_grades"),
                "evidence_gaps": [_clip(g, 120) for g in (p.get("evidence_gaps") or [])[:6]],
                "required_due_diligence": [
                    _clip(item, 160) for item in (p.get("required_due_diligence") or [])[:4]],
                "llm_rationale": _clip(p.get("llm_rationale"), 400),
            }
            for p in (record.get("proposals") or [])[:5]
        ],
    }


def _qualitative_judge_view(record: dict[str, Any], clip_scale: float = 1.0) -> dict[str, Any]:
    """One method's qualitative site discussions, clipped per paragraph."""

    limit = max(int(600 * clip_scale), 60)
    return {
        "method": record.get("method_alias") or record.get("judge_alias"),
        "qualitative_site_discussions": [
            {
                name: str(text or "")[:limit]
                for name, text in ((p.get("qualitative_discussion") or {}).get("paragraphs") or {}).items()
            }
            for p in (record.get("proposals") or [])
            if p.get("qualitative_discussion")
        ][:5],
    }


def _fit_views(records: list[dict[str, Any]], build, budget: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Views for every record within ``budget`` serialized chars: shrink text
    clips first, and only as a last resort drop trailing methods (returned as
    the omitted-alias list so the run status can report partial coverage)."""

    views = [build(record, 1.0) for record in records]
    for scale in (1.0, 0.5, 0.25):
        views = [build(record, scale) for record in records]
        if len(json.dumps(views, ensure_ascii=False)) <= budget:
            return views, []
    omitted: list[str] = []
    while views and len(json.dumps(views, ensure_ascii=False)) > budget:
        omitted.append(str(views[-1].get("method")))
        views = views[:-1]
    return views, omitted


def _run_live_judge(
    anonymized: list[dict[str, Any]],
    alias_map: dict[str, str],
) -> tuple[dict[str, dict[str, float]], str]:
    """One live judge call scoring anonymized records; returns
    (scores per condition_group per dimension, status)."""

    if os.environ.get("E13_DISABLE_LIVE_JUDGE") == "1":
        return {}, "disabled"
    provider = os.environ.get("E13_JUDGE_PROVIDER", "gemini")
    compact_records, omitted_records = _fit_views(
        anonymized, _compact_judge_view, 120000)
    qualitative_sections, omitted_qualitative = _fit_views(
        anonymized, _qualitative_judge_view, 160000)
    omitted = sorted({*omitted_records, *omitted_qualitative})
    prompt = (
        "Score each anonymized method's hospital-strategy proposal record on "
        "these dimensions from 1 (poor) to 5 (excellent): "
        + ", ".join(LIVE_JUDGE_DIMENSIONS) + ".\n"
        "Reward grounded rankings, provenance, disclosed uncertainty, and "
        "actionable next steps; penalize concrete claims without a source or "
        "explicit unverified grade. Methods are identified only by alias.\n"
        "For qualitative_site_reasoning, evaluate each method's qualitative "
        "site discussions (provided below) against these criteria:\n"
        + "\n".join(f"- {criterion}" for criterion in _QUALITATIVE_CRITERIA) + "\n"
        "Score EVERY method listed.\n"
        'Reply as JSON: {"scores": [{"method": "Method A", "<dimension>": <1-5>, ...}]}\n\n'
        "CONDITION RECORDS:\n"
        + json.dumps(compact_records, ensure_ascii=False)
        + "\n\nQUALITATIVE SITE DISCUSSIONS PER METHOD:\n"
        + json.dumps(qualitative_sections, ensure_ascii=False)
    )
    result = None
    try:
        if provider == "gemini":
            from geo_strategist.providers.gemini_client import GeminiClient

            result = GeminiClient(model=os.environ.get("E13_JUDGE_MODEL")).generate(
                prompt, purpose="e13_judge")
        elif provider == "openrouter":
            from geo_strategist.providers.openrouter_client import OpenRouterClient

            result = OpenRouterClient().generate(prompt, purpose="e13_judge")
        else:
            from geo_strategist.providers.opencode_go_client import OpenCodeGoClient

            result = OpenCodeGoClient(model=os.environ.get("E13_JUDGE_MODEL")).generate(
                prompt, purpose="e13_judge")
    except Exception as exc:
        return {}, f"judge_error:{type(exc).__name__}"
    if result is None or not result.ok:
        return {}, f"judge_{result.error_class if result else 'error'}"

    from geo_strategist.experiments.live_common import extract_json_block

    parsed = extract_json_block(result.text) or {}
    group_by_alias = {alias: group for group, alias in alias_map.items()}
    scores: dict[str, dict[str, float]] = {}
    for row in parsed.get("scores") or []:
        if not isinstance(row, dict):
            continue
        group = group_by_alias.get(str(row.get("method")))
        if not group:
            continue
        values = {}
        for dimension in LIVE_JUDGE_DIMENSIONS:
            value = row.get(dimension)
            if isinstance(value, (int, float)) and 1 <= float(value) <= 5:
                values[dimension] = float(value)
        if values:
            scores[group] = values
    if not scores:
        return scores, "judge_unparseable"
    if omitted:
        return scores, "ok_partial:omitted_methods=" + ",".join(omitted)
    return scores, "ok"


def _qualitative_reasoning_section(
    comparable_rows: list[dict[str, Any]],
    waiting_rows: list[dict[str, Any]],
    rows_by_group: dict[str, dict[str, Any]],
) -> list[str]:
    """The Qualitative Site Reasoning table + comparative prose."""

    if not comparable_rows:
        return []
    dimension = "qualitative_site_reasoning"

    def score(group: str) -> float | None:
        row = rows_by_group.get(group)
        return row["score_dimensions"][dimension] if row and row.get("comparable") else None

    rows = []
    for row in sorted(comparable_rows, key=lambda r: -r["score_dimensions"][dimension]):
        value = row["score_dimensions"][dimension]
        note = (
            "candidate-differentiated, fully covered discussion" if value >= 5
            else "mostly covered; some candidates/dimensions weaker" if value >= 4
            else "present but generic across candidates" if value >= 3
            else "short or missing dimensions" if value >= 2
            else "missing or contains unverifiable claims")
        rows.append([row["condition_group"], row["label"], value, note])
    lines = [
        "## Qualitative Site Reasoning",
        "",
        "Scores the candidate-level qualitative discussion in each report "
        "(regional characteristics, population composition, elderly demand, "
        "medical supply, real-estate/cost caveats, recommended action, and "
        "reviewer comments), rubric 1-5.",
        "",
        markdown_table(["condition", "label", "score", "assessment"], rows),
        "",
    ]
    prose: list[str] = []
    strong = [row for row in comparable_rows
              if row["condition_group"] in ("C11", "C12")]
    if strong:
        prose.append(
            "High-ranking orchestration conditions (C11/C12) carry the richest "
            "review inputs — evolutionary critique and node-level "
            "criticize/refine feedback respectively — which flows directly into "
            "their per-candidate review-comment paragraphs and keeps their "
            "discussions candidate-specific.")
    vanilla_scores = [score(g) for g in ("C1", "C2", "C3", "C4") if score(g) is not None]
    agentic_scores = [
        score(g) for g in ("C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13")
        if score(g) is not None]
    if vanilla_scores and agentic_scores:
        comparison = ("below" if mean(vanilla_scores) < mean(agentic_scores)
                      else "on par with" if mean(vanilla_scores) == mean(agentic_scores) else "above")
        prose.append(
            f"The vanilla single-pass baselines (C1-C4) average {mean(vanilla_scores):.1f} — "
            f"{comparison} the agentic conditions' average of {mean(agentic_scores):.1f}: "
            "without review rounds or branch rationale, their discussions rest on "
            "the single-pass model rationale alone.")
    c4_score, c8_score, c10_score = score("C4"), score("C8"), score("C10")
    if c8_score is not None and c10_score is not None:
        verdict = ("identical" if c8_score == c10_score else
                   "stronger in C8" if c8_score > c10_score else "stronger in C10")
        prose.append(
            f"Under the same provider/model, narrative quality is {verdict} "
            f"(C8 {c8_score} vs C10 {c10_score}): the interactive Skills harness "
            "and the free-form AI-Scientist draft tree feed different amounts of "
            "branch rationale and reviewer output into the discussions.")
    if c4_score is not None and c8_score is not None:
        prose.append(
            f"C4 (vanilla direct DeepSeek) scores {c4_score} vs C8 {c8_score} on "
            "the same provider/model: the delta isolates what the Skills-unified "
            "contract adds to the discussion quality over a single pass.")
    if waiting_rows:
        prose.append(
            "C2/C3/C5/C6/C7/C8 will enter this comparison after their "
            "manual-harness executions are ingested; their handoff prompts "
            "already require the same qualitative discussion per candidate.")
    for paragraph in prose:
        lines.extend([paragraph, ""])
    return lines


def _pairwise_section(rows_by_group: dict[str, dict[str, Any]],
                      group_a: str, group_b: str, heading: str,
                      note: str) -> list[str]:
    lines = [f"## {heading}", ""]
    row_a, row_b = rows_by_group.get(group_a), rows_by_group.get(group_b)
    if not row_a or not row_b:
        missing = [g for g in (group_a, group_b) if g not in rows_by_group]
        lines.extend([f"Not available: {', '.join(missing)} did not produce a scored record.", ""])
        return lines
    if not (row_a.get("comparable") and row_b.get("comparable")):
        lines.extend([
            f"Comparison is informational only: "
            f"{group_a} execution `{row_a['execution_mode']}`, "
            f"{group_b} execution `{row_b['execution_mode']}`.", "",
        ])
    lines.extend([
        note, "",
        markdown_table(
            ["dimension", group_a, group_b, "delta"],
            [[d.replace("_", " "),
              row_a["score_dimensions"][d], row_b["score_dimensions"][d],
              row_a["score_dimensions"][d] - row_b["score_dimensions"][d]]
             for d in SCORE_DIMENSIONS],
        ),
        "",
        f"Mean: {group_a} {row_a['mean_score']} vs {group_b} {row_b['mean_score']}.",
        "",
    ])
    return lines


def _strict_pair_note(status: dict[str, Any], group_a: str, group_b: str,
                      treatment: str) -> str:
    key_a, key_b = group_a.lower(), group_b.lower()
    if not status["confounded"]:
        return (
            "Both conditions use the same provider and model "
            f"(`{status[key_a]['provider']}` / `{status[key_a]['model']}`); "
            f"differences isolate {treatment}."
        )
    return (
        f"**CONFOUNDED**: {group_a} and {group_b} do not share provider and model "
        f"({group_a} `{status[key_a]['provider']}/{status[key_a]['model']}`, "
        f"{group_b} `{status[key_b]['provider']}/{status[key_b]['model']}`); "
        "the isolated effect cannot be separated from model effects."
    )


_CANDIDATE_DELIBERATION_METRICS: tuple[tuple[str, str], ...] = (
    ("candidate_level_assessment_coverage", "assessment coverage"),
    ("reviewer_finding_coverage", "reviewer finding coverage"),
    ("blocking_findings", "blocking findings"),
    ("major_findings", "major findings"),
    ("major_blocking_issue_handling", "major/blocking issue handling"),
    ("author_response_quality", "author response quality"),
    ("evidence_ref_grounding_rate", "evidence-ref grounding rate"),
    ("unsupported_finding_filter_rate", "unsupported finding filter rate"),
    ("review_revision_quality", "review revision quality"),
)


def _candidate_deliberation_section(
    records: list[dict[str, Any]],
    advanced_rows: list[dict[str, Any]],
) -> list[str]:
    """Post-hoc candidate-level deliberation quality for C11-C13.

    Deliberately reported as its own informational block, never folded into
    ``score_dimensions``/``mean_score``: this measures the quality of the
    post-hoc qualitative-assessment/review layer, not the core
    evolutionary-search / AB-MCTS / dynamic-routing slate-generation
    algorithm each condition is primarily being compared on.
    """

    records_by_group = {str(r.get("condition_group")): r for r in records}
    rows = []
    for row in advanced_rows:
        group = row["condition_group"]
        summary = records_by_group.get(group, {}).get("candidate_deliberation_summary") or {}
        if summary:
            rows.append((group, summary))
    if not rows:
        return []
    lines = [
        "## C11-C13 candidate-level qualitative deliberation (post-hoc, informational)",
        "",
        "These metrics describe the agent-assessment -> reviewer-critique -> "
        "author-response -> provenance-judge layer applied after each "
        "condition's own slate was finalized. They are **not** part of "
        "`score_dimensions`/`mean_score` above, so they cannot obscure the "
        "core algorithmic slate-generation comparison.",
        "",
        markdown_table(
            ["condition", *[label for _key, label in _CANDIDATE_DELIBERATION_METRICS]],
            [[group, *[summary.get(key, "n/a") for key, _label in _CANDIDATE_DELIBERATION_METRICS]]
             for group, summary in rows],
        ),
        "",
    ]
    return lines


def run_condition_comparison_judge(
    repo_root: str | Path = ".",
    *,
    proposals_dir: str | Path | None = None,
    allow_live_judge: bool = True,
) -> ConditionJudgeResult:
    repo_root = Path(repo_root).resolve()
    proposals_path = Path(proposals_dir) if proposals_dir else DEFAULT_PROPOSALS_DIR
    if not proposals_path.is_absolute():
        proposals_path = repo_root / proposals_path

    run_id = str(uuid.uuid4())
    out_dir = proposals_path
    generated_at = _now_iso()

    records_path = proposals_path / "condition_records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(
            f"no condition records at {records_path}; run "
            "run-condition-proposals first or pass --proposals-dir")
    records = _read_jsonl(records_path)
    registry = build_condition_registry()
    c1_c9_status = c1_c9_strict_comparison_status(registry)
    c2_c6_status = c2_c6_strict_comparison_status(registry)
    c3_c7_status = c3_c7_strict_comparison_status(registry)
    c4_c8_c10_status = c4_c8_c10_strict_comparison_status(registry)

    c9_record = next((r for r in records if r.get("condition_group") == "C9"), None)
    c9_slate = [p.get("candidate_id") for p in (c9_record or {}).get("proposals") or []]

    scored_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: str(r.get("condition_group"))):
        group = str(record.get("condition_group"))
        dimensions = _dimension_scores(record, c9_slate)
        comparable = record.get("comparable_for_e13") is True and record.get("eligible_for_judge") is True
        scored_rows.append({
            "score_id": _stable_id("condition_judge_score", {"condition_group": group}),
            "condition_group": group,
            "label": record.get("label"),
            "workflow_type": record.get("workflow_type"),
            "provider": record.get("provider") or record.get("provider_family"),
            "model": record.get("model"),
            "harness": record.get("harness"),
            "execution_mode": record.get("execution_mode", "unknown"),
            "comparable": comparable,
            "exclusion_reason": record.get("exclusion_reason"),
            "proposal_report_path": record.get("proposal_report_path"),
            "score_dimensions": dimensions,
            "mean_score": round(mean(dimensions.values()), 4),
            "judge_type": "deterministic_structural_judge",
            "judgment_scope": JUDGMENT_SCOPE,
            "generated_at": generated_at,
        })

    # ---- live judge over anonymized comparable records ----------------------
    comparable_rows = [row for row in scored_rows if row["comparable"]]
    alias_map = {row["condition_group"]: f"Method {chr(ord('A') + index)}"
                 for index, row in enumerate(comparable_rows)}
    anonymized = [
        anonymize_condition_record_for_judge(
            record, alias_map[str(record.get("condition_group"))])
        for record in records
        if str(record.get("condition_group")) in alias_map
    ]
    live_scores: dict[str, dict[str, float]] = {}
    live_status = "skipped_no_comparable_records"
    if allow_live_judge and anonymized:
        live_scores, live_status = _run_live_judge(anonymized, alias_map)
    for row in scored_rows:
        judged = live_scores.get(row["condition_group"])
        if judged:
            merged = dict(row["score_dimensions"])
            for dimension, value in judged.items():
                merged[dimension] = round((merged[dimension] + value) / 2, 2)
            row["score_dimensions"] = merged
            row["mean_score"] = round(mean(merged.values()), 4)
            row["judge_type"] = "structural_plus_live_llm_judge"
            row["live_judge_dimensions"] = judged

    rows_by_group = {row["condition_group"]: row for row in scored_rows}
    comparable_rows = [row for row in scored_rows if row["comparable"]]
    waiting_rows = [row for row in scored_rows
                    if row["execution_mode"] == "waiting_for_manual_harness"]
    fallback_rows = [row for row in scored_rows
                     if not row["comparable"]
                     and row["execution_mode"] != "waiting_for_manual_harness"]

    # ---- figures --------------------------------------------------------------
    figures_dir = out_dir / "figures"
    figure_refs: list[str] = []
    if comparable_rows:
        figure = condition_comparison_figure(
            [(row["condition_group"], row["mean_score"]) for row in comparable_rows],
            figures_dir / "judge_condition_comparison.png",
            title="Mean proposal-quality score by condition (comparable results only)",
            xlabel="mean score (1-5 across 18 dimensions)")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")
        radar_series = {
            row["condition_group"]: [row["score_dimensions"][d] for d in SCORE_DIMENSIONS]
            for row in comparable_rows[:7]
        }
        radar = radar_figure(list(SCORE_DIMENSIONS), radar_series,
                             figures_dir / "judge_dimension_radar.png",
                             title="Condition-judge dimension radar (comparable conditions)")
        if radar:
            figure_refs.append(f"../figures/{radar.name}")
    for pair, filename, title in (
        (("C1", "C9"), "judge_c1_c9_orchestration.png", "C1 vs C9 agentic-orchestration comparison (same provider/model, Gemini)"),
        (("C2", "C6"), "judge_c2_c6_skills.png", "C2 vs C6 Skills-unification comparison (same provider/model, Codex)"),
        (("C3", "C7"), "judge_c3_c7_skills.png", "C3 vs C7 Skills-unification comparison (same provider/model, Claude Code)"),
        (("C4", "C8"), "judge_c4_c8_skills.png", "C4 vs C8 Skills-unification comparison (same provider/model, DeepSeek)"),
        (("C4", "C10"), "judge_c4_c10_orchestration.png", "C4 vs C10 agentic-orchestration comparison (same provider/model, DeepSeek)"),
        (("C8", "C10"), "judge_c8_c10_harness.png", "C8 vs C10 Skills-harness vs direct-API comparison (same provider/model, DeepSeek)"),
        (("C9", "C10"), "judge_c9_c10_models.png", "C9 vs C10 model/provider comparison (same algorithm family)"),
    ):
        row_a, row_b = rows_by_group.get(pair[0]), rows_by_group.get(pair[1])
        if row_a and row_b:
            figure = grouped_bar_figure(
                [d.replace("_", " ") for d in SCORE_DIMENSIONS],
                {pair[0]: [row_a["score_dimensions"][d] for d in SCORE_DIMENSIONS],
                 pair[1]: [row_b["score_dimensions"][d] for d in SCORE_DIMENSIONS]},
                figures_dir / filename, title=title, ylabel="score (1-5)")
            if figure:
                figure_refs.append(f"../figures/{figure.name}")
    vanilla_rows = [rows_by_group[g] for g in ("C1", "C2", "C3", "C4") if g in rows_by_group]
    if len(vanilla_rows) >= 2:
        figure = grouped_bar_figure(
            [d.replace("_", " ") for d in SCORE_DIMENSIONS],
            {row["condition_group"]: [row["score_dimensions"][d] for d in SCORE_DIMENSIONS]
             for row in vanilla_rows},
            figures_dir / "judge_c1_c4_vanilla.png",
            title="C1-C4 vanilla-baseline comparison (provider/harness differ together)",
            ylabel="score (1-5)")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")
    skills_rows = [rows_by_group[g] for g in ("C5", "C6", "C7", "C8") if g in rows_by_group]
    if len(skills_rows) >= 2:
        figure = grouped_bar_figure(
            [d.replace("_", " ") for d in SCORE_DIMENSIONS],
            {row["condition_group"]: [row["score_dimensions"][d] for d in SCORE_DIMENSIONS]
             for row in skills_rows},
            figures_dir / "judge_c5_c8_skills_harness.png",
            title="C5-C8 Skills-in-harness comparison (harness+model differ together)",
            ylabel="score (1-5)")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")
    novelty_scores = [(row["condition_group"], float(row["score_dimensions"]["novelty_vs_c9"]))
                      for row in comparable_rows if row["condition_group"] != "C9"]
    if novelty_scores:
        figure = condition_comparison_figure(
            novelty_scores, figures_dir / "judge_novelty_vs_c9.png",
            title="Novelty of each condition's slate vs the C9 baseline",
            xlabel="novelty score (1-5)")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")

    # ---- report ------------------------------------------------------------------
    lines = [
        "# Condition Comparison Report (cross-condition judge)",
        "",
        f"Generated: {generated_at} — live judge status: `{live_status}`",
        "",
        "## Result groups",
        "",
        f"- Comparable live-agent results: "
        f"{', '.join(row['condition_group'] for row in comparable_rows) or 'none'}",
        "- Non-comparable fallback/failed results: "
        + (", ".join(
            "{} ({})".format(row["condition_group"], row["execution_mode"])
            for row in fallback_rows) or "none"),
        f"- Waiting for manual harness: "
        f"{', '.join(row['condition_group'] for row in waiting_rows) or 'none'}",
        "",
        "Waiting/manual conditions are excluded from the comparable "
        "live-agent ranking until their manual results are ingested, and a "
        "deterministic fallback ranking is never treated as a comparable "
        "live result.",
        "",
        "## Score matrix (comparable results)",
        "",
        markdown_table(
            ["condition", *[d.replace("_", " ") for d in SCORE_DIMENSIONS], "mean"],
            [[row["condition_group"],
              *[row["score_dimensions"][d] for d in SCORE_DIMENSIONS],
              row["mean_score"]] for row in comparable_rows],
        ) if comparable_rows else "No comparable results to score.",
        "",
    ]
    for ref in figure_refs:
        lines.extend([f"![]({ref})", ""])
    if fallback_rows or waiting_rows:
        lines.extend([
            "## Excluded results",
            "",
            markdown_table(
                ["condition", "execution mode", "exclusion reason"],
                [[row["condition_group"], row["execution_mode"],
                  row.get("exclusion_reason") or "not_comparable"]
                 for row in [*fallback_rows, *waiting_rows]],
            ),
            "",
        ])

    lines.extend(_pairwise_section(rows_by_group, "C1", "C9",
                                   "C1 vs C9 — strict agentic-orchestration comparison (Gemini)",
                                   _strict_pair_note(c1_c9_status, "C1", "C9",
                                                      "the AI Scientist-style agentic loop over a single vanilla pass")))
    lines.extend(_pairwise_section(rows_by_group, "C2", "C6",
                                   "C2 vs C6 — strict Skills-unification comparison (Codex)",
                                   _strict_pair_note(c2_c6_status, "C2", "C6",
                                                      "the Skills contract and branch search")))
    lines.extend(_pairwise_section(rows_by_group, "C3", "C7",
                                   "C3 vs C7 — strict Skills-unification comparison (Claude Code)",
                                   _strict_pair_note(c3_c7_status, "C3", "C7",
                                                      "the Skills contract and branch search")))
    lines.extend(_pairwise_section(rows_by_group, "C4", "C8",
                                   "C4 vs C8 — strict Skills-unification comparison (DeepSeek)",
                                   "Part of the C4/C8/C10 strict trio (see below); isolates the "
                                   "Skills contract and branch search over a single vanilla pass."))
    lines.extend(_pairwise_section(rows_by_group, "C4", "C10",
                                   "C4 vs C10 — strict agentic-orchestration comparison (DeepSeek)",
                                   "Part of the C4/C8/C10 strict trio; isolates the full "
                                   "AI Scientist-style agentic loop over a single vanilla pass."))
    lines.extend(_pairwise_section(rows_by_group, "C8", "C10",
                                   "C8 vs C10 — strict Skills-harness vs direct-API comparison (DeepSeek)",
                                   "Part of the C4/C8/C10 strict trio; isolates the interactive "
                                   "Skills harness against the direct-API AI Scientist-style draft tree."))
    lines.extend(_pairwise_section(
        rows_by_group, "C9", "C10",
        "C9 vs C10 — model/provider comparison",
        "Same AI Scientist-style algorithm family and aligned budget envelope; "
        "the comparison isolates provider and model effects."))

    trio_confounded = c4_c8_c10_status["confounded"]
    trio_note = (
        "C4, C8, and C10 share the same provider and model "
        f"(`{c4_c8_c10_status['c4']['provider']}` / `{c4_c8_c10_status['c4']['model']}`); "
        "the three pairwise comparisons above isolate harness and "
        "orchestration effects cleanly."
        if not trio_confounded else
        "**CONFOUNDED**: C4, C8, and C10 do not all share provider and model "
        f"(C4 `{c4_c8_c10_status['c4']['provider']}/{c4_c8_c10_status['c4']['model']}`, "
        f"C8 `{c4_c8_c10_status['c8']['provider']}/{c4_c8_c10_status['c8']['model']}`, "
        f"C10 `{c4_c8_c10_status['c10']['provider']}/{c4_c8_c10_status['c10']['model']}`); "
        "harness/orchestration effects cannot be separated from model effects "
        "in the three pairwise comparisons above."
    )
    lines.extend(["## C4/C8/C10 strict trio status", "", trio_note, ""])

    vanilla_rows_table = [rows_by_group[g] for g in ("C1", "C2", "C3", "C4")
                          if g in rows_by_group]
    if vanilla_rows_table:
        lines.extend([
            "## C1-C4 vanilla-baseline comparison",
            "",
            "Four single-pass, no-Skills, no-branch-search baselines across "
            "four provider/harness stacks (Gemini direct, Codex CLI, Claude "
            "Code CLI, DeepSeek direct). Providers and models differ "
            "together, so this compares provider+harness stacks, not "
            "harnesses in isolation.",
            "",
            markdown_table(
                ["condition", "harness", "provider/model", "execution mode", "mean"],
                [[row["condition_group"], row["harness"],
                  f"{row['provider']}/{row['model']}",
                  row["execution_mode"], row["mean_score"]]
                 for row in vanilla_rows_table],
            ),
            "",
        ])

    skills_harness_rows = [rows_by_group[g] for g in ("C5", "C6", "C7", "C8")
                           if g in rows_by_group]
    if skills_harness_rows:
        lines.extend([
            "## C5-C8 Skills-in-harness comparison",
            "",
            "One Skills-unified, branch-search manual run per CLI harness "
            "(Antigravity / Codex / Claude Code / OpenCode). Providers and "
            "models differ together with the harness, so this compares "
            "harness+model stacks, not harnesses in isolation.",
            "",
            markdown_table(
                ["condition", "harness", "provider/model", "execution mode", "mean"],
                [[row["condition_group"], row["harness"],
                  f"{row['provider']}/{row['model']}",
                  row["execution_mode"], row["mean_score"]]
                 for row in skills_harness_rows],
            ),
            "",
        ])

    lines.extend(_qualitative_reasoning_section(comparable_rows, waiting_rows, rows_by_group))

    advanced = [row for row in comparable_rows if row["condition_group"] in ("C11", "C12", "C13")]
    if advanced:
        lines.extend([
            "## C11-C13 orchestration comparison",
            "",
            markdown_table(
                ["condition", "orchestration", "mean", "branch diversity",
                 "call efficiency", "code quality"],
                [[row["condition_group"], row["workflow_type"], row["mean_score"],
                  row["score_dimensions"]["branch_diversity"],
                  row["score_dimensions"]["model_call_efficiency"],
                  row["score_dimensions"]["generated_code_quality"]]
                 for row in advanced],
            ),
            "",
        ])
        lines.extend(_candidate_deliberation_section(records, advanced))

    ordered = sorted(comparable_rows, key=lambda row: -row["mean_score"])
    weakest_dimension = None
    if ordered:
        dimension_means = {
            d: mean(row["score_dimensions"][d] for row in comparable_rows)
            for d in SCORE_DIMENSIONS
        }
        weakest_dimension = min(dimension_means, key=dimension_means.get)
    recommended_next = (
        waiting_rows[0]["condition_group"] if waiting_rows
        else (ordered[-1]["condition_group"] if len(ordered) > 1 else None))
    lines.extend([
        "## Recommendations",
        "",
        f"- Strongest comparable conditions: "
        f"{', '.join(row['condition_group'] for row in ordered[:2]) or 'none scored'}.",
        f"- Recommended next condition to improve: "
        f"{recommended_next or 'n/a'}"
        + (f" (weakest shared dimension across the track: {weakest_dimension.replace('_', ' ')})"
           if weakest_dimension else "") + ".",
        *( [f"- Unblock manual-harness conditions "
            f"({', '.join(row['condition_group'] for row in waiting_rows)}) via the "
            "prompt files under `manual_harness/` in the live output dir."] if waiting_rows else [] ),
        "",
        "## Proposal files a human should inspect next",
        "",
        *[f"- `{row['condition_group']}`: {row['proposal_report_path'] or 'no report'}"
          for row in ordered[:4]],
        *( [f"- `{row['condition_group']}` (excluded, for debugging): "
            f"{row['proposal_report_path']}" for row in fallback_rows[:2]] ),
        "",
        required_due_diligence_section([
            "E13 scores compare proposal artifacts and agentic process quality, not real-world outcomes.",
            f"Live judge status: {live_status}; when unavailable, scores are structural-only.",
            "Adoption decisions require the due diligence listed in each proposal report.",
        ]),
    ])

    report_path = out_dir / "reports" / "condition_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    output_paths = {
        "comparison_report": str(report_path),
        "scores": str(out_dir / "condition_judge_scores.jsonl"),
        "anonymized_judge_inputs": str(out_dir / "condition_judge_anonymized_inputs.jsonl"),
        "manifest": str(out_dir / "condition_judge_manifest.json"),
    }
    _write_jsonl(Path(output_paths["scores"]), scored_rows)
    _write_jsonl(Path(output_paths["anonymized_judge_inputs"]), anonymized)
    _write_json(Path(output_paths["manifest"]), {
        "run_id": run_id,
        "generated_at": generated_at,
        "judgment_scope": JUDGMENT_SCOPE,
        "scored_condition_count": len(scored_rows),
        "comparable_condition_count": len(comparable_rows),
        "alias_map": alias_map,
        "live_judge_status": live_status,
        "c1_c9_strict_comparison": c1_c9_status,
        "c2_c6_strict_comparison": c2_c6_status,
        "c3_c7_strict_comparison": c3_c7_status,
        "c4_c8_c10_strict_comparison": c4_c8_c10_status,
        "score_dimensions": list(SCORE_DIMENSIONS),
        "result_groups": {
            "comparable_live_agent_results": [row["condition_group"] for row in comparable_rows],
            "non_comparable_fallback_results": [row["condition_group"] for row in fallback_rows],
            "waiting_for_manual_harness_results": [row["condition_group"] for row in waiting_rows],
        },
        "output_artifacts": {key: _rel(Path(path), repo_root) for key, path in output_paths.items()},
    })

    return ConditionJudgeResult(
        run_id=run_id,
        output_dir=out_dir,
        scored_condition_count=len(scored_rows),
        eligible_condition_count=len(comparable_rows),
        live_judge_enabled=allow_live_judge,
        live_judge_status=live_status,
        comparison_report_path=str(report_path),
        output_paths=output_paths,
    )
