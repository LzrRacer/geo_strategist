"""Condition comparison judge: cross-condition qualitative + deterministic
document evaluation for C0-C14 (historically labeled "E13").

**Primary score — report-visible decision-analysis quality
(``report_visible_decision_analysis_quality_v3``).** Every comparable condition's
*final Markdown report only* (never the structured condition record,
provider/model/harness metadata, branch/candidate-review JSON, or internal
logs) is sanitized (``report_judge_view.sanitize_report_for_judge``) and
scored against a fixed 25-question, 5-category yes/no checklist
(``checklist_judge_panel.py``) by every available judge. A "yes" only counts
if the judge supplies an exact, verifiable evidence excerpt from the
supplied report text. Each of the five categories (A. Decision-space
exploration, B. Executed validation and evidence use, C. Cross-alternative
synthesis, D. Robustness and decision refinement, E. Business decision
utility — see ``checklist_judge_panel._CATEGORY_LABELS``) scores 0-5; the
five sum to ``llm_points_total`` (0-25), the metric conditions are
**ranked on**. The rubric never scores an algorithm/workflow/provider name
— differences must come from what a judge can read and quote in the report.

**Two evaluation dimensions over the same 25 questions** (never a second
scoring pass, purely a neutral decomposition of the same 25-question
total): ``process_score`` (0-17, ``checklist_judge_panel.
QUESTION_DIMENSIONS``) reflects exploration, executed validation, synthesis
traceability, robustness, and critique content the report visibly
contains. ``decision_support_score`` (0-8) is the practical-usability
half: actionable, candidate-specific, portfolio-aware decision content.
The two always sum to ``llm_points_total``. Every condition, including C0,
can earn any point either dimension awards for content its own report
genuinely contains — neither dimension is gated by condition identity.

**C0 is the deterministic-calculation baseline.** It is scored on the same
25-question rubric and competes normally in every ranking, recommendation,
and summary table: no score cap, no condition-specific penalty, no
exclusion. C0's report supplies factual/numerical evidence from
deterministic scoring; the other C1-C14 conditions are read for the
practical decision-support value their reports visibly add *beyond* that
baseline (interpretation, strategic comparison, risk assessment,
uncertainty management, candidate-specific next steps, contingencies).
Whether C0 or an agentic condition scores higher is an empirical outcome
of what each report actually contains, never an assumption baked into the
rubric or the aggregation code. C0 is naturally absent from the per-model
total-score tables only because it has no provider/model to attribute a
score to (see ``MODEL_STACKS``) — that is a fact about the table's axis,
not a ranking exclusion.

**Per-model total scores** (``comparison_families.MODEL_STACKS``): within
each vanilla/native-agent/Skills/AI-Scientist family the framework is
identical (see the family-contract validation work), so summing/averaging
a model's ``llm_points_total`` across its conditions measures only that
model's output quality, never a framework difference.

**Auxiliary — deterministic document checklist.** A 25-item PASS/FAIL
structural checklist (``deterministic_document_checklist.py``) runs
unconditionally for every comparable condition, needs no LLM call, and is
reported in its own section/JSONL/CSV. It is never added into
``llm_points_total`` and never used as the primary ranking — see
``deterministic_passed`` on each scored row, kept strictly separate.

If a condition's ``proposal_report_path`` is missing or unreadable, this
module never falls back to the structured condition record — that
condition's LLM checklist evaluation is marked unavailable and it receives
no invented score.

Artifacts (first-reader-friendly names): ``reports/condition_comparison_report.md``,
``condition_judge_scores.jsonl``, ``condition_judge_manifest.json``,
``condition_judge_anonymized_inputs.jsonl`` (alias + sanitized report text
only — no condition/provider/model identity), ``condition_deterministic_checklist.jsonl``,
``condition_llm_checklist.jsonl``, and the CSV exports under ``reports/``:
``condition_agentic_reasoning_scores.csv`` (the primary score export),
``condition_llm_checklist_matrix.csv``, ``condition_deterministic_checklist_matrix.csv``,
``condition_checklist_totals.csv``. The judge explicitly separates
``comparable_live_agent_results`` from ``non_comparable_fallback_results``
and ``waiting_for_manual_harness_results`` and includes four strict
same-provider/model comparisons — C1-vs-C13 (Gemini), C2-vs-C10 (Codex),
C3-vs-C11 (Claude Code), and the C4/C12/C14 trio (OpenCode Go/DeepSeek) — plus
the C13-vs-C14 model/provider comparison, the C1-C4 vanilla-baseline table,
and the C9-C12 Skills-in-harness table.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from geo_strategist.experiments.checklist_judge_panel import (
    CATEGORY_SCORE_FIELDS,
    DIMENSION_MAX,
    LLM_CHECKLIST_QUESTIONS,
    QUESTION_DIMENSIONS,
    _CATEGORY_LABELS,
    aggregate_checklist,
    checklist_panel_summary,
    default_judge_backends,
    run_checklist_panel,
)
from geo_strategist.experiments.comparison_families import (
    COMPARISON_FAMILIES,
    MODEL_STACKS,
)
from geo_strategist.experiments.condition_checklist_csv import (
    write_agentic_reasoning_scores_csv,
    write_checklist_totals_csv,
    write_deterministic_checklist_matrix_csv,
    write_llm_checklist_matrix_csv,
)
from geo_strategist.experiments.condition_output_contract import JUDGMENT_SCOPE
from geo_strategist.experiments.deterministic_document_checklist import (
    DETERMINISTIC_CHECKLIST_ITEMS,
    checklist_summary,
    run_deterministic_checklist,
)
from geo_strategist.experiments.condition_registry import (
    build_condition_registry,
    c1_c13_strict_comparison_status,
    c2_c10_strict_comparison_status,
    c3_c11_strict_comparison_status,
    c4_c12_c14_strict_comparison_status,
    c5_c9_strict_comparison_status,
    c6_c10_strict_comparison_status,
    c7_c11_strict_comparison_status,
    c8_c12_strict_comparison_status,
    c9_c13_discovery_comparison_status,
    c12_c14_discovery_comparison_status,
)
from geo_strategist.experiments.condition_utils import (
    _read_jsonl,
    _rel,
    _stable_id,
    _write_json,
    _write_jsonl,
)
from geo_strategist.experiments.report_judge_view import sanitize_report_for_judge
from geo_strategist.reporting import markdown_table, required_due_diligence_section
from geo_strategist.reporting.figures import condition_comparison_figure, grouped_bar_figure

DEFAULT_PROPOSALS_DIR = Path("outputs/condition_proposals/live")

QUALITATIVE_RUBRIC_VERSION = "report_visible_decision_analysis_quality_v3"
LLM_CHECKLIST_MAX = len(LLM_CHECKLIST_QUESTIONS)
DETERMINISTIC_CHECKLIST_MAX = len(DETERMINISTIC_CHECKLIST_ITEMS)
_CATEGORY_LABEL_ORDER: tuple[str, ...] = tuple(_CATEGORY_LABELS.values())
_CATEGORY_SCORE_FIELD_ORDER: tuple[str, ...] = tuple(CATEGORY_SCORE_FIELDS.values())


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _delta(value_a: Any, value_b: Any) -> Any:
    if not isinstance(value_a, (int, float)) or not isinstance(value_b, (int, float)):
        return "n/a"
    return round(value_a - value_b, 4)


def _category_scores(questions: list[dict[str, Any]]) -> dict[str, float]:
    """Each category's score (0-5): the sum of its 5 questions' points."""

    totals = {field: 0.0 for field in _CATEGORY_SCORE_FIELD_ORDER}
    for question in questions:
        field_name = CATEGORY_SCORE_FIELDS.get(question.get("category"))
        point = question.get("point")
        if field_name and point is not None:
            totals[field_name] += point
    return {field: round(value, 4) for field, value in totals.items()}


def _dimension_scores(questions: list[dict[str, Any]]) -> dict[str, float]:
    """process_score (0-17) + decision_support_score (0-8), always summing
    to llm_points_total -- a decomposition of the same 25-question score,
    never a second scoring pass. Recomputable offline from persisted
    per-question points (see QUESTION_DIMENSIONS)."""

    totals = {"process_score": 0.0, "decision_support_score": 0.0}
    for question in questions:
        dimension = QUESTION_DIMENSIONS.get(question.get("question_id"))
        point = question.get("point")
        if dimension and point is not None:
            totals[f"{dimension}_score"] += point
    return {field: round(value, 4) for field, value in totals.items()}


def _read_report_text(repo_root: Path, report_path: str | None) -> str | None:
    if not report_path:
        return None
    path = Path(report_path)
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _unavailable_llm_checklist_row(group: str, judge_names: list[str], generated_at: str) -> dict[str, Any]:
    """A condition whose final report is missing/unreadable: no panel call
    is made (never falls back to the structured record), and every
    configured judge is recorded unavailable so the row's absence is
    explicit rather than silently scored as zero."""

    judge_status = {name: "unavailable" for name in judge_names}
    return {
        "condition_group": group,
        "judge_status": judge_status,
        "questions": [],
        "summary": {
            "total_questions": LLM_CHECKLIST_MAX,
            "scored_questions": 0,
            "overall_mean_yes_ratio": None,
            "mean_yes_ratio_by_category": {},
            "judge_availability": judge_status,
            "category_scores": {field: None for field in _CATEGORY_SCORE_FIELD_ORDER},
            "llm_points_total": None,
            "llm_max": LLM_CHECKLIST_MAX,
        },
        "report_status": "unavailable",
        "generated_at": generated_at,
    }


def _deterministic_validation_section(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = ["## Deterministic Validation (auxiliary structural checklist)", "",
             "25-item PASS/FAIL document checklist per comparable condition "
             "(structure, consistency, numerical information, sources/"
             "assumptions, document quality and safety). Needs no LLM call. "
             "**Auxiliary only** — never added into the primary qualitative "
             "score (`llm_points_total`) and never used for ranking.",
             ""]
    for row in rows:
        lines.extend([f"### {row['condition_group']}", "",
                      markdown_table(
                          ["item", "category", "status", "detail"],
                          [[item["item_id"], item["category"], item["status"], item["detail"]]
                           for item in row["items"]]),
                      f"**{row['condition_group']} deterministic checklist total (auxiliary):** "
                      f"{row['summary']['passed_items']} / {DETERMINISTIC_CHECKLIST_MAX}",
                      ""])
    return lines


def _llm_checklist_section(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return []
    lines = ["## LLM Checklist Evaluation (primary qualitative score)", "",
             "25 binary (yes/no) questions across 5 categories, evaluated "
             "**only from each condition's sanitized final report** (never "
             "condition metadata, provider/model identity, or internal "
             "logs), answered independently by every available judge (GPT "
             "via Codex, Claude via Claude Code, Gemini via Antigravity, "
             "DeepSeek via OpenCode Go). A \"yes\" only counts if the judge "
             "supplies an exact evidence excerpt verified against the "
             "report text — an unverifiable \"yes\" is recorded as "
             "`yes_invalid_evidence` and scores zero, the same as a \"no\". "
             "Each category (A-E) scores 0-5; the total (`llm_points_total`) "
             "is the primary ranking metric.", ""]
    for row in rows:
        status_note = ", ".join(f"{judge}: {status}" for judge, status in row["judge_status"].items())
        lines.extend([f"### {row['condition_group']}", "", f"Judge availability: {status_note}", ""])
        if row.get("report_status") == "unavailable":
            lines.extend([
                "**Report unavailable** — `proposal_report_path` was missing or "
                "unreadable for this condition. No LLM checklist panel was run "
                "and no score is reported (never inferred from the structured "
                "condition record).", "",
            ])
            continue
        panel_size = len(row["judge_status"])
        for question in row["questions"]:
            lines.extend([f"**Question:** {question['text']}", ""])
            for judge, answer in question["judges"].items():
                evidence = (question.get("evidence") or {}).get(judge) or {}
                if answer in ("yes", "yes_invalid_evidence"):
                    lines.append(
                        f"- {judge.upper()}: {answer}"
                        + (f" — *{evidence.get('evidence_location')}*: "
                           f"“{evidence.get('evidence_excerpt')}”"
                           if evidence.get("evidence_excerpt") else ""))
                else:
                    lines.append(f"- {judge.upper()}: {answer}")
            lines.extend([
                "",
                f"Aggregated Yes Ratio (available judges only): "
                f"{question['yes_ratio'] if question['yes_ratio'] is not None else 'n/a (no judges available)'}",
                f"Point (verified yes_count / panel size {panel_size}): {question.get('point', 'n/a')}",
                "",
            ])
        category_scores = row["summary"].get("category_scores") or {}
        lines.extend([
            markdown_table(
                ["category", "score (/5)"],
                [[_CATEGORY_LABELS[category], category_scores.get(field, "n/a")]
                 for category, field in CATEGORY_SCORE_FIELDS.items()]),
            "",
            f"**{row['condition_group']} decision-analysis quality total:** "
            f"{row['summary'].get('llm_points_total', 'n/a')} / {LLM_CHECKLIST_MAX} "
            f"(process: {row['summary'].get('process_score', 'n/a')} / "
            f"{row['summary'].get('process_max', 'n/a')}; decision-support: "
            f"{row['summary'].get('decision_support_score', 'n/a')} / "
            f"{row['summary'].get('decision_support_max', 'n/a')})",
            "",
        ])
    return lines


def _checklist_summary_section(
    deterministic_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    scored_comparable_rows: list[dict[str, Any]],
) -> list[str]:
    lines = ["## Summary", ""]
    lines.extend(["### Deterministic Validation Summary (auxiliary)", ""])
    if deterministic_rows:
        lines.extend([markdown_table(
            ["condition", "passed", "applicable", "pass rate"],
            [[row["condition_group"], row["summary"]["passed_items"],
              row["summary"]["applicable_items"], row["summary"]["pass_rate"]]
             for row in deterministic_rows]), ""])
    else:
        lines.extend(["No comparable conditions to validate.", ""])

    lines.extend(["### Report-visible Decision-analysis Summary (primary ranking)", ""])
    scored = [row for row in scored_comparable_rows if row.get("llm_points_total") is not None]
    unavailable = [row for row in scored_comparable_rows if row.get("llm_points_total") is None]
    if scored:
        ranked = sorted(scored, key=lambda r: -r["llm_points_total"])
        category_labels = [label.split(". ", 1)[0] for label in _CATEGORY_LABEL_ORDER]
        lines.extend([markdown_table(
            ["condition", *[f"{label} (/5)" for label in category_labels],
             f"total (/{LLM_CHECKLIST_MAX})", "process (/17)", "decision-support (/8)"],
            [[row["condition_group"],
              *[(row.get("category_scores") or {}).get(field, "n/a") for field in _CATEGORY_SCORE_FIELD_ORDER],
              row["llm_points_total"], row.get("process_score", "n/a"),
              row.get("decision_support_score", "n/a")]
             for row in ranked]), ""])
    else:
        lines.extend(["No comparable conditions scored.", ""])
    if unavailable:
        lines.extend([
            f"Not scored (report unavailable for the LLM checklist judge): "
            f"{', '.join(row['condition_group'] for row in unavailable)}. "
            "No score is invented for these conditions.", "",
        ])
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
    lines.extend([note, ""])
    cat_a, cat_b = row_a.get("category_scores"), row_b.get("category_scores")
    if cat_a and cat_b:
        lines.extend([
            markdown_table(
                ["category (max 5 pts)", group_a, group_b, "delta"],
                [[_CATEGORY_LABELS[category], cat_a.get(field, "n/a"), cat_b.get(field, "n/a"),
                  _delta(cat_a.get(field), cat_b.get(field))]
                 for category, field in CATEGORY_SCORE_FIELDS.items()]),
            "",
        ])
    lines.extend([
        markdown_table(
            ["metric", group_a, group_b, "delta"],
            [
                [f"Report-visible decision-analysis total (primary, /{LLM_CHECKLIST_MAX})",
                 row_a.get("llm_points_total", "n/a"), row_b.get("llm_points_total", "n/a"),
                 _delta(row_a.get("llm_points_total"), row_b.get("llm_points_total"))],
                [f"Deterministic checklist passed (auxiliary, /{DETERMINISTIC_CHECKLIST_MAX})",
                 row_a.get("deterministic_passed", "n/a"), row_b.get("deterministic_passed", "n/a"),
                 _delta(row_a.get("deterministic_passed"), row_b.get("deterministic_passed"))],
            ],
        ), "",
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


_FAMILY_TITLES: dict[str, str] = {
    "vanilla_model": "C1-C4 vanilla model comparison",
    "native_agent_model": "C5-C8 native-agent model comparison",
    "skills_agent_model": "C9-C12 Skills model comparison",
    "ai_scientist_model": "C13-C14 AI-Scientist model comparison",
}


def _family_comparison_section(
    rows_by_group: dict[str, dict[str, Any]], records_by_group: dict[str, dict[str, Any]],
) -> list[str]:
    """One table per model-comparison family (vanilla/native-agent/Skills/
    AI-Scientist), replacing the previous ad-hoc per-pair sections. Within
    each family every condition shares the same framework/output contract
    (see comparison_families.py + the family_validation.py checks that ran
    when the condition's manual_result.json was ingested), so a score
    difference here reflects only the underlying model's output."""

    lines = ["## Family model comparisons", "",
             "Within each family below, every member condition runs the identical "
             "output contract, resource budget, and report/judge pipeline (enforced "
             "by family_validation.py at ingestion) — so a score difference is "
             "attributable to the provider/model, not a difference in what the "
             "condition was allowed or required to do.", ""]
    any_family = False
    for family_id, title in _FAMILY_TITLES.items():
        spec = COMPARISON_FAMILIES[family_id]
        member_rows = [rows_by_group[g] for g in spec.member_condition_ids if g in rows_by_group]
        if not member_rows:
            continue
        any_family = True
        rows_out = []
        for row in member_rows:
            family_validation = (records_by_group.get(row["condition_group"], {})
                                 .get("family_validation") or {})
            contract_status = (family_validation.get(family_id) or {}).get("passed")
            rows_out.append([
                row["condition_group"], f"{row['provider']}/{row['model']}", row["harness"],
                row["execution_mode"], row.get("llm_points_total", "n/a"),
                row.get("process_score", "n/a"), row.get("decision_support_score", "n/a"),
                "n/a" if contract_status is None else ("yes" if contract_status else "no"),
            ])
        lines.extend([
            f"### {title}", "",
            markdown_table(
                ["condition", "provider/model", "harness", "execution mode",
                 f"total (/{LLM_CHECKLIST_MAX})", "process (/17)", "decision-support (/8)",
                 "family contract satisfied"],
                rows_out,
            ), "",
        ])
    if not any_family:
        lines.append("No family had more than zero comparable, scored member conditions.")
        lines.append("")
    return lines


def _skills_ablation_pairwise_section(rows_by_group: dict[str, dict[str, Any]]) -> list[str]:
    """The four strict Skills-ablation pairs (C5/C9, C6/C10, C7/C11, C8/C12)
    as mechanism views: same provider/model/harness/budget on both sides,
    isolating the project Skills package as the sole treatment difference.
    Previously recorded only as a manifest contract with no rendered table."""

    pairs = COMPARISON_FAMILIES["skills_ablation_pair"].strict_pairs
    lines = ["## Skills-ablation pairwise comparisons (mechanism view)", "",
             "Same provider/model/harness/budget on both sides of each pair; the only "
             "treatment difference is whether the project Skills package was available. "
             "The no-Skills side is never held to the five-objective Skills contract.", ""]
    any_pair = False
    for no_skills, skills in pairs:
        row_a, row_b = rows_by_group.get(no_skills), rows_by_group.get(skills)
        if not row_a or not row_b:
            continue
        any_pair = True
        lines.extend([
            f"### {no_skills} (no Skills) vs {skills} (Skills)", "",
            markdown_table(
                ["metric", no_skills, skills, "delta"],
                [
                    [f"total (/{LLM_CHECKLIST_MAX})", row_a.get("llm_points_total", "n/a"),
                     row_b.get("llm_points_total", "n/a"),
                     _delta(row_a.get("llm_points_total"), row_b.get("llm_points_total"))],
                    ["process (/17)", row_a.get("process_score", "n/a"), row_b.get("process_score", "n/a"),
                     _delta(row_a.get("process_score"), row_b.get("process_score"))],
                    ["decision-support (/8)", row_a.get("decision_support_score", "n/a"),
                     row_b.get("decision_support_score", "n/a"),
                     _delta(row_a.get("decision_support_score"), row_b.get("decision_support_score"))],
                ],
            ), "",
        ])
    if not any_pair:
        lines.append("No Skills-ablation pair had both sides scored.")
        lines.append("")
    return lines


def _model_total_scores_section(rows_by_group: dict[str, dict[str, Any]]) -> list[str]:
    """Per-model-stack total score (sum AND per-condition mean) across
    C1-C14's four-family model ladder. C0 is naturally absent here — it has
    no provider/model to key a row on, not a ranking exclusion; it is still
    scored and ranked in full elsewhere (see module docstring). The mean
    guards the total against condition-count bias: Gemini/DeepSeek have 4
    member conditions, GPT/Claude 3."""

    lines = ["## Model comparison: total scores by provider/model", "",
             "Sum and per-condition mean of llm_points_total across each model's "
             "conditions in the C1-C14 family ladder (vanilla / native-agent / "
             "Skills / AI-Scientist). The mean is what should be compared across "
             "models with a different condition count; the sum is provided for "
             "reference only.", ""]
    rows_out = []
    for model, groups in MODEL_STACKS.items():
        scored = [rows_by_group[g] for g in groups
                  if g in rows_by_group and rows_by_group[g].get("llm_points_total") is not None]
        if not scored:
            continue
        total = round(sum(row["llm_points_total"] for row in scored), 4)
        mean_score = round(total / len(scored), 4)
        rows_out.append([
            model, ", ".join(row["condition_group"] for row in scored),
            len(scored), total, mean_score,
        ])
    if rows_out:
        lines.extend([markdown_table(
            ["model", "scored conditions", "n", f"sum (/{LLM_CHECKLIST_MAX} x n)",
             f"mean (/{LLM_CHECKLIST_MAX})"], rows_out), ""])
    else:
        lines.append("No model stack had a scored condition yet.")
        lines.append("")
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
    c1_c13_status = c1_c13_strict_comparison_status(registry)
    c2_c10_status = c2_c10_strict_comparison_status(registry)
    c3_c11_status = c3_c11_strict_comparison_status(registry)
    c4_c12_c14_status = c4_c12_c14_strict_comparison_status(registry)
    comparison_contracts = {
        "c5_c9_skills_comparison": c5_c9_strict_comparison_status(registry),
        "c6_c10_skills_comparison": c6_c10_strict_comparison_status(registry),
        "c7_c11_skills_comparison": c7_c11_strict_comparison_status(registry),
        "c8_c12_skills_comparison": c8_c12_strict_comparison_status(registry),
        "c9_c13_discovery_comparison": c9_c13_discovery_comparison_status(registry),
        "c12_c14_discovery_comparison": c12_c14_discovery_comparison_status(registry),
    }

    scored_rows: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: str(r.get("condition_group"))):
        group = str(record.get("condition_group"))
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
            "judge_type": "report_visible_decision_analysis_quality_checklist",
            "judgment_scope": JUDGMENT_SCOPE,
            "generated_at": generated_at,
        })

    comparable_rows = [row for row in scored_rows if row["comparable"]]
    alias_map = {row["condition_group"]: f"Method {chr(ord('A') + index)}"
                 for index, row in enumerate(comparable_rows)}
    records_by_group = {str(r.get("condition_group")): r for r in records}

    # ---- sanitized final-report text: the ONLY evidence source for the LLM
    # checklist judge — never the structured record. A condition whose
    # report is missing/unreadable gets an empty string here and is marked
    # unavailable below; it is never scored from the structured record. -----
    report_text_by_group: dict[str, str] = {}
    for row in comparable_rows:
        group = row["condition_group"]
        real_record = records_by_group.get(group, {})
        raw_text = _read_report_text(repo_root, real_record.get("proposal_report_path"))
        report_text_by_group[group] = (
            sanitize_report_for_judge(raw_text, alias_map.get(group, group)) if raw_text else "")

    anonymized_inputs = [
        {
            "method_alias": alias_map[group],
            "report_text": report_text_by_group.get(group, ""),
            "report_available": bool(report_text_by_group.get(group)),
        }
        for group in alias_map
    ]

    # ---- deterministic checklist: always runs for comparable rows, no LLM
    # call required, fully reproducible offline from the artifact files. ----
    deterministic_checklist_rows: list[dict[str, Any]] = []
    for row in comparable_rows:
        group = row["condition_group"]
        real_record = records_by_group.get(group, {})
        det_results = run_deterministic_checklist(
            real_record, _read_report_text(repo_root, real_record.get("proposal_report_path")))
        deterministic_checklist_rows.append({
            "condition_group": group,
            "items": [item.to_dict() for item in det_results],
            "summary": checklist_summary(det_results),
            "generated_at": generated_at,
        })

    # ---- LLM checklist panel: gated behind allow_live_judge / the disable
    # env var, since it makes real judge CLI/API calls. -----------------------
    llm_checklist_rows: list[dict[str, Any]] = []
    live_status = "skipped_no_comparable_records"
    if allow_live_judge and comparable_rows:
        if os.environ.get("E13_DISABLE_LIVE_JUDGE") == "1":
            live_status = "disabled"
        else:
            configured_judge_names = [backend.name for backend in default_judge_backends()]
            any_judge_ok = False
            for row in comparable_rows:
                group = row["condition_group"]
                report_text = report_text_by_group.get(group) or ""
                if not report_text:
                    llm_checklist_rows.append(
                        _unavailable_llm_checklist_row(group, configured_judge_names, generated_at))
                    continue
                panel = run_checklist_panel({
                    "condition_group": group,
                    "method_alias": alias_map.get(group),
                    "report_text": report_text,
                })
                if any(status == "ok" for status in panel.judge_status.values()):
                    any_judge_ok = True
                agg = aggregate_checklist(panel)
                panel_size = len(panel.judge_status)
                questions = []
                for item in agg:
                    entry = item.to_dict()
                    entry["point"] = round(entry["yes_count"] / panel_size, 4) if panel_size else None
                    questions.append(entry)
                category_scores = _category_scores(questions)
                dimension_scores = _dimension_scores(questions)
                summary = checklist_panel_summary(agg, panel.judge_status)
                summary["category_scores"] = category_scores
                summary["llm_points_total"] = round(sum(category_scores.values()), 4)
                summary["llm_max"] = LLM_CHECKLIST_MAX
                summary.update(dimension_scores)
                summary["process_max"] = DIMENSION_MAX["process"]
                summary["decision_support_max"] = DIMENSION_MAX["decision_support"]
                llm_checklist_rows.append({
                    "condition_group": group,
                    "judge_status": panel.judge_status,
                    "questions": questions,
                    "summary": summary,
                    "report_status": "ok",
                    "generated_at": generated_at,
                })
            live_status = "ok" if any_judge_ok else "no_judges_available"

    llm_by_group = {row["condition_group"]: row for row in llm_checklist_rows}
    det_by_group = {row["condition_group"]: row for row in deterministic_checklist_rows}
    for row in scored_rows:
        group = row["condition_group"]
        if group in det_by_group:
            det_summary = det_by_group[group]["summary"]
            row["deterministic_checklist"] = det_summary
            row["deterministic_passed"] = det_summary["passed_items"]
        if group in llm_by_group:
            llm_row = llm_by_group[group]
            summary = llm_row["summary"]
            row["llm_checklist"] = summary
            row["category_scores"] = summary.get("category_scores")
            row["llm_points_total"] = summary.get("llm_points_total")
            row["llm_max"] = LLM_CHECKLIST_MAX
            row["process_score"] = summary.get("process_score")
            row["decision_support_score"] = summary.get("decision_support_score")
            row["process_max"] = DIMENSION_MAX["process"]
            row["decision_support_max"] = DIMENSION_MAX["decision_support"]
            row["report_status"] = llm_row.get("report_status")

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
    scored_for_figures = [row for row in comparable_rows if row.get("llm_points_total") is not None]
    if scored_for_figures:
        figure = condition_comparison_figure(
            [(row["condition_group"], row["llm_points_total"]) for row in scored_for_figures],
            figures_dir / "judge_condition_comparison.png",
            title="Report-visible decision-analysis total by condition (comparable, scored results only)",
            xlabel=f"llm_points_total (0-{LLM_CHECKLIST_MAX})")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")
    for pair, filename, title in (
        (("C1", "C13"), "judge_c1_c13_orchestration.png", "C1 vs C13 agentic-orchestration comparison (same provider/model, Gemini)"),
        (("C2", "C10"), "judge_c2_c10_skills.png", "C2 vs C10 Skills-unification comparison (same provider/model, Codex)"),
        (("C3", "C11"), "judge_c3_c11_skills.png", "C3 vs C11 Skills-unification comparison (same provider/model, Claude Code)"),
        (("C4", "C12"), "judge_c4_c12_skills.png", "C4 vs C12 Skills-unification comparison (same provider/model, DeepSeek)"),
        (("C4", "C14"), "judge_c4_c14_orchestration.png", "C4 vs C14 agentic-orchestration comparison (same provider/model, DeepSeek)"),
        (("C12", "C14"), "judge_c12_c14_harness.png", "C12 vs C14 Skills-harness vs direct-API comparison (same provider/model, DeepSeek)"),
        (("C13", "C14"), "judge_c13_c14_models.png", "C13 vs C14 model/provider comparison (same algorithm family)"),
    ):
        row_a, row_b = rows_by_group.get(pair[0]), rows_by_group.get(pair[1])
        if row_a and row_b and row_a.get("category_scores") and row_b.get("category_scores"):
            figure = grouped_bar_figure(
                _CATEGORY_LABEL_ORDER,
                {pair[0]: [row_a["category_scores"].get(field, 0.0) for field in _CATEGORY_SCORE_FIELD_ORDER],
                 pair[1]: [row_b["category_scores"].get(field, 0.0) for field in _CATEGORY_SCORE_FIELD_ORDER]},
                figures_dir / filename, title=title, ylabel="qualitative score (0-5 per category)")
            if figure:
                figure_refs.append(f"../figures/{figure.name}")
    vanilla_groups = [g for g in ("C1", "C2", "C3", "C4")
                      if g in rows_by_group and rows_by_group[g].get("category_scores")]
    if len(vanilla_groups) >= 2:
        figure = grouped_bar_figure(
            _CATEGORY_LABEL_ORDER,
            {g: [rows_by_group[g]["category_scores"].get(field, 0.0) for field in _CATEGORY_SCORE_FIELD_ORDER]
             for g in vanilla_groups},
            figures_dir / "judge_c1_c4_vanilla.png",
            title="C1-C4 vanilla-baseline comparison (provider/harness differ together)",
            ylabel="qualitative score (0-5 per category)")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")
    skills_groups = [g for g in ("C9", "C10", "C11", "C12")
                     if g in rows_by_group and rows_by_group[g].get("category_scores")]
    if len(skills_groups) >= 2:
        figure = grouped_bar_figure(
            _CATEGORY_LABEL_ORDER,
            {g: [rows_by_group[g]["category_scores"].get(field, 0.0) for field in _CATEGORY_SCORE_FIELD_ORDER]
             for g in skills_groups},
            figures_dir / "judge_c9_c12_skills_harness.png",
            title="C9-C12 Skills-in-harness comparison (harness+model differ together)",
            ylabel="qualitative score (0-5 per category)")
        if figure:
            figure_refs.append(f"../figures/{figure.name}")

    # ---- report ------------------------------------------------------------------
    lines = [
        "# Condition Comparison Report (cross-condition judge)",
        "",
        f"Generated: {generated_at} — live LLM checklist judge status: `{live_status}` "
        f"— rubric: `{QUALITATIVE_RUBRIC_VERSION}` — primary ranking metric: `llm_points_total`",
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

    lines.extend(_deterministic_validation_section(deterministic_checklist_rows))
    lines.extend(_llm_checklist_section(llm_checklist_rows))
    lines.extend(_checklist_summary_section(deterministic_checklist_rows, llm_checklist_rows, comparable_rows))

    lines.extend(_pairwise_section(rows_by_group, "C1", "C13",
                                   "C1 vs C13 — strict agentic-orchestration comparison (Gemini)",
                                   _strict_pair_note(c1_c13_status, "C1", "C13",
                                                      "the AI Scientist-style agentic loop over a single vanilla pass")))
    lines.extend(_pairwise_section(rows_by_group, "C2", "C10",
                                   "C2 vs C10 — strict Skills-unification comparison (Codex)",
                                   _strict_pair_note(c2_c10_status, "C2", "C10",
                                                      "the Skills contract and branch search")))
    lines.extend(_pairwise_section(rows_by_group, "C3", "C11",
                                   "C3 vs C11 — strict Skills-unification comparison (Claude Code)",
                                   _strict_pair_note(c3_c11_status, "C3", "C11",
                                                      "the Skills contract and branch search")))
    lines.extend(_pairwise_section(rows_by_group, "C4", "C12",
                                   "C4 vs C12 — strict Skills-unification comparison (DeepSeek)",
                                   "Part of the C4/C12/C14 strict trio (see below); isolates the "
                                   "Skills contract and branch search over a single vanilla pass."))
    lines.extend(_pairwise_section(rows_by_group, "C4", "C14",
                                   "C4 vs C14 — strict agentic-orchestration comparison (DeepSeek)",
                                   "Part of the C4/C12/C14 strict trio; isolates the full "
                                   "AI Scientist-style agentic loop over a single vanilla pass."))
    lines.extend(_pairwise_section(rows_by_group, "C12", "C14",
                                   "C12 vs C14 — strict Skills-harness vs direct-API comparison (DeepSeek)",
                                   "Part of the C4/C12/C14 strict trio; isolates the interactive "
                                   "Skills harness against the direct-API AI Scientist-style draft tree."))
    lines.extend(_pairwise_section(
        rows_by_group, "C13", "C14",
        "C13 vs C14 — model/provider comparison",
        "Same AI Scientist-style algorithm family and aligned budget envelope; "
        "the comparison isolates provider and model effects."))

    trio_confounded = c4_c12_c14_status["confounded"]
    trio_note = (
        "C4, C12, and C14 share the same provider and model "
        f"(`{c4_c12_c14_status['c4']['provider']}` / `{c4_c12_c14_status['c4']['model']}`); "
        "the three pairwise comparisons above isolate harness and "
        "orchestration effects cleanly."
        if not trio_confounded else
        "**CONFOUNDED**: C4, C12, and C14 do not all share provider and model "
        f"(C4 `{c4_c12_c14_status['c4']['provider']}/{c4_c12_c14_status['c4']['model']}`, "
        f"C12 `{c4_c12_c14_status['c12']['provider']}/{c4_c12_c14_status['c12']['model']}`, "
        f"C14 `{c4_c12_c14_status['c14']['provider']}/{c4_c12_c14_status['c14']['model']}`); "
        "harness/orchestration effects cannot be separated from model effects "
        "in the three pairwise comparisons above."
    )
    lines.extend(["## C4/C12/C14 strict trio status", "", trio_note, ""])

    lines.extend(_model_total_scores_section(rows_by_group))
    lines.extend(_family_comparison_section(rows_by_group, records_by_group))
    lines.extend(_skills_ablation_pairwise_section(rows_by_group))

    scored_comparable = [row for row in comparable_rows if row.get("llm_points_total") is not None]
    unavailable_comparable = [row for row in comparable_rows if row.get("llm_points_total") is None]
    # C0 competes on the same rubric as every other condition here (see
    # module docstring) — this section mixes conditions from different
    # frameworks together, which is exactly why it is informational context
    # rather than the primary comparison (see the per-family and per-model
    # tables above for apples-to-apples views), not because any condition is
    # excluded from ranking.
    ordered = sorted(scored_comparable, key=lambda row: -row["llm_points_total"])
    weakest_category = None
    if scored_comparable:
        category_means = {
            category: mean((row.get("category_scores") or {}).get(field, 0.0) for row in scored_comparable)
            for category, field in CATEGORY_SCORE_FIELDS.items()
        }
        weakest_category = min(category_means, key=category_means.get)
    recommended_next = (
        waiting_rows[0]["condition_group"] if waiting_rows
        else (ordered[-1]["condition_group"] if len(ordered) > 1 else None))
    lines.extend([
        "## Recommendations (supplemental cross-family view, not a primary comparison)",
        "",
        "The per-family and per-model tables above are the primary comparison "
        "views (same framework/contract on every side); this section mixes "
        "conditions from different frameworks together and should be read as "
        "informational context, not a definitive ranking.",
        "",
        f"- Strongest comparable conditions (by report-visible decision-analysis total): "
        f"{', '.join(row['condition_group'] for row in ordered[:2]) or 'none scored'}.",
        f"- Recommended next condition to improve: "
        f"{recommended_next or 'n/a'}"
        + (f" (weakest shared checklist category across the track: {_CATEGORY_LABELS[weakest_category]})"
           if weakest_category else "") + ".",
        *( [f"- Unblock manual-harness conditions "
            f"({', '.join(row['condition_group'] for row in waiting_rows)}) via the "
            "prompt files under `manual_harness/` in the live output dir."] if waiting_rows else [] ),
        *( [f"- Not scored (report unavailable for the LLM checklist judge): "
            f"{', '.join(row['condition_group'] for row in unavailable_comparable)}."]
           if unavailable_comparable else [] ),
        "",
        "## Proposal files a human should inspect next",
        "",
        *[f"- `{row['condition_group']}`: {row['proposal_report_path'] or 'no report'}"
          for row in ordered[:4]],
        *( [f"- `{row['condition_group']}` (excluded, for debugging): "
            f"{row['proposal_report_path']}" for row in fallback_rows[:2]] ),
        "",
        required_due_diligence_section([
            "The checklist judge compares report-visible decision analysis and "
            "document-structural evidence, not real-world outcomes.",
            f"Live LLM checklist judge status: {live_status}; when unavailable, no "
            "primary qualitative score is reported (never inferred from the "
            "structured condition record).",
            "Adoption decisions require the due diligence listed in each proposal report.",
        ]),
    ])

    report_path = out_dir / "reports" / "condition_comparison_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")

    reports_dir = out_dir / "reports"
    output_paths = {
        "comparison_report": str(report_path),
        "scores": str(out_dir / "condition_judge_scores.jsonl"),
        "anonymized_judge_inputs": str(out_dir / "condition_judge_anonymized_inputs.jsonl"),
        "deterministic_checklist": str(out_dir / "condition_deterministic_checklist.jsonl"),
        "llm_checklist": str(out_dir / "condition_llm_checklist.jsonl"),
        "manifest": str(out_dir / "condition_judge_manifest.json"),
        "agentic_reasoning_scores_csv": str(reports_dir / "condition_agentic_reasoning_scores.csv"),
        "llm_checklist_matrix_csv": str(reports_dir / "condition_llm_checklist_matrix.csv"),
        "deterministic_checklist_matrix_csv": str(reports_dir / "condition_deterministic_checklist_matrix.csv"),
        "checklist_totals_csv": str(reports_dir / "condition_checklist_totals.csv"),
    }
    _write_jsonl(Path(output_paths["scores"]), scored_rows)
    _write_jsonl(Path(output_paths["anonymized_judge_inputs"]), anonymized_inputs)
    _write_jsonl(Path(output_paths["deterministic_checklist"]), deterministic_checklist_rows)
    _write_jsonl(Path(output_paths["llm_checklist"]), llm_checklist_rows)
    write_agentic_reasoning_scores_csv(
        Path(output_paths["agentic_reasoning_scores_csv"]), scored_rows, llm_max=LLM_CHECKLIST_MAX)
    write_llm_checklist_matrix_csv(Path(output_paths["llm_checklist_matrix_csv"]), llm_checklist_rows)
    write_deterministic_checklist_matrix_csv(
        Path(output_paths["deterministic_checklist_matrix_csv"]), deterministic_checklist_rows)
    write_checklist_totals_csv(
        Path(output_paths["checklist_totals_csv"]), scored_rows,
        llm_max=LLM_CHECKLIST_MAX, deterministic_max=DETERMINISTIC_CHECKLIST_MAX)
    _write_json(Path(output_paths["manifest"]), {
        "run_id": run_id,
        "generated_at": generated_at,
        "judgment_scope": JUDGMENT_SCOPE,
        "qualitative_rubric_version": QUALITATIVE_RUBRIC_VERSION,
        "qualitative_score_max": LLM_CHECKLIST_MAX,
        "primary_ranking_metric": "llm_points_total",
        "judge_evidence_source": "sanitized_final_markdown_report_only",
        "scored_condition_count": len(scored_rows),
        "comparable_condition_count": len(comparable_rows),
        "alias_map": alias_map,
        "live_judge_status": live_status,
        "c1_c13_strict_comparison": c1_c13_status,
        "c2_c10_strict_comparison": c2_c10_status,
        "c3_c11_strict_comparison": c3_c11_status,
        "c4_c12_c14_strict_comparison": c4_c12_c14_status,
        **comparison_contracts,
        "checklist_evaluation": {
            "llm_checklist_questions": LLM_CHECKLIST_MAX,
            "deterministic_checklist_items": DETERMINISTIC_CHECKLIST_MAX,
            "llm_max": LLM_CHECKLIST_MAX,
            "deterministic_max": DETERMINISTIC_CHECKLIST_MAX,
            "llm_point_rule": "verified yes_count / panel_size; panel_size is the "
                              "configured judge count. A \"yes\" without a verified "
                              "evidence excerpt scores zero, the same as a \"no\".",
            "panel_size": len(llm_checklist_rows[0]["judge_status"]) if llm_checklist_rows else None,
        },
        "result_groups": {
            "comparable_live_agent_results": [row["condition_group"] for row in comparable_rows],
            "non_comparable_fallback_results": [row["condition_group"] for row in fallback_rows],
            "waiting_for_manual_harness_results": [row["condition_group"] for row in waiting_rows],
        },
        "unavailable_for_llm_checklist": [row["condition_group"] for row in unavailable_comparable],
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
