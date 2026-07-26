"""Checklist-based, panel-of-judges qualitative LLM evaluation.

Scores the report-visible **decision-analysis** quality of each condition's
final Markdown report against a fixed 25-question, 5-category checklist
(``report_visible_decision_analysis_quality_v3``) — not proposal-content fields, not
condition metadata, and not internal logs. Every question is independently
answered yes/no by every available judge, with a mandatory exact evidence
excerpt for every "yes"; both individual answers and the aggregated per-
question point are kept.

Design goals:

- The judge sees only the sanitized final report text supplied by the
  caller (``context["report_text"]``) — never the structured condition
  record, provider/model/harness metadata, or internal branch/candidate-
  review JSON. Building that sanitized text is this module's caller's job
  (``report_judge_view.py`` / ``condition_comparison_judge.py``); this
  module only prompts, parses, and validates evidence against whatever text
  it is given.
- A "yes" answer is only counted if it carries both an ``evidence_location``
  and an ``evidence_excerpt`` that is verified as a literal exact substring
  of the supplied report text. A
  "yes" that fails this check is recorded as ``"yes_invalid_evidence"`` and
  contributes zero, exactly like a "no" — it is never silently accepted on
  rationale alone.
- Judges are pluggable ``JudgeBackend`` entries reusing existing
  provider/harness wrappers already in this project (Codex CLI, Claude Code
  CLI, the Antigravity CLI, and the OpenCode Go direct API client) — no new
  evaluation infrastructure. Adding a fifth judge later is one entry in
  ``default_judge_backends()``.
- A judge that is unavailable (CLI/credential missing, non-zero exit,
  timeout, unparseable output) is marked ``"unavailable"`` and remains in
  the configured four-judge denominator; it never crashes the run.
- Gemini defaults to the Antigravity CLI adapter rather than the direct
  Gemini API client, because the direct API has a documented, very tight
  quota (``GEMINI_REQUESTS_PER_DAY`` in ``.env.example``) that a 4-judge
  panel would exhaust quickly; set ``E13_GEMINI_JUDGE_VIA=api`` to use the
  direct API client instead.

The rubric intentionally scores only what is observable in the report text:
it never references a condition's algorithm, provider, model, or workflow
label, and does not hard-code any expectation that a particular condition
(AI Scientist, Skills, or any other) scores higher — differences must come
from what the judge can read and quote.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

LLM_CHECKLIST_CATEGORIES: tuple[str, ...] = (
    "decision_space_exploration",
    "executed_validation_and_evidence_use",
    "cross_alternative_synthesis",
    "robustness_and_decision_refinement",
    "business_decision_utility",
)

_CATEGORY_LABELS: dict[str, str] = {
    "decision_space_exploration": "A. Decision-space exploration",
    "executed_validation_and_evidence_use": "B. Executed validation and evidence use",
    "cross_alternative_synthesis": "C. Cross-alternative synthesis",
    "robustness_and_decision_refinement": "D. Robustness and decision refinement",
    "business_decision_utility": "E. Business decision utility",
}

# Category key -> the manifest/CSV/report field prefix used for the primary
# qualitative score (e.g. "A_alternative_space_exploration").
CATEGORY_SCORE_FIELDS: dict[str, str] = {
    "decision_space_exploration": "A_decision_space_exploration",
    "executed_validation_and_evidence_use": "B_executed_validation_and_evidence_use",
    "cross_alternative_synthesis": "C_cross_alternative_synthesis",
    "robustness_and_decision_refinement": "D_robustness_and_decision_refinement",
    "business_decision_utility": "E_business_decision_utility",
}


@dataclass(frozen=True)
class ChecklistQuestion:
    question_id: str
    category: str
    text: str


LLM_CHECKLIST_QUESTIONS: tuple[ChecklistQuestion, ...] = (
    ChecklistQuestion("a1_materially_different_strategies", LLM_CHECKLIST_CATEGORIES[0], "Does the report present at least two materially different decision strategies, regimes, scenarios, or analytical alternatives?"),
    ChecklistQuestion("a2_alternative_differences_explained", LLM_CHECKLIST_CATEGORIES[0], "Does the report explain how the alternatives differ in objectives, assumptions, eligibility constraints, evaluation logic, risk tolerance, or portfolio rules?"),
    ChecklistQuestion("a3_candidate_outcomes_two_alternatives", LLM_CHECKLIST_CATEGORIES[0], "Does the report provide candidate-level numerical, ordinal, categorical, or decision-status outcomes for at least two alternatives?"),
    ChecklistQuestion("a4_divergence_or_convergence_analyzed", LLM_CHECKLIST_CATEGORIES[0], "Does the report analyze either meaningful divergence or meaningful convergence across the alternatives, rather than merely listing them?"),
    ChecklistQuestion("a5_rejected_strategy_lesson", LLM_CHECKLIST_CATEGORIES[0], "Does the report describe at least one rejected, failed, dominated, or non-selected strategy and explain a decision-relevant lesson from it?"),
    ChecklistQuestion("b1_executed_quantitative_analysis", LLM_CHECKLIST_CATEGORIES[1], "Does the report show that at least one recommendation or strategy is based on an executed quantitative analysis rather than unsupported narrative assertion?"),
    ChecklistQuestion("b2_evidence_or_metrics_linked", LLM_CHECKLIST_CATEGORIES[1], "Does the report link material conclusions to external deterministic metrics, source evidence, evidence grades, or explicit uncertainty labels?"),
    ChecklistQuestion("b3_performed_validation_test", LLM_CHECKLIST_CATEGORIES[1], "Does the report contain at least one actually performed sensitivity test, ablation, stress test, scenario test, or counterfactual comparison?"),
    ChecklistQuestion("b4_evidence_quality_affects_decision", LLM_CHECKLIST_CATEGORIES[1], "Does evidence quality or missingness visibly affect candidate eligibility, rank, confidence, risk status, or decision status?"),
    ChecklistQuestion("b5_negative_result_interpreted", LLM_CHECKLIST_CATEGORIES[1], "Does the report identify and interpret at least one negative, inconclusive, contradictory, or failed result rather than hiding it?"),
    ChecklistQuestion("c1_synthesis_rule", LLM_CHECKLIST_CATEGORIES[2], "Does the report state an explicit rule or rationale for converting alternative results into the final recommendation or portfolio?"),
    ChecklistQuestion("c2_candidate_source_strategy", LLM_CHECKLIST_CATEGORIES[2], "For at least one final candidate, does the report identify which strategy, objective, scenario, or alternative supported its inclusion?"),
    ChecklistQuestion("c3_tradeoff_explained", LLM_CHECKLIST_CATEGORIES[2], "Does the report explain at least one substantive conflict or trade-off between objectives, strategies, or candidate attributes?"),
    ChecklistQuestion("c4_default_leader_exception", LLM_CHECKLIST_CATEGORIES[2], "Does the report explain why at least one candidate was included despite not being the default composite-score leader, or why a default leader was excluded?"),
    ChecklistQuestion("c5_final_vs_excluded", LLM_CHECKLIST_CATEGORIES[2], "Does the report compare the final recommendation with at least one credible excluded alternative, rejected candidate, or contingency portfolio?"),
    ChecklistQuestion("d1_major_issue_evidence", LLM_CHECKLIST_CATEGORIES[3], "Does the report identify at least one major candidate-specific or portfolio-specific issue and connect it to concrete evidence?"),
    ChecklistQuestion("d2_observable_issue_effect", LLM_CHECKLIST_CATEGORIES[3], "Does at least one issue produce an observable change in rank, candidate inclusion, decision status, confidence, risk classification, evidence label, or due-diligence requirement?"),
    ChecklistQuestion("d3_candidate_reversal_condition", LLM_CHECKLIST_CATEGORIES[3], "Does the report state at least one candidate-specific reversal condition, threshold, scenario, or finding that would change the recommendation?"),
    ChecklistQuestion("d4_stability_classification", LLM_CHECKLIST_CATEGORIES[3], "Does the report classify at least one candidate or conclusion as stable, sensitive, provisional, conditional, or unresolved and explain why?"),
    ChecklistQuestion("d5_unresolved_issue_propagated", LLM_CHECKLIST_CATEGORIES[3], "Does an unresolved major issue affect the final recommendation status, decision gate, next step, replacement option, or residual-risk statement?"),
    ChecklistQuestion("e1_candidate_specific_reasoning", LLM_CHECKLIST_CATEGORIES[4], "Does the report provide candidate-specific reasoning that connects demand, supply, finance, evidence, access, feasibility, or other relevant factors to the recommended action?"),
    ChecklistQuestion("e2_relative_preference", LLM_CHECKLIST_CATEGORIES[4], "Does the report explain at least one relative ranking or preference between two concrete candidates or portfolios?"),
    ChecklistQuestion("e3_portfolio_composition", LLM_CHECKLIST_CATEGORIES[4], "Does the report explain the geographic, action-type, risk, or strategic composition of the recommended portfolio?"),
    ChecklistQuestion("e4_prioritized_due_diligence", LLM_CHECKLIST_CATEGORIES[4], "Does the report provide prioritized, candidate-linked due diligence or information-acquisition steps?"),
    ChecklistQuestion("e5_actionable_status", LLM_CHECKLIST_CATEGORIES[4], "Does the report assign an actionable status such as proceed, conditional, revise, defer, replace, or reject and connect it to a concrete next step or decision condition?"),
)

assert len(LLM_CHECKLIST_QUESTIONS) == 25
assert {q.category for q in LLM_CHECKLIST_QUESTIONS} == set(LLM_CHECKLIST_CATEGORIES)
assert all(
    sum(1 for q in LLM_CHECKLIST_QUESTIONS if q.category == category) == 5
    for category in LLM_CHECKLIST_CATEGORIES
)

# Two evaluation dimensions over the same 25 questions/categories -- a
# neutral decomposition, never a second scoring pass or a different rubric,
# and never gated by condition identity: "process" is where a report earns
# credit for reasoning content it visibly contains (exploration, executed
# validation, synthesis traceability, robustness, critique). "decision_support"
# is the practical-usability half: does the report give a reader an
# actionable, candidate-specific, portfolio-aware decision. Every question
# belongs to exactly one dimension; the two subtotals always sum to
# llm_points_total (0-25). Any condition, including C0, earns whichever
# points its own report content supports on either dimension -- neither
# dimension carries a per-condition cap or penalty. Recomputable offline
# from persisted per-question yes-counts -- no re-judging required to add
# or audit this split.
DECISION_SUPPORT_QUESTION_IDS: frozenset[str] = frozenset({
    "b4_evidence_quality_affects_decision",
    "d3_candidate_reversal_condition",
    "d4_stability_classification",
    "e1_candidate_specific_reasoning",
    "e2_relative_preference",
    "e3_portfolio_composition",
    "e4_prioritized_due_diligence",
    "e5_actionable_status",
})
QUESTION_DIMENSIONS: dict[str, str] = {
    q.question_id: ("decision_support" if q.question_id in DECISION_SUPPORT_QUESTION_IDS else "process")
    for q in LLM_CHECKLIST_QUESTIONS
}
DIMENSION_MAX: dict[str, int] = {
    "process": len(LLM_CHECKLIST_QUESTIONS) - len(DECISION_SUPPORT_QUESTION_IDS),
    "decision_support": len(DECISION_SUPPORT_QUESTION_IDS),
}
assert set(QUESTION_DIMENSIONS.values()) == {"process", "decision_support"}
assert DIMENSION_MAX == {"process": 17, "decision_support": 8}

_QUESTIONS_BY_ID: dict[str, ChecklistQuestion] = {q.question_id: q for q in LLM_CHECKLIST_QUESTIONS}

_MAX_EVIDENCE_EXCERPT_CHARS = 300
_MAX_EVIDENCE_LOCATION_CHARS = 200

_JUDGE_OPERATIONAL_INSTRUCTIONS = (
    "1. Evaluate only information visible in the submitted final report.\n"
    "2. Ignore condition names, model names, provider names, algorithm names, "
    "token counts, call counts, run times, and claims that the workflow is "
    "agentic or AI Scientist-style.\n"
    '3. "At least one" means one clearly identifiable candidate, alternative, '
    "critique, uncertainty, or decision condition.\n"
    "4. A generic or boilerplate statement does not satisfy a criterion. "
    "Evidence must be substantive and decision-relevant.\n"
    "5. Do not infer that exploration, critique, revision, or sensitivity analysis "
    "occurred unless its result is observable in the report.\n"
    "6. A claim that alternatives were considered is insufficient unless at least "
    "one alternative and its candidate-level consequence are shown.\n"
    "7. An author response is not an observable revision unless the report shows a "
    "concrete change, or gives a substantive evidence-based justification for "
    "retaining the original decision.\n"
    "8. Numerical tables alone do not establish causal reasoning, synthesis, or "
    "trade-off analysis unless the report explicitly interprets them.\n"
    "9. Every yes answer must include an exact evidence excerpt and report location.\n"
    "10. If sufficient evidence cannot be quoted, answer no.\n"
    "11. Every question is positively oriented. Yes always indicates higher "
    "report quality.\n"
    "12. Each yes receives one point and each no receives zero points for a "
    "single-judge run.\n"
    "13. Do not award points for shared report-template text that merely restates "
    "standard data fields, evidence-grade definitions, generic due-diligence "
    "disclaimers, or standard pipeline descriptions."
)


@dataclass(frozen=True)
class JudgeBackend:
    name: str
    provider: str
    call: Callable[[str], str | None]


@dataclass(frozen=True)
class JudgeAnswer:
    question_id: str
    answer: str  # "yes" | "no"
    evidence_location: str = ""
    evidence_excerpt: str = ""
    rationale: str = ""
    # False only for a "yes" whose evidence_location/evidence_excerpt is
    # missing, or whose excerpt cannot be found verbatim (after whitespace
    # normalization) in the report text supplied to the judge. A "no" is
    # always evidence_valid=True — no excerpt is required to say no.
    evidence_valid: bool = True


@dataclass
class PanelResult:
    condition_group: str
    judge_status: dict[str, str] = field(default_factory=dict)  # judge_name -> "ok"|"unavailable"|error detail
    answers_by_judge: dict[str, dict[str, JudgeAnswer]] = field(default_factory=dict)  # judge -> question_id -> answer


@dataclass(frozen=True)
class AggregatedChecklistRow:
    question_id: str
    category: str
    text: str
    # judge_name -> "yes" | "no" | "yes_invalid_evidence" | "unavailable".
    # "yes_invalid_evidence" contributes zero to yes_count, exactly like "no"
    # — it is kept distinct only so the report/CSV can show *why* a judge's
    # "yes" did not score.
    judges: dict[str, str]
    # judge_name -> {"evidence_location":..., "evidence_excerpt":..., "rationale":...}
    evidence: dict[str, dict[str, str]]
    yes_count: int
    available_judge_count: int
    yes_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "category": self.category,
            "category_label": _CATEGORY_LABELS[self.category],
            "text": self.text,
            "judges": self.judges,
            "evidence": self.evidence,
            "yes_count": self.yes_count,
            "available_judge_count": self.available_judge_count,
            "yes_ratio": self.yes_ratio,
        }


def _prompt_for_panel(context: dict[str, Any]) -> str:
    method = context.get("method_alias") or "the report"
    questions_block = "\n".join(
        f'{q.question_id}: [{_CATEGORY_LABELS[q.category]}] {q.text}'
        for q in LLM_CHECKLIST_QUESTIONS
    )
    report_text = str(context.get("report_text") or "")
    return (
        f"You are evaluating the decision-analysis quality of a hospital-strategy "
        f"site-selection decision-support report ({method}). Answer EVERY question "
        "below independently with a strict yes/no, using only the final report text "
        "supplied below as evidence.\n\n"
        f"INSTRUCTIONS:\n{_JUDGE_OPERATIONAL_INSTRUCTIONS}\n\n"
        f"QUESTIONS:\n{questions_block}\n\n"
        "FINAL REPORT (the only evidence you may use):\n-----\n"
        + report_text + "\n-----\n\n"
        'Reply as strict JSON only: {"answers": {"<question_id>": '
        '{"answer": "yes"|"no", "evidence_location": "<report heading or candidate '
        'subsection>", "evidence_excerpt": "<exact text copied verbatim from the '
        f'report above, at most {_MAX_EVIDENCE_EXCERPT_CHARS} characters>", '
        '"rationale": "<concise explanation>"}, ...}}. '
        'A "yes" without both evidence_location and evidence_excerpt is invalid and '
        "will be scored as no; the excerpt must be an exact substring of the report "
        "text above. Include every question_id listed above."
    )


def _extract_json_block(text: str) -> Any | None:
    from geo_strategist.experiments.live_common import extract_json_block

    return extract_json_block(text)


def _excerpt_verifiable(excerpt: str, report_text: str) -> bool:
    """True only for a literal excerpt from the exact sanitized report."""

    if not excerpt:
        return False
    return excerpt in report_text


def _parse_panel_response(text: str, report_text: str) -> dict[str, JudgeAnswer] | None:
    parsed = _extract_json_block(text)
    # extract_json_block is a best-effort JSON parse of raw LLM text and can
    # return any JSON type (list, string, number) -- e.g. a judge replying
    # with a bare `[...]` instead of the required `{"answers": {...}}`
    # object. Treat anything that isn't a dict as unparseable rather than
    # crashing the whole panel/judge run on a single bad backend reply.
    if not isinstance(parsed, dict):
        return None
    raw_answers = parsed.get("answers")
    if not isinstance(raw_answers, dict):
        return None
    out: dict[str, JudgeAnswer] = {}
    for question_id, row in raw_answers.items():
        if question_id not in _QUESTIONS_BY_ID or not isinstance(row, dict):
            continue
        answer = str(row.get("answer", "")).strip().lower()
        if answer not in ("yes", "no"):
            continue
        evidence_location = str(row.get("evidence_location") or "").strip()[:_MAX_EVIDENCE_LOCATION_CHARS]
        evidence_excerpt = str(row.get("evidence_excerpt") or "").strip()[:_MAX_EVIDENCE_EXCERPT_CHARS]
        rationale = str(row.get("rationale") or "")[:400]
        evidence_valid = True
        if answer == "yes":
            evidence_valid = bool(evidence_location) and bool(evidence_excerpt) and (
                _excerpt_verifiable(evidence_excerpt, report_text))
        out[question_id] = JudgeAnswer(
            question_id, answer, evidence_location, evidence_excerpt, rationale, evidence_valid)
    return out or None


# ---------------------------------------------------------------------------
# Judge backends — reuse existing provider/harness wrappers, no new plumbing.
# ---------------------------------------------------------------------------

def _cli_judge_call(harness: str, timeout: int | None = None) -> Callable[[str], str | None]:
    """Non-interactive CLI judge call reusing the existing harness adapters
    (same ``HarnessAdapter``/subprocess pattern as the Skills-harness
    launcher in ``harnesses/agentic_runner.py``, without the manual-result
    file expectations a Skills run has)."""

    def call(prompt: str) -> str | None:
        from geo_strategist.harnesses.adapters import adapter_for

        adapter = adapter_for(harness)
        repo_root = Path(".").resolve()
        command = adapter.build_command(
            repo_root=repo_root, run_dir=repo_root,
            launcher_prompt_path=Path("/dev/null"),
            manual_result_path=Path("/dev/null"),
        )
        if not command:
            return None
        stdin_input = prompt if adapter.prompt_mode == "stdin" else None
        if adapter.prompt_mode == "argument":
            command = [*command, prompt]
        # A judge call is one short generate-and-answer request, not an
        # agentic session — bound it far below the adapter's Skills-harness
        # timeout (minutes, not the 7200s a real agentic run may need) so an
        # unresponsive/interactive CLI degrades to "unavailable" quickly
        # instead of stalling the whole judge run. 300s (not 60s): a real
        # `claude -p` call against the full 25-question panel prompt
        # measured ~71s — 60s silently killed every Claude judge call
        # before it could return.
        effective_timeout = timeout if timeout is not None else int(os.environ.get(
            "E13_CHECKLIST_JUDGE_CLI_TIMEOUT_SECONDS", "300"))
        try:
            completed = subprocess.run(
                command, cwd=repo_root, env=os.environ.copy(),
                input=stdin_input, capture_output=True, text=True,
                timeout=min(effective_timeout, adapter.timeout_seconds), check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout or None

    return call


def _gpt_backend() -> JudgeBackend:
    return JudgeBackend("gpt", "codex", _cli_judge_call("codex"))


def _claude_backend() -> JudgeBackend:
    return JudgeBackend("claude", "claude_code", _cli_judge_call("claude_code"))


def _gemini_backend() -> JudgeBackend:
    if os.environ.get("E13_GEMINI_JUDGE_VIA", "antigravity") == "api":
        def call(prompt: str) -> str | None:
            from geo_strategist.providers.gemini_client import GeminiClient

            result = GeminiClient(model=os.environ.get("E13_JUDGE_MODEL")).generate(
                prompt, purpose="e13_checklist_judge")
            return result.text if result and result.ok else None

        return JudgeBackend("gemini", "gemini_api", call)
    return JudgeBackend("gemini", "antigravity", _cli_judge_call("antigravity"))


def _deepseek_backend() -> JudgeBackend:
    def call(prompt: str) -> str | None:
        from geo_strategist.providers.opencode_go_client import OpenCodeGoClient

        result = OpenCodeGoClient(model=os.environ.get("OPENCODE_GO_JUDGE_MODEL")).generate(
            prompt, purpose="e13_checklist_judge")
        return result.text if result and result.ok else None

    return JudgeBackend("deepseek", "opencode_go", call)


_BACKEND_FACTORIES: dict[str, Callable[[], JudgeBackend]] = {
    "gpt": _gpt_backend,
    "claude": _claude_backend,
    "gemini": _gemini_backend,
    "deepseek": _deepseek_backend,
}


def default_judge_backends() -> list[JudgeBackend]:
    """The default judge panel (env-configured; ``gpt,claude,gemini,deepseek``
    unless ``E13_CHECKLIST_JUDGES`` overrides it — e.g. ``gpt`` alone for a
    single-judge re-evaluation run). Add a new backend by adding one factory
    to ``_BACKEND_FACTORIES`` and its key to ``E13_CHECKLIST_JUDGES``."""

    names = [n.strip() for n in os.environ.get(
        "E13_CHECKLIST_JUDGES", "gpt,claude,gemini,deepseek").split(",") if n.strip()]
    return [_BACKEND_FACTORIES[name]() for name in names if name in _BACKEND_FACTORIES]


def run_checklist_panel(
    context: dict[str, Any],
    backends: list[JudgeBackend] | None = None,
) -> PanelResult:
    """Run the 25-question checklist against one condition's sanitized final
    report text (``context["report_text"]``), once per judge backend. A
    backend that is unavailable, times out, or returns unparseable output is
    recorded as ``"unavailable"`` and contributes no answers — it never
    raises. The caller is responsible for ensuring ``report_text`` is
    already anonymized/sanitized and for never calling this with a
    structured condition record in its place."""

    backends = backends if backends is not None else default_judge_backends()
    result = PanelResult(condition_group=str(context.get("condition_group") or "unknown"))
    report_text = str(context.get("report_text") or "")
    prompt = _prompt_for_panel(context)
    for backend in backends:
        try:
            raw = backend.call(prompt)
        except Exception as exc:  # a judge backend must never crash the panel
            result.judge_status[backend.name] = f"error:{type(exc).__name__}"
            continue
        if raw is None:
            result.judge_status[backend.name] = "unavailable"
            continue
        parsed = _parse_panel_response(raw, report_text)
        if parsed is None:
            result.judge_status[backend.name] = "unparseable"
            continue
        result.judge_status[backend.name] = "ok"
        result.answers_by_judge[backend.name] = parsed
    return result


def aggregate_checklist(panel_result: PanelResult) -> list[AggregatedChecklistRow]:
    """Per question: every judge's answer (with evidence) plus the
    aggregated yes-ratio. The configured panel size is the denominator;
    ``available_judge_count`` is reported separately for availability. A
    "yes" that fails evidence validation is recorded as
    ``"yes_invalid_evidence"`` and does count toward the denominator, but
    contributes zero to ``yes_count`` — the same as an honest "no"."""

    rows: list[AggregatedChecklistRow] = []
    for question in LLM_CHECKLIST_QUESTIONS:
        judges: dict[str, str] = {}
        evidence: dict[str, dict[str, str]] = {}
        yes_count = 0
        available = 0
        for judge_name, answers in panel_result.answers_by_judge.items():
            answer = answers.get(question.question_id)
            if answer is None:
                judges[judge_name] = "unavailable"
                continue
            available += 1
            evidence[judge_name] = {
                "evidence_location": answer.evidence_location,
                "evidence_excerpt": answer.evidence_excerpt,
                "rationale": answer.rationale,
            }
            if answer.answer == "yes" and answer.evidence_valid:
                judges[judge_name] = "yes"
                yes_count += 1
            elif answer.answer == "yes":
                judges[judge_name] = "yes_invalid_evidence"
            else:
                judges[judge_name] = "no"
        for judge_name, status in panel_result.judge_status.items():
            if judge_name not in judges:
                judges[judge_name] = "unavailable"
        rows.append(AggregatedChecklistRow(
            question_id=question.question_id,
            category=question.category,
            text=question.text,
            judges=judges,
            evidence=evidence,
            yes_count=yes_count,
            available_judge_count=available,
                yes_ratio=(round(yes_count / len(panel_result.judge_status), 4)
                           if panel_result.judge_status else None),
        ))
    return rows


def checklist_panel_summary(
    rows: list[AggregatedChecklistRow],
    judge_status: dict[str, str] | None = None,
) -> dict[str, Any]:
    scored = [r for r in rows if r.yes_ratio is not None]
    by_category: dict[str, list[float]] = {}
    for row in scored:
        by_category.setdefault(row.category, []).append(row.yes_ratio)
    return {
        "total_questions": len(rows),
        "scored_questions": len(scored),
        "overall_mean_yes_ratio": round(sum(r.yes_ratio for r in scored) / len(scored), 4) if scored else None,
        "mean_yes_ratio_by_category": {
            category: round(sum(values) / len(values), 4)
            for category, values in by_category.items()
        },
        "judge_availability": dict(judge_status or {}),
    }
