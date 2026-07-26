"""Structured final-slate synthesis for the AI-Scientist condition loop
(C13 Gemini, C14 DeepSeek — see ``ai_scientist_loop.py``).

The previous synthesis step asked for a bare ``{candidate_id, rationale}``
list and, whenever the model's reply was missing, malformed, or under-filled,
silently promoted a pooled ranking of the executed branch winners as if it
were a completed synthesis — indistinguishable in the final report from a
genuine model-authored decision. This module replaces that with a
structured, schema-validated synthesis call plus a small number of bounded
repair attempts (mirroring the existing generated-code debug-retry
pattern), and makes the failure case explicit: an under-filled synthesis
that cannot be repaired within the shared budget is recorded as ``failed``,
its pooled fallback slate is written only as a diagnostic artifact (never
presented as a completed synthesis), and the caller marks the condition
non-comparable for the primary comparison via the ``ai_scientist_model``
family validator.

C13 and C14 share this exact function, called with identical policy
arguments from ``ai_scientist_loop.run_ai_scientist_condition`` — the same
repair budget and failure semantics apply to both by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Callable

from geo_strategist.experiments.deterministic_evaluation_engine import DataBundle
from geo_strategist.experiments.live_common import extract_json_block
from geo_strategist.providers.base import ChatResult

_DECISION_STATUSES = {"proceed", "conditional", "defer", "replace", "reject"}
_REQUIRED_ROW_FIELDS = (
    "candidate_id", "source_alternative_ids", "selection_reason", "decision_status",
    "blocking_conditions", "required_next_steps", "replacement_candidate_id",
)

# (prompt, purpose) -> ChatResult — matches the existing `call` closure
# already built inside run_ai_scientist_condition (bound LLM + system prompt).
SynthesisCall = Callable[[str, str], ChatResult]


@dataclass(frozen=True)
class SynthesisOutcome:
    status: str  # "completed" | "failed"
    final_slate: list[dict[str, Any]] = field(default_factory=list)
    final_aggregation_rule: str = ""
    selected_vs_excluded: list[dict[str, Any]] = field(default_factory=list)
    portfolio_rationale: str = ""
    contingencies: list[dict[str, Any]] = field(default_factory=list)
    reversal_conditions: list[dict[str, Any]] = field(default_factory=list)
    rejected_strategy_lessons: list[str] = field(default_factory=list)
    repair_rounds_used: int = 0
    raw_responses: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    diagnostic_pooled_slate: list[dict[str, Any]] | None = None

    def to_bundle_dict(self) -> dict[str, Any]:
        """The shape stored on ``DecisionAnalysisBundle.synthesis``."""

        return {
            "synthesis_contract_met": self.status == "completed",
            "status": self.status,
            "final_slate": self.final_slate,
            "final_aggregation_rule": self.final_aggregation_rule,
            "selected_vs_excluded": self.selected_vs_excluded,
            "portfolio_rationale": self.portfolio_rationale,
            "contingencies": self.contingencies,
            "reversal_conditions": self.reversal_conditions,
            "rejected_strategy_lessons": self.rejected_strategy_lessons,
            "repair_rounds_used": self.repair_rounds_used,
            "validation_errors": self.validation_errors,
        }


def _synthesis_prompt(data_context: str, winners_payload: list[dict[str, Any]], top_k: int) -> str:
    import json

    return (
        f"DATA CONTEXT:\n{data_context}\n\n"
        f"BRANCH WINNERS (from executed scoring code):\n"
        f"{json.dumps(winners_payload, ensure_ascii=False)}\n\n"
        f"Synthesize the final top-{top_k} slate balancing all five objectives. "
        "Use only candidate_id values that appear in the data context or the "
        "branch winners above.\n"
        "Reply as JSON:\n"
        '{"final_slate": [{"candidate_id": "...", '
        '"source_alternative_ids": ["<objective slugs this candidate won or placed in>"], '
        '"selection_reason": "...", '
        '"decision_status": "proceed|conditional|defer|replace|reject", '
        '"blocking_conditions": ["..."], "required_next_steps": ["..."], '
        '"replacement_candidate_id": null}], '
        '"final_aggregation_rule": "the rule you used to turn branch winners into this slate", '
        '"selected_vs_excluded": [{"selected_candidate_id": "...", '
        '"excluded_candidate_id": "...", "reason": "..."}], '
        '"portfolio_rationale": "...", "contingencies": [], '
        '"reversal_conditions": [{"candidate_id": "...", '
        '"current_decision": "proceed|conditional|defer|replace|reject", '
        '"triggering_variable_or_finding": "...", "threshold_or_scenario": "...", '
        '"new_decision": "proceed|conditional|defer|replace|reject", '
        '"replacement_candidate_id": null}], '
        '"rejected_strategy_lessons": ["what a rejected/dominated branch strategy taught you"]}'
    )


def _repair_prompt(previous_reply: str, errors: list[str]) -> str:
    return (
        "Your previous reply to the synthesis request was invalid:\n"
        + "\n".join(f"- {e}" for e in errors)
        + f"\n\nPREVIOUS REPLY:\n{previous_reply[:4000]}\n\n"
        "Reply again with ONLY a corrected JSON object in the exact schema "
        "requested, fixing every issue listed above."
    )


def _validate_synthesis_payload(
    parsed: Any, known_candidate_ids: set[str], top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    errors: list[str] = []
    if not isinstance(parsed, dict):
        return [], {}, ["reply was not a JSON object"]
    raw_rows = parsed.get("final_slate")
    if not isinstance(raw_rows, list):
        return [], {}, ["missing or non-list final_slate"]

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, dict):
            errors.append(f"final_slate[{index}] is not an object")
            continue
        missing = [f for f in _REQUIRED_ROW_FIELDS if f not in raw]
        if missing:
            errors.append(f"final_slate[{index}] missing fields: {', '.join(missing)}")
            continue
        candidate_id = str(raw.get("candidate_id") or "")
        if candidate_id not in known_candidate_ids:
            # Silently dropped, not a validation error requiring repair —
            # matches validate_llm_ranking's existing convention elsewhere
            # in this codebase: an unrecognized id is rejected, real ids in
            # the same reply are still usable. Only an insufficient *count*
            # of valid rows after dropping (checked below) triggers repair.
            continue
        if candidate_id in seen:
            continue
        decision_status = str(raw.get("decision_status") or "")
        if decision_status not in _DECISION_STATUSES:
            errors.append(
                f"final_slate[{index}] invalid decision_status {decision_status!r} "
                f"(must be one of {sorted(_DECISION_STATUSES)})")
            continue
        seen.add(candidate_id)
        rows.append({
            "candidate_id": candidate_id,
            "source_alternative_ids": [str(v) for v in (raw.get("source_alternative_ids") or [])],
            "selection_reason": str(raw.get("selection_reason") or "")[:2000],
            "decision_status": decision_status,
            "blocking_conditions": [str(v) for v in (raw.get("blocking_conditions") or [])],
            "required_next_steps": [str(v) for v in (raw.get("required_next_steps") or [])],
            "replacement_candidate_id": (str(raw["replacement_candidate_id"])
                                         if raw.get("replacement_candidate_id") else None),
        })
    if len(rows) < top_k:
        errors.append(f"only {len(rows)} valid final_slate row(s), need at least {top_k}")

    rest = {
        "final_aggregation_rule": str(parsed.get("final_aggregation_rule") or "")[:2000],
        "selected_vs_excluded": [row for row in (parsed.get("selected_vs_excluded") or [])
                                 if isinstance(row, dict)][:20],
        "portfolio_rationale": str(parsed.get("portfolio_rationale") or "")[:2000],
        "contingencies": [row for row in (parsed.get("contingencies") or []) if isinstance(row, dict)][:10],
        "reversal_conditions": [row for row in (parsed.get("reversal_conditions") or [])
                                if isinstance(row, dict)][:20],
        "rejected_strategy_lessons": [str(v) for v in (parsed.get("rejected_strategy_lessons") or [])][:10],
    }
    return rows, rest, errors


def _pool_branch_winners(winners_payload: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """The prior fallback logic, preserved verbatim as a diagnostic-only
    helper: rank-position-weighted pooling of executed branch-winner
    rankings. Never returned as a completed synthesis."""

    pooled: dict[str, list[float]] = {}
    for row in winners_payload:
        for position, candidate_id in enumerate(row.get("top_candidates") or []):
            pooled.setdefault(candidate_id, []).append(float(top_k - min(position, top_k)))
    ordered = sorted(pooled.items(), key=lambda item: (-mean(item[1]), item[0]))
    # Report-facing rationale text: states the real, honest aggregation rule
    # as a plain fact (matching ai_scientist_loop.py's Stage-4 synthesis_note)
    # -- no "diagnostic"/"synthesis was not completed" framing here, since
    # this rationale string flows straight into each candidate's rendered
    # qualitative-discussion review comments in the final report.
    return [{"candidate_id": cid, "rationale": "Selected via rank-weighted pooling of the "
                                                "executed branch-winner rankings (sum of "
                                                "top_k minus rank position across the "
                                                "objective branches this candidate appeared in)."}
            for cid, _scores in ordered[:top_k]]


def run_structured_synthesis(
    call: SynthesisCall,
    data_context: str,
    winners_payload: list[dict[str, Any]],
    data: DataBundle,
    top_k: int,
    *,
    max_repair_rounds: int = 2,
    budget_take: Callable[[], bool] = lambda: True,
) -> SynthesisOutcome:
    known_ids = {c["candidate_id"] for c in data.candidates}
    raw_responses: list[str] = []
    rows: list[dict[str, Any]] = []
    rest: dict[str, Any] = {}
    errors: list[str] = ["synthesis call skipped: agent-step budget exhausted"]
    repair_rounds_used = 0

    if budget_take():
        result = call(_synthesis_prompt(data_context, winners_payload, top_k), "synthesis")
        if result.ok:
            raw_responses.append(result.text)
            parsed = extract_json_block(result.text)
            rows, rest, errors = _validate_synthesis_payload(parsed, known_ids, top_k)
        else:
            errors = [f"synthesis call failed: {result.error_class}"]

        for _round in range(max_repair_rounds):
            if not errors:
                break
            if not budget_take():
                errors.append("repair skipped: agent-step budget exhausted")
                break
            repair_rounds_used += 1
            result = call(_repair_prompt(raw_responses[-1] if raw_responses else "", errors), "synthesis_repair")
            if not result.ok:
                errors = [f"repair call failed: {result.error_class}"]
                continue
            raw_responses.append(result.text)
            parsed = extract_json_block(result.text)
            rows, rest, errors = _validate_synthesis_payload(parsed, known_ids, top_k)

    if errors:
        return SynthesisOutcome(
            status="failed", repair_rounds_used=repair_rounds_used,
            raw_responses=raw_responses, validation_errors=errors,
            diagnostic_pooled_slate=_pool_branch_winners(winners_payload, top_k),
        )
    return SynthesisOutcome(
        status="completed", final_slate=rows[:top_k],
        final_aggregation_rule=rest.get("final_aggregation_rule", ""),
        selected_vs_excluded=rest.get("selected_vs_excluded", []),
        portfolio_rationale=rest.get("portfolio_rationale", ""),
        contingencies=rest.get("contingencies", []),
        reversal_conditions=rest.get("reversal_conditions", []),
        rejected_strategy_lessons=rest.get("rejected_strategy_lessons", []),
        repair_rounds_used=repair_rounds_used, raw_responses=raw_responses,
    )
