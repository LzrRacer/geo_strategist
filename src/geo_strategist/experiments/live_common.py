"""Shared infrastructure for live-agent conditions (C1/C4, C13-C14).

Grounding contract: LLMs may only rank candidates that exist in
``candidate_actions.jsonl`` (validated by candidate_id), facility targets are
only attached from source-traceable facility records, and every branch/search
condition is steered and scored by the *external* objective metrics computed
from the deterministic evaluation-model components — never by hard-coded
rankings and never by unverifiable LLM assertions. Free-text LLM rationale is
kept, but always labeled as model output.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from geo_strategist.experiments.branch_objectives import (
    BranchObjective,
    objective_weights,
)
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DEFAULT_WEIGHTS,
    DataBundle,
    proposal_for_candidate,
    rank_candidates,
)
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _write_json, _write_jsonl
from geo_strategist.providers.base import CallLedger, ChatResult, redact_secrets

# ---------------------------------------------------------------------------
# JSON extraction from LLM output
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json_block(text: str) -> Any | None:
    """Best-effort extraction of the first JSON object/array in LLM text."""

    if not text:
        return None
    candidates: list[str] = []
    candidates.extend(match.strip() for match in _FENCE_RE.findall(text))
    stripped = text.strip()
    candidates.append(stripped)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            candidates.append(stripped[start:end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


# ---------------------------------------------------------------------------
# Data context for prompts
# ---------------------------------------------------------------------------

def candidate_component_rows(data: DataBundle) -> list[dict[str, Any]]:
    """All candidates with their evaluation-model components (real data)."""

    return [
        {
            "candidate_id": row["candidate_id"],
            "prefecture": row["prefecture"],
            "municipality": row["municipality"],
            "action": row["action_type"],
            "components": row["score_components"],
        }
        for row in rank_candidates(data, DEFAULT_WEIGHTS)
    ]


def data_context_json(data: DataBundle, *, max_candidates: int | None = None) -> str:
    rows = candidate_component_rows(data)
    if max_candidates:
        rows = rows[:max_candidates]
    context = {
        "study_area": "tokyo_aichi_osaka",
        "candidate_count": len(rows),
        "component_glossary": {
            "demand": "healthcare demand pressure (census-derived, 0-1)",
            "aging": "population aging pressure (census-derived, 0-1)",
            "supply_shortage": "supply gap for build / density for reorganize (0-1)",
            "financial": "prefecture-level payback plausibility (model_estimate, 0-1)",
            "land": "land price availability score (MLIT, 0-1; null = not_available)",
            "demographic_risk": "population stability (1=stable, 0=declining; null = not_available)",
            "evidence_completeness": "data completeness for this municipality (0-1)",
        },
        "financial_note": "financial values are prefecture-median model estimates, not audited figures",
        "candidates": rows,
        "rules": [
            "Only rank candidates by their exact candidate_id from this list.",
            "Never invent hospital names, addresses, coordinates, parcels, or financial values.",
            "Missing components are evidence gaps to disclose, not values to guess.",
        ],
    }
    return json.dumps(context, ensure_ascii=False)


# ---------------------------------------------------------------------------
# External objective metrics (the non-LLM search signal)
# ---------------------------------------------------------------------------

def external_objective_metric(
    data: DataBundle,
    ranking_ids: list[str],
    objective: BranchObjective,
    *,
    top_k: int = 5,
) -> float:
    """Mean objective-weighted composite of the slate head, availability-weighted."""

    weights = objective_weights(objective, DEFAULT_WEIGHTS)
    by_id = {row["candidate_id"]: row for row in rank_candidates(data, weights)}
    head = [by_id[cid] for cid in ranking_ids[:top_k] if cid in by_id]
    if not head:
        return 0.0
    return round(
        mean(r["composite_score"] for r in head) * mean(r["component_availability"] for r in head),
        6,
    )


def slate_metrics_all_objectives(
    data: DataBundle,
    ranking_ids: list[str],
    objectives: list[BranchObjective],
    *,
    top_k: int = 5,
) -> dict[str, float]:
    return {
        objective.slug: external_objective_metric(data, ranking_ids, objective, top_k=top_k)
        for objective in objectives
    }


def slate_overlap(ids_a: list[str], ids_b: list[str], *, top_k: int = 5) -> float:
    """Jaccard overlap of two slate heads (novelty = 1 - overlap)."""

    a, b = set(ids_a[:top_k]), set(ids_b[:top_k])
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


# ---------------------------------------------------------------------------
# Validation of LLM rankings against the real candidate universe
# ---------------------------------------------------------------------------

def validate_llm_ranking(
    rows: Any,
    data: DataBundle,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep rows whose candidate_id exists; report the rejected ids.

    Accepts a list of dicts (``candidate_id`` [+ optional ``rationale``]) or a
    list of bare id strings. Duplicates keep first occurrence.
    """

    known = {candidate["candidate_id"] for candidate in data.candidates}
    valid: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen: set[str] = set()
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, str):
            row = {"candidate_id": row}
        if not isinstance(row, dict):
            continue
        cid = str(row.get("candidate_id") or "")
        if cid in seen:
            continue
        if cid not in known:
            if cid:
                rejected.append(cid)
            continue
        seen.add(cid)
        valid.append({"candidate_id": cid, "rationale": str(row.get("rationale") or "")[:2000]})
    return valid, rejected


def build_live_proposals(
    ranked_rows: list[dict[str, Any]],
    data: DataBundle,
    spec: ConditionSpec,
    *,
    top_k: int = 5,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Turn a validated LLM slate into evidence-graded proposal records."""

    weights = weights or DEFAULT_WEIGHTS
    by_id = {row["candidate_id"]: row for row in rank_candidates(data, weights)}
    proposals: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked_rows[:top_k], start=1):
        base = dict(by_id[item["candidate_id"]])
        base["rank"] = rank
        proposal = proposal_for_candidate(
            base, data,
            condition_id=spec.condition_id,
            condition_group=spec.group,
        )
        if item.get("rationale"):
            proposal["llm_rationale"] = item["rationale"]
            proposal.setdefault("evidence_grades", {})["llm_rationale"] = "model_estimate"
        proposals.append(proposal)
    return proposals


# ---------------------------------------------------------------------------
# Live run context: artifact dir, ledger, redacted call trace
# ---------------------------------------------------------------------------

class LiveRunContext:
    def __init__(self, run_dir: Path, *, condition_group: str) -> None:
        self.run_dir = Path(run_dir)
        self.condition_group = condition_group
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = CallLedger()
        self._trace_lock = threading.Lock()
        self._trace_path = self.run_dir / "model_call_trace.jsonl"
        self._reasoning_path = self.run_dir / "reasoning_traces.jsonl"
        self._call_index = 0

    def record_call(self, purpose: str, prompt: str, result: ChatResult) -> None:
        """Persist a redacted call trace row; reasoning goes to its own file
        (trace artifact only — never pasted into final reports)."""

        self.ledger.record(result, purpose=purpose)
        with self._trace_lock:
            self._call_index += 1
            index = self._call_index
            row = {
                "call_index": index,
                "condition_group": self.condition_group,
                "purpose": purpose,
                "provider": result.provider,
                "model": result.model,
                "status": "ok" if result.ok else result.error_class,
                "error_detail": result.error_detail,
                "finish_reason": result.finish_reason,
                "request_count": result.request_count,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "reasoning_tokens": result.reasoning_tokens,
                "latency_seconds": result.latency_seconds,
                "prompt_excerpt": redact_secrets(prompt[:600]),
                "response_excerpt": redact_secrets(result.text[:600]),
            }
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if result.reasoning_text:
                with self._reasoning_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({
                        "call_index": index,
                        "purpose": purpose,
                        "model": result.model,
                        "reasoning_content": redact_secrets(result.reasoning_text[:20000]),
                    }, ensure_ascii=False) + "\n")

    @property
    def trace_path(self) -> Path:
        return self._trace_path


# ---------------------------------------------------------------------------
# Result surface shared by every live condition runner
# ---------------------------------------------------------------------------

@dataclass
class LiveConditionResult:
    spec: ConditionSpec
    execution_mode: str
    comparable_for_e13: bool
    exclusion_reason: str | None
    proposals: list[dict[str, Any]] = field(default_factory=list)
    ranked_rows: list[dict[str, Any]] = field(default_factory=list)
    branch_results: list[dict[str, Any]] = field(default_factory=list)
    search_nodes: list[dict[str, Any]] = field(default_factory=list)
    generated_code_stats: dict[str, Any] = field(default_factory=dict)
    model_call_summary: dict[str, Any] = field(default_factory=dict)
    review_rows: list[dict[str, Any]] = field(default_factory=list)
    robustness_results: list[dict[str, Any]] = field(default_factory=list)
    due_diligence: list[str] = field(default_factory=list)
    narrative_sections: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    failure_notes: list[str] = field(default_factory=list)
    steps_run: int = 0
    candidate_review_packets: list[dict[str, Any]] = field(default_factory=list)
    candidate_review_threads: list[dict[str, Any]] = field(default_factory=list)
    candidate_deliberation_summary: dict[str, Any] = field(default_factory=dict)
    analysis_bundle: Any | None = None
    # Structured final-slate synthesis outcome (currently produced by the
    # AI-Scientist loop; see ai_scientist_synthesis.SynthesisOutcome.to_bundle_dict).
    # Empty for runners that have no separate synthesis step.
    synthesis: dict[str, Any] = field(default_factory=dict)
    # Per-candidate decision status/next-steps/reversal-condition content, in
    # the exact DecisionAnalysisBundle.final_decision_rows / reversal_conditions
    # shape, when a runner's own synthesis step produced it (currently
    # AI-Scientist's structured synthesis). Empty for runners that don't.
    final_decision_rows: list[dict[str, Any]] = field(default_factory=list)
    reversal_conditions: list[dict[str, Any]] = field(default_factory=list)

    def finalize_artifacts(self, context: LiveRunContext) -> None:
        self.model_call_summary = context.ledger.summary()
        _write_json(context.run_dir / "model_call_summary.json", self.model_call_summary)
        if self.branch_results:
            _write_jsonl(context.run_dir / "branch_results.jsonl", self.branch_results)
            self.artifacts["branch_results"] = str(context.run_dir / "branch_results.jsonl")
        if self.ranked_rows:
            _write_jsonl(context.run_dir / "ranked_candidates.jsonl", self.ranked_rows)
            self.artifacts["ranked_candidates"] = str(context.run_dir / "ranked_candidates.jsonl")
        self.artifacts.setdefault("model_call_trace", str(context.trace_path))
        self.artifacts.setdefault("model_call_summary", str(context.run_dir / "model_call_summary.json"))


def failure_result(
    spec: ConditionSpec,
    error_class: str,
    detail: str,
) -> LiveConditionResult:
    """A non-comparable failure record (never a silent deterministic swap)."""

    mode = error_class if error_class in (
        "live_auth_failed", "live_rate_limited", "output_truncated",
        "waiting_for_manual_harness",
    ) else "live_error"
    return LiveConditionResult(
        spec=spec,
        execution_mode=mode,
        comparable_for_e13=False,
        exclusion_reason=f"{mode}: {detail}",
        failure_notes=[detail],
    )


# ---------------------------------------------------------------------------
# Evidence / fabrication guard on LLM narrative text
# ---------------------------------------------------------------------------

_NUMERIC_CLAIM_RE = re.compile(r"[¥$€]\s?[\d,]+|\d+\s*(?:億|万)円")


def narrative_fabrication_flags(text: str, data: DataBundle) -> list[str]:
    """Flag currency-amount claims in narrative text (financial inputs are
    prefecture-median model estimates, so absolute currency claims are
    unsupported)."""

    flags: list[str] = []
    for match in _NUMERIC_CLAIM_RE.findall(text or ""):
        flags.append(f"unsupported_currency_claim:{match.strip()}")
    return flags[:20]
