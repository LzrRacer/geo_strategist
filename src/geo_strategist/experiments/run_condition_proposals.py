"""Condition-proposal orchestrator for the canonical C0-C13 track.

Dispatches each requested condition to its live runner (or the deterministic
baseline for C0, or a manual-harness handoff for C2/C3/C5/C6/C7/C8), writes
one proposal report + record per condition, and finishes with the E13
cross-condition judge.

Live-agent rules:

- ``require_live_agents`` (default on): a failed live condition is recorded
  with its true failure mode (``live_auth_failed`` / ``live_rate_limited`` /
  ``output_truncated`` / ``live_error`` / ``waiting_for_manual_harness``) and
  ``comparable_for_e13 = false`` — never silently replaced by a
  deterministic ranking.
- ``disable_deterministic_fallback_for_comparison`` (default on): even the
  debug-only deterministic fallback report is skipped; when off, a fallback
  report may be written but is always marked ``deterministic_fallback`` and
  excluded from comparison.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_strategist.experiments.ai_scientist_loop import (
    LlmCall,
    run_ai_scientist_condition,
)
from geo_strategist.experiments.branch_objectives import (
    BranchObjective,
    parse_branch_objectives,
)
from geo_strategist.experiments.vanilla_llm import run_vanilla_condition
from geo_strategist.experiments.c11_shinka_evolution import (
    MultiModelCall,
    run_c11_evolution_condition,
)
from geo_strategist.experiments.c12_ab_mcts import run_c12_ab_mcts_condition
from geo_strategist.experiments.c13_fugu_router import run_c13_router_condition
from geo_strategist.experiments.candidate_deliberation_runtime import (
    condition_supports_candidate_deliberation,
    run_candidate_deliberation,
)
from geo_strategist.experiments.candidate_qualitative_deliberation import (
    run_candidate_qualitative_deliberation,
)
from geo_strategist.experiments.condition_output_contract import (
    condition_record_evidence_passed,
    make_condition_record,
    merge_condition_record,
)
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DataBundle,
    build_deterministic_outcome,
    load_data_bundle,
)
from geo_strategist.experiments.condition_registry import (
    COMPARABLE_EXECUTION_MODES,
    CONDITION_ORDER,
    ConditionSpec,
    build_condition_registry,
    c1_c9_strict_comparison_status,
    c2_c6_strict_comparison_status,
    c3_c7_strict_comparison_status,
    c4_c8_c10_strict_comparison_status,
)
from geo_strategist.experiments.condition_utils import (
    _read_json,
    _read_jsonl,
    _rel,
    _write_json,
    _write_jsonl,
)
from geo_strategist.experiments.live_common import LiveConditionResult, failure_result
from geo_strategist.experiments.live_report import write_condition_report
from geo_strategist.harnesses.agentic_runner import (
    execute_agentic_skills_harness,
    prepare_agentic_skills_harness,
)
from geo_strategist.harnesses.prompts import write_manual_harness_prompts
from geo_strategist.providers.gemini_client import GeminiClient
from geo_strategist.providers.opencode_go_client import OpenCodeGoClient
from geo_strategist.providers.base import RetryPolicy

# C11/C12 extend the base reviewer-only candidate deliberation with the
# agent-qualitative-assessment-first layer (see
# candidate_qualitative_deliberation.py); C13 runs the equivalent steps
# in-line as routed tasks (c13_fugu_router.py) so they show up in its own
# routing log, so it is excluded from this post-hoc dispatch entirely.
_QUALITATIVE_DELIBERATION_GROUPS: frozenset[str] = frozenset({"C11", "C12"})
_INLINE_QUALITATIVE_DELIBERATION_GROUPS: frozenset[str] = frozenset({"C13"})


@dataclass(frozen=True)
class ConditionProposalsResult:
    run_id: str
    output_dir: Path
    conditions_run: list[str]
    report_paths: dict[str, str]
    e13_report_path: str | None
    summary_path: str
    manifest_path: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


_AI_SCIENTIST_SHARED_BUDGETS = {
    "num_drafts": 10,
    "agent_steps": 40,
    "variants_per_branch": 6,
    "max_debug_depth": 4,
    "concurrency": 4,
}


def _gemini_call(model: str | None, *, retry: RetryPolicy | None = None) -> LlmCall:
    client = GeminiClient(model=model, retry=retry)

    def call(prompt: str, system: str | None, purpose: str):
        return client.generate(prompt, system=system, purpose=purpose)

    return call


def _opencode_call(model: str | None, *, retry: RetryPolicy | None = None) -> LlmCall:
    client = OpenCodeGoClient(model=model, retry=retry)

    def call(prompt: str, system: str | None, purpose: str):
        max_tokens = None
        if purpose in ("report", "write_final_condition_proposal", "synthesis"):
            max_tokens = _env_int("OPENCODE_GO_REPORT_MAX_TOKENS", 8192)
        return client.generate(prompt, system=system, purpose=purpose, max_tokens=max_tokens)

    return call


def _opencode_multimodel() -> MultiModelCall:
    client = OpenCodeGoClient()
    # Reasoning-heavy models need headroom beyond the task default before any
    # content appears; truncation retries escalate from here.
    base_tokens = _env_int("OPENCODE_GO_MULTIMODEL_MAX_TOKENS", 8192)

    def call(prompt: str, system: str | None, purpose: str, model: str):
        return client.generate(prompt, system=system, purpose=purpose, model=model,
                               max_tokens=base_tokens)

    return call


def _llm_for_spec(spec: ConditionSpec) -> LlmCall:
    if spec.provider == "gemini":
        return _gemini_call(spec.model)
    return _opencode_call(spec.model)


def _candidate_reviewers() -> list[str]:
    raw = os.environ.get("CANDIDATE_REVIEW_REVIEWERS", "healthcare_strategy")
    reviewers = [part.strip() for part in raw.split(",") if part.strip()]
    return reviewers or ["healthcare_strategy"]


def _deliberation_llm_call(spec: ConditionSpec) -> LlmCall | None:
    """LLM used for the candidate-deliberation reviewer/author roles.

    Defaults to the same direct-API provider as the condition to avoid
    spending Gemini quota on non-Gemini conditions. Override with
    ``CANDIDATE_REVIEW_PROVIDER``/``CANDIDATE_REVIEW_MODEL`` when a fixed
    reviewer provider is intentional.
    """

    provider = os.environ.get("CANDIDATE_REVIEW_PROVIDER", "same_provider")
    model = os.environ.get("CANDIDATE_REVIEW_MODEL")
    retry = RetryPolicy(
        max_attempts=_env_int("CANDIDATE_REVIEW_RETRY_ATTEMPTS", 1),
        base_delay=_env_float("CANDIDATE_REVIEW_RETRY_BASE_DELAY", 10.0),
        max_delay=_env_float("CANDIDATE_REVIEW_RETRY_MAX_DELAY", 65.0),
    )
    if provider == "same_provider":
        if spec.provider == "gemini":
            provider = "gemini"
            model = model or spec.model
        elif spec.provider == "opencode_go":
            provider = "opencode_go"
            model = model or spec.model
        else:
            return None
    if provider in {"none", "disabled", "off"}:
        return None
    if provider == "opencode_go":
        client = OpenCodeGoClient(model=model, retry=retry)

        def opencode_call(prompt: str, system: str | None, purpose: str):
            return client.generate(prompt, system=system, purpose=purpose)

        return opencode_call
    client = GeminiClient(model=model, retry=retry)

    def gemini_call(prompt: str, system: str | None, purpose: str):
        return client.generate(prompt, system=system, purpose=purpose)

    return gemini_call


def _c0_result(repo_root: Path, spec: ConditionSpec, data: DataBundle,
               *, top_k: int, run_dir: Path) -> LiveConditionResult:
    outcome = build_deterministic_outcome(
        repo_root, condition_id=spec.condition_id, condition_group=spec.group,
        top_k=top_k, review_rounds=1, data=data,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(run_dir / "ranked_candidates.jsonl", outcome.candidates_ranked[:50])
    return LiveConditionResult(
        spec=spec,
        execution_mode="deterministic_baseline",
        comparable_for_e13=True,
        exclusion_reason=None,
        proposals=outcome.proposals,
        ranked_rows=outcome.candidates_ranked[:top_k],
        review_rows=outcome.review_rows,
        due_diligence=outcome.due_diligence_items,
        narrative_sections={
            "method_summary": (
                "Deterministic weighted composite over the evaluation-model "
                f"components with fixed default weights {outcome.weights_used}; "
                "no LLM calls."),
        },
        artifacts={"ranked_candidates": str(run_dir / "ranked_candidates.jsonl")},
    )


def _manual_harness_result(
    repo_root: Path,
    spec: ConditionSpec,
    data: DataBundle,
    out_dir: Path,
    *,
    top_k: int,
    manual_result_path: Path | None = None,
    strict_skill_trace: bool = False,
    extra_artifacts: dict[str, str] | None = None,
    method_prefix: str = "Manual",
    auto_harness: bool = False,
) -> LiveConditionResult:
    """Ingest a completed manual-harness run, or pause the condition.

    The manual harness returns ``manual_result.json`` (schema documented in the
    handoff prompt). Its slate is validated against the real candidate
    universe, rebuilt through the shared evidence-graded proposal builder, and
    the harness's per-candidate qualitative text is attached as
    ``model_estimate``-graded reviewer input — facility names in manual prose
    are never promoted to verified fields.
    """

    candidates = [
        manual_result_path,
        out_dir / "runs" / spec.padded_id / "manual_result.json",
        repo_root / "outputs/condition_proposals/manual_harness" / spec.group / "manual_result.json",
    ]
    found = next((p for p in candidates if p and Path(p).exists()), None)
    if found is not None:
        payload = _read_json(Path(found))
        from geo_strategist.experiments.live_common import validate_llm_ranking
        from geo_strategist.experiments.live_common import build_live_proposals

        ranked, rejected = validate_llm_ranking(payload.get("ranked_candidates") or [], data)
        if not ranked:
            return failure_result(
                spec, "live_error",
                f"manual result {found} contained no valid candidate_ids "
                f"(rejected: {rejected[:5]})")
        proposals = build_live_proposals(ranked, data, spec, top_k=top_k)
        manual_discussions = {
            str(row.get("candidate_id")): row.get("qualitative_discussion")
            for row in payload.get("ranked_candidates") or []
            if isinstance(row, dict)
        }
        for proposal in proposals:
            manual_text = manual_discussions.get(proposal.get("candidate_id"))
            if manual_text:
                proposal["manual_harness_discussion"] = manual_text
                proposal.setdefault("evidence_grades", {})["manual_harness_discussion"] = "model_estimate"
        due_diligence = [item for p in proposals for item in p.get("required_due_diligence") or []]
        due_diligence.extend(str(c) for c in (payload.get("review_comments") or [])[:10])
        due_diligence.extend(data.data_notes)

        trace_issues: list[str] = []
        if spec.skills_unified:
            # Skills-unified conditions (C5-C8) promise the ten-skill
            # contract; non-Skills conditions may return a free-form trace.
            if strict_skill_trace:
                from geo_strategist.agent.skill_registry import (
                    validate_skill_trace_against_io,
                )

                trace_issues = validate_skill_trace_against_io(
                    payload.get("skill_trace") or [],
                    Path(found).parent,
                    manual_result_path=Path(found),
                )
            else:
                from geo_strategist.agent.skill_registry import validate_skill_trace

                trace_issues = validate_skill_trace(payload.get("skill_trace") or [])
        narrative = {
            "method_summary": f"{method_prefix} {spec.harness} harness run ingested from {found}; "
                              "slate validated against the candidate universe "
                              f"({len(rejected)} unknown id(s) rejected).",
        }
        if trace_issues:
            narrative["skill_trace_validation"] = (
                "Skill-trace contract issues: "
                + "; ".join(trace_issues[:6]))
            due_diligence.append(
                "Manual harness skill trace deviates from the Skills-unified "
                "contract: " + "; ".join(trace_issues[:3]))
        comparable = not (strict_skill_trace and trace_issues)
        artifacts = {"manual_result": str(found)}
        artifacts.update(extra_artifacts or {})
        return LiveConditionResult(
            spec=spec,
            execution_mode="live_manual_harness",
            comparable_for_e13=comparable,
            exclusion_reason=None if comparable else "invalid_skill_trace",
            proposals=proposals,
            ranked_rows=[
                {"rank": index + 1, "candidate_id": p["candidate_id"],
                 "municipality": p["municipality"], "prefecture": p["prefecture"],
                 "action_type": p["action_type"], "composite_score": p["composite_score"]}
                for index, p in enumerate(proposals)
            ],
            due_diligence=list(dict.fromkeys(due_diligence)),
            narrative_sections=narrative,
            model_call_summary=payload.get("model_call_summary") or {},
            steps_run=len(payload.get("skill_trace") or []),
            artifacts=artifacts,
        )
    prompts = write_manual_harness_prompts(repo_root, [spec.group], output_dir=out_dir)
    prompt_path = Path(prompts[spec.group])
    if auto_harness:
        from geo_strategist.harnesses.agentic_runner import AgenticHarnessLaunch

        run_dir = out_dir / "runs" / spec.padded_id
        run_dir.mkdir(parents=True, exist_ok=True)
        launch = AgenticHarnessLaunch(
            run_dir=run_dir,
            launcher_prompt_path=prompt_path,
            validation_issues=[],
        )
        execution = execute_agentic_skills_harness(repo_root, spec, launch)
        execution_artifacts = {
            "manual_prompt": str(prompt_path),
            "agent_stdout": str(execution.stdout_path),
            "agent_stderr": str(execution.stderr_path),
            "agent_execution": str(execution.metadata_path),
        }
        if execution.status == "succeeded":
            result = _manual_harness_result(
                repo_root, spec, data, out_dir, top_k=top_k,
                manual_result_path=execution.manual_result_path,
                strict_skill_trace=strict_skill_trace,
                extra_artifacts=execution_artifacts,
                method_prefix="Automated",
            )
            result.narrative_sections["agent_execution"] = (
                f"Non-interactive {spec.harness} adapter completed with "
                f"return code {execution.returncode}.")
            return result
        result = failure_result(
            spec,
            "live_error" if execution.status != "not_supported" else "waiting_for_manual_harness",
            f"{execution.status}: {execution.detail or 'adapter did not produce manual_result.json'}",
        )
        result.exclusion_reason = execution.status
        result.artifacts.update(execution_artifacts)
        return result

    result = failure_result(
        spec, "waiting_for_manual_harness",
        f"{spec.harness} requires an interactive subscription session; run the "
        f"handoff prompt at {prompt_path} and re-run this condition "
        "with --manual-result pointing at the returned manual_result.json.",
    )
    result.artifacts["manual_prompt"] = str(prompt_path)
    return result


def _agentic_skills_harness_result(
    repo_root: Path,
    spec: ConditionSpec,
    data: DataBundle,
    out_dir: Path,
    *,
    top_k: int,
    manual_result_path: Path | None = None,
    auto_agentic_harness: bool = False,
) -> LiveConditionResult:
    """Ingest or launch a C5-C8 AGENTS.md + Skill package harness run."""

    candidates = [
        manual_result_path,
        out_dir / "runs" / spec.padded_id / "manual_result.json",
        repo_root / "outputs/condition_proposals/manual_harness" / spec.group / "manual_result.json",
    ]
    found = next((p for p in candidates if p and Path(p).exists()), None)
    if found is not None:
        return _manual_harness_result(
            repo_root, spec, data, out_dir, top_k=top_k,
            manual_result_path=Path(found), strict_skill_trace=True)

    launch = prepare_agentic_skills_harness(repo_root, spec, out_dir)
    if launch.validation_issues:
        result = failure_result(
            spec,
            "live_error",
            "AGENTS.md + Skill package validation failed: "
            + "; ".join(launch.validation_issues[:8]),
        )
        result.artifacts["launcher_prompt"] = str(launch.launcher_prompt_path)
        result.failure_notes.extend(launch.validation_issues)
        return result

    if auto_agentic_harness:
        execution = execute_agentic_skills_harness(repo_root, spec, launch)
        execution_artifacts = {
            "launcher_prompt": str(launch.launcher_prompt_path),
            "agent_stdout": str(execution.stdout_path),
            "agent_stderr": str(execution.stderr_path),
            "agent_execution": str(execution.metadata_path),
        }
        if execution.status == "succeeded":
            result = _manual_harness_result(
                repo_root, spec, data, out_dir, top_k=top_k,
                manual_result_path=execution.manual_result_path,
                strict_skill_trace=True,
                extra_artifacts=execution_artifacts,
                method_prefix="Automated",
            )
            result.narrative_sections["agent_execution"] = (
                f"Non-interactive {spec.harness} adapter completed with "
                f"return code {execution.returncode}.")
            return result
        if execution.status == "not_supported":
            result = failure_result(
                spec,
                "waiting_for_manual_harness",
                f"{spec.harness} adapter does not support non-interactive "
                f"execution; run launcher prompt at {launch.launcher_prompt_path}.",
            )
            result.artifacts.update(execution_artifacts)
            return result
        result = failure_result(
            spec,
            "live_error",
            f"{execution.status}: {execution.detail}",
        )
        result.exclusion_reason = execution.status
        result.artifacts.update(execution_artifacts)
        return result

    result = failure_result(
        spec,
        "waiting_for_manual_harness",
        f"{spec.harness} requires an interactive subscription session; run the "
        f"launcher prompt at {launch.launcher_prompt_path} and re-run this "
        "condition with --manual-result pointing at the returned manual_result.json.",
    )
    result.artifacts["launcher_prompt"] = str(launch.launcher_prompt_path)
    return result


def _dispatch_condition(
    repo_root: Path,
    spec: ConditionSpec,
    data: DataBundle,
    objectives: list[BranchObjective],
    out_dir: Path,
    *,
    top_k: int,
    max_review_rounds: int,
    c9_baseline_slate: list[str] | None,
    manual_result_path: Path | None = None,
    auto_agentic_harness: bool = False,
    enable_candidate_deliberation: bool = True,
) -> LiveConditionResult:
    run_dir = out_dir / "runs" / spec.padded_id
    if spec.runner == "c0_deterministic":
        return _c0_result(repo_root, spec, data, top_k=top_k, run_dir=run_dir)
    if spec.runner == "manual_harness":
        return _manual_harness_result(
            repo_root, spec, data, out_dir, top_k=top_k,
            manual_result_path=manual_result_path,
            auto_harness=auto_agentic_harness)
    if spec.runner == "agentic_skills_harness":
        return _agentic_skills_harness_result(
            repo_root, spec, data, out_dir, top_k=top_k,
            manual_result_path=manual_result_path,
            auto_agentic_harness=auto_agentic_harness)
    if manual_result_path is not None and spec.skills_unified:
        # Any Skills-unified condition can be driven manually; an explicit
        # --manual-result must never be silently ignored in favor of a fresh
        # automated run.
        return _manual_harness_result(
            repo_root, spec, data, out_dir, top_k=top_k,
            manual_result_path=manual_result_path)
    if spec.runner == "vanilla_llm":
        return run_vanilla_condition(
            repo_root, spec, _llm_for_spec(spec), data, objectives, run_dir, top_k=top_k)
    if spec.runner == "c9_ai_scientist_gemini":
        return run_ai_scientist_condition(
            repo_root, spec, _llm_for_spec(spec), data, objectives, run_dir,
            top_k=top_k, max_review_rounds=max_review_rounds,
            num_drafts=_env_int("C9_NUM_DRAFTS", _AI_SCIENTIST_SHARED_BUDGETS["num_drafts"]),
            agent_steps=_env_int("C9_AGENT_STEPS", _AI_SCIENTIST_SHARED_BUDGETS["agent_steps"]),
            variants_per_branch=_env_int("C9_VARIANTS_PER_BRANCH", _AI_SCIENTIST_SHARED_BUDGETS["variants_per_branch"]),
            max_debug_depth=_env_int("C9_MAX_DEBUG_DEPTH", _AI_SCIENTIST_SHARED_BUDGETS["max_debug_depth"]),
            concurrency=_env_int("C9_CONCURRENCY", _AI_SCIENTIST_SHARED_BUDGETS["concurrency"]),
        )
    if spec.runner == "c10_ai_scientist_deepseek":
        return run_ai_scientist_condition(
            repo_root, spec, _llm_for_spec(spec), data, objectives, run_dir,
            top_k=top_k, max_review_rounds=max_review_rounds,
            num_drafts=_env_int("C10_NUM_DRAFTS", _AI_SCIENTIST_SHARED_BUDGETS["num_drafts"]),
            agent_steps=_env_int("C10_AGENT_STEPS", _AI_SCIENTIST_SHARED_BUDGETS["agent_steps"]),
            variants_per_branch=_env_int("C10_VARIANTS_PER_BRANCH", _AI_SCIENTIST_SHARED_BUDGETS["variants_per_branch"]),
            max_debug_depth=_env_int("C10_MAX_DEBUG_DEPTH", _AI_SCIENTIST_SHARED_BUDGETS["max_debug_depth"]),
            concurrency=_env_int("C10_CONCURRENCY", _AI_SCIENTIST_SHARED_BUDGETS["concurrency"]),
        )
    if spec.runner == "c11_evolution":
        return run_c11_evolution_condition(
            repo_root, spec, _opencode_multimodel(), data, objectives, run_dir,
            top_k=top_k,
            population_size=_env_int("C11_POPULATION_SIZE", 40),
            generations=_env_int("C11_GENERATIONS", 3),
            elite_count=_env_int("C11_ELITE_COUNT", 8),
            c9_baseline_slate=c9_baseline_slate,
        )
    if spec.runner == "c12_ab_mcts":
        return run_c12_ab_mcts_condition(
            repo_root, spec, _opencode_multimodel(), data, objectives, run_dir,
            top_k=top_k, max_llm_calls=_env_int("C12_MAX_LLM_CALLS", 24))
    if spec.runner == "c13_router":
        return run_c13_router_condition(
            repo_root, spec, _opencode_multimodel(), data, objectives, run_dir,
            top_k=top_k,
            enable_candidate_deliberation=enable_candidate_deliberation,
            candidate_reviewers=_candidate_reviewers(),
            max_data_requests_per_reviewer=_env_int("CANDIDATE_REVIEW_MAX_DATA_REQUESTS", 0))
    raise ValueError(f"unknown runner {spec.runner!r} for {spec.group}")


def _load_c9_baseline_slate(out_dir: Path, results: dict[str, LiveConditionResult]) -> list[str] | None:
    """The C9 (Gemini AI Scientist-style) slate used as the novelty baseline
    for C11-C13."""

    if "C9" in results and results["C9"].proposals:
        return [p["candidate_id"] for p in results["C9"].proposals]
    for base in (out_dir, out_dir.parent / "run_stage1", out_dir.parent / "live"):
        for run_name in ("C09", "C9"):
            record_path = base / "runs" / run_name / "final_slate.json"
            if record_path.exists():
                payload = _read_json(record_path)
                slate = payload.get("final_slate") or []
                ids = [row.get("candidate_id") for row in slate if row.get("candidate_id")]
                if ids:
                    return ids
    return None


def _record_for(spec: ConditionSpec, result: LiveConditionResult,
                report_path: Path | None) -> dict[str, Any]:
    record = make_condition_record(
        condition_id=spec.condition_id,
        condition_group=spec.group,
        workflow_type=spec.algorithm,
        provider_family=spec.provider,
        agentic_loop=spec.runner not in ("c0_deterministic", "vanilla_llm"),
        tree_search=spec.branch_search,
        llm_required=spec.provider != "none",
        live_api_calls_made=result.execution_mode in ("live", "live_manual_harness"),
        raw_llm_outputs_read=result.execution_mode in ("live", "live_manual_harness"),
        proposals=result.proposals,
        source_artifacts=sorted(result.artifacts.values()),
        proposal_report_path=str(report_path) if report_path else None,
        required_due_diligence=result.due_diligence,
        extra={
            **spec.to_public_dict(),
            "execution_mode": result.execution_mode,
            "comparable_for_e13": result.comparable_for_e13,
            "steps_run": result.steps_run,
            "branch_results": result.branch_results,
            "generated_code_stats": result.generated_code_stats,
            "model_call_summary": result.model_call_summary,
            "failure_notes": result.failure_notes,
            "narrative_sections": result.narrative_sections,
            "proposals": result.proposals,
            "candidate_review_packets": result.candidate_review_packets,
            "candidate_qualitative_assessments": result.candidate_qualitative_assessments,
            "candidate_assessment_reviews": result.candidate_assessment_reviews,
            "candidate_deliberation_summary": result.candidate_deliberation_summary,
        },
    )
    if result.comparable_for_e13 and result.proposals:
        eligible, exclusion_reason = condition_record_evidence_passed(record, result.proposals)
        if not eligible:
            record["comparable_for_e13"] = False
    else:
        eligible, exclusion_reason = False, result.exclusion_reason or "not_comparable"
    record["eligible_for_judge"] = eligible
    record["exclusion_reason"] = exclusion_reason or result.exclusion_reason
    return record


def _infer_code_stage(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "inspect" in name:
        return "inspect_available_data"
    if "branch" in name:
        return "run_branch_search"
    if "score" in name or "scoring" in name:
        return "evaluation_scoring_code"
    if "repair" in name or "_a" in name:
        return "generated_code_attempt"
    if "sandbox" in parent:
        return "sandbox_execution"
    return "generated_evaluation_code"


def _write_code_manifest(
    spec: ConditionSpec,
    result: LiveConditionResult,
    run_dir: Path,
    repo_root: Path,
) -> None:
    """Index generated/evaluation code by condition and execution stage."""

    code_files: list[dict[str, Any]] = []
    for base in (run_dir / "generated_code", run_dir / "sandbox"):
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            try:
                rel = path.relative_to(repo_root)
            except ValueError:
                rel = path
            code_files.append({
                "condition_group": spec.group,
                "condition_id": spec.condition_id,
                "algorithm": spec.algorithm,
                "harness": spec.harness,
                "stage": _infer_code_stage(path),
                "path": str(rel),
                "bytes": path.stat().st_size,
            })
    manual_result = result.artifacts.get("manual_result")
    skill_trace: list[dict[str, Any]] = []
    if manual_result and Path(manual_result).exists():
        try:
            payload = _read_json(Path(manual_result))
            skill_trace = [
                {
                    "skill_id": row.get("skill_id"),
                    "status": row.get("status"),
                    "produced_outputs": row.get("produced_outputs") or [],
                    "output_refs": row.get("output_refs") or [],
                }
                for row in payload.get("skill_trace") or []
                if isinstance(row, dict)
            ]
        except Exception:
            skill_trace = []
    manifest = {
        "condition_group": spec.group,
        "condition_id": spec.condition_id,
        "label": spec.label,
        "algorithm": spec.algorithm,
        "provider": spec.provider,
        "model": spec.model,
        "harness": spec.harness,
        "execution_mode": result.execution_mode,
        "skills_unified": spec.skills_unified,
        "branch_search": spec.branch_search,
        "generated_code_stats": result.generated_code_stats,
        "branch_results": result.branch_results,
        "skill_trace": skill_trace,
        "code_files": code_files,
        "notes": (
            "Indexes generated/evaluation code and Skill stages for this "
            "condition run; source code content remains in the referenced files."
        ),
    }
    manifest_path = run_dir / "code_manifest.json"
    _write_json(manifest_path, manifest)
    result.artifacts["code_manifest"] = str(manifest_path)


def run_condition_proposals(
    repo_root: str | Path = ".",
    *,
    conditions: list[str] | None = None,
    output_dir: str | Path = "outputs/condition_proposals/live",
    top_k_sites: int = 5,
    max_review_rounds: int = 2,
    require_live_agents: bool = True,
    disable_deterministic_fallback_for_comparison: bool = True,
    branch_objectives_csv: str | None = None,
    run_e13: bool = True,
    manual_result_path: str | Path | None = None,
    enable_candidate_deliberation: bool = True,
    allow_candidate_replacement: bool = False,
    auto_agentic_harness: bool = False,
) -> ConditionProposalsResult:
    root = Path(repo_root).resolve()
    run_id = str(uuid.uuid4())
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = build_condition_registry()
    requested = conditions or list(CONDITION_ORDER)
    unknown = [c for c in requested if c not in registry]
    if unknown:
        raise ValueError(f"unknown conditions: {unknown}")
    if manual_result_path is not None and not Path(manual_result_path).exists():
        raise FileNotFoundError(
            f"--manual-result file not found: {manual_result_path}")
    objectives = parse_branch_objectives(branch_objectives_csv)

    data = load_data_bundle(root)
    results: dict[str, LiveConditionResult] = {}
    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    report_paths: dict[str, str] = {}

    ordered = [c for c in CONDITION_ORDER if c in requested]
    for group in ordered:
        spec = registry[group]
        try:
            result = _dispatch_condition(
                root, spec, data, objectives, out_dir,
                top_k=top_k_sites, max_review_rounds=max_review_rounds,
                c9_baseline_slate=_load_c9_baseline_slate(out_dir, results),
                manual_result_path=Path(manual_result_path) if manual_result_path else None,
                auto_agentic_harness=auto_agentic_harness,
                enable_candidate_deliberation=enable_candidate_deliberation,
            )
        except Exception as exc:  # a crashed runner is a live_error, not a fallback
            result = failure_result(spec, "live_error", f"{type(exc).__name__}: {exc}")

        if (not result.comparable_for_e13
                and result.execution_mode not in COMPARABLE_EXECUTION_MODES
                and not disable_deterministic_fallback_for_comparison
                and not require_live_agents):
            # Debug-only fallback: clearly labeled, never comparable.
            outcome = build_deterministic_outcome(
                root, condition_id=spec.condition_id, condition_group=spec.group,
                top_k=top_k_sites, review_rounds=0, data=data)
            result.proposals = outcome.proposals
            result.due_diligence = outcome.due_diligence_items
            result.execution_mode = "deterministic_fallback"
            result.comparable_for_e13 = False
            result.exclusion_reason = (
                result.exclusion_reason or "deterministic_fallback_debug_only")
            result.failure_notes.append(
                "Debug-only deterministic fallback ranking attached; excluded "
                "from E13 comparison.")

        if (enable_candidate_deliberation and result.proposals
                and condition_supports_candidate_deliberation(spec)
                and spec.group not in _INLINE_QUALITATIVE_DELIBERATION_GROUPS):
            run_dir = out_dir / "runs" / spec.padded_id
            deliberation_llm = _deliberation_llm_call(spec)
            try:
                if deliberation_llm is not None:
                    if spec.group in _QUALITATIVE_DELIBERATION_GROUPS:
                        # C11/C12: extend the base reviewer-only pipeline with
                        # the agent-assessment-first layer (post-hoc, after
                        # each condition's own final synthesis + judge).
                        result = run_candidate_qualitative_deliberation(
                            result, data, deliberation_llm,
                            reviewers=_candidate_reviewers(),
                            allow_replacement=allow_candidate_replacement,
                            max_workers=_env_int("CANDIDATE_REVIEW_MAX_WORKERS", 1),
                            max_data_requests_per_reviewer=_env_int(
                                "CANDIDATE_REVIEW_MAX_DATA_REQUESTS", 0),
                            run_dir=run_dir,
                            model_name=f"{spec.provider}/{spec.model}")
                    else:
                        result = run_candidate_deliberation(
                            result, data, deliberation_llm,
                            reviewers=_candidate_reviewers(),
                            allow_replacement=allow_candidate_replacement,
                            max_workers=_env_int("CANDIDATE_REVIEW_MAX_WORKERS", 1),
                            max_data_requests_per_reviewer=_env_int(
                                "CANDIDATE_REVIEW_MAX_DATA_REQUESTS", 0),
                            run_dir=run_dir)
            except Exception as exc:  # deliberation failures never sink the base result
                result.failure_notes.append(
                    f"candidate_deliberation_error: {type(exc).__name__}: {exc}")

        results[group] = result
        _write_code_manifest(spec, result, out_dir / "runs" / spec.padded_id, root)
        report_path = write_condition_report(spec, result, out_dir, data=data)
        report_paths[group] = str(report_path)
        record = _record_for(spec, result, report_path)
        records.append(record)
        summaries.append({
            "condition_group": group,
            "label": spec.label,
            "provider": spec.provider,
            "model": spec.model,
            "harness": spec.harness,
            "execution_mode": result.execution_mode,
            "comparable_for_e13": record["comparable_for_e13"],
            "eligible_for_judge": record["eligible_for_judge"],
            "exclusion_reason": record["exclusion_reason"],
            "proposal_count": len(result.proposals),
            "steps_run": result.steps_run,
            "top_candidate": (
                f"{result.proposals[0].get('municipality')} "
                f"({result.proposals[0].get('action_type')})"
                if result.proposals else None),
            "report_path": str(report_path),
        })

    # Merge with any existing records so partial runs (e.g. ingesting one
    # manual-harness condition into the live dir) update only their own rows;
    # the shared precedence rule keeps ingested manual results from
    # regressing to waiting placeholders.
    records_path = out_dir / "condition_records.jsonl"
    existing = {
        str(row.get("condition_group")): row
        for row in _read_jsonl(records_path)
    }
    for record in records:
        group_key = str(record.get("condition_group"))
        existing[group_key] = merge_condition_record(existing.get(group_key), record)
    merged = [existing[g] for g in CONDITION_ORDER if g in existing]
    _write_jsonl(records_path, merged)

    e13_report_path: str | None = None
    if run_e13:
        from geo_strategist.experiments.condition_comparison_judge import (
            run_condition_comparison_judge,
        )

        e13 = run_condition_comparison_judge(root, proposals_dir=out_dir)
        e13_report_path = e13.comparison_report_path

    summary_payload = {
        "run_id": run_id,
        "generated_at": _now_iso(),
        "require_live_agents": require_live_agents,
        "disable_deterministic_fallback_for_comparison": disable_deterministic_fallback_for_comparison,
        "auto_agentic_harness": auto_agentic_harness,
        "branch_objectives": [o.key for o in objectives],
        "c1_c9_strict_comparison": c1_c9_strict_comparison_status(registry),
        "c2_c6_strict_comparison": c2_c6_strict_comparison_status(registry),
        "c3_c7_strict_comparison": c3_c7_strict_comparison_status(registry),
        "c4_c8_c10_strict_comparison": c4_c8_c10_strict_comparison_status(registry),
        "conditions": summaries,
        "e13_report_path": e13_report_path,
        "data_notes": data.data_notes,
    }
    summary_path = out_dir / "condition_outputs_summary.json"
    _write_json(summary_path, summary_payload)

    manifest = {
        "run_id": run_id,
        "generated_at": summary_payload["generated_at"],
        "output_dir": _rel(out_dir, root),
        "conditions_run": [s["condition_group"] for s in summaries],
        "top_k_sites": top_k_sites,
        "max_review_rounds": max_review_rounds,
        "require_live_agents": require_live_agents,
        "auto_agentic_harness": auto_agentic_harness,
        "branch_objectives": [o.key for o in objectives],
        "artifacts": {
            "condition_records": str(records_path),
            "condition_outputs_summary": str(summary_path),
            "e13_report": e13_report_path,
            **{f"{group}_report": path for group, path in report_paths.items() if path},
        },
    }
    manifest_path = out_dir / "run_manifest.json"
    _write_json(manifest_path, manifest)
    _write_artifact_index(out_dir, summaries, results, e13_report_path)

    return ConditionProposalsResult(
        run_id=run_id,
        output_dir=out_dir,
        conditions_run=[s["condition_group"] for s in summaries],
        report_paths=report_paths,
        e13_report_path=e13_report_path,
        summary_path=str(summary_path),
        manifest_path=str(manifest_path),
    )


def _write_artifact_index(
    out_dir: Path,
    summaries: list[dict[str, Any]],
    results: dict[str, LiveConditionResult],
    e13_report_path: str | None,
) -> None:
    lines = [
        "# Artifact Index",
        "",
        "| condition | execution mode | comparable | report | key run artifacts |",
        "| --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        group = summary["condition_group"]
        artifacts = results[group].artifacts if group in results else {}
        lines.append(
            f"| {group} | {summary['execution_mode']} | {summary['comparable_for_e13']} "
            f"| `{Path(summary['report_path']).name}` "
            f"| {', '.join(f'`{Path(p).name}`' for p in list(artifacts.values())[:4]) or '—'} |"
        )
    lines.extend([
        "",
        f"- E13 comparison report: `{e13_report_path or 'not generated'}`",
        "- `condition_records.jsonl` — machine-readable condition records",
        "- `condition_outputs_summary.json` / `run_manifest.json` — run metadata",
        "- `runs/<Cx>/` — per-condition traces, journals, generated code, model calls",
        "",
    ])
    (out_dir / "artifact_index.md").write_text("\n".join(lines), encoding="utf-8")
