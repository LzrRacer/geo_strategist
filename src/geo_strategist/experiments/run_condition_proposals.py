"""Condition-proposal orchestrator for the canonical C0-C14 track.

Dispatches each requested condition to its live runner (or the deterministic
baseline for C0, or a manual-harness handoff for C2/C3/C9/C10/C11/C12), writes
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

import json
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
from geo_strategist.experiments.candidate_deliberation_runtime import (
    condition_supports_candidate_deliberation,
    run_candidate_deliberation,
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
from geo_strategist.experiments.family_validation import (
    FamilyValidationResult,
    apply_family_validation,
)
from geo_strategist.experiments.condition_utils import (
    _read_json,
    _read_jsonl,
    _rel,
    _write_json,
    _write_jsonl,
)
from geo_strategist.experiments.live_common import LiveConditionResult, failure_result
from geo_strategist.experiments.live_report import (
    BUSINESS_REPORT_CONTRACT_VERSION,
    write_condition_report,
)
from geo_strategist.harnesses.agentic_runner import (
    execute_agentic_skills_harness_with_retry,
    prepare_agentic_skills_harness,
    recover_from_sanitized_workspace_mirror,
)
from geo_strategist.harnesses.prompts import write_manual_harness_prompts
from geo_strategist.providers.gemini_client import GeminiClient
from geo_strategist.providers.opencode_go_client import OpenCodeGoClient
from geo_strategist.providers.base import RetryPolicy


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
        try:
            payload = _read_json(Path(found))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            return failure_result(
                spec, "live_error", f"manual result JSON parse error at $: {exc}")
        if not isinstance(payload, dict):
            return failure_result(
                spec, "live_error", "manual result schema error at $: expected a JSON object")
        # C2/C3/C5-C8 always carry the reporting contract; C9-C12 (Skills) carry
        # it once re-run with the contract-bearing launcher. Validate whenever it
        # is present so a contract-bearing Skills result is checked and attached,
        # while a legacy Skills result (no contract) still ingests via the
        # compatible path rather than being rejected.
        _reporting_required = spec.group in {"C2", "C3", "C5", "C6", "C7", "C8"}
        _reporting_present = bool(payload.get("reporting_contract_version"))
        if _reporting_required or (
            spec.group in {"C9", "C10", "C11", "C12"} and _reporting_present
        ):
            from geo_strategist.experiments.decision_reporting_contract import (
                validate_reporting_payload,
            )

            candidate_ids = {
                str(row.get("candidate_id")) for row in data.candidates
                if isinstance(row, dict) and row.get("candidate_id")
            }
            try:
                # Resolve evidence refs against the run dir, the output root
                # (agents reference artifacts as `runs/Cxx/...` per the launcher
                # save-path convention), and the repo root.
                validate_reporting_payload(
                    payload, condition_group=spec.group, candidate_ids=candidate_ids,
                    evidence_roots=[Path(found).parent, out_dir, Path(repo_root)])
            except (TypeError, ValueError) as exc:
                return failure_result(
                    spec, "live_error", f"manual result reporting schema error: {exc}")
        from geo_strategist.experiments.decision_analysis import (
            default_discovery_contract,
            load_decision_analysis_bundle,
        )
        normalized_path = Path(found).parent / "decision_analysis_bundle_v1.json"
        try:
            analysis_bundle = load_decision_analysis_bundle(
                found,
                repo_root=repo_root,
                condition_id=spec.condition_id,
                discovery_contract=default_discovery_contract(
                    advanced=spec.runner not in ("c0_deterministic", "vanilla_llm")),
                output_path=normalized_path,
            )
        except (OSError, ValueError, TypeError) as exc:
            return failure_result(spec, "live_error", f"invalid decision-analysis bundle: {exc}")
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
                proposal["model_reported_qualitative_discussion"] = manual_text
                proposal.setdefault("evidence_grades", {})["manual_harness_discussion"] = "model_estimate"
                proposal.setdefault("evidence_grades", {})[
                    "model_reported_qualitative_discussion"] = "model_estimate"
        due_diligence = [item for p in proposals for item in p.get("required_due_diligence") or []]
        due_diligence.extend(str(c) for c in (payload.get("review_comments") or [])[:10])
        due_diligence.extend(data.data_notes)

        trace_issues: list[str] = []
        fundamental_trace_issues: list[str] = []
        if spec.skills_unified:
            # Skills-unified conditions (C9-C12) promise the Skills
            # contract; non-Skills conditions may return a free-form trace.
            if strict_skill_trace:
                from geo_strategist.agent.skill_registry import (
                    validate_skill_trace_against_io,
                )
                from geo_strategist.experiments.family_validation import (
                    split_skill_trace_issues,
                )

                trace_issues = validate_skill_trace_against_io(
                    payload.get("skill_trace") or [],
                    Path(found).parent,
                    manual_result_path=Path(found),
                )
                # Only an unsupported-claim issue (a declared output/code/
                # execution/review/branch-lineage with no backing artifact)
                # is a fundamental evidence-policy violation; trace-shape,
                # lifecycle-order, hypothesis-format, and five-objective-
                # coverage differences are real process variation, not
                # exclusionary (see family_validation.py module docstring).
                fundamental_trace_issues, _deviations = split_skill_trace_issues(trace_issues)
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
        provenance_issues: list[str] = []
        provenance = analysis_bundle.execution_provenance
        if provenance.analysis_completion_status != "complete":
            provenance_issues.append("analysis_completion_status_not_complete")
        if (isinstance(payload.get("execution_provenance"), dict)
                and provenance.execution_channel == "manual_interactive"
                and not provenance.comparable_manual_execution):
            provenance_issues.append("manual_execution_provenance_not_comparable")
        comparable = not (strict_skill_trace and fundamental_trace_issues) and not provenance_issues
        if provenance_issues:
            due_diligence.extend(provenance_issues)
        artifacts = {"manual_result": str(found)}
        artifacts["decision_analysis_bundle"] = str(normalized_path)
        for entry in analysis_bundle.artifact_manifest:
            if entry.exists:
                artifacts.setdefault(entry.role, entry.path)
        artifacts.update(extra_artifacts or {})

        if spec.skills_unified:
            from geo_strategist.agent.skills_budget import resource_usage_summary

            usage = resource_usage_summary(spec.group, payload.get("skill_trace") or [])
            usage_path = Path(found).parent / "resource_usage_summary.json"
            usage_path.write_text(json.dumps(usage, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts["resource_usage_summary"] = str(usage_path)
            narrative["resource_usage"] = json.dumps(usage, ensure_ascii=False)
            if not all(usage["within_budget"].values()):
                due_diligence.append(
                    "Skills-search resource usage exceeded the configured "
                    f"budget: {usage['consumed_resources']} vs "
                    f"{usage['configured_budget']}.")
        return LiveConditionResult(
            spec=spec,
            execution_mode="live_manual_harness",
            comparable_for_e13=comparable,
            exclusion_reason=(None if comparable else
                              "invalid_skill_trace" if fundamental_trace_issues else "invalid_execution_provenance"),
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
            branch_results=analysis_bundle.branch_results,
            review_rows=[row.model_dump(mode="json") for row in analysis_bundle.critical_issues],
            analysis_bundle=analysis_bundle,
        )
    prompts = write_manual_harness_prompts(repo_root, [spec.group], output_dir=out_dir)
    prompt_path = Path(prompts[spec.group])
    if auto_harness:
        from geo_strategist.harnesses.agentic_runner import AgenticHarnessLaunch

        run_dir = out_dir / "runs" / spec.padded_id
        run_dir.mkdir(parents=True, exist_ok=True)
        harness_cwd = repo_root
        sanitization_note: str | None = None
        if spec.runner == "coding_agent_no_skills":
            # C5-C8: enforce Skills isolation for real (not just a prompt
            # instruction) by running from a sanitized symlink workspace
            # with .agents/skills, .claude/skills, and Skills-referencing
            # AGENTS.md lines withheld.
            from geo_strategist.harnesses.no_skills_isolation import (
                prepare_no_skills_workspace,
                skills_dirs_present,
            )

            harness_cwd = prepare_no_skills_workspace(repo_root, run_dir)
            contaminated = skills_dirs_present(harness_cwd)
            sanitization_note = (
                "no_skills_workspace_isolation: skills_dirs_present="
                f"{contaminated}")
            # Regenerate the prompt so the automated agent is told to stay
            # inside the sanitized workspace it is actually launched in,
            # instead of the earlier "run from the repository root" text
            # (which pointed at the real, unsanitized repo root and
            # contradicted the isolation this run depends on).
            prompts = write_manual_harness_prompts(
                repo_root, [spec.group], output_dir=out_dir,
                sanitized_workspace=harness_cwd)
            prompt_path = Path(prompts[spec.group])
        launch = AgenticHarnessLaunch(
            run_dir=run_dir,
            launcher_prompt_path=prompt_path,
            validation_issues=[],
        )
        execution = execute_agentic_skills_harness_with_retry(harness_cwd, spec, launch)
        execution_artifacts = {
            "manual_prompt": str(prompt_path),
            "agent_stdout": str(execution.stdout_path),
            "agent_stderr": str(execution.stderr_path),
            "agent_execution": str(execution.metadata_path),
        }
        execution, recovered_from_mirror = recover_from_sanitized_workspace_mirror(
            execution, harness_cwd=harness_cwd, repo_root=repo_root)
        if sanitization_note or recovered_from_mirror:
            metadata = _read_json(execution.metadata_path)
            if sanitization_note:
                metadata["isolation"] = sanitization_note
            if recovered_from_mirror:
                metadata["status"] = execution.status
                metadata["recovered_from_sanitized_workspace_mirror"] = True
                metadata["original_failure_state"] = metadata.get("detail")
            _write_json(execution.metadata_path, metadata)
        if execution.status in ("succeeded", "succeeded_with_nonzero_exit"):
            result = _manual_harness_result(
                repo_root, spec, data, out_dir, top_k=top_k,
                manual_result_path=execution.manual_result_path,
                strict_skill_trace=strict_skill_trace,
                extra_artifacts=execution_artifacts,
                method_prefix="Automated",
            )
            exit_note = (
                f"return code {execution.returncode}" if execution.status == "succeeded"
                else (
                    f"return code {execution.returncode} (nonzero — a valid "
                    "manual_result.json existed anyway and was validated; "
                    "see agent_execution.json for the abnormal-exit diagnostic)"
                )
            )
            result.narrative_sections["agent_execution"] = (
                f"Non-interactive {spec.harness} adapter completed with {exit_note}.")
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
    """Ingest or launch a C9-C12 AGENTS.md + Skill package harness run."""

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
        execution = execute_agentic_skills_harness_with_retry(repo_root, spec, launch)
        execution_artifacts = {
            "launcher_prompt": str(launch.launcher_prompt_path),
            "agent_stdout": str(execution.stdout_path),
            "agent_stderr": str(execution.stderr_path),
            "agent_execution": str(execution.metadata_path),
        }
        if execution.status in ("succeeded", "succeeded_with_nonzero_exit"):
            result = _manual_harness_result(
                repo_root, spec, data, out_dir, top_k=top_k,
                manual_result_path=execution.manual_result_path,
                strict_skill_trace=True,
                extra_artifacts=execution_artifacts,
                method_prefix="Automated",
            )
            result.narrative_sections["agent_execution"] = (
                f"Non-interactive {spec.harness} adapter completed with "
                f"return code {execution.returncode}"
                + (" (nonzero exit; valid output recovered anyway)."
                   if execution.status == "succeeded_with_nonzero_exit" else "."))
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
    manual_result_path: Path | None = None,
    auto_agentic_harness: bool = False,
) -> LiveConditionResult:
    run_dir = out_dir / "runs" / spec.padded_id
    if spec.runner == "c0_deterministic":
        return _c0_result(repo_root, spec, data, top_k=top_k, run_dir=run_dir)
    if spec.runner == "manual_harness":
        return _manual_harness_result(
            repo_root, spec, data, out_dir, top_k=top_k,
            manual_result_path=manual_result_path,
            auto_harness=auto_agentic_harness)
    if spec.runner == "coding_agent_no_skills":
        # C5-C8: a full native coding-agent session on the same harness/
        # provider/model as their C9-C12 Skills-unified strict pair, but
        # with project Skills withheld. Reuses the same manual-harness
        # ingestion path as C2/C3 with strict_skill_trace left False (its
        # skill_trace is optional, matching the no-Skills contract).
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
    if spec.runner == "c13_ai_scientist_gemini":
        return run_ai_scientist_condition(
            repo_root, spec, _llm_for_spec(spec), data, objectives, run_dir,
            top_k=top_k, max_review_rounds=max_review_rounds,
            num_drafts=_env_int("C13_NUM_DRAFTS", _AI_SCIENTIST_SHARED_BUDGETS["num_drafts"]),
            agent_steps=_env_int("C13_AGENT_STEPS", _AI_SCIENTIST_SHARED_BUDGETS["agent_steps"]),
            variants_per_branch=_env_int("C13_VARIANTS_PER_BRANCH", _AI_SCIENTIST_SHARED_BUDGETS["variants_per_branch"]),
            max_debug_depth=_env_int("C13_MAX_DEBUG_DEPTH", _AI_SCIENTIST_SHARED_BUDGETS["max_debug_depth"]),
            concurrency=_env_int("C13_CONCURRENCY", _AI_SCIENTIST_SHARED_BUDGETS["concurrency"]),
        )
    if spec.runner == "c14_ai_scientist_deepseek":
        return run_ai_scientist_condition(
            repo_root, spec, _llm_for_spec(spec), data, objectives, run_dir,
            top_k=top_k, max_review_rounds=max_review_rounds,
            num_drafts=_env_int("C14_NUM_DRAFTS", _AI_SCIENTIST_SHARED_BUDGETS["num_drafts"]),
            agent_steps=_env_int("C14_AGENT_STEPS", _AI_SCIENTIST_SHARED_BUDGETS["agent_steps"]),
            variants_per_branch=_env_int("C14_VARIANTS_PER_BRANCH", _AI_SCIENTIST_SHARED_BUDGETS["variants_per_branch"]),
            max_debug_depth=_env_int("C14_MAX_DEBUG_DEPTH", _AI_SCIENTIST_SHARED_BUDGETS["max_debug_depth"]),
            concurrency=_env_int("C14_CONCURRENCY", _AI_SCIENTIST_SHARED_BUDGETS["concurrency"]),
        )
    raise ValueError(f"unknown runner {spec.runner!r} for {spec.group}")


def _record_for(spec: ConditionSpec, result: LiveConditionResult,
                report_path: Path | None,
                family_validation_results: dict[str, FamilyValidationResult] | None = None,
                ) -> dict[str, Any]:
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
            "search_nodes": result.search_nodes,
            "robustness_results": result.robustness_results,
            "generated_code_stats": result.generated_code_stats,
            "model_call_summary": result.model_call_summary,
            "failure_notes": result.failure_notes,
            "narrative_sections": result.narrative_sections,
            "proposals": result.proposals,
            "candidate_review_packets": result.candidate_review_packets,
            "candidate_deliberation_summary": result.candidate_deliberation_summary,
            "decision_analysis_bundle": (
                result.analysis_bundle.model_dump(mode="json")
                if result.analysis_bundle is not None else None
            ),
            "family_validation": (
                {family_id: fv.to_dict() for family_id, fv in family_validation_results.items()}
                if family_validation_results else {}
            ),
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
    if spec.branch_search and result.analysis_bundle is not None:
        search_status = result.analysis_bundle.branch_search_status
        if search_status == "succeeded" and not result.branch_results:
            raise ValueError(
                "branch_search is succeeded but normalized branch_results is empty")
    else:
        search_status = "succeeded" if result.branch_results else "not_performed"
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
        "branch_search_status": search_status,
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
                manual_result_path=Path(manual_result_path) if manual_result_path else None,
                auto_agentic_harness=auto_agentic_harness,
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

        # condition_supports_candidate_deliberation() gates on spec.algorithm
        # (deterministic_baseline, vanilla_llm), not on spec.group or
        # spec.runner — C2/C3 are Vanilla LLM conditions dispatched through
        # the manual_harness runner and must be excluded here too. Do not
        # reintroduce a hard-coded C1-C4 group list.
        if (enable_candidate_deliberation and result.proposals
                and condition_supports_candidate_deliberation(spec)):
            run_dir = out_dir / "runs" / spec.padded_id
            deliberation_llm = _deliberation_llm_call(spec)
            try:
                if deliberation_llm is not None:
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

        from geo_strategist.experiments.decision_analysis import (
            bundle_artifact_manifest_hash,
            bundle_from_condition_result,
            derive_search_decision_analysis,
        )

        if result.analysis_bundle is None:
            # A branch-search condition that produces a ranked slate and
            # per-objective branch winners but no explicit decision-analysis
            # surface. Derive it deterministically from that real executed
            # data so the condition's genuine work is observable to the
            # judge, on the same footing as the reporting-contract conditions.
            if (not result.synthesis and not result.final_decision_rows
                    and result.proposals and (result.branch_results or result.search_nodes)):
                derived = derive_search_decision_analysis(
                    proposals=result.proposals,
                    branch_results=result.branch_results,
                    algorithm_label=spec.label,
                    artifacts=result.artifacts,
                )
                result.synthesis = derived["synthesis"]
                result.final_decision_rows = derived["final_decision_rows"]
                result.robustness_results = derived["robustness_results"]
                result.reversal_conditions = derived["reversal_conditions"]
            result.analysis_bundle = bundle_from_condition_result(
                condition_group=spec.group,
                condition_id=spec.condition_id,
                execution_mode=result.execution_mode,
                comparable=result.comparable_for_e13,
                ranked_candidates=result.ranked_rows,
                branch_results=result.branch_results,
                review_rows=result.review_rows,
                narrative_sections=result.narrative_sections,
                artifacts=result.artifacts,
                branch_search=spec.branch_search,
                candidate_universe_size=len(data.candidates),
                raw_search_nodes=result.search_nodes,
                robustness_results=result.robustness_results,
                synthesis=result.synthesis,
                final_decision_rows=result.final_decision_rows,
                reversal_conditions=result.reversal_conditions,
            )
        normalized_path = out_dir / "runs" / spec.padded_id / "decision_analysis_bundle_v1.json"
        result.analysis_bundle.execution_provenance.artifact_manifest_hash = (
            bundle_artifact_manifest_hash(result.analysis_bundle))
        _write_json(normalized_path, result.analysis_bundle.model_dump(mode="json"))
        result.artifacts["decision_analysis_bundle"] = str(normalized_path)

        # Family-contract validation is the single choke point for every
        # runner path (deterministic, vanilla, manual-harness, Skills,
        # AI-Scientist, orchestration): a primary-family violation marks the
        # condition non-comparable with an explicit family_contract:<id>:...
        # reason (never overwriting a more specific existing reason), while
        # secondary-family results are recorded without affecting overall
        # comparability. See family_validation.py.
        family_validation_results = apply_family_validation(
            spec, result, out_dir / "runs" / spec.padded_id)

        results[group] = result
        _write_code_manifest(spec, result, out_dir / "runs" / spec.padded_id, root)
        report_path = write_condition_report(spec, result, out_dir, data=data)
        report_paths[group] = str(report_path)
        record = _record_for(spec, result, report_path, family_validation_results)
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
        "business_report_contract_version": BUSINESS_REPORT_CONTRACT_VERSION,
        "decision_analysis_bundle_version": "1.0",
        "branch_objectives": [o.key for o in objectives],
        "c1_c13_strict_comparison": c1_c13_strict_comparison_status(registry),
        "c2_c10_strict_comparison": c2_c10_strict_comparison_status(registry),
        "c3_c11_strict_comparison": c3_c11_strict_comparison_status(registry),
        "c4_c12_c14_strict_comparison": c4_c12_c14_strict_comparison_status(registry),
        "c5_c9_skills_comparison": c5_c9_strict_comparison_status(registry),
        "c6_c10_skills_comparison": c6_c10_strict_comparison_status(registry),
        "c7_c11_skills_comparison": c7_c11_strict_comparison_status(registry),
        "c8_c12_skills_comparison": c8_c12_strict_comparison_status(registry),
        "c9_c13_discovery_comparison": c9_c13_discovery_comparison_status(registry),
        "c12_c14_discovery_comparison": c12_c14_discovery_comparison_status(registry),
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
        "business_report_contract_version": BUSINESS_REPORT_CONTRACT_VERSION,
        "decision_analysis_bundle_version": "1.0",
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
