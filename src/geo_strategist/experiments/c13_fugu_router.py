"""C13: Fugu-style dynamic orchestrator (optional condition).

A router model assigns each subtask in the proposal workflow to a model
role at run time (Sakana Fugu-style heterogeneous orchestration): routing/
planning and review/synthesis on the synthesis model, fast generation on the
flash model, hard critique on the pro model, code/debug on the code model,
and a bounded final judge. Every routing decision is logged with its reason
and expected/actual artifacts in ``c13_routing_log.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from geo_strategist.agent.candidate_llm_review import DEFAULT_REVIEWERS
from geo_strategist.agent.codeexec.interpreter import Interpreter
from geo_strategist.agent.codeexec.sandbox_guard import scan_generated_code
from geo_strategist.experiments.ai_scientist_loop import (
    _CODE_RULES,
    _data_files_block,
    _parse_exec_stdout,
    _strip_code_fences,
)
from geo_strategist.experiments.branch_objectives import BranchObjective
from geo_strategist.experiments.c11_shinka_evolution import MultiModelCall
from geo_strategist.agent.candidate_review_schemas import CandidateReviewPacket, ReviewThread
from geo_strategist.experiments.candidate_deliberation_runtime import (
    _max_review_candidates,
    _write_artifacts,
    build_candidate_dossier,
)
from geo_strategist.experiments.candidate_data_explorer import CandidateDataExplorer
from geo_strategist.experiments.candidate_qualitative_deliberation import (
    build_packets_for_slate,
    compute_deliberation_summary,
    disallowed_municipalities_for_slate,
    generate_assessments_for_slate,
    run_reviews_for_slate,
)
from geo_strategist.experiments.deterministic_evaluation_engine import DataBundle, apply_reviews
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _read_json, _write_json, _write_jsonl
from geo_strategist.experiments.live_common import (
    LiveConditionResult,
    LiveRunContext,
    build_live_proposals,
    data_context_json,
    extract_json_block,
    failure_result,
    slate_metrics_all_objectives,
    validate_llm_ranking,
)

_SYSTEM_PROMPT = (
    "You are one specialist inside a dynamically routed multi-model "
    "orchestrator for hospital-location strategy. Never invent facts; "
    "candidates exist only as candidate_id values from the provided data. "
    "Reply with valid JSON or code exactly as asked."
)

_ROUTER_SYSTEM = (
    "You are the routing/planning model of a multi-model orchestrator. For "
    "each task, choose the best model role and explain why in one sentence. "
    "Reply with valid JSON."
)

_TASKS: tuple[dict[str, str], ...] = (
    {"task_id": "t1_slate_options", "task_type": "fast_generation",
     "description": "Generate three alternative ranked slate options with rationales.",
     "expected_artifact": "slate_options.json"},
    {"task_id": "t2_critique", "task_type": "hard_critique",
     "description": "Critique the slate options: evidence gaps, risk blind spots, groupthink.",
     "expected_artifact": "critique.json"},
    {"task_id": "t3_scoring_code", "task_type": "code_generation",
     "description": "Write and execute a scoring script that ranks all candidates.",
     "expected_artifact": "scoring_output.json"},
    {"task_id": "t4_synthesis", "task_type": "review_synthesis",
     "description": "Synthesize slate options, critique, and executed scoring into the final slate.",
     "expected_artifact": "final_slate.json"},
    {"task_id": "t5_candidate_assessments", "task_type": "candidate_assessment",
     "description": "Write a structured qualitative assessment for each selected candidate.",
     "expected_artifact": "candidate_qualitative_assessments.jsonl"},
    {"task_id": "t6_candidate_reviews", "task_type": "candidate_review",
     "description": "Critically review each candidate's dossier and agent assessment.",
     "expected_artifact": "candidate_assessment_reviews.jsonl"},
    {"task_id": "t7_author_responses", "task_type": "author_response",
     "description": "Respond to reviewer findings and run the provenance/consistency judge.",
     "expected_artifact": "candidate_review_packets.json"},
    {"task_id": "t8_final_judge", "task_type": "final_judge",
     "description": "Final tie-break/confirmation of the synthesized slate.",
     "expected_artifact": "judge_decision.json"},
)

_DEFAULT_ROLE_FOR_TASK = {
    "fast_generation": "fast",
    "hard_critique": "pro",
    "code_generation": "code",
    "review_synthesis": "review",
    "candidate_assessment": "review",
    "candidate_review": "pro",
    "author_response": "review",
    "final_judge": "judge",
}


def run_c13_router_condition(
    repo_root: str | Path,
    spec: ConditionSpec,
    llm: MultiModelCall,
    data: DataBundle,
    objectives: list[BranchObjective],
    run_dir: Path,
    *,
    top_k: int = 5,
    exec_timeout: int = 90,
    max_debug_depth: int = 2,
    enable_candidate_deliberation: bool = True,
    candidate_reviewers: list[str] | None = None,
    max_data_requests_per_reviewer: int = 0,
) -> LiveConditionResult:
    root = Path(repo_root).resolve()
    context = LiveRunContext(run_dir, condition_group=spec.group)
    roles = spec.model_roles
    routing_log: list[dict[str, Any]] = []
    data_context = data_context_json(data)
    data_files = _data_files_block(root)
    code_stats = {"generated": 0, "executed_ok": 0, "repaired": 0, "failed": 0}
    reviewers_list = candidate_reviewers or DEFAULT_REVIEWERS

    def call(prompt: str, purpose: str, model: str, system: str = _SYSTEM_PROMPT):
        result = llm(prompt, system, purpose, model)
        context.record_call(purpose, prompt, result)
        return result

    def route(task: dict[str, str]) -> tuple[str, str]:
        """Ask the router model to choose a role; fall back to the default map."""

        available = {role: roles.get(role) for role in
                     ("fast", "pro", "code", "review", "judge") if roles.get(role)}
        result = call(
            "TASK:\n" + json.dumps(task, ensure_ascii=False)
            + "\nAVAILABLE MODEL ROLES:\n" + json.dumps(available, ensure_ascii=False)
            + '\nReply as JSON: {"chosen_role": "<role>", "reason": "..."}',
            f"route_{task['task_id']}", roles.get("router", "qwen3.7-plus"),
            system=_ROUTER_SYSTEM,
        )
        chosen = _DEFAULT_ROLE_FOR_TASK.get(task["task_type"], "fast")
        reason = "default routing (router unavailable or invalid reply)"
        if result.ok:
            parsed = extract_json_block(result.text) or {}
            candidate_role = str(parsed.get("chosen_role") or "")
            if candidate_role in available:
                chosen = candidate_role
                reason = str(parsed.get("reason") or "")[:400]
        return chosen, reason

    def log_route(task: dict[str, str], role: str, reason: str,
                  actual_artifact: str | None, success: bool) -> None:
        routing_log.append({
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "chosen_role": role,
            "chosen_model": roles.get(role),
            "reason": reason,
            "expected_artifact": task["expected_artifact"],
            "actual_artifact": actual_artifact,
            "success": success,
        })

    # ---- t1: slate options -----------------------------------------------------
    task = _TASKS[0]
    role, reason = route(task)
    result = call(
        f"DATA CONTEXT:\n{data_context}\n\n"
        f"Generate 3 alternative top-{top_k} slates with different strategic "
        "angles (demand-led, feasibility-led, risk-led). candidate_id values "
        "only from the data context.\n"
        'Reply as JSON: {"options": [{"angle": "...", '
        '"slate": [{"candidate_id": "...", "rationale": "..."}]}]}',
        task["task_id"], roles.get(role, "deepseek-v4-flash"),
    )
    options: list[dict[str, Any]] = []
    if result.ok:
        parsed = extract_json_block(result.text) or {}
        for option in parsed.get("options") or []:
            if not isinstance(option, dict):
                continue
            valid, _rej = validate_llm_ranking(option.get("slate") or [], data)
            if valid:
                options.append({"angle": str(option.get("angle") or ""), "slate": valid})
    _write_json(context.run_dir / "slate_options.json", options)
    log_route(task, role, reason, "slate_options.json", bool(options))
    if not options:
        if not result.ok and result.error_class in ("live_auth_failed", "live_rate_limited"):
            failed = failure_result(spec, result.error_class, result.error_detail or "t1 failed")
        else:
            failed = failure_result(spec, "live_error", "no valid slate options generated")
        _write_jsonl(context.run_dir / "c13_routing_log.jsonl", routing_log)
        failed.finalize_artifacts(context)
        return failed

    # ---- t2: critique ------------------------------------------------------------
    task = _TASKS[1]
    role, reason = route(task)
    critique_text = ""
    result = call(
        "Critique these slate options for evidence gaps, risk blind spots, and "
        "overlap/groupthink:\n"
        + json.dumps([{"angle": o["angle"],
                       "slate": [r["candidate_id"] for r in o["slate"]][:top_k]}
                      for o in options], ensure_ascii=False)
        + '\nReply as JSON: {"critique": "...", "must_address": ["..."]}',
        task["task_id"], roles.get(role, "deepseek-v4-pro"),
    )
    must_address: list[str] = []
    if result.ok:
        parsed = extract_json_block(result.text) or {}
        critique_text = str(parsed.get("critique") or "")[:4000]
        must_address = [str(p) for p in (parsed.get("must_address") or [])][:8]
    _write_json(context.run_dir / "critique.json",
                {"critique": critique_text, "must_address": must_address})
    log_route(task, role, reason, "critique.json", bool(critique_text))

    # ---- t3: scoring code (generated + executed) -----------------------------------
    task = _TASKS[2]
    role, reason = route(task)
    scoring_ids: list[str] = []
    result = call(
        f"{data_files}\nWrite a Python script that scores ALL candidates with a "
        "balanced model over the five objectives (elderly demand, emergency "
        'access, reorganization feasibility, financial risk, evidence '
        f'completeness), objective_slug "combined".\n{_CODE_RULES}\n'
        "Reply with only the Python code.",
        task["task_id"], roles.get(role, "kimi-k2.7-code"),
    )
    if result.ok:
        code = _strip_code_fences(result.text)
        for attempt in range(max_debug_depth + 1):
            code_path = context.run_dir / "generated_code" / f"t3_scoring_a{attempt}.py"
            code_path.parent.mkdir(parents=True, exist_ok=True)
            code_path.write_text(code, encoding="utf-8")
            code_stats["generated"] += 1
            if scan_generated_code(code, repo_root=root, run_dir=context.run_dir):
                code_stats["failed"] += 1
                break
            execution = Interpreter(context.run_dir / "sandbox" / f"t3_a{attempt}",
                                    timeout=exec_timeout).run(code)
            if execution.exc_type is None:
                payload = _parse_exec_stdout(execution.term_out)
                valid, _rej = validate_llm_ranking((payload or {}).get("ranking") or [], data)
                if valid:
                    scoring_ids = [row["candidate_id"] for row in valid]
                    code_stats["executed_ok"] += 1
                    _write_json(context.run_dir / "scoring_output.json", {
                        "ranking_head": scoring_ids[: top_k * 2],
                        "metrics": (payload or {}).get("metrics") or {},
                    })
                    break
                failure_tail = "stdout lacked required JSON ranking"
            else:
                failure_tail = execution.stderr_tail[-1500:]
            if attempt >= max_debug_depth:
                code_stats["failed"] += 1
                break
            repair = call(
                f"{data_files}\nFix this script.\n{_CODE_RULES}\nFAILURE:\n{failure_tail}\n\n"
                f"SCRIPT:\n```python\n{code[:10000]}\n```\nReply with only the Python code.",
                "t3_code_repair", roles.get("code", "kimi-k2.7-code"),
            )
            if not repair.ok:
                code_stats["failed"] += 1
                break
            code = _strip_code_fences(repair.text)
            code_stats["repaired"] += 1
    log_route(task, role, reason, "scoring_output.json" if scoring_ids else None, bool(scoring_ids))

    # ---- t4: synthesis --------------------------------------------------------------
    task = _TASKS[3]
    role, reason = route(task)
    final_rows: list[dict[str, Any]] = []
    synthesis_note = ""
    result = call(
        f"Synthesize the final top-{top_k} slate from:\n"
        + json.dumps({
            "slate_options": [{"angle": o["angle"],
                               "slate": [r["candidate_id"] for r in o["slate"]][:top_k]}
                              for o in options],
            "critique_must_address": must_address,
            "executed_scoring_head": scoring_ids[: top_k * 2],
        }, ensure_ascii=False)
        + '\nReply as JSON: {"final_slate": [{"candidate_id": "...", "rationale": "..."}], '
          '"synthesis_summary": "..."}',
        task["task_id"], roles.get(role, "qwen3.7-plus"),
    )
    if result.ok:
        parsed = extract_json_block(result.text) or {}
        final_rows, _rej = validate_llm_ranking(parsed.get("final_slate") or [], data)
        synthesis_note = str(parsed.get("synthesis_summary") or "")[:4000]
    if len(final_rows) < top_k:
        pooled: dict[str, float] = {}
        for option in options:
            for position, row in enumerate(option["slate"][:top_k]):
                pooled[row["candidate_id"]] = pooled.get(row["candidate_id"], 0.0) + (top_k - position)
        for position, cid in enumerate(scoring_ids[:top_k]):
            pooled[cid] = pooled.get(cid, 0.0) + (top_k - position)
        seen = {row["candidate_id"] for row in final_rows}
        for cid, _score in sorted(pooled.items(), key=lambda item: (-item[1], item[0])):
            if len(final_rows) >= top_k:
                break
            if cid not in seen:
                final_rows.append({"candidate_id": cid, "rationale": "aggregated across routed outputs"})
                seen.add(cid)
    _write_json(context.run_dir / "final_slate.json",
                {"final_slate": final_rows, "synthesis_summary": synthesis_note})
    log_route(task, role, reason, "final_slate.json", bool(final_rows))

    # ---- t5-t7: candidate qualitative deliberation ------------------------------------
    # Routed as explicit tasks (per-candidate agent assessment -> reviewer
    # critique of dossier+assessment -> author response + provenance/
    # consistency judge) so they appear in c13_routing_log.jsonl alongside
    # every other orchestration decision. Runs over the draft slate so the
    # dossiers exist; entries are filtered down to the post-judge (t8) final
    # slate below since the judge only reorders/subsets candidate_ids.
    draft_proposals = build_live_proposals(final_rows, data, spec, top_k=top_k)
    assessments: list[dict[str, Any]] = []
    assessment_reviews: list[dict[str, Any]] = []
    review_packets: list[dict[str, Any]] = []
    review_threads: list[dict[str, Any]] = []
    deliberation_summary: dict[str, Any] = {}
    if enable_candidate_deliberation and draft_proposals:
        max_candidates = _max_review_candidates()
        slate = draft_proposals[:max_candidates] if max_candidates else draft_proposals
        dossier_map = {p["candidate_id"]: build_candidate_dossier(p, slate, data) for p in slate}
        explorer = CandidateDataExplorer(data)

        task = _TASKS[4]
        role, reason = route(task)
        assessment_model = roles.get(role, "qwen3.7-plus")

        def assessment_llm(prompt: str, system: str | None, purpose: str):
            return call(prompt, purpose, assessment_model, system=system or _SYSTEM_PROMPT)

        assessment_objs = generate_assessments_for_slate(
            assessment_llm, spec.condition_id, slate, dossier_map,
            model_name=f"{spec.provider}/{spec.model}")
        assessments = [a.model_dump() for a in assessment_objs]
        _write_jsonl(context.run_dir / "candidate_qualitative_assessments.jsonl", assessments)
        log_route(task, role, reason, "candidate_qualitative_assessments.jsonl", bool(assessments))

        task = _TASKS[5]
        role, reason = route(task)
        review_model = roles.get(role, "deepseek-v4-pro")

        def review_llm(prompt: str, system: str | None, purpose: str):
            return call(prompt, purpose, review_model, system=system or _SYSTEM_PROMPT)

        assessment_by_candidate = {a.candidate_id: a for a in assessment_objs}
        threads, assessment_review_objs = run_reviews_for_slate(
            review_llm, slate, dossier_map, assessment_by_candidate, explorer, reviewers_list,
            condition_id=spec.condition_id, model_name=f"{spec.provider}/{spec.model}",
            max_workers=1, max_data_requests=max_data_requests_per_reviewer)
        assessment_reviews = [r.model_dump() for r in assessment_review_objs]
        review_threads = [t.model_dump() for t in threads]
        _write_jsonl(context.run_dir / "candidate_assessment_reviews.jsonl", assessment_reviews)
        log_route(task, role, reason, "candidate_assessment_reviews.jsonl", bool(assessment_reviews))

        task = _TASKS[6]
        role, reason = route(task)
        author_model = roles.get(role, "qwen3.7-plus")

        def author_llm(prompt: str, system: str | None, purpose: str):
            return call(prompt, purpose, author_model, system=system or _SYSTEM_PROMPT)

        disallowed = disallowed_municipalities_for_slate(data, slate, dossier_map)
        packet_objs = build_packets_for_slate(
            author_llm, slate, dossier_map, threads, disallowed_municipalities=disallowed)
        review_packets = [p.model_dump() for p in packet_objs]
        _write_json(context.run_dir / "candidate_review_packets.json", review_packets)
        log_route(task, role, reason, "candidate_review_packets.json", bool(review_packets))

        deliberation_summary = compute_deliberation_summary(
            slate, assessment_objs, assessment_review_objs, packet_objs)

    # ---- t8: judge -------------------------------------------------------------------
    task = _TASKS[7]
    role, reason = route(task)
    judge_note = ""
    result = call(
        "As final judge, confirm or reorder this slate (same candidate_ids only):\n"
        + json.dumps(final_rows, ensure_ascii=False)
        + '\nReply as JSON: {"final_slate": [{"candidate_id": "..."}], "rationale": "..."}',
        task["task_id"], roles.get(role, "qwen3.7-max"),
    )
    judge_success = False
    if result.ok:
        parsed = extract_json_block(result.text) or {}
        reordered, _rej = validate_llm_ranking(parsed.get("final_slate") or [], data)
        if reordered and {r["candidate_id"] for r in reordered} <= {r["candidate_id"] for r in final_rows}:
            rationale_by_id = {r["candidate_id"]: r.get("rationale", "") for r in final_rows}
            final_rows = [{"candidate_id": r["candidate_id"],
                           "rationale": rationale_by_id.get(r["candidate_id"], "")}
                          for r in reordered]
            judge_success = True
        judge_note = str(parsed.get("rationale") or "")[:1000]
    _write_json(context.run_dir / "judge_decision.json",
                {"rationale": judge_note, "applied": judge_success})
    log_route(task, role, reason, "judge_decision.json", judge_success)

    proposals = build_live_proposals(final_rows, data, spec, top_k=top_k)
    for proposal in proposals:
        existing = set(proposal.get("required_due_diligence") or [])
        for point in must_address[:5]:
            item = f"Critique follow-up: {point}"
            if item not in existing:
                proposal.setdefault("required_due_diligence", []).append(item)
                existing.add(item)
    proposals, review_rows = apply_reviews(proposals, 1)

    # The judge (t8) only reorders/subsets final_rows, never introduces a new
    # candidate_id, so filtering the t5-t7 deliberation artifacts down to the
    # post-judge slate keeps them consistent with what actually shipped.
    final_candidate_ids = {p["candidate_id"] for p in proposals}
    assessments = [a for a in assessments if a.get("candidate_id") in final_candidate_ids]
    assessment_reviews = [r for r in assessment_reviews if r.get("candidate_id") in final_candidate_ids]
    review_packets = [p for p in review_packets if p.get("candidate_id") in final_candidate_ids]
    review_threads = [t for t in review_threads if t.get("candidate_id") in final_candidate_ids]
    if enable_candidate_deliberation and draft_proposals:
        _write_jsonl(context.run_dir / "candidate_qualitative_assessments.jsonl", assessments)
        _write_jsonl(context.run_dir / "candidate_assessment_reviews.jsonl", assessment_reviews)

    # ---- artifacts ----------------------------------------------------------------------
    _write_jsonl(context.run_dir / "c13_routing_log.jsonl", routing_log)
    call_summary = context.ledger.summary()
    _write_json(context.run_dir / "c13_model_call_summary.json", call_summary)
    (context.run_dir / "c13_orchestrator_decisions.md").write_text(
        "# C13 Orchestrator Decisions\n\n"
        "| task | type | chosen model | success | reason |\n"
        "| --- | --- | --- | --- | --- |\n"
        + "\n".join(
            f"| {row['task_id']} | {row['task_type']} | {row['chosen_model']} "
            f"| {row['success']} | {row['reason'].replace('|', '/')} |"
            for row in routing_log)
        + f"\n\nSynthesis summary: {synthesis_note or 'not_available'}\n"
        + f"\nJudge rationale: {judge_note or 'not_available'}\n",
        encoding="utf-8",
    )
    (context.run_dir / "c13_final_report.md").write_text(
        "# C13 Final Report (working copy)\n\n"
        "See the orchestrator-level C11_proposal_report.md for the full report.\n\n"
        + "\n".join(
            f"{index + 1}. {p['municipality']} ({p['prefecture']}) — {p['action_type']}"
            for index, p in enumerate(proposals)) + "\n",
        encoding="utf-8",
    )

    due_diligence = [item for p in proposals for item in p.get("required_due_diligence") or []]
    due_diligence.extend(data.data_notes)

    outcome = LiveConditionResult(
        spec=spec,
        execution_mode="live",
        comparable_for_e13=True,
        exclusion_reason=None,
        proposals=proposals,
        ranked_rows=[
            {"rank": index + 1, "candidate_id": p["candidate_id"],
             "municipality": p["municipality"], "prefecture": p["prefecture"],
             "action_type": p["action_type"], "composite_score": p["composite_score"]}
            for index, p in enumerate(proposals)
        ],
        branch_results=[
            {"objective": o.slug, "objective_label": o.label,
             "winner_external_metric": slate_metrics_all_objectives(
                 data, [row["candidate_id"] for row in final_rows], [o], top_k=top_k)[o.slug],
             "winner_top_candidates": [row["candidate_id"] for row in final_rows][:top_k]}
            for o in objectives
        ],
        generated_code_stats=code_stats,
        review_rows=review_rows,
        due_diligence=list(dict.fromkeys(due_diligence)),
        narrative_sections={
            "method_summary": synthesis_note or "Dynamically routed multi-model pipeline.",
            "critique": critique_text,
            "judge_rationale": judge_note,
        },
        steps_run=len(routing_log) * 2,
        model_call_summary=call_summary,
        candidate_review_packets=review_packets,
        candidate_review_threads=review_threads,
        candidate_qualitative_assessments=assessments,
        candidate_assessment_reviews=assessment_reviews,
        candidate_deliberation_summary=deliberation_summary,
        artifacts={
            "c13_routing_log": str(context.run_dir / "c13_routing_log.jsonl"),
            "c13_model_call_summary": str(context.run_dir / "c13_model_call_summary.json"),
            "c13_orchestrator_decisions": str(context.run_dir / "c13_orchestrator_decisions.md"),
            "c13_final_report": str(context.run_dir / "c13_final_report.md"),
        },
    )
    outcome.finalize_artifacts(context)
    outcome.model_call_summary = call_summary
    if review_packets:
        _write_artifacts(
            outcome,
            [CandidateReviewPacket(**p) for p in review_packets],
            [ReviewThread(**t) for t in review_threads],
            context.run_dir)
        summary_path = context.run_dir / "candidate_review_summary.json"
        existing_summary = _read_json(summary_path) if summary_path.exists() else {}
        existing_summary = existing_summary if isinstance(existing_summary, dict) else {}
        existing_summary["candidate_qualitative_deliberation"] = deliberation_summary
        _write_json(summary_path, existing_summary)
        outcome.artifacts["candidate_qualitative_assessments"] = str(
            context.run_dir / "candidate_qualitative_assessments.jsonl")
        outcome.artifacts["candidate_assessment_reviews"] = str(
            context.run_dir / "candidate_assessment_reviews.jsonl")
    return outcome


def mean_objective_value(data: DataBundle, ids: list[str],
                         objectives: list[BranchObjective], top_k: int) -> float:
    metrics = slate_metrics_all_objectives(data, ids, objectives, top_k=top_k)
    return round(mean(metrics.values()), 6) if metrics else 0.0
