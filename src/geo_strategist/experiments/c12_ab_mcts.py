"""C12: AB-MCTS-style adaptive branching (optional condition).

Search tree over *proposal programs*: each node carries an evaluation_spec,
generated scoring code, the executed ranking, external metrics, a proposal
draft, and review feedback. Inspired by Sakana AI's AB-MCTS: at every step
the controller decides whether to **go wider** (expand a new sibling under
the root — explore) or **go deeper** (refine/criticize/revise the current
best node — exploit) based on the externally measured improvement rate, not
model self-scoring. Actions: expand, refine, crossover, criticize, revise.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

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
from geo_strategist.experiments.deterministic_evaluation_engine import DataBundle, apply_reviews
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _write_json, _write_jsonl
from geo_strategist.experiments.live_common import (
    LiveConditionResult,
    LiveRunContext,
    build_live_proposals,
    extract_json_block,
    failure_result,
    slate_metrics_all_objectives,
    validate_llm_ranking,
)

_SYSTEM_PROMPT = (
    "You are a node-program generator in an adaptive branching search over "
    "hospital-location proposal programs. Never invent facts; candidates "
    "exist only as candidate_id values in the given data files. Reply with "
    "valid JSON or code exactly as asked."
)


def run_c12_ab_mcts_condition(
    repo_root: str | Path,
    spec: ConditionSpec,
    llm: MultiModelCall,
    data: DataBundle,
    objectives: list[BranchObjective],
    run_dir: Path,
    *,
    top_k: int = 5,
    max_llm_calls: int = 24,
    max_debug_depth: int = 2,
    exec_timeout: int = 90,
) -> LiveConditionResult:
    root = Path(repo_root).resolve()
    context = LiveRunContext(run_dir, condition_group=spec.group)
    roles = spec.model_roles
    calls_used = 0
    code_dir = context.run_dir / "generated_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    data_files = _data_files_block(root)
    objective_text = "\n".join(f"- {o.slug}: {o.description}" for o in objectives)
    nodes: list[dict[str, Any]] = []
    node_metrics: list[dict[str, Any]] = []
    code_stats = {"generated": 0, "executed_ok": 0, "repaired": 0, "failed": 0}

    def call(prompt: str, purpose: str, role: str):
        nonlocal calls_used
        if calls_used >= max_llm_calls:
            return None
        calls_used += 1
        model = roles.get(role) or roles.get("expand", "deepseek-v4-flash")
        result = llm(prompt, _SYSTEM_PROMPT, purpose, model)
        context.record_call(purpose, prompt, result)
        return result

    def node_fitness(ranking_ids: list[str]) -> float:
        metrics = slate_metrics_all_objectives(data, ranking_ids, objectives, top_k=top_k)
        return round(mean(metrics.values()), 6) if metrics else 0.0

    def execute_program(node_id: str, spec_text: str, code: str) -> dict[str, Any]:
        """Run generated scoring code with code-model repair on failure."""

        nonlocal code_stats
        current = code
        for attempt in range(max_debug_depth + 1):
            code_path = code_dir / f"{node_id}_a{attempt}.py"
            code_path.write_text(current, encoding="utf-8")
            code_stats["generated"] += 1
            if scan_generated_code(current, repo_root=root, run_dir=context.run_dir):
                code_stats["failed"] += 1
                return {"status": "sandbox_violation"}
            execution = Interpreter(context.run_dir / "sandbox" / f"{node_id}_a{attempt}",
                                    timeout=exec_timeout).run(current)
            if execution.exc_type is None:
                payload = _parse_exec_stdout(execution.term_out)
                valid, _rejected = validate_llm_ranking((payload or {}).get("ranking") or [], data)
                if valid:
                    code_stats["executed_ok"] += 1
                    ids = [row["candidate_id"] for row in valid]
                    return {"status": "success", "ranking_ids": ids,
                            "internal_metrics": (payload or {}).get("metrics") or {},
                            "code_path": str(code_path)}
                failure_tail = "stdout lacked required JSON ranking with valid candidate_ids"
            else:
                failure_tail = execution.stderr_tail[-1500:]
            if attempt >= max_debug_depth:
                break
            repair = call(
                f"{data_files}\nRepair this proposal-program scoring script.\n{_CODE_RULES}\n"
                f"EVALUATION SPEC:\n{spec_text[:1500]}\nFAILURE:\n{failure_tail}\n\n"
                f"SCRIPT:\n```python\n{current[:10000]}\n```\nReply with only the Python code.",
                "code_repair", "code_repair",
            )
            if repair is None or not repair.ok:
                break
            current = _strip_code_fences(repair.text)
            code_stats["repaired"] += 1
        code_stats["failed"] += 1
        return {"status": "failed"}

    def make_node(parent_id: str | None, action: str, prompt_extra: str, role: str) -> dict[str, Any] | None:
        node_id = f"n{len(nodes)}"
        result = call(
            f"{data_files}\nOBJECTIVES:\n{objective_text}\n{prompt_extra}\n\n"
            "Produce a proposal program: an evaluation_spec (how to score "
            "candidates and why) and a Python script implementing it with "
            'objective_slug "combined".\n'
            f"{_CODE_RULES}\n"
            'Reply as JSON: {"evaluation_spec": "...", "proposal_draft": "...", '
            '"python_code": "<the full script as a string>"}',
            f"{action}_{node_id}", role,
        )
        if result is None or not result.ok:
            return None
        parsed = extract_json_block(result.text) or {}
        code = _strip_code_fences(str(parsed.get("python_code") or ""))
        if not code and "```" in result.text:
            # Model replied with a bare code block instead of the JSON wrapper.
            code = _strip_code_fences(result.text)
            if not code.strip().startswith(("import", "from", "#", '"""')):
                code = ""
        if not code:
            return None
        spec_text = str(parsed.get("evaluation_spec") or "")[:4000]
        execution = execute_program(node_id, spec_text, code)
        node = {
            "node_id": node_id, "parent_id": parent_id, "action": action,
            "evaluation_spec": spec_text,
            "proposal_draft": str(parsed.get("proposal_draft") or "")[:4000],
            "status": execution["status"],
            "ranking_ids": execution.get("ranking_ids", []),
            "internal_metrics": execution.get("internal_metrics", {}),
            "code_path": execution.get("code_path"),
            "review_feedback": "",
            "value": node_fitness(execution.get("ranking_ids", [])) if execution["status"] == "success" else 0.0,
        }
        nodes.append(node)
        node_metrics.append({
            "node_id": node_id, "parent_id": parent_id, "action": action,
            "status": node["status"], "value": node["value"],
            "objective_metrics": slate_metrics_all_objectives(
                data, node["ranking_ids"], objectives, top_k=top_k) if node["ranking_ids"] else {},
        })
        return node

    # ---- root expansion -------------------------------------------------------
    root_node = make_node(None, "expand", "This is the ROOT program: a balanced first attempt.", "expand")
    if root_node is None or root_node["status"] != "success":
        # one retry as a fresh sibling before declaring failure
        root_node = make_node(None, "expand", "Previous root attempt failed; produce a simpler, robust program.", "expand")
    if root_node is None or not any(n["status"] == "success" for n in nodes):
        failed = failure_result(spec, "live_error", "no root proposal program executed successfully")
        failed.generated_code_stats = code_stats
        _write_json(context.run_dir / "c12_tree.json", {"nodes": nodes})
        failed.finalize_artifacts(context)
        return failed

    # ---- adaptive loop: width vs depth from external improvement --------------
    last_improvement = 1.0
    consecutive_failures = 0
    while calls_used < max_llm_calls - 4:
        successes = [n for n in nodes if n["status"] == "success"]
        best = max(successes, key=lambda n: n["value"])
        go_deeper = last_improvement > 0.005 and best["parent_id"] is not None or (
            last_improvement > 0.02
        )
        before = best["value"]
        if go_deeper:
            # criticize -> revise/refine the current best node
            critique = call(
                "Criticize this proposal program's evaluation spec and executed "
                "slate; name concrete weaknesses.\n"
                + json.dumps({"evaluation_spec": best["evaluation_spec"],
                              "slate_head": best["ranking_ids"][:top_k],
                              "value": best["value"]}, ensure_ascii=False)
                + '\nReply as JSON: {"critique": "..."}',
                f"criticize_{best['node_id']}", "node_evaluation",
            )
            critique_text = ""
            if critique is not None and critique.ok:
                critique_text = str((extract_json_block(critique.text) or {}).get("critique") or "")[:2000]
                best["review_feedback"] = critique_text
            child = make_node(
                best["node_id"], "refine",
                f"PARENT SPEC:\n{best['evaluation_spec'][:1500]}\n"
                f"PARENT VALUE: {best['value']}\nCRITIQUE:\n{critique_text}\n"
                "Refine the parent program to address the critique (go deeper).",
                "refine",
            )
        else:
            if len([n for n in nodes if n["parent_id"] is None]) >= 2 and len(nodes) >= 3:
                top_two = sorted(successes, key=lambda n: -n["value"])[:2]
                child = make_node(
                    top_two[0]["node_id"], "crossover",
                    "PARENT A SPEC:\n" + top_two[0]["evaluation_spec"][:1200]
                    + "\nPARENT B SPEC:\n" + top_two[-1]["evaluation_spec"][:1200]
                    + "\nCross the two parents into one stronger program.",
                    "expand",
                )
            else:
                child = make_node(
                    None, "expand",
                    "Existing programs plateaued; expand a NEW sibling with a "
                    "substantially different scoring angle (go wider).",
                    "expand",
                )
        after = max((n["value"] for n in nodes if n["status"] == "success"), default=before)
        last_improvement = after - before
        if child is None:
            # A malformed reply is not a terminal condition: push the search
            # toward widening and only stop after repeated failures.
            consecutive_failures += 1
            last_improvement = 0.0
            if consecutive_failures >= 3:
                break
        else:
            consecutive_failures = 0

    successes = [n for n in nodes if n["status"] == "success"]
    best = max(successes, key=lambda n: n["value"])

    # ---- final judge over the best path ---------------------------------------
    best_path = []
    cursor: dict[str, Any] | None = best
    by_id = {n["node_id"]: n for n in nodes}
    while cursor is not None:
        best_path.append(cursor["node_id"])
        cursor = by_id.get(cursor["parent_id"]) if cursor["parent_id"] else None
    best_path.reverse()

    final_rows = [{"candidate_id": cid, "rationale": f"best node {best['node_id']} executed ranking"}
                  for cid in best["ranking_ids"][:top_k]]
    judge_note = ""
    judge = call(
        "As final judge, confirm or reorder this slate (same candidate_ids only):\n"
        + json.dumps({"slate": final_rows, "node_value": best["value"],
                      "evaluation_spec": best["evaluation_spec"][:1500]}, ensure_ascii=False)
        + '\nReply as JSON: {"final_slate": [{"candidate_id": "..."}], "rationale": "..."}',
        "final_judge", "judge",
    )
    if judge is not None and judge.ok:
        parsed = extract_json_block(judge.text) or {}
        reordered, _rej = validate_llm_ranking(parsed.get("final_slate") or [], data)
        if reordered and {r["candidate_id"] for r in reordered} <= {r["candidate_id"] for r in final_rows}:
            final_rows = [{"candidate_id": r["candidate_id"],
                           "rationale": f"best node {best['node_id']} executed ranking (judge-ordered)"}
                          for r in reordered]
        judge_note = str(parsed.get("rationale") or "")[:1000]

    proposals = build_live_proposals(final_rows, data, spec, top_k=top_k)
    proposals, review_rows = apply_reviews(proposals, 1)

    # ---- artifacts --------------------------------------------------------------
    _write_json(context.run_dir / "c12_tree.json", {
        "nodes": [
            {key: node[key] for key in
             ("node_id", "parent_id", "action", "status", "value",
              "evaluation_spec", "proposal_draft", "review_feedback", "code_path")}
            | {"ranking_head": node["ranking_ids"][:top_k]}
            for node in nodes
        ],
        "best_node_id": best["node_id"],
        "best_path": best_path,
    })
    _write_jsonl(context.run_dir / "c12_node_metrics.jsonl", node_metrics)
    (context.run_dir / "c12_best_path.md").write_text(
        "# C12 Best Path\n\n"
        + "\n".join(
            f"- `{node_id}` ({by_id[node_id]['action']}) — value {by_id[node_id]['value']}"
            for node_id in best_path)
        + f"\n\nJudge rationale: {judge_note or 'not_available'}\n",
        encoding="utf-8",
    )
    (context.run_dir / "c12_final_report.md").write_text(
        "# C12 Final Report (working copy)\n\n"
        "See the orchestrator-level C10_proposal_report.md for the full report.\n\n"
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
                 data, best["ranking_ids"], [o], top_k=top_k)[o.slug],
             "winner_top_candidates": best["ranking_ids"][:top_k]}
            for o in objectives
        ],
        generated_code_stats=code_stats,
        review_rows=review_rows,
        due_diligence=list(dict.fromkeys(due_diligence)),
        narrative_sections={
            "method_summary": (
                f"AB-MCTS-style search: {len(nodes)} nodes, best value {best['value']}, "
                f"best path {' -> '.join(best_path)}."),
            "judge_rationale": judge_note,
        },
        steps_run=calls_used,
        artifacts={
            "c12_tree": str(context.run_dir / "c12_tree.json"),
            "c12_node_metrics": str(context.run_dir / "c12_node_metrics.jsonl"),
            "c12_best_path": str(context.run_dir / "c12_best_path.md"),
            "c12_final_report": str(context.run_dir / "c12_final_report.md"),
        },
    )
    outcome.finalize_artifacts(context)
    return outcome
