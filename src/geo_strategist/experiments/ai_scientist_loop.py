"""AI Scientist-style condition loop (C9 Gemini, C10 DeepSeek).

Follows the AI Scientist-v2 shape from the local reference
(``references/local/ai_scientist/ai_scientist.txt``): idea/hypothesis
generation → experiment design → LLM-generated experiment code → sandboxed
execution → debug/rerun → best-first search over a journal of nodes →
metrics → proposal write-up → review → revision.

Adaptations for this domain:

- Nodes are scoring-code variants per branch objective; the five shared
  branch objectives replace the reference's experiment stages.
- Node fitness is the *external* objective metric computed from the
  deterministic evaluation-model components over the ranking that the
  generated code actually produced when executed — the LLM cannot win by
  asserting numbers, only by producing code whose executed output ranks
  well-evidenced candidates highly.
- The final proposal is derived from executed generated-code artifacts
  (branch-winner rankings + synthesis over them), never recomputed by the
  deterministic baseline.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from geo_strategist.agent.codeexec.interpreter import Interpreter
from geo_strategist.agent.codeexec.sandbox_guard import scan_generated_code
from geo_strategist.experiments.branch_objectives import BranchObjective
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DataBundle,
    apply_reviews,
)
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _write_json, _write_jsonl
from geo_strategist.experiments.live_common import (
    LiveConditionResult,
    LiveRunContext,
    build_live_proposals,
    data_context_json,
    extract_json_block,
    external_objective_metric,
    failure_result,
    narrative_fabrication_flags,
    slate_metrics_all_objectives,
    validate_llm_ranking,
)
from geo_strategist.providers.base import ChatResult

# An LLM caller: (prompt, system, purpose) -> ChatResult. Providers are
# adapted to this signature by the runners in run_condition_proposals.
LlmCall = Callable[[str, str | None, str], ChatResult]

_SYSTEM_PROMPT = (
    "You are a careful healthcare-facility strategy researcher. You must never "
    "invent hospital names, addresses, coordinates, parcels, or financial "
    "values. Candidates exist only as candidate_id values from the provided "
    "data. Reply with valid JSON when asked for JSON."
)

_CODE_RULES = """\
Rules for the Python script (violations abort execution):
- Python 3.11, standard library only (json, csv, math, statistics, pathlib, collections).
- Read ONLY the data files listed above, using the exact absolute paths given.
- Do NOT import socket/requests/urllib/http/subprocess; do NOT use os.system,
  os.environ, getenv, eval, exec, or file deletion; do NOT write files.
- Print exactly one JSON object to stdout as the LAST line:
  {"objective": "<objective_slug>", "ranking": [{"candidate_id": "...", "score": <float>}, ...],
   "metrics": {"<name>": <float>, ...}}
- "ranking" must contain at least 20 candidate_id values copied exactly from
  candidate_actions.jsonl, best first. Scores are your computed values.
- Handle missing/null fields defensively; missing data must lower confidence,
  not crash the script.
"""


class _StepBudget:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used = 0
        self._lock = threading.Lock()

    def take(self, count: int = 1) -> bool:
        with self._lock:
            if self._used + count > self._limit:
                return False
            self._used += count
            return True

    @property
    def used(self) -> int:
        with self._lock:
            return self._used

    def remaining(self) -> int:
        with self._lock:
            return self._limit - self._used


def _data_files_block(repo_root: Path) -> str:
    area = (repo_root / ".data/interim/study_area/tokyo_aichi_osaka").resolve()
    return f"""\
Data files (JSONL, one object per line), absolute paths:
- {area}/candidate_actions.jsonl
  fields: candidate_id, prefecture, municipality, candidate_action (build|reorganize|consolidate), priority_score
- {area}/municipality_scores_enriched.jsonl
  fields: prefecture, municipality, demand_pressure_score, population_aging_pressure_score,
  healthcare_supply_score, land_score, land_score_available, data_completeness_score
- {area}/municipality_feature_base.jsonl
  fields: prefecture, municipality, population_total_pct_change
- {area}/hospital_features.jsonl
  fields: prefecture, payback_years (hospital cash-flow workbook; model estimate)
"""


def _parse_exec_stdout(term_out: list[str]) -> dict[str, Any] | None:
    for line in reversed(term_out):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    parsed = extract_json_block("\n".join(term_out))
    return parsed if isinstance(parsed, dict) else None


def _strip_code_fences(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    return (match.group(1) if match else text).strip()


def run_ai_scientist_condition(
    repo_root: str | Path,
    spec: ConditionSpec,
    llm: LlmCall,
    data: DataBundle,
    objectives: list[BranchObjective],
    run_dir: Path,
    *,
    top_k: int = 5,
    max_review_rounds: int = 2,
    num_drafts: int = 10,
    agent_steps: int = 40,
    variants_per_branch: int = 6,
    max_debug_depth: int = 4,
    concurrency: int = 4,
    exec_timeout: int = 90,
) -> LiveConditionResult:
    root = Path(repo_root).resolve()
    context = LiveRunContext(run_dir, condition_group=spec.group)
    budget = _StepBudget(agent_steps)
    journal: list[dict[str, Any]] = []
    journal_lock = threading.Lock()
    code_dir = context.run_dir / "generated_code"
    code_dir.mkdir(parents=True, exist_ok=True)

    def call(prompt: str, purpose: str) -> ChatResult:
        result = llm(prompt, _SYSTEM_PROMPT, purpose)
        context.record_call(purpose, prompt, result)
        return result

    # ---- Stage 1: ideation ------------------------------------------------
    drafts: list[dict[str, Any]] = []
    data_context = data_context_json(data)
    ideation_batches = max(1, -(-num_drafts // 5))
    hard_failure: ChatResult | None = None
    for batch in range(ideation_batches):
        if not budget.take():
            break
        want = min(5, num_drafts - len(drafts))
        if want <= 0:
            break
        prompt = (
            f"DATA CONTEXT:\n{data_context}\n\n"
            f"Generate {want} distinct research hypotheses for ranking hospital "
            "location/reorganization candidates in this study area. Each "
            "hypothesis must state a scoring approach that could be implemented "
            "as a Python script over the given components.\n"
            'Reply as JSON: {"drafts": [{"draft_id": "d<n>", "hypothesis": "...", '
            '"scoring_approach": "...", "component_focus": ["demand", ...]}]}'
        )
        result = call(prompt, "ideation")
        if not result.ok:
            if result.error_class in ("live_auth_failed", "live_rate_limited"):
                hard_failure = result
                break
            continue
        parsed = extract_json_block(result.text)
        for row in (parsed or {}).get("drafts", []) if isinstance(parsed, dict) else []:
            if isinstance(row, dict) and row.get("hypothesis"):
                row.setdefault("draft_id", f"d{len(drafts) + 1}")
                drafts.append(row)
    if hard_failure is not None:
        failed = failure_result(spec, hard_failure.error_class, hard_failure.error_detail or "provider failure")
        failed.finalize_artifacts(context)
        return failed
    if not drafts:
        failed = failure_result(spec, "live_error", "ideation produced no usable drafts")
        failed.finalize_artifacts(context)
        return failed

    # ---- Stage 2: per-objective variant tree with debug loops -------------
    data_files = _data_files_block(root)
    reserve = 6  # keep steps for synthesis/report/review

    def run_variant(objective: BranchObjective, variant_index: int, draft: dict[str, Any]) -> dict[str, Any] | None:
        node_id = f"{objective.slug}_v{variant_index}"
        if budget.remaining() <= reserve or not budget.take():
            return None
        prompt = (
            f"{data_files}\n"
            f"BRANCH OBJECTIVE: {objective.label} — {objective.description}\n"
            f"HYPOTHESIS (draft {draft.get('draft_id')}): {draft.get('hypothesis')}\n"
            f"SCORING APPROACH: {draft.get('scoring_approach')}\n\n"
            "Write a complete Python script implementing this scoring approach "
            f"for the branch objective, with objective_slug \"{objective.slug}\".\n"
            f"{_CODE_RULES}\n"
            "Reply with only the Python code (a single ```python block is fine)."
        )
        result = call(prompt, f"codegen_{objective.slug}")
        node: dict[str, Any] = {
            "node_id": node_id, "objective": objective.slug,
            "draft_id": draft.get("draft_id"), "attempts": [], "status": "failed",
            "external_metric": None, "ranking_ids": [], "internal_metrics": {},
        }
        if not result.ok:
            node["attempts"].append({"stage": "codegen", "status": result.error_class})
            return node
        code = _strip_code_fences(result.text)
        for attempt in range(max_debug_depth + 1):
            code_path = code_dir / f"{node_id}_a{attempt}.py"
            code_path.write_text(code, encoding="utf-8")
            violations = scan_generated_code(code, repo_root=root, run_dir=context.run_dir)
            if violations:
                exec_summary = {"exc_type": "SandboxViolation",
                                "detail": [v["violation"] for v in violations]}
                stderr_tail = json.dumps(violations)
            else:
                execution = Interpreter(
                    context.run_dir / "sandbox" / f"{node_id}_a{attempt}",
                    timeout=exec_timeout,
                ).run(code)
                exec_summary = {"exc_type": execution.exc_type,
                                "returncode": execution.returncode,
                                "exec_time": round(execution.exec_time, 2)}
                stderr_tail = execution.stderr_tail
                if execution.exc_type is None:
                    payload = _parse_exec_stdout(execution.term_out)
                    ranking = (payload or {}).get("ranking")
                    valid, rejected = validate_llm_ranking(ranking or [], data)
                    if valid:
                        ids = [row["candidate_id"] for row in valid]
                        node.update({
                            "status": "success",
                            "ranking_ids": ids,
                            "rejected_candidate_ids": rejected,
                            "internal_metrics": (payload or {}).get("metrics") or {},
                            "external_metric": external_objective_metric(
                                data, ids, objective, top_k=top_k),
                            "code_path": str(code_path),
                        })
                        node["attempts"].append({"stage": f"exec_a{attempt}", "status": "success",
                                                 **exec_summary})
                        return node
                    exec_summary["exc_type"] = "InvalidRankingOutput"
                    stderr_tail = (
                        "Script ran but stdout did not contain the required JSON "
                        "with valid candidate_id values. Rejected ids: "
                        + ", ".join(rejected[:5])
                    )
            node["attempts"].append({"stage": f"exec_a{attempt}", "status": "failed", **exec_summary})
            if attempt >= max_debug_depth or budget.remaining() <= reserve or not budget.take():
                return node
            debug_prompt = (
                f"{data_files}\nThe following script failed. Fix it and reply with "
                "only the corrected Python code.\n"
                f"{_CODE_RULES}\n"
                f"FAILURE ({exec_summary.get('exc_type')}):\n{stderr_tail[-2000:]}\n\n"
                f"SCRIPT:\n```python\n{code[:12000]}\n```"
            )
            debug_result = call(debug_prompt, f"debug_{objective.slug}")
            if not debug_result.ok:
                node["attempts"].append({"stage": f"debug_a{attempt}", "status": debug_result.error_class})
                return node
            code = _strip_code_fences(debug_result.text)
        return node

    tasks: list[tuple[BranchObjective, int, dict[str, Any]]] = []
    for objective_index, objective in enumerate(objectives):
        for variant in range(variants_per_branch):
            draft = drafts[(objective_index * variants_per_branch + variant) % len(drafts)]
            tasks.append((objective, variant, draft))

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(run_variant, o, v, d) for o, v, d in tasks]
        for future in futures:
            node = future.result()
            if node is not None:
                with journal_lock:
                    journal.append(node)

    _write_jsonl(context.run_dir / "journal.jsonl", journal)

    # ---- Stage 3: best-first branch winners --------------------------------
    branch_results: list[dict[str, Any]] = []
    for objective in objectives:
        nodes = [n for n in journal if n["objective"] == objective.slug and n["status"] == "success"]
        winner = max(nodes, key=lambda n: n["external_metric"] or 0.0, default=None)
        branch_results.append({
            "objective": objective.slug,
            "objective_label": objective.label,
            "nodes_evaluated": len([n for n in journal if n["objective"] == objective.slug]),
            "nodes_succeeded": len(nodes),
            "winner_node_id": winner["node_id"] if winner else None,
            "winner_external_metric": winner["external_metric"] if winner else None,
            "winner_top_candidates": winner["ranking_ids"][:top_k] if winner else [],
        })

    successful_nodes = [n for n in journal if n["status"] == "success"]
    if not successful_nodes:
        failed = failure_result(spec, "live_error", "no generated-code node executed successfully")
        failed.branch_results = branch_results
        failed.generated_code_stats = _code_stats(journal)
        failed.finalize_artifacts(context)
        return failed

    # ---- Stage 4: synthesis of the final slate from branch winners ---------
    winners_payload = [
        {
            "objective": row["objective"],
            "external_metric": row["winner_external_metric"],
            "top_candidates": row["winner_top_candidates"],
        }
        for row in branch_results if row["winner_node_id"]
    ]
    final_rows: list[dict[str, Any]] = []
    synthesis_note = ""
    if budget.take():
        prompt = (
            f"DATA CONTEXT:\n{data_context}\n\n"
            f"BRANCH WINNERS (from executed scoring code):\n{json.dumps(winners_payload, ensure_ascii=False)}\n\n"
            f"Synthesize the final top-{top_k} slate balancing all five objectives. "
            "Use only candidate_id values that appear in the data context.\n"
            'Reply as JSON: {"final_slate": [{"candidate_id": "...", "rationale": "..."}], '
            '"strategy_summary": "..."}'
        )
        result = call(prompt, "synthesis")
        if result.ok:
            parsed = extract_json_block(result.text)
            if isinstance(parsed, dict):
                final_rows, _rejected = validate_llm_ranking(parsed.get("final_slate") or [], data)
                synthesis_note = str(parsed.get("strategy_summary") or "")[:4000]
    if len(final_rows) < top_k:
        # Fall back to aggregating the executed branch-winner rankings
        # (still live-agent artifacts, not the deterministic baseline).
        pooled: dict[str, list[float]] = {}
        for row in winners_payload:
            for position, cid in enumerate(row["top_candidates"]):
                pooled.setdefault(cid, []).append(float(top_k - min(position, top_k)))
        ordered = sorted(pooled.items(), key=lambda item: (-mean(item[1]), item[0]))
        existing = {row["candidate_id"] for row in final_rows}
        for cid, _scores in ordered:
            if len(final_rows) >= top_k:
                break
            if cid not in existing:
                final_rows.append({"candidate_id": cid,
                                   "rationale": "aggregated across executed branch-winner rankings"})
        synthesis_note = synthesis_note or (
            "Final slate aggregated from executed branch-winner rankings "
            "(synthesis call unavailable or under-filled)."
        )

    proposals = build_live_proposals(final_rows, data, spec, top_k=top_k)

    # ---- Stage 5: write-up narrative ---------------------------------------
    narrative: dict[str, str] = {"method_summary": synthesis_note}
    if budget.take():
        prompt = (
            "Write an executive summary (<= 250 words) for a hospital "
            "location/reorganization proposal report. Facts you may use are "
            "ONLY in this JSON (validated candidates and executed metrics):\n"
            + json.dumps({
                "final_slate": [
                    {"municipality": p["municipality"], "prefecture": p["prefecture"],
                     "action": p["action_type"], "composite_score": p["composite_score"]}
                    for p in proposals
                ],
                "branch_winners": winners_payload,
            }, ensure_ascii=False)
            + "\nDo not state absolute financial amounts; financial evidence is a "
              "prefecture-median model estimate. Reply with plain Markdown text."
        )
        result = call(prompt, "report")
        if result.ok:
            flags = narrative_fabrication_flags(result.text, data)
            narrative["executive_summary"] = result.text.strip()
            if flags:
                narrative["executive_summary_flags"] = "; ".join(flags)

    # ---- Stage 6: review + revision ----------------------------------------
    review_rows: list[dict[str, Any]] = []
    for round_index in range(max_review_rounds):
        if not budget.take():
            break
        prompt = (
            "Review this proposal slate as a skeptical healthcare-strategy "
            "reviewer. Score 1-10 and list concrete revision requests.\n"
            + json.dumps({
                "slate": [
                    {"municipality": p["municipality"], "action": p["action_type"],
                     "evidence_gaps": p.get("evidence_gaps"),
                     "rationale": p.get("llm_rationale", "")}
                    for p in proposals
                ],
                "executive_summary": narrative.get("executive_summary", ""),
            }, ensure_ascii=False)
            + '\nReply as JSON: {"score": <1-10>, "revision_requests": ["..."]}'
        )
        result = call(prompt, "review")
        if not result.ok:
            break
        parsed = extract_json_block(result.text) or {}
        requests = [str(r) for r in (parsed.get("revision_requests") or [])][:10]
        review_rows.append({
            "review_round": round_index + 1,
            "reviewer": f"{spec.provider}:{spec.model}",
            "score": parsed.get("score"),
            "revision_requests": requests,
        })
        if not requests:
            break
        for proposal in proposals:
            existing = set(proposal.get("required_due_diligence") or [])
            for request in requests[:5]:
                item = f"LLM reviewer follow-up: {request}"
                if item not in existing:
                    proposal.setdefault("required_due_diligence", []).append(item)
                    existing.add(item)
            proposal["proposal_status"] = "revised_proposal"
            proposal["revision_summary"] = f"revised for {len(requests)} live-reviewer findings"

    proposals, persona_rows = apply_reviews(proposals, 1)
    review_rows.extend(persona_rows)

    due_diligence: list[str] = []
    for proposal in proposals:
        due_diligence.extend(proposal.get("required_due_diligence") or [])
    due_diligence.extend(data.data_notes)

    ranked_rows = [
        {"rank": index + 1, "candidate_id": p["candidate_id"],
         "municipality": p["municipality"], "prefecture": p["prefecture"],
         "action_type": p["action_type"], "composite_score": p["composite_score"],
         "objective_metrics": slate_metrics_all_objectives(
             data, [p["candidate_id"]], objectives, top_k=1)}
        for index, p in enumerate(proposals)
    ]

    outcome = LiveConditionResult(
        spec=spec,
        execution_mode="live",
        comparable_for_e13=True,
        exclusion_reason=None,
        proposals=proposals,
        ranked_rows=ranked_rows,
        branch_results=branch_results,
        generated_code_stats=_code_stats(journal),
        review_rows=review_rows,
        due_diligence=list(dict.fromkeys(due_diligence)),
        narrative_sections=narrative,
        steps_run=budget.used,
        artifacts={
            "journal": str(context.run_dir / "journal.jsonl"),
            "generated_code_dir": str(code_dir),
        },
    )
    _write_json(context.run_dir / "final_slate.json", {
        "final_slate": final_rows,
        "strategy_summary": synthesis_note,
        "objective_metrics": slate_metrics_all_objectives(
            data, [row["candidate_id"] for row in final_rows], objectives, top_k=top_k),
    })
    outcome.artifacts["final_slate"] = str(context.run_dir / "final_slate.json")
    outcome.finalize_artifacts(context)
    return outcome


def _code_stats(journal: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = [a for node in journal for a in node.get("attempts", [])]
    exec_attempts = [a for a in attempts if str(a.get("stage", "")).startswith("exec")]
    return {
        "nodes": len(journal),
        "nodes_succeeded": sum(1 for n in journal if n.get("status") == "success"),
        "exec_attempts": len(exec_attempts),
        "exec_failures": sum(1 for a in exec_attempts if a.get("status") == "failed"),
        "sandbox_violations": sum(1 for a in attempts if a.get("exc_type") == "SandboxViolation"),
        "debug_rounds": sum(1 for a in attempts if str(a.get("stage", "")).startswith("debug")),
    }
