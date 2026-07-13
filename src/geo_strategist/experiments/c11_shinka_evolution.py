"""C11: Multi-model Shinka-style evolution (optional condition).

Evolutionary search over *scoring strategies* (LLM-proposed evaluation
programs: component-emphasis multipliers + action preferences), inspired by
ShinkaEvolve's evolve/evaluate loop: high-throughput mutation on the fast
model, selected critique on the pro model, generated population-scoring code
repaired by the code model and executed in the sandbox, synthesis on the
synthesis model, and a bounded judge call for the final tie-break.

Fitness is external and data-grounded: a strategy's slate is scored by the
mean of the five branch-objective metrics; novelty rejection compares slate
overlap against the C9 baseline slate and existing population members.

Model-call budget guards (enforced, then reported in
``c11_model_call_summary.json``): pro <= 15% of calls, judge <= 3%,
fast >= 70%.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Callable

from geo_strategist.agent.codeexec.interpreter import Interpreter
from geo_strategist.agent.codeexec.sandbox_guard import scan_generated_code
from geo_strategist.experiments.ai_scientist_loop import _strip_code_fences
from geo_strategist.experiments.branch_objectives import BranchObjective
from geo_strategist.experiments.deterministic_evaluation_engine import (
    DEFAULT_WEIGHTS,
    DataBundle,
    apply_reviews,
    rank_candidates,
)
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _write_json, _write_jsonl
from geo_strategist.experiments.live_common import (
    LiveConditionResult,
    LiveRunContext,
    build_live_proposals,
    extract_json_block,
    failure_result,
    slate_metrics_all_objectives,
    slate_overlap,
    validate_llm_ranking,
)
from geo_strategist.providers.base import ChatResult

# (prompt, system, purpose, model) -> ChatResult
MultiModelCall = Callable[[str, str | None, str, str], ChatResult]

_SYSTEM_PROMPT = (
    "You are part of an evolutionary search over hospital-location scoring "
    "strategies. Strategies are JSON programs over fixed data components; you "
    "never invent facts, only strategy parameters. Reply with valid JSON."
)

_COMPONENTS = tuple(DEFAULT_WEIGHTS)
_ACTIONS = ("build", "reorganize", "consolidate")
# Weight-space strategies legitimately converge on overlapping slates, so
# novelty rejection distinguishes the C9-baseline check (encourage genuinely
# different slates) from the population check (only reject near-duplicates).
_NOVELTY_MAX_BASELINE_OVERLAP = 0.8
_NOVELTY_MAX_POPULATION_OVERLAP = 0.95


def _strategy_slate(data: DataBundle, strategy: dict[str, Any], top_k: int) -> list[str]:
    """Deterministically evaluate one strategy program into a slate."""

    emphasis = strategy.get("emphasis") or {}
    weights = {
        name: DEFAULT_WEIGHTS[name] * float(_safe_mult(emphasis.get(name)))
        for name in _COMPONENTS
    }
    total = sum(weights.values()) or 1.0
    weights = {name: value / total for name, value in weights.items()}
    ranked = rank_candidates(data, weights)
    action_preference = strategy.get("action_preference") or {}
    for row in ranked:
        row["_adjusted"] = row["composite_score"] * _safe_mult(
            action_preference.get(row["action_type"]))
    ranked.sort(key=lambda row: (-row["_adjusted"], row["candidate_id"]))
    return [row["candidate_id"] for row in ranked[: top_k * 4]]


def _safe_mult(value: Any) -> float:
    try:
        return min(max(float(value), 0.25), 4.0)
    except (TypeError, ValueError):
        return 1.0


def _fitness(data: DataBundle, slate: list[str], objectives: list[BranchObjective], top_k: int) -> float:
    metrics = slate_metrics_all_objectives(data, slate, objectives, top_k=top_k)
    return round(mean(metrics.values()), 6) if metrics else 0.0


def run_c11_evolution_condition(
    repo_root: str | Path,
    spec: ConditionSpec,
    llm: MultiModelCall,
    data: DataBundle,
    objectives: list[BranchObjective],
    run_dir: Path,
    *,
    top_k: int = 5,
    population_size: int = 40,
    generations: int = 3,
    elite_count: int = 8,
    c9_baseline_slate: list[str] | None = None,
    exec_timeout: int = 60,
) -> LiveConditionResult:
    root = Path(repo_root).resolve()
    context = LiveRunContext(run_dir, condition_group=spec.group)
    roles = spec.model_roles
    call_counts: dict[str, int] = {"fast": 0, "pro": 0, "code": 0, "synthesis": 0, "judge": 0}

    def call(prompt: str, purpose: str, role: str) -> ChatResult | None:
        # Share gates apply from the second call of a role onwards; blocking
        # the first call would silently skip the documented single judge
        # tie-break (and the first critique) on efficient small runs.
        total = sum(call_counts.values()) or 1
        if role == "pro" and call_counts["pro"] >= 1 and (call_counts["pro"] + 1) / (total + 1) > 0.15:
            return None
        if role == "judge" and call_counts["judge"] >= 1 and (call_counts["judge"] + 1) / (total + 1) > 0.03:
            return None
        model = {
            "fast": roles.get("mutation"), "pro": roles.get("critique"),
            "code": roles.get("code_repair"), "synthesis": roles.get("synthesis"),
            "judge": roles.get("judge"),
        }[role] or roles.get("mutation", "deepseek-v4-flash")
        result = llm(prompt, _SYSTEM_PROMPT, purpose, model)
        context.record_call(purpose, prompt, result)
        call_counts[role] += 1
        return result

    strategy_schema = (
        '{"strategies": [{"description": "...", '
        '"emphasis": {' + ", ".join(f'"{c}": <0.25-4.0>' for c in _COMPONENTS) + '}, '
        '"action_preference": {' + ", ".join(f'"{a}": <0.25-4.0>' for a in _ACTIONS) + '}}]}'
    )
    objective_text = "\n".join(f"- {o.slug}: {o.description}" for o in objectives)

    population: list[dict[str, Any]] = []
    diversity_rows: list[dict[str, Any]] = []
    rejected_for_novelty = 0
    next_id = 0

    def admit(strategy: dict[str, Any], generation: int, parents: list[str], origin: str) -> bool:
        nonlocal next_id, rejected_for_novelty
        slate = _strategy_slate(data, strategy, top_k)
        if not slate:
            return False
        baseline_overlap = slate_overlap(slate, c9_baseline_slate or [], top_k=top_k)
        max_member_overlap = max(
            (slate_overlap(slate, member["slate"], top_k=top_k) for member in population),
            default=0.0,
        )
        rejected = (baseline_overlap > _NOVELTY_MAX_BASELINE_OVERLAP
                    or max_member_overlap > _NOVELTY_MAX_POPULATION_OVERLAP)
        individual_id = f"g{generation}_i{next_id}"
        next_id += 1
        row = {
            "individual_id": individual_id, "generation": generation,
            "parent_ids": parents, "origin": origin,
            "strategy": strategy, "slate": slate[: top_k * 2],
            "fitness": _fitness(data, slate, objectives, top_k),
            "overlap_vs_c9_baseline": baseline_overlap,
            "max_overlap_vs_population": max_member_overlap,
        }
        diversity_rows.append({
            "individual_id": individual_id, "generation": generation,
            "origin": origin, "fitness": row["fitness"],
            "overlap_vs_c9_baseline": baseline_overlap,
            "max_overlap_vs_population": max_member_overlap,
            "admitted": not rejected,
        })
        if rejected:
            rejected_for_novelty += 1
            return False
        population.append(row)
        return True

    # ---- population initialization -----------------------------------------
    batch = 8
    while len(population) < population_size:
        want = min(batch, population_size - len(population))
        result = call(
            f"OBJECTIVES:\n{objective_text}\n\n"
            f"Propose {want} diverse scoring strategies for ranking hospital "
            "location/reorganization candidates. Components: "
            + ", ".join(_COMPONENTS) + ".\nReply as JSON: " + strategy_schema,
            "population_init", "fast",
        )
        if result is None or not result.ok:
            if result is not None and result.error_class in ("live_auth_failed", "live_rate_limited"):
                failed = failure_result(spec, result.error_class, result.error_detail or "init failed")
                failed.finalize_artifacts(context)
                return failed
            break
        parsed = extract_json_block(result.text) or {}
        admitted_any = False
        for strategy in (parsed.get("strategies") or [])[:want]:
            if isinstance(strategy, dict):
                admitted_any = admit(strategy, 0, [], "init") or admitted_any
        if not admitted_any and call_counts["fast"] > population_size:
            break
    if not population:
        failed = failure_result(spec, "live_error", "population initialization produced no admissible strategies")
        failed.finalize_artifacts(context)
        return failed

    # ---- generated population-scoring code (sandbox) -----------------------
    code_dir = context.run_dir / "generated_code"
    code_dir.mkdir(parents=True, exist_ok=True)
    code_stats = {"generated": 0, "executed_ok": 0, "repaired": 0, "failed": 0}
    strategies_path = context.run_dir / "population_strategies.json"
    _write_json(strategies_path, [
        {"individual_id": m["individual_id"], "strategy": m["strategy"], "slate": m["slate"]}
        for m in population
    ])
    code_prompt = (
        f"Write a Python 3.11 stdlib-only script that reads {strategies_path} "
        "(JSON list of {individual_id, strategy, slate}) and prints one JSON "
        'object: {"diversity": {"mean_pairwise_slate_overlap": <float>}, '
        '"per_individual": [{"individual_id": "...", "slate_size": <int>}]}. '
        "Slate overlap = |A∩B|/|A∪B| over the first 5 entries. No imports "
        "beyond json/itertools/statistics/pathlib; no network, no os.environ, "
        "no subprocess, no file writes. Reply with only the Python code."
    )
    result = call(code_prompt, "population_scoring_code", "code")
    executed_payload: dict[str, Any] | None = None
    if result is not None and result.ok:
        code = _strip_code_fences(result.text)
        for attempt in range(2):
            code_path = code_dir / f"population_scoring_a{attempt}.py"
            code_path.write_text(code, encoding="utf-8")
            code_stats["generated"] += 1
            if scan_generated_code(code, repo_root=root, run_dir=context.run_dir):
                code_stats["failed"] += 1
                break
            execution = Interpreter(context.run_dir / "sandbox" / f"scoring_a{attempt}",
                                    timeout=exec_timeout).run(code)
            if execution.exc_type is None:
                parsed = extract_json_block("\n".join(execution.term_out))
                if isinstance(parsed, dict):
                    executed_payload = parsed
                    code_stats["executed_ok"] += 1
                break
            repair = call(
                f"Fix this script (stdlib only, same output contract). FAILURE:\n"
                f"{execution.stderr_tail[-1500:]}\n\nSCRIPT:\n```python\n{code[:10000]}\n```\n"
                "Reply with only the Python code.",
                "code_repair", "code",
            )
            if repair is None or not repair.ok:
                code_stats["failed"] += 1
                break
            code = _strip_code_fences(repair.text)
            code_stats["repaired"] += 1

    # ---- generations: parent sampling, mutation, crossover, elitism --------
    generation_summaries: list[dict[str, Any]] = []
    critique_text = ""
    for generation in range(1, generations + 1):
        population.sort(key=lambda m: -m["fitness"])
        elites = population[:elite_count]
        if generation == generations or generation == 1:
            critique = call(
                "Critique these elite scoring strategies; suggest 3 refinement "
                "directions the next generation should explore.\n"
                + json.dumps([{"strategy": e["strategy"], "fitness": e["fitness"]}
                              for e in elites[:4]], ensure_ascii=False)
                + '\nReply as JSON: {"refinement_directions": ["..."]}',
                "elite_critique", "pro",
            )
            if critique is not None and critique.ok:
                parsed = extract_json_block(critique.text) or {}
                critique_text = "; ".join(
                    str(d) for d in (parsed.get("refinement_directions") or [])[:3])

        children_target = max(population_size // 2, 8)
        admitted = 0
        mutation_batches = -(-children_target // batch)
        for _batch_index in range(mutation_batches):
            # Parent sampling balances exploitation (elites) and exploration.
            import random
            parents = [
                random.choice(elites) if index % 2 == 0 else random.choice(population)
                for index in range(min(batch, children_target - admitted))
            ]
            if not parents:
                break
            result = call(
                f"OBJECTIVES:\n{objective_text}\n"
                + (f"REFINEMENT DIRECTIONS: {critique_text}\n" if critique_text else "")
                + "Mutate each parent strategy into one child (change 1-3 "
                  "parameters meaningfully):\n"
                + json.dumps([{"parent_id": p["individual_id"], "strategy": p["strategy"],
                               "fitness": p["fitness"]} for p in parents], ensure_ascii=False)
                + "\nReply as JSON: " + strategy_schema,
                "mutation", "fast",
            )
            if result is None or not result.ok:
                continue
            parsed = extract_json_block(result.text) or {}
            for parent, strategy in zip(parents, parsed.get("strategies") or []):
                if isinstance(strategy, dict) and admit(
                        strategy, generation, [parent["individual_id"]], "mutation"):
                    admitted += 1
        # one crossover batch per generation
        if len(elites) >= 2:
            result = call(
                "Cross over these parent pairs into one child strategy each "
                "(combine strengths):\n"
                + json.dumps([
                    {"parent_a": elites[i]["strategy"], "parent_b": elites[i + 1]["strategy"]}
                    for i in range(0, min(len(elites) - 1, 4), 2)
                ], ensure_ascii=False)
                + "\nReply as JSON: " + strategy_schema,
                "crossover", "fast",
            )
            if result is not None and result.ok:
                parsed = extract_json_block(result.text) or {}
                for strategy in (parsed.get("strategies") or [])[:2]:
                    if isinstance(strategy, dict):
                        admit(strategy, generation, [e["individual_id"] for e in elites[:2]], "crossover")

        # survivor selection: elites + best children up to population_size
        population.sort(key=lambda m: -m["fitness"])
        population[:] = population[:population_size]
        generation_summaries.append({
            "generation": generation,
            "population_size": len(population),
            "best_fitness": population[0]["fitness"],
            "mean_fitness": round(mean(m["fitness"] for m in population), 6),
            "children_admitted": admitted,
            "rejected_for_novelty_cumulative": rejected_for_novelty,
            "refinement_directions": critique_text,
        })

    population.sort(key=lambda m: -m["fitness"])
    elites = population[:elite_count]

    # ---- final synthesis + judge tie-break ----------------------------------
    final_rows: list[dict[str, Any]] = []
    synthesis_note = ""
    result = call(
        "Synthesize the final top-{k} candidate slate from these elite "
        "strategies' slates (choose candidate_ids appearing in them, balancing "
        "all objectives):\n".format(k=top_k)
        + json.dumps([{"individual_id": e["individual_id"], "fitness": e["fitness"],
                       "slate_head": e["slate"][:top_k]} for e in elites], ensure_ascii=False)
        + '\nReply as JSON: {"final_slate": [{"candidate_id": "...", "rationale": "..."}], '
          '"strategy_summary": "..."}',
        "final_synthesis", "synthesis",
    )
    if result is not None and result.ok:
        parsed = extract_json_block(result.text) or {}
        final_rows, _rejected = validate_llm_ranking(parsed.get("final_slate") or [], data)
        synthesis_note = str(parsed.get("strategy_summary") or "")[:4000]
    if len(final_rows) < top_k:
        seen = {row["candidate_id"] for row in final_rows}
        for elite in elites:
            for cid in elite["slate"]:
                if len(final_rows) >= top_k:
                    break
                if cid not in seen:
                    final_rows.append({"candidate_id": cid,
                                       "rationale": f"from elite {elite['individual_id']}"})
                    seen.add(cid)
    judge_note = ""
    judge = call(
        "As final judge, confirm or reorder this slate (same candidate_ids "
        "only) and state the tie-break rationale:\n"
        + json.dumps(final_rows, ensure_ascii=False)
        + '\nReply as JSON: {"final_slate": [{"candidate_id": "..."}], "rationale": "..."}',
        "final_judge", "judge",
    )
    if judge is not None and judge.ok:
        parsed = extract_json_block(judge.text) or {}
        reordered, _rej = validate_llm_ranking(parsed.get("final_slate") or [], data)
        if len(reordered) >= min(top_k, len(final_rows)) and {
            r["candidate_id"] for r in reordered
        } <= {r["candidate_id"] for r in final_rows}:
            rationale_by_id = {r["candidate_id"]: r.get("rationale", "") for r in final_rows}
            final_rows = [{"candidate_id": r["candidate_id"],
                           "rationale": rationale_by_id.get(r["candidate_id"], "")}
                          for r in reordered]
        judge_note = str(parsed.get("rationale") or "")[:1000]

    proposals = build_live_proposals(final_rows, data, spec, top_k=top_k)
    proposals, review_rows = apply_reviews(proposals, 1)

    # ---- artifacts -----------------------------------------------------------
    _write_jsonl(context.run_dir / "c11_population.jsonl", population)
    _write_json(context.run_dir / "c11_generation_summary.json", generation_summaries)
    _write_json(context.run_dir / "c11_elites.json", elites)
    diversity_path = context.run_dir / "c11_variant_diversity.csv"
    with diversity_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(diversity_rows[0].keys()) if diversity_rows else
                                ["individual_id", "generation", "origin", "fitness",
                                 "overlap_vs_c9_baseline", "max_overlap_vs_population", "admitted"])
        writer.writeheader()
        writer.writerows(diversity_rows)

    total_calls = sum(call_counts.values()) or 1
    call_summary = {
        **context.ledger.summary(),
        "role_call_counts": call_counts,
        "role_shares": {role: round(count / total_calls, 4) for role, count in call_counts.items()},
        "budget_checks": {
            # The first call of a gated role is always permitted (the judge
            # tie-break is defined as exactly one bounded call), so each
            # budget allows max(1, share-of-total).
            "pro_share_le_15pct": call_counts["pro"] <= max(1, int(0.15 * total_calls)),
            "judge_share_le_3pct": call_counts["judge"] <= max(1, int(0.03 * total_calls)),
            "fast_share_ge_70pct": call_counts["fast"] / total_calls >= 0.70,
        },
    }
    _write_json(context.run_dir / "c11_model_call_summary.json", call_summary)

    synthesis_report = context.run_dir / "c11_final_synthesis_report.md"
    synthesis_report.write_text(
        "# C11 Final Synthesis\n\n"
        f"Strategy summary (synthesis model): {synthesis_note or 'not_available'}\n\n"
        f"Judge tie-break rationale: {judge_note or 'not_available'}\n\n"
        f"Population diversity code executed: {bool(executed_payload)}; "
        f"external diversity metrics: {json.dumps(executed_payload or {}, ensure_ascii=False)[:2000]}\n",
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
                 data, [r["candidate_id"] for r in final_rows], [o], top_k=top_k)[o.slug],
             "winner_top_candidates": [r["candidate_id"] for r in final_rows][:top_k]}
            for o in objectives
        ],
        generated_code_stats=code_stats,
        review_rows=review_rows,
        due_diligence=list(dict.fromkeys(due_diligence)),
        narrative_sections={
            "method_summary": synthesis_note,
            "judge_rationale": judge_note,
        },
        steps_run=total_calls,
        model_call_summary=call_summary,
        artifacts={
            "c11_population": str(context.run_dir / "c11_population.jsonl"),
            "c11_generation_summary": str(context.run_dir / "c11_generation_summary.json"),
            "c11_elites": str(context.run_dir / "c11_elites.json"),
            "c11_model_call_summary": str(context.run_dir / "c11_model_call_summary.json"),
            "c11_variant_diversity": str(diversity_path),
            "c11_final_synthesis_report": str(synthesis_report),
        },
    )
    outcome.finalize_artifacts(context)
    outcome.model_call_summary = call_summary
    return outcome
