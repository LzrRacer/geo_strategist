"""C1/C2: Vanilla direct LLM baseline — one live call, no tools, no code.

The model receives the same data context every other condition sees and
returns a ranked slate in a single pass. No tool use, no generated-code
execution, no review loop. Output is validated against the real candidate
universe like every live condition.
"""

from __future__ import annotations

import json
from pathlib import Path

from geo_strategist.experiments.ai_scientist_loop import LlmCall
from geo_strategist.experiments.branch_objectives import BranchObjective
from geo_strategist.experiments.deterministic_evaluation_engine import DataBundle
from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.condition_utils import _write_json
from geo_strategist.experiments.live_common import (
    LiveConditionResult,
    LiveRunContext,
    build_live_proposals,
    data_context_json,
    extract_json_block,
    failure_result,
    narrative_fabrication_flags,
    slate_metrics_all_objectives,
    validate_llm_ranking,
)

_SYSTEM_PROMPT = (
    "You are a healthcare-facility strategy analyst. Answer in a single pass. "
    "Never invent hospital names, addresses, coordinates, parcels, or "
    "financial values; candidates exist only as candidate_id values from the "
    "provided data. Reply with valid JSON."
)


def run_vanilla_condition(
    repo_root: str | Path,
    spec: ConditionSpec,
    llm: LlmCall,
    data: DataBundle,
    objectives: list[BranchObjective],
    run_dir: Path,
    *,
    top_k: int = 5,
) -> LiveConditionResult:
    context = LiveRunContext(run_dir, condition_group=spec.group)
    prompt = (
        f"DATA CONTEXT:\n{data_context_json(data)}\n\n"
        f"Rank the best {top_k} candidates for hospital location / "
        "reorganization action in this study area, balancing elderly demand, "
        "emergency access, reorganization feasibility, financial risk, and "
        "evidence completeness. Use only candidate_id values from the data "
        "context.\n"
        'Reply as JSON: {"ranked": [{"candidate_id": "...", "rationale": "..."}], '
        '"summary": "..."}'
    )
    result = llm(prompt, _SYSTEM_PROMPT, "vanilla_single_pass")
    context.record_call("vanilla_single_pass", prompt, result)
    if not result.ok:
        failed = failure_result(spec, result.error_class, result.error_detail or "provider failure")
        failed.finalize_artifacts(context)
        return failed

    parsed = extract_json_block(result.text)
    ranked_rows, rejected = validate_llm_ranking(
        (parsed or {}).get("ranked") if isinstance(parsed, dict) else None, data,
    )
    if not ranked_rows:
        failed = failure_result(spec, "live_error", "no valid candidate_ids in single-pass response")
        failed.finalize_artifacts(context)
        return failed

    proposals = build_live_proposals(ranked_rows, data, spec, top_k=top_k)
    summary_text = str((parsed or {}).get("summary") or "")[:4000]
    narrative = {"executive_summary": summary_text}
    flags = narrative_fabrication_flags(summary_text, data)
    if flags:
        narrative["executive_summary_flags"] = "; ".join(flags)

    due_diligence: list[str] = []
    for proposal in proposals:
        due_diligence.extend(proposal.get("required_due_diligence") or [])
    if rejected:
        due_diligence.append(
            f"C1 proposed {len(rejected)} unknown candidate_id value(s); they were "
            "rejected by the grounding validator and are listed in the trace."
        )
    due_diligence.extend(data.data_notes)

    _write_json(context.run_dir / "final_slate.json", {
        "final_slate": ranked_rows[:top_k],
        "rejected_candidate_ids": rejected,
        "objective_metrics": slate_metrics_all_objectives(
            data, [row["candidate_id"] for row in ranked_rows], objectives, top_k=top_k),
    })

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
        due_diligence=list(dict.fromkeys(due_diligence)),
        narrative_sections=narrative,
        steps_run=1,
        artifacts={"final_slate": str(context.run_dir / "final_slate.json")},
    )
    outcome.finalize_artifacts(context)
    return outcome
