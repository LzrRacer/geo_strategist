"""Skill registry metadata for the Skills-unified contract.

Declares each skill's required inputs and produced outputs plus the
filesystem Skill package paths used by C9-C12 agentic Skills harnesses. The
machine-readable IO contract stays here; the agent-readable execution
procedure lives in ``.agents/skills/*/SKILL.md`` and the compatibility
``.claude/skills/*/SKILL.md`` tree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from geo_strategist.agent.skills import SKILLS_UNIFIED_CONTRACT

SkillStatus = Literal["succeeded", "blocked", "skipped", "failed", "degraded"]
SkillRun = Callable[[dict[str, Any]], "SkillResult"]


@dataclass(frozen=True)
class SkillResult:
    skill_id: str
    status: SkillStatus
    output_refs: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    issue_records: list[dict[str, Any]] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Skill:
    skill_id: str
    skill_name: str
    required_inputs: list[str]
    produced_outputs: list[str]
    skill_package_name: str
    codex_skill_path: str
    claude_skill_path: str
    run: SkillRun | None = None


# skill_id -> (required context keys, produced context keys)
SKILL_IO: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "inspect_available_data": ((), ("data_bundle", "data_inventory")),
    "generate_research_hypotheses": (("data_bundle",), ("hypotheses",)),
    "design_evaluation_model": (("hypotheses",), ("designs",)),
    "write_experiment_code": (("designs",), ("generated_code",)),
    "execute_generated_code": (("generated_code",), ("execution_results",)),
    "debug_failed_code": (("execution_results",), ("debugged_designs",)),
    "run_branch_search": (("execution_results",), ("branch_results", "proposals")),
    "expand_decision_regime": (("branch_results",), ("decision_regimes",)),
    "compare_search_branches": (("branch_results",), ("branch_comparison",)),
    "analyze_branch_divergence": (("branch_results",), ("search_diagnostics",)),
    "design_robustness_test": (("branch_results",), ("robustness_designs",)),
    "run_robustness_ablations": (("robustness_designs",), ("robustness_results",)),
    "identify_reversal_conditions": (("branch_results",), ("reversal_conditions",)),
    "review_proposal": (("proposals",), ("reviews", "revision_requests")),
    "revise_proposal": (("reviews",), ("proposals",)),
    "synthesize_decision_portfolios": (("branch_results",), ("portfolios", "proposals")),
    "prioritize_due_diligence": (("proposals",), ("due_diligence_plan",)),
    "write_final_condition_proposal": (("proposals",), ("condition_reports",)),
}

assert tuple(SKILL_IO) == SKILLS_UNIFIED_CONTRACT


def skill_package_name(skill_id: str) -> str:
    return "geo-" + skill_id.replace("_", "-")


def _skill_for(skill_id: str, required: tuple[str, ...], produced: tuple[str, ...]) -> Skill:
    package = skill_package_name(skill_id)
    return Skill(
        skill_id=skill_id,
        skill_name=skill_id.replace("_", " ").title(),
        required_inputs=list(required),
        produced_outputs=list(produced),
        skill_package_name=package,
        codex_skill_path=f".agents/skills/{package}/SKILL.md",
        claude_skill_path=f".claude/skills/{package}/SKILL.md",
    )


def build_default_skill_registry() -> dict[str, Skill]:
    return {
        skill_id: _skill_for(skill_id, required, produced)
        for skill_id, (required, produced) in SKILL_IO.items()
    }


def select_next_admissible_skill(
    registry: dict[str, Skill],
    context: dict[str, Any],
    completed_skill_ids: set[str] | None = None,
) -> Skill | None:
    completed = completed_skill_ids or set()
    for skill_id in registry:
        if skill_id in completed:
            continue
        skill = registry[skill_id]
        if all(context.get(required) is not None for required in skill.required_inputs):
            return skill
    return None


def validate_skill_trace(trace: list[dict[str, Any]]) -> list[str]:
    """Check a recorded skill trace (from a manual Skills-harness run
    run) against the contract; returns a list of issues."""

    issues: list[str] = []
    seen: list[str] = []
    for row in trace:
        skill_id = str(row.get("skill_id"))
        if skill_id not in SKILL_IO:
            issues.append(f"unknown_skill:{skill_id}")
            continue
        seen.append(skill_id)
    return issues


def _frontmatter_value(text: str, key: str) -> str | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end == -1:
        return None
    for line in text[4:end].splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return None


def validate_skill_packages(repo_root: Path) -> list[str]:
    """Validate that every registry skill has an agent-readable package."""

    issues: list[str] = []
    registry = build_default_skill_registry()
    if set(registry) != set(SKILLS_UNIFIED_CONTRACT):
        issues.append("skill_package_registry_mismatch")
    for skill in registry.values():
        for label, rel_path in (
            ("codex", skill.codex_skill_path),
            ("claude", skill.claude_skill_path),
        ):
            path = repo_root / rel_path
            if not path.exists():
                issues.append(f"missing_{label}_skill_package:{rel_path}")
                continue
            text = path.read_text(encoding="utf-8")
            name = _frontmatter_value(text, "name")
            description = _frontmatter_value(text, "description")
            if name != skill.skill_package_name:
                issues.append(
                    f"{rel_path}:frontmatter_name_mismatch:"
                    f"{name or '<missing>'}!={skill.skill_package_name}")
            if not description:
                issues.append(f"{rel_path}:missing_description")
            for required in skill.required_inputs:
                if required not in text:
                    issues.append(f"{rel_path}:missing_required_input:{required}")
            for produced in skill.produced_outputs:
                if produced not in text:
                    issues.append(f"{rel_path}:missing_produced_output:{produced}")
    return issues


def _row_has_output(row: dict[str, Any], output: str, artifacts_dir: Path) -> bool:
    produced = row.get("produced_outputs") or []
    if isinstance(produced, list) and output in {str(item) for item in produced}:
        return True
    payload = row.get("payload") or {}
    if isinstance(payload, dict) and output in payload:
        return True
    refs = row.get("output_refs") or row.get("artifact_refs") or []
    if isinstance(refs, list) and any(output in str(ref) for ref in refs):
        return True
    return any(artifacts_dir.glob(f"**/{output}*"))


def _rows_for_skill(trace: list[dict[str, Any]], skill_id: str) -> list[dict[str, Any]]:
    return [row for row in trace if str(row.get("skill_id")) == skill_id]


def _referenced_review_file_has_content(row: dict[str, Any], artifacts_dir: Path) -> bool:
    """True if the row's ``output_refs``/``artifact_refs`` name a review
    artifact that actually exists under ``artifacts_dir`` and is non-empty.

    A reviewer that writes real findings to a standalone file (e.g.
    ``reviews.json`` or a nested ``generated_code/review_proposal.json``)
    and only records the reference in ``output_refs`` — rather than
    inlining the content into ``payload["reviews"]`` — still produced a
    genuine, separate reviewer artifact; only an empty/missing reference or
    an inline-but-empty payload should be rejected. Refs may be given
    relative to the repo root, the run directory, or as a bare filename, and
    the file may live directly in the run directory or in a subdirectory
    (e.g. ``generated_code/``), so this searches recursively by basename —
    the same fallback pattern ``_row_has_output`` already uses.
    """

    refs = row.get("output_refs") or row.get("artifact_refs") or []
    if not isinstance(refs, list):
        return False
    for ref in refs:
        ref_text = str(ref)
        if "review" not in ref_text.lower():
            continue
        name = Path(ref_text).name
        direct = artifacts_dir / name
        if direct.is_file() and direct.stat().st_size > 0:
            return True
        for candidate in artifacts_dir.glob(f"**/{name}"):
            if candidate.is_file() and candidate.stat().st_size > 0:
                return True
    return False


_REQUIRED_HYPOTHESIS_FIELDS: tuple[str, ...] = (
    "mechanism", "required_data", "implementation_plan", "expected_contribution",
    "evaluation_method", "failure_modes", "acceptance_evidence",
)

_REQUIRED_BRANCH_LINEAGE_FIELDS: tuple[str, ...] = (
    "branch_id", "parent_id", "objective", "hypothesis_id", "status",
)

_MIN_DISTINCT_HYPOTHESES = 2


def _hypotheses_from_payload(payload: dict[str, Any]) -> list[Any]:
    hypotheses = payload.get("hypotheses") if isinstance(payload, dict) else None
    return hypotheses if isinstance(hypotheses, list) else []


def _branches_from_payload(payload: dict[str, Any]) -> list[Any]:
    branches = payload.get("branches") if isinstance(payload, dict) else None
    return branches if isinstance(branches, list) else []


def _stable_content_key(row: dict[str, Any]) -> tuple:
    """A debug-retry row's content, excluding bookkeeping keys that would
    otherwise make two functionally-identical retries look distinct (e.g. a
    bare attempt counter or timestamp with no actual change recorded)."""

    payload = dict(row.get("payload") or {})
    for bookkeeping_key in ("debug_attempt", "attempt", "timestamp", "generated_at"):
        payload.pop(bookkeeping_key, None)
    return (row.get("status"), tuple(sorted(payload.items(), key=lambda kv: kv[0])))


def validate_skill_trace_against_io(
    trace: list[dict[str, Any]],
    artifacts_dir: Path,
    *,
    manual_result_path: Path | None = None,
) -> list[str]:
    """Strict C9-C12 trace validation for AGENTS.md + Skill package runs.

    Beyond skill order and declared-output presence, this enforces the
    auditable-loop requirements: hypotheses carry the required evaluable
    fields, branch results carry lineage (branch/parent/objective/hypothesis
    IDs), a debug retry that repeats an earlier attempt unmodified does not
    count as new search depth, and the final synthesis traces back to
    branch artifacts actually produced during the run (never narrative-only).
    """

    issues = validate_skill_trace(trace)
    seen = [str(row.get("skill_id")) for row in trace if row.get("skill_id")]
    required_lifecycle = {
        "inspect_available_data", "generate_research_hypotheses",
        "execute_generated_code", "run_branch_search", "review_proposal",
        "revise_proposal", "write_final_condition_proposal",
    }
    if any(skill_id in seen for skill_id in {
        "expand_decision_regime", "compare_search_branches", "analyze_branch_divergence",
        "design_robustness_test", "run_robustness_ablations",
        "identify_reversal_conditions", "synthesize_decision_portfolios",
        "prioritize_due_diligence",
    }):
        required_lifecycle.add("synthesize_decision_portfolios")
    for skill_id in sorted(required_lifecycle):
        if skill_id not in seen:
            issues.append(f"missing_lifecycle_skill:{skill_id}")

    allowed_statuses = {"succeeded", "blocked", "skipped", "failed", "degraded"}
    for index, row in enumerate(trace):
        status = str(row.get("status") or "")
        if status not in allowed_statuses:
            issues.append(f"invalid_skill_status:{index}:{status or '<missing>'}")

    registry = build_default_skill_registry()
    available_outputs: set[str] = set()
    for index, row in enumerate(trace):
        skill_id = str(row.get("skill_id") or "")
        skill = registry.get(skill_id)
        if skill is None:
            continue
        missing_inputs = [value for value in skill.required_inputs if value not in available_outputs]
        if missing_inputs:
            issues.append(
                f"skill_lineage_missing_inputs:{index}:{skill_id}:{','.join(missing_inputs)}")
        if row.get("status") in {"succeeded", "degraded"}:
            available_outputs.update(str(value) for value in row.get("produced_outputs") or [])
    for skill_id, skill in registry.items():
        rows = _rows_for_skill(trace, skill_id)
        if not rows:
            continue
        latest = rows[-1]
        refs = latest.get("output_refs") or latest.get("artifact_refs") or []
        if latest.get("status") == "succeeded" and skill.produced_outputs and not refs:
            issues.append(f"artifact_producing_skill_missing_output_refs:{skill_id}")
        for output in skill.produced_outputs:
            if not _row_has_output(latest, output, artifacts_dir):
                issues.append(f"missing_produced_output:{skill_id}:{output}")

    code_rows = _rows_for_skill(trace, "write_experiment_code")
    if code_rows and not any(
        _row_has_output(row, "generated_code", artifacts_dir) for row in code_rows
    ):
        issues.append("write_experiment_code_missing_generated_code")

    exec_rows = _rows_for_skill(trace, "execute_generated_code")
    if exec_rows and not any(
        _row_has_output(row, "execution_results", artifacts_dir) for row in exec_rows
    ):
        issues.append("execute_generated_code_missing_execution_results")

    # --- hypothesis completeness: materially distinct, evaluable hypotheses ---
    hypothesis_rows = _rows_for_skill(trace, "generate_research_hypotheses")
    hypothesis_ids: set[str] = set()
    if hypothesis_rows:
        hypotheses = _hypotheses_from_payload(hypothesis_rows[-1].get("payload") or {})
        if len(hypotheses) < _MIN_DISTINCT_HYPOTHESES:
            issues.append(
                f"insufficient_distinct_hypotheses:{len(hypotheses)}<{_MIN_DISTINCT_HYPOTHESES}")
        for index, hypothesis in enumerate(hypotheses):
            if not isinstance(hypothesis, dict):
                issues.append(f"hypothesis_not_structured:{index}")
                continue
            missing = [f for f in _REQUIRED_HYPOTHESIS_FIELDS if not hypothesis.get(f)]
            if missing:
                issues.append(f"hypothesis_missing_fields:{index}:{','.join(missing)}")
            hyp_id = hypothesis.get("hypothesis_id")
            if hyp_id:
                hypothesis_ids.add(str(hyp_id))

    for skill_id in ("inspect_available_data", "generate_research_hypotheses"):
        for index, row in enumerate(_rows_for_skill(trace, skill_id)):
            payload = row.get("payload") or {}
            if isinstance(payload, dict) and (
                payload.get("selected_candidate_ids") or payload.get("final_candidate_ids")
                or payload.get("final_slate")
            ):
                issues.append(f"premature_final_selection:{skill_id}:{index}")

    # --- branch lineage: branch_id/parent_id/objective/hypothesis_id/status ---
    branch_rows = _rows_for_skill(trace, "run_branch_search")
    branch_ids: set[str] = set()
    if branch_rows:
        payload = branch_rows[-1].get("payload") or {}
        objectives = payload.get("branch_objectives") if isinstance(payload, dict) else None
        expected = {
            "elderly-demand", "emergency-access", "reorganization-feasibility",
            "financial-risk", "evidence-completeness",
        }
        if set(objectives or []) != expected:
            issues.append("run_branch_search_missing_five_objectives")
        branches = _branches_from_payload(payload)
        if not branches:
            issues.append("run_branch_search_missing_branch_lineage")
        for index, branch in enumerate(branches):
            if not isinstance(branch, dict):
                issues.append(f"branch_not_structured:{index}")
                continue
            # parent_id may legitimately be None (a root branch has no
            # parent) — only its *absence* is an error, not a null value.
            missing = [
                f for f in _REQUIRED_BRANCH_LINEAGE_FIELDS
                if (f not in branch) or (f != "parent_id" and branch.get(f) in (None, ""))
            ]
            if missing:
                issues.append(f"branch_missing_lineage_fields:{index}:{','.join(missing)}")
            branch_id = branch.get("branch_id")
            if branch_id:
                branch_ids.add(str(branch_id))
            hyp_ref = branch.get("hypothesis_id")
            if hypothesis_ids and hyp_ref and str(hyp_ref) not in hypothesis_ids:
                issues.append(f"branch_references_unknown_hypothesis:{index}:{hyp_ref}")

    # --- debug depth: an identical unmodified retry is not new search depth ---
    debug_rows = _rows_for_skill(trace, "debug_failed_code")
    seen_content_keys: list[tuple] = []
    for index, row in enumerate(debug_rows):
        key = _stable_content_key(row)
        if key in seen_content_keys:
            issues.append(f"debug_repeated_unmodified_retry:{index}")
        seen_content_keys.append(key)

    # --- reviewer/revision artifacts must be separate from the branch/proposal
    # artifact they critique, never overwriting it in place. ---
    review_rows = _rows_for_skill(trace, "review_proposal")
    if review_rows:
        latest_review = review_rows[-1]
        payload = latest_review.get("payload") or {}
        reviews = payload.get("reviews") if isinstance(payload, dict) else None
        has_reviews = bool(reviews) or _referenced_review_file_has_content(
            latest_review, artifacts_dir)
        if not has_reviews:
            issues.append("review_proposal_missing_reviewer_artifact")

    # --- final synthesis must trace to branch artifacts actually produced,
    # never a narrative-only report. ---
    final_rows = _rows_for_skill(trace, "write_final_condition_proposal")
    if final_rows:
        payload = final_rows[-1].get("payload") or {}
        source_refs = payload.get("source_branch_ids") if isinstance(payload, dict) else None
        source_refs = source_refs if isinstance(source_refs, list) else []
        if not source_refs:
            issues.append("write_final_condition_proposal_not_traceable_to_branches")
        elif branch_ids and not any(str(ref) in branch_ids for ref in source_refs):
            issues.append("write_final_condition_proposal_source_branch_ids_not_found")
        if manual_result_path is not None and not manual_result_path.exists():
            issues.append("write_final_condition_proposal_missing_manual_result")
    return list(dict.fromkeys(issues))
