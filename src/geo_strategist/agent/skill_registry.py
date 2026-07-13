"""Skill registry metadata for the Skills-unified contract.

Declares each skill's required inputs and produced outputs plus the
filesystem Skill package paths used by C5-C8 agentic Skills harnesses. The
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
    "run_branch_search": (("execution_results",), ("branch_results",)),
    "review_proposal": (("proposals",), ("reviews", "revision_requests")),
    "revise_proposal": (("reviews",), ("proposals",)),
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
    # duplicates are allowed (retries/rounds); first-occurrence order must
    # follow the contract order
    first_occurrence = list(dict.fromkeys(seen))
    expected = [s for s in SKILLS_UNIFIED_CONTRACT if s in first_occurrence]
    if first_occurrence != expected:
        issues.append("skill_order_violation: " + " -> ".join(first_occurrence))
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
    if tuple(registry) != SKILLS_UNIFIED_CONTRACT:
        issues.append("skill_package_order_mismatch")
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


def validate_skill_trace_against_io(
    trace: list[dict[str, Any]],
    artifacts_dir: Path,
    *,
    manual_result_path: Path | None = None,
) -> list[str]:
    """Strict C5-C8 trace validation for AGENTS.md + Skill package runs."""

    issues = validate_skill_trace(trace)
    seen = [str(row.get("skill_id")) for row in trace if row.get("skill_id")]
    for skill_id in SKILLS_UNIFIED_CONTRACT:
        if skill_id not in seen:
            issues.append(f"missing_skill:{skill_id}")

    allowed_statuses = {"succeeded", "blocked", "skipped", "failed", "degraded"}
    for index, row in enumerate(trace):
        status = str(row.get("status") or "")
        if status not in allowed_statuses:
            issues.append(f"invalid_skill_status:{index}:{status or '<missing>'}")

    registry = build_default_skill_registry()
    for skill_id, skill in registry.items():
        rows = _rows_for_skill(trace, skill_id)
        if not rows:
            continue
        latest = rows[-1]
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

    branch_rows = _rows_for_skill(trace, "run_branch_search")
    if branch_rows:
        payload = branch_rows[-1].get("payload") or {}
        objectives = payload.get("branch_objectives") if isinstance(payload, dict) else None
        expected = {
            "elderly-demand", "emergency-access", "reorganization-feasibility",
            "financial-risk", "evidence-completeness",
        }
        if set(objectives or []) != expected:
            issues.append("run_branch_search_missing_five_objectives")

    final_rows = _rows_for_skill(trace, "write_final_condition_proposal")
    if final_rows and manual_result_path is not None and not manual_result_path.exists():
        issues.append("write_final_condition_proposal_missing_manual_result")
    return list(dict.fromkeys(issues))
