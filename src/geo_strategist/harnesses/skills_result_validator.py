"""Standalone validator for a C9-C12 Skills-unified ``manual_result.json``.

Built for C9 (Antigravity) but reusable for C10/C11/C12 (same
``agentic_skills_harness`` family, same output contract) via
``expected_condition_group``. Reuses the shared ranked_candidates /
qualitative_discussion / evidence / C0-substitution checks from
``c5_result_validator.py`` (the two families' contract is identical for
that part) and adds the Skills-unified-only requirement: a complete
``skill_trace``.

The skill_trace check wraps the SAME real trace validator used at
ingestion (``agent.skill_registry.validate_skill_trace_against_io`` +
``experiments.family_validation.split_skill_trace_issues``) rather than
approximating it, so a pre-stop self-check catches exactly what ingestion
would later reject or record as a deviation — not a slightly different
guess at the same contract. Only the "fundamental" (unsupported-claim)
trace issues are hard errors here, matching the ingestion-time rule that
only those can exclude a run from the comparison; everything else
(lifecycle order, hypothesis format, five-objective coverage, branch
lineage) is reported as a deviation, not a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geo_strategist.harnesses.c5_result_validator import (
    DEFAULT_C0_RANKED_CANDIDATES_PATH,
    DEFAULT_CANDIDATE_ACTIONS_PATH,
    ValidationIssue,
    _validate_ranked_candidates_and_evidence,
)


@dataclass
class SkillsValidationReport:
    ok: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    c0_substitution_flags: list[ValidationIssue] = field(default_factory=list)
    skill_trace_deviations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "c0_substitution_flags": [issue.to_dict() for issue in self.c0_substitution_flags],
            "skill_trace_deviations": list(self.skill_trace_deviations),
        }

    def human_report(self) -> str:
        lines = [f"Skills-unified result validation: {'PASS' if self.ok else 'FAIL'}"]
        if self.errors:
            lines.append(f"\n{len(self.errors)} error(s):")
            lines.extend(f"  - [{issue.code}] {issue.message}" for issue in self.errors)
        if self.c0_substitution_flags:
            lines.append(f"\n{len(self.c0_substitution_flags)} possible C0-substitution flag(s):")
            lines.extend(f"  - [{issue.code}] {issue.message}" for issue in self.c0_substitution_flags)
        if self.warnings:
            lines.append(f"\n{len(self.warnings)} warning(s):")
            lines.extend(f"  - [{issue.code}] {issue.message}" for issue in self.warnings)
        if self.skill_trace_deviations:
            lines.append(
                f"\n{len(self.skill_trace_deviations)} skill_trace deviation(s) "
                "(recorded, not exclusionary):")
            lines.extend(f"  - {issue}" for issue in self.skill_trace_deviations)
        if not any((self.errors, self.c0_substitution_flags, self.warnings,
                    self.skill_trace_deviations)):
            lines.append("No issues found.")
        return "\n".join(lines)


def validate_skills_agent_result(
    result_path: Path,
    *,
    repo_root: Path | None = None,
    candidate_actions_path: Path | None = None,
    c0_ranked_candidates_path: Path | None = None,
    run_dir: Path | None = None,
    expected_condition_group: str = "C9",
) -> SkillsValidationReport:
    """Validate one C9-C12 Skills-unified ``manual_result.json``.

    ``run_dir`` defaults to ``result_path.parent`` and is where
    ``generated_code/`` and any skill_trace output artifacts are searched
    for (matching ``validate_skill_trace_against_io``'s own lookup).
    """

    repo_root = (repo_root or Path(".")).resolve()
    run_dir = run_dir or result_path.parent

    if not result_path.exists():
        return SkillsValidationReport(
            ok=False, errors=[ValidationIssue("missing_file", f"{result_path} does not exist")])

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return SkillsValidationReport(
            ok=False,
            errors=[ValidationIssue("invalid_json", f"{result_path} is not valid JSON: {exc}")])

    if not isinstance(payload, dict):
        return SkillsValidationReport(
            ok=False,
            errors=[ValidationIssue("invalid_shape", "top-level JSON must be an object")])

    errors: list[ValidationIssue] = []
    if payload.get("condition_group") != expected_condition_group:
        errors.append(ValidationIssue(
            "wrong_condition_group",
            f"condition_group is {payload.get('condition_group')!r}, "
            f"expected {expected_condition_group!r}"))

    skill_trace = payload.get("skill_trace")
    has_skill_trace = isinstance(skill_trace, list) and bool(skill_trace)
    if not has_skill_trace:
        errors.append(ValidationIssue(
            "missing_skill_trace",
            "skill_trace is missing or empty — required for the Skills-unified contract "
            "(C5-C8's no-Skills control may leave this empty; C9-C12 may not)"))
        skill_trace = []

    candidate_errors, warnings, c0_flags = _validate_ranked_candidates_and_evidence(
        payload, repo_root=repo_root,
        candidate_actions_path=candidate_actions_path or (repo_root / DEFAULT_CANDIDATE_ACTIONS_PATH),
        c0_ranked_candidates_path=(
            c0_ranked_candidates_path or (repo_root / DEFAULT_C0_RANKED_CANDIDATES_PATH)),
        run_dir=run_dir,
        # A generated_code/ directory is a strong signal but not the only
        # one for Skills-unified runs — the dynamic skill_trace itself
        # (write_experiment_code/execute_generated_code/run_branch_search
        # rows) is independent evidence of a real analysis even if code
        # artifacts are referenced via output_refs rather than a fixed
        # directory, so its absence is a warning here, not a hard error.
        require_generated_code=False,
        extra_independent_analysis_evidence=has_skill_trace,
    )
    errors.extend(candidate_errors)
    generated_code_dir = run_dir / "generated_code"
    if not (generated_code_dir.is_dir() and any(generated_code_dir.iterdir())):
        warnings.append(ValidationIssue(
            "missing_generated_code",
            f"{generated_code_dir} does not exist or is empty; keeping executed analysis code "
            "alongside the result is expected even when also referenced via skill_trace output_refs"))

    from geo_strategist.agent.skill_registry import validate_skill_trace_against_io
    from geo_strategist.experiments.family_validation import split_skill_trace_issues

    trace_issues = validate_skill_trace_against_io(
        skill_trace, run_dir, manual_result_path=result_path)
    fundamental, deviations = split_skill_trace_issues(trace_issues)
    for issue in fundamental:
        errors.append(ValidationIssue("skill_trace_fundamental", issue))

    ok = not errors and not c0_flags
    return SkillsValidationReport(
        ok=ok, errors=errors, warnings=warnings, c0_substitution_flags=c0_flags,
        skill_trace_deviations=deviations,
    )
