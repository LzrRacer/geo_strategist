"""Family-aware minimal-validity checks for the C0-C14 condition track.

Every condition's ``primary_comparison_family`` (see ``comparison_families.py``)
names the shared inputs, output requirements, and resource budget a family's
members are held to — comparability comes from that shared setup (data,
candidate universe, response contract, budgets), not from a post-hoc
validator policing how a model chose to reason. A condition that produces a
valid, evidence-grounded, complete final report stays evaluable even when
its internal execution differs from the "expected" pattern for its family
(more analysis, a different reasoning path, an unanticipated tool use) —
that is real, decision-relevant variation the judge should see and score,
not a contract violation to exclude.

``validate_condition_family`` runs the validator for every family a
condition belongs to (primary and secondary). ``apply_family_validation``
is the single call site (``run_condition_proposals._record_for``) that
turns a *primary*-family failure into ``comparable_for_e13 = False`` — but
only for fundamental failures, reserved for:

- no usable final report (an empty/missing ranked slate);
- a structured claim of executed analysis (a produced skill output, written
  code, an executed test, a cited review, a synthesis's source branches)
  that resolves to no backing artifact at all — an unsupported factual
  claim, which the evidence policy (AGENTS.md) never permits regardless of
  which condition made it;
- Skills-package contamination of a no-Skills condition (invalidates the
  ablation treatment itself, not "normal model behavior" — AGENTS.md calls
  this out as requiring a re-run);
- a deterministic condition (C0) whose bundle claims exploration it could
  not have performed (only possible via a pipeline bug, never real model
  behavior, since C0 has no model).

Everything else observed (extra analysis, a fuller or thinner Skills trace
than anticipated, branch-search variations, an incomplete hypothesis
format, a repeated debug attempt) is recorded as a ``deviation`` — visible
in the condition's record and reportable, but never exclusionary. Failed
conditions always keep a visible report; a fundamental-failure exclusion is
recorded and explained, never silently swapped for a different result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from geo_strategist.experiments.condition_registry import ConditionSpec
from geo_strategist.experiments.live_common import LiveConditionResult


@dataclass(frozen=True)
class FamilyValidationResult:
    family_id: str
    passed: bool
    missing_outputs: list[str] = field(default_factory=list)
    # Fundamental issues only (see module docstring): unsupported claims,
    # contamination, or a genuinely empty result. These are the only things
    # that can flip `passed` to False.
    prohibited_extras: list[str] = field(default_factory=list)
    # Informational: real, sometimes decision-relevant differences from the
    # "expected" family pattern that a report should still be evaluated on
    # its own merits for. Never affects `passed` or comparability.
    deviations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def first_issue(self) -> str | None:
        for issue in (*self.prohibited_extras, *self.missing_outputs):
            return issue
        return None

    def to_dict(self) -> dict:
        return {
            "family_id": self.family_id,
            "passed": self.passed,
            "missing_outputs": list(self.missing_outputs),
            "prohibited_extras": list(self.prohibited_extras),
            "deviations": list(self.deviations),
            "notes": list(self.notes),
        }


FamilyValidator = Callable[[ConditionSpec, LiveConditionResult, Path], FamilyValidationResult]


def _bundle(result: LiveConditionResult):
    return result.analysis_bundle


def _validate_deterministic_reference(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> FamilyValidationResult:
    """C0: a fixed deterministic policy with no model. It may produce a
    strong evidence-backed report, but a bundle claiming branch results,
    search nodes, robustness tests, or critique/revision activity can only
    come from a pipeline bug (C0 has no model to generate any of that) —
    this is the one case where "extra analysis" is not real model behavior,
    so it stays a fundamental check."""

    family_id = "deterministic_reference"
    extras: list[str] = []
    bundle = _bundle(result)
    if bundle is not None:
        if bundle.branch_results:
            extras.append("branch_results")
        if bundle.search_nodes:
            extras.append("search_nodes")
        if bundle.decision_alternatives:
            extras.append("decision_alternatives")
        if any(r.status == "succeeded" for r in bundle.robustness_analysis):
            extras.append("robustness_analysis")
        if bundle.critique_dispositions:
            extras.append("critique_dispositions")
        if bundle.revision_effects:
            extras.append("revision_effects")
    missing: list[str] = [] if result.proposals else ["ranked_candidates"]
    return FamilyValidationResult(
        family_id=family_id, passed=not extras and not missing,
        missing_outputs=missing, prohibited_extras=extras)


def _validate_vanilla_model(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> FamilyValidationResult:
    """C1-C4: comparability comes from the shared single-pass runner/prompt
    definition (C1/C4 via ``run_vanilla_condition``: exactly one model call,
    no tools, by construction; C2/C3 via the manual-harness single-pass
    prompt), not from excluding a condition after the fact for producing
    more than expected. If a model under this prompt chose to reason
    through branch-like structure, request a review, or write code, that is
    real, observable model behavior the judge's report-content questions
    should see and score — not a contract violation. This validator only
    confirms the output is usable: a real ranked slate with a stated
    rationale per candidate."""

    family_id = "vanilla_model"
    bundle = _bundle(result)
    deviations: list[str] = []
    if bundle is not None:
        if bundle.branch_search_status == "succeeded" or bundle.branch_results:
            deviations.append("branch_search_present")
        if bundle.search_nodes or bundle.decision_alternatives:
            deviations.append("search_structure_present")
        if any(r.status == "succeeded" for r in bundle.robustness_analysis):
            deviations.append("robustness_analysis_present")
        if bundle.critique_dispositions or bundle.revision_effects:
            deviations.append("review_or_revision_present")
        total_requests = bundle.model_call_summary.get("total_requests")
        if isinstance(total_requests, int) and total_requests > 1:
            deviations.append("more_than_one_model_call")
    generated_code_dir = Path(run_dir) / "generated_code"
    if generated_code_dir.is_dir() and any(generated_code_dir.iterdir()):
        deviations.append("generated_code_present")
    deviations = sorted(set(deviations))

    missing: list[str] = [] if result.proposals else ["ranked_candidates"]
    if result.proposals and any(not p.get("llm_rationale") for p in result.proposals):
        missing.append("rationale")

    return FamilyValidationResult(
        family_id=family_id, passed=not missing,
        missing_outputs=missing, deviations=deviations)


def _validate_native_agent_model(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> FamilyValidationResult:
    """C5-C8: open-ended native coding-agent control. No fixed branch count
    or five-objective coverage is required or expected. Two fundamental
    checks only:

    1. Claimed-analysis-requires-execution-artifact, mechanical/non-LLM: the
       legacy manual-result loader (``decision_analysis.load_decision_analysis_bundle``)
       already resolves every skill_trace row's ``output_refs``/``artifact_refs``
       against real files and records any that do not resolve as
       ``unresolved_output_ref:<ref>`` in ``bundle.migration_warnings`` — a
       structured claim (branch result, executed test, revision) with no
       backing artifact on disk is an unsupported factual claim under the
       evidence policy, not a difference in reasoning style.
    2. Skills-contamination: the isolation contract for C5-C8 is much
       stronger than for vanilla (a sanitized workspace with `.agents/skills`
       and `.claude/skills` withheld; AGENTS.md: "reading or referencing the
       Skill packages contaminates the comparison and requires a re-run").
       A skill_trace entry naming an actual Skills-unified operator id
       invalidates the no-Skills ablation treatment itself.
    """

    from geo_strategist.agent.skills import SKILLS_UNIFIED_CONTRACT

    family_id = "native_agent_model"
    extras: list[str] = []
    bundle = _bundle(result)
    if bundle is not None:
        unresolved = [w[len("unresolved_output_ref:"):] for w in bundle.migration_warnings
                      if w.startswith("unresolved_output_ref:")]
        extras.extend(f"unbacked_claim:{ref}" for ref in unresolved)
        contaminated = sorted({
            str(row.get("skill_id")) for row in bundle.skill_trace
            if str(row.get("skill_id")) in SKILLS_UNIFIED_CONTRACT
        })
        if contaminated:
            extras.append(f"skills_package_contamination:{','.join(contaminated)}")
    missing: list[str] = [] if result.proposals else ["ranked_candidates"]
    return FamilyValidationResult(
        family_id=family_id, passed=not extras and not missing,
        missing_outputs=missing, prohibited_extras=extras)


# validate_skill_trace_against_io issue prefixes that indicate a structured
# claim of executed analysis with literally nothing backing it (an
# unsupported factual claim under the evidence policy) -- the only
# skill-trace issues treated as fundamental. Everything else that validator
# checks (trace shape, lifecycle ordering, hypothesis format, five-objective
# coverage, branch lineage completeness, repeated debug attempts) is a
# process/format deviation: real variation in how the run unfolded, not a
# claim unsupported by evidence.
_FUNDAMENTAL_SKILL_ISSUE_PREFIXES: tuple[str, ...] = (
    "missing_produced_output:",
    "write_experiment_code_missing_generated_code",
    "execute_generated_code_missing_execution_results",
    "review_proposal_missing_reviewer_artifact",
    "write_final_condition_proposal_not_traceable_to_branches",
    "write_final_condition_proposal_source_branch_ids_not_found",
    "write_final_condition_proposal_missing_manual_result",
)


def split_skill_trace_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    fundamental = [i for i in issues if i.startswith(_FUNDAMENTAL_SKILL_ISSUE_PREFIXES)]
    deviations = [i for i in issues if i not in fundamental]
    return fundamental, deviations


def _validate_skills_agent_model(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> FamilyValidationResult:
    """C9-C12: shared Skills package + branch-search contract. Wraps the
    existing strict trace validator (``skill_registry.validate_skill_trace_against_io``)
    but only its unsupported-claim findings (see
    ``_FUNDAMENTAL_SKILL_ISSUE_PREFIXES``) can exclude a condition; trace
    shape, lifecycle-order, hypothesis-format, five-objective-coverage, and
    branch-lineage issues are recorded as deviations — a Skills run that
    covered four of five objectives or used a different lifecycle order
    produced real, evaluable output and stays comparable.

    Also recorded as notes (never exclusionary): failed skill invocations
    (status ``failed``/``blocked``) and a ``degenerate_search``
    classification — both are valid, reportable outcomes of a real run.
    """

    from geo_strategist.agent.skill_registry import validate_skill_trace_against_io

    family_id = "skills_agent_model"
    bundle = _bundle(result)
    extras: list[str] = []
    deviations: list[str] = []
    notes: list[str] = []
    if bundle is not None:
        issues = validate_skill_trace_against_io(bundle.skill_trace, Path(run_dir))
        extras, deviations = split_skill_trace_issues(issues)
        failed = sorted({
            str(row.get("skill_id")) for row in bundle.skill_trace
            if str(row.get("status")) in ("failed", "blocked")
        })
        if failed:
            notes.append(f"failed_skill_invocations:{','.join(failed)}")
        if bundle.search_diagnostics.classification == "degenerate_search":
            notes.append("degenerate_search: " + (bundle.search_diagnostics.explanation or
                                                     "search converged to an indistinguishable shortlist"))
    missing: list[str] = [] if result.proposals else ["ranked_candidates"]
    return FamilyValidationResult(
        family_id=family_id, passed=not extras and not missing,
        missing_outputs=missing, prohibited_extras=extras,
        deviations=deviations, notes=notes)


def _validate_ai_scientist_model(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> FamilyValidationResult:
    """C13/C14: identical AI-Scientist runner, budgets, and structured
    synthesis contract. An under-filled structured synthesis is handled
    internally by the runner (bounded repair, then a stated rank-pooling
    aggregation over executed branch winners as the actual final slate —
    see ``ai_scientist_loop``/``ai_scientist_synthesis``) and never excludes
    the condition here: the report it produces is still a genuine,
    evidence-grounded decision document and should be judged on its own
    content, not on whether the model-authored synthesis call itself
    succeeded on the first attempt."""

    family_id = "ai_scientist_model"
    missing = [] if result.proposals else ["ranked_candidates"]
    return FamilyValidationResult(family_id=family_id, passed=not missing, missing_outputs=missing)


FAMILY_VALIDATORS: dict[str, FamilyValidator] = {
    "deterministic_reference": _validate_deterministic_reference,
    "vanilla_model": _validate_vanilla_model,
    "native_agent_model": _validate_native_agent_model,
    "skills_agent_model": _validate_skills_agent_model,
    # skills_ablation_pair is a cross-condition comparison view (rendered as
    # pairwise tables), not an independent per-condition output contract —
    # a condition's own contract is fully determined by its primary family
    # (native_agent_model or skills_agent_model), so no separate validator
    # is registered for it.
    "ai_scientist_model": _validate_ai_scientist_model,
}


def validate_condition_family(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> dict[str, FamilyValidationResult]:
    """Run the family validator for every family ``spec`` belongs to
    (primary and secondary). ``skills_ablation_pair`` membership is recorded
    on the spec but has no standalone validator (see ``FAMILY_VALIDATORS``
    docstring), so it is skipped here."""

    results: dict[str, FamilyValidationResult] = {}
    for family_id in spec.comparison_families:
        validator = FAMILY_VALIDATORS.get(family_id)
        if validator is None:
            continue
        results[family_id] = validator(spec, result, run_dir)
    return results


def apply_family_validation(
    spec: ConditionSpec, result: LiveConditionResult, run_dir: Path,
) -> dict[str, FamilyValidationResult]:
    """Run family validation and, on a *primary*-family fundamental failure,
    mark ``result`` non-comparable — unless an earlier, more specific
    exclusion reason is already set (never overwritten; family validation
    adds a reason, it does not relitigate one). Non-fundamental deviations
    are recorded in due diligence for transparency but never touch
    comparability. Returns the full per-family result dict (including
    secondary families) so the caller can record it regardless of overall
    comparability."""

    family_results = validate_condition_family(spec, result, run_dir)
    primary = spec.primary_comparison_family
    primary_result = family_results.get(primary)
    if primary_result is not None:
        if not primary_result.passed:
            issue = primary_result.first_issue or "contract_violation"
            if result.comparable_for_e13:
                result.comparable_for_e13 = False
                result.exclusion_reason = f"family_contract:{primary}:{issue}"
            result.due_diligence.append(
                f"Family contract violation ({primary}): "
                + "; ".join([*primary_result.prohibited_extras, *primary_result.missing_outputs]))
        elif primary_result.deviations:
            result.due_diligence.append(
                f"Observed deviation from the typical {primary} pattern (informational, not a "
                "contract violation): " + "; ".join(primary_result.deviations))
    return family_results
