"""Comparison-family registry for the C0-C14 condition track.

A "comparison family" groups conditions that share an identical output
contract, resource-budget profile, and report profile, so that within a
family the only thing that can explain a score difference is the underlying
provider/model output — not a difference in what the condition was allowed
or required to do. ``condition_registry.ConditionSpec`` records which
families a condition belongs to; this module is the single source of truth
for what each family actually means and which conditions are its members,
so registry wiring and family-aware validation (``family_validation.py``)
and reporting (``live_report.py``, ``condition_comparison_judge.py``) all
read from one place instead of re-deriving family membership from ad hoc
condition-id checks.

A condition may belong to more than one family (e.g. C11 is a member of
both ``skills_agent_model``, comparing it against C9/C10/C12 on the same
Skills contract, and ``skills_ablation_pair`` with C7, isolating the Skills
treatment on the same Claude Code/sonnet-5.0 stack). ``primary_comparison_family``
on the spec names the one used for the condition's headline family-contract
validation (family_validation.py dispatches on it); membership in any other
family is a secondary, non-exclusionary comparison view.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from geo_strategist.experiments.condition_registry import CONDITION_ORDER


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    member_condition_ids: tuple[str, ...]
    contract_summary: str
    validator_id: str
    strict_pairs: tuple[tuple[str, str], ...] = field(default_factory=tuple)


COMPARISON_FAMILIES: dict[str, FamilySpec] = {
    "deterministic_reference": FamilySpec(
        family_id="deterministic_reference",
        member_condition_ids=("C0",),
        contract_summary=(
            "Fixed deterministic scoring policy; not a live-agent condition. "
            "Never credited with exploration, branch search, generated-code "
            "experiments, or review/revision it did not perform."
        ),
        validator_id="deterministic_reference",
    ),
    "vanilla_model": FamilySpec(
        family_id="vanilla_model",
        member_condition_ids=("C1", "C2", "C3", "C4"),
        contract_summary=(
            "True single-pass contract: identical data, candidate universe, "
            "prompt, response schema, max-1 model request, parser/failure "
            "policy, report renderer, and judge across all four providers. "
            "No branch search, generated code, robustness tests, review "
            "loops, or candidate deliberation permitted."
        ),
        validator_id="vanilla_model",
    ),
    "native_agent_model": FamilySpec(
        family_id="native_agent_model",
        member_condition_ids=("C5", "C6", "C7", "C8"),
        contract_summary=(
            "Open-ended native coding-agent control: full repository/data "
            "access, free-form multi-step exploration, no project Skills "
            "package. No fixed branch count or five-objective requirement; "
            "only a minimum execution surface (valid slate, traceable "
            "execution artifacts, final rationale) is required."
        ),
        validator_id="native_agent_model",
    ),
    "skills_agent_model": FamilySpec(
        family_id="skills_agent_model",
        member_condition_ids=("C9", "C10", "C11", "C12"),
        contract_summary=(
            "Shared project Skills package, Skills registry, required "
            "lifecycle, five mandatory branch objectives (elderly-demand, "
            "emergency-access, reorganization-feasibility, financial-risk, "
            "evidence-completeness), resource budgets, sandbox rules, output "
            "schema, review limits, synthesis contract, report renderer, "
            "and judge."
        ),
        validator_id="skills_agent_model",
    ),
    "skills_ablation_pair": FamilySpec(
        family_id="skills_ablation_pair",
        member_condition_ids=("C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12"),
        contract_summary=(
            "Four strict same-provider/model/harness pairs isolating the "
            "project Skills package as the sole treatment difference: "
            "C5/C9 (Antigravity, gemini-3.5-flash), C6/C10 (Codex, gpt-5.5), "
            "C7/C11 (Claude Code, sonnet-5.0), C8/C12 (OpenCode, "
            "deepseek-v4-flash). The no-Skills side is never retrofitted "
            "with a Skills-equivalent mandatory five-branch process."
        ),
        validator_id="skills_ablation_pair",
        strict_pairs=(("C5", "C9"), ("C6", "C10"), ("C7", "C11"), ("C8", "C12")),
    ),
    "ai_scientist_model": FamilySpec(
        family_id="ai_scientist_model",
        member_condition_ids=("C13", "C14"),
        contract_summary=(
            "Identical AI-Scientist runner: ideation procedure, draft/"
            "variant counts, five branch objectives, resource limits, "
            "structured synthesis with bounded repair, repair/fallback "
            "policy, leave-one-objective-out ablation, review rounds, "
            "report renderer, and judge. Only the provider/model differs."
        ),
        validator_id="ai_scientist_model",
    ),
}


# Provider/model stack -> the condition groups run on that exact stack,
# across the vanilla/native-agent/Skills/AI-Scientist families. Used for the
# per-model total-score aggregation (condition_comparison_judge.py): within
# each family the framework is identical, so summing/averaging a stack's
# scores measures only that model's output quality, never a framework
# difference. C0 (no model) is intentionally excluded from this map.
MODEL_STACKS: dict[str, tuple[str, ...]] = {
    "gemini-3.5-flash": ("C1", "C5", "C9", "C13"),
    "gpt-5.5": ("C2", "C6", "C10"),
    "sonnet-5.0": ("C3", "C7", "C11"),
    "deepseek-v4-flash": ("C4", "C8", "C12", "C14"),
}


def family_ids_for_condition(group: str) -> tuple[str, ...]:
    """Every family a condition group belongs to, in registry order."""

    return tuple(
        family_id for family_id, spec in COMPARISON_FAMILIES.items()
        if group in spec.member_condition_ids
    )


def validate_family_registry_consistency(
    registry: dict[str, "object"] | None = None,
) -> list[str]:
    """Cross-check the family registry against ``CONDITION_ORDER`` and,
    when a live ``ConditionSpec`` registry is supplied, against each spec's
    own ``comparison_families``/``primary_comparison_family`` wiring and the
    strict-pair provider/model/budget-profile agreement. Returns a list of
    human-readable issues; an empty list means the registry is consistent."""

    issues: list[str] = []
    known = set(CONDITION_ORDER)
    for family_id, spec in COMPARISON_FAMILIES.items():
        unknown_members = [g for g in spec.member_condition_ids if g not in known]
        if unknown_members:
            issues.append(
                f"family {family_id!r} references unknown condition id(s): {unknown_members}")
        for a, b in spec.strict_pairs:
            if a not in spec.member_condition_ids or b not in spec.member_condition_ids:
                issues.append(
                    f"family {family_id!r} strict pair ({a}, {b}) is not a subset "
                    "of its own member_condition_ids")

    if registry is not None:
        for group, cond_spec in registry.items():
            declared = set(getattr(cond_spec, "comparison_families", ()))
            expected = set(family_ids_for_condition(group))
            if declared != expected:
                issues.append(
                    f"{group}: ConditionSpec.comparison_families {sorted(declared)} "
                    f"does not match COMPARISON_FAMILIES membership {sorted(expected)}")
            primary = getattr(cond_spec, "primary_comparison_family", "")
            if primary and primary not in declared:
                issues.append(
                    f"{group}: primary_comparison_family {primary!r} is not in its "
                    f"own comparison_families {sorted(declared)}")
        for family_id, spec in COMPARISON_FAMILIES.items():
            for a, b in spec.strict_pairs:
                spec_a, spec_b = registry.get(a), registry.get(b)
                if spec_a is None or spec_b is None:
                    continue
                if getattr(spec_a, "provider", None) != getattr(spec_b, "provider", None):
                    issues.append(
                        f"skills_ablation_pair {a}/{b}: provider mismatch "
                        f"({spec_a.provider!r} vs {spec_b.provider!r})")
                if getattr(spec_a, "model", None) != getattr(spec_b, "model", None):
                    issues.append(
                        f"skills_ablation_pair {a}/{b}: model mismatch "
                        f"({spec_a.model!r} vs {spec_b.model!r})")
                if getattr(spec_a, "budget_profile_id", None) != getattr(spec_b, "budget_profile_id", None):
                    issues.append(
                        f"skills_ablation_pair {a}/{b}: budget_profile_id mismatch "
                        f"({spec_a.budget_profile_id!r} vs {spec_b.budget_profile_id!r})")
    return issues
