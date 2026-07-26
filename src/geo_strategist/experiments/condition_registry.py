"""Canonical C0-C14 condition registry.

Single source of truth for condition identity: provider, model, harness,
algorithm family, Skills unification, and branch-search participation.
Model/provider values may be overridden through ``Cx_*`` environment
variables (see ``.env.example``); the strict comparison checks read from here.

Schema history: this is schema version 4 of the condition track (see
``configs/experiment_conditions.yaml`` ``version: 8``). Version 1 (``C0-C13``,
YAML ``version: 5``) numbered the Skills-unified conditions ``C5-C8`` and the
AI-Scientist/advanced-orchestration conditions ``C9-C13``. Version 2 inserted
four coding-agent, non-Skills control conditions as the new ``C5-C8`` and
shifted every old ``C5`` onward up by four (old C5->C9 ... old C13->C17).
Version 3 (YAML ``version: 7``) made C5-C8/C9-C12/C13-C14 open-ended
discovery variants. Version 4 (YAML ``version: 8``) removes the optional
advanced-orchestration conditions C15-C17 entirely, restricting the track to
C0-C14; see ``legacy_condition_id_migration`` for translating a pre-revision
artifact's condition IDs — a pre-revision ID that maps to a removed C15-C17
condition now raises instead of resolving. Do not reinterpret a pre-revision
``C9`` artifact (Skills-Antigravity, old numbering) as the current ``C9`` (a
no-Skills coding-agent control) — they are different conditions.

Canonical order:

- C0  Deterministic Python baseline
- C1  Vanilla direct LLM baseline — Gemini (no tools, no code, no Skills)
- C2  Vanilla LLM baseline — Codex CLI, no Skills contract (manual harness)
- C3  Vanilla LLM baseline — Claude Code CLI, no Skills contract (manual harness)
- C4  Vanilla direct LLM baseline — DeepSeek via OpenCode Go direct API
- C5  Antigravity native open-ended discovery — no project Skills
- C6  Codex native open-ended discovery — no project Skills
- C7  Claude Code native open-ended discovery — no project Skills
- C8  OpenCode native open-ended discovery — no project Skills
- C9  Skills-Antigravity — AGENTS.md + filesystem Skills harness,
      Antigravity CLI (``agy``), Gemini, Skills-unified contract + branch search
- C10  Skills-Codex — AGENTS.md + filesystem Skills harness, Codex CLI
      (``codex``), Skills + branch search
- C11  Skills-Claude Code — AGENTS.md + filesystem Skills harness, Claude Code
      CLI (``claude``), Skills + branch search
- C12  Skills-OpenCode — AGENTS.md + filesystem Skills harness, OpenCode CLI
      (``opencode``), Skills + branch search
- C13  Traditional AI Scientist-style Gemini
- C14 Traditional AI Scientist-style DeepSeek
- E13 Cross-condition proposal and agentic-process judge (not a condition)

Four provider/model stacks each pair a vanilla baseline against its
Skills-and/or-agentic-loop counterpart on the *same* provider+model, so the
delta isolates the treatment rather than the model:

- Gemini (gemini-3.5-flash):        C1 (vanilla) vs C13 (AI Scientist)      — strict pair
- Codex (gpt-5.5):                  C2 (vanilla) vs C10 (Skills)            — strict pair
- Claude Code (sonnet-5.0):         C3 (vanilla) vs C11 (Skills)            — strict pair
- OpenCode Go (deepseek-v4-flash):  C4 (vanilla) vs C12 (Skills) vs C14 (AI Scientist) — strict trio

A second family of four strict pairs isolates the project-specific Skills
contract itself, holding provider+model+harness fixed (coding-agent control
vs Skills-unified, same CLI):

- Gemini (gemini-3.5-flash), Antigravity CLI: C5 (no Skills) vs C9 (Skills)
- GPT (gpt-5.5), Codex CLI:                   C6 (no Skills) vs C10 (Skills)
- Sonnet (sonnet-5.0), Claude Code CLI:       C7 (no Skills) vs C11 (Skills)
- DeepSeek (deepseek-v4-flash), OpenCode CLI: C8 (no Skills) vs C12 (Skills)

C9 now defaults to gemini-3.5-flash (not gemini-3.5-pro) so its Gemini/
Antigravity stack matches C1, C5, and C13 for the cross-cutting Gemini
comparisons.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

CONDITION_ORDER: tuple[str, ...] = (
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
    "C12", "C13", "C14",
)

# Pre-revision (schema version 1) condition IDs that now refer to a
# different condition under the current numbering. A caller that encounters
# one of these IDs in a previously generated manifest/run-lineage record must
# migrate it via ``legacy_condition_id_migration`` before treating it as a
# current condition — silently reusing the bare ID would misclassify e.g. a
# pre-revision "C9" (Skills-Antigravity) as the current C9 (a no-Skills
# coding-agent control). Old C11/C12/C13 (Shinka-style evolution, AB-MCTS,
# Fugu-style orchestrator) mapped to the now-removed C15/C16/C17 and are
# intentionally absent: ``legacy_condition_id_migration`` raises for them
# instead of resolving to a condition that no longer exists.
LEGACY_CONDITION_ID_MAP: dict[str, str] = {
    "C5": "C9", "C6": "C10", "C7": "C11", "C8": "C12",
    "C9": "C13", "C10": "C14",
}

# Old schema-version-1 IDs whose migrated target (C15/C16/C17) was removed
# from the condition track entirely, rather than merely renumbered.
_REMOVED_LEGACY_TARGETS: dict[str, str] = {
    "C11": "C15 (Shinka-style evolution)",
    "C12": "C16 (AB-MCTS-style adaptive branching)",
    "C13": "C17 (Fugu-style dynamic orchestrator)",
}


def legacy_condition_id_migration(old_id: str) -> str:
    """Translate a schema-version-1 (pre-revision) condition ID to its
    current equivalent. C0-C4 are unchanged across schema versions and are
    returned as-is. Raises ``ValueError`` for an ID this map does not cover
    (including any ID already valid only under the current schema, such as a
    bare unrecognized string, and any ID whose migrated target was later
    removed from the track) so a caller cannot silently misclassify an
    artifact instead of migrating or rejecting it."""

    if old_id in ("C0", "C1", "C2", "C3", "C4"):
        return old_id
    if old_id in LEGACY_CONDITION_ID_MAP:
        return LEGACY_CONDITION_ID_MAP[old_id]
    if old_id in _REMOVED_LEGACY_TARGETS:
        raise ValueError(
            f"{old_id!r} is a schema-version-1 condition ID that migrates to "
            f"{_REMOVED_LEGACY_TARGETS[old_id]}, which has been removed from "
            "the condition track (C0-C14 only). This artifact cannot be "
            "reinterpreted under the current schema.")
    raise ValueError(
        f"{old_id!r} is not a recognized schema-version-1 condition ID; "
        "refusing to guess a migration. If this artifact predates the "
        "condition-track renumbering, check it by hand before reusing it.")

# execution_mode vocabulary used on every condition record.
EXECUTION_MODES: tuple[str, ...] = (
    "live",                       # true live-agent run through the automated path
    "live_manual_harness",        # true live run performed by the user in a harness
    "deterministic_baseline",     # C0 only — the intended non-LLM baseline
    "deterministic_fallback",     # debug-only fallback ranking (never comparable)
    "live_auth_failed",
    "live_rate_limited",
    "output_truncated",
    "live_error",
    "waiting_for_manual_harness",
)

COMPARABLE_EXECUTION_MODES: frozenset[str] = frozenset(
    {"live", "live_manual_harness", "deterministic_baseline"}
)


@dataclass(frozen=True)
class ConditionSpec:
    group: str
    condition_id: str
    label: str
    provider: str            # none | gemini | opencode_go | antigravity | codex
                             # | claude_code
    model: str | None
    harness: str             # deterministic_python | direct_api | antigravity
                             # | codex | claude_code | opencode
    algorithm: str
    skills_unified: bool
    branch_search: bool
    runner: str              # dispatch key in run_condition_proposals
    instruction_mode: str | None = None
    optional: bool = False
    model_roles: dict[str, str] = field(default_factory=dict)
    discovery_controller: str = "none"
    skills_access: Literal["enabled", "disabled"] = "disabled"
    search_policy: str = "none"
    routing_policy: Literal["none", "fixed_roles", "dynamic"] = "none"
    discovery_contract_version: str = "1.0"
    maximum_model_requests: int = 40
    maximum_code_executions: int = 30
    maximum_external_evaluations: int = 60
    maximum_review_rounds: int = 2
    maximum_wall_time_seconds: int = 3600
    # Comparison-family wiring (see comparison_families.py for the
    # authoritative FamilySpec registry; validate_family_registry_consistency
    # cross-checks these against it so the two never silently drift).
    comparison_families: tuple[str, ...] = ()
    primary_comparison_family: str = ""
    comparison_role: str = ""
    output_contract_id: str = ""
    budget_profile_id: str = ""
    report_profile_id: str = ""

    @property
    def padded_id(self) -> str:
        """Zero-padded id used for run/report directories (C0 -> C00)."""

        return f"C{int(self.group[1:]):02d}"

    @property
    def report_slug(self) -> str:
        """Self-explanatory report basename, e.g. C13_ai_scientist_gemini."""

        return f"{self.padded_id}_{_REPORT_SLUGS[self.group]}"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "condition_group": self.group,
            "condition_id": self.condition_id,
            "label": self.label,
            "provider": self.provider,
            "model": self.model,
            "harness": self.harness,
            "algorithm": self.algorithm,
            "skills_unified": self.skills_unified,
            "branch_search": self.branch_search,
            "instruction_mode": self.instruction_mode,
            "optional": self.optional,
            "model_roles": dict(self.model_roles),
            "discovery_controller": self.discovery_controller,
            "skills_access": self.skills_access,
            "search_policy": self.search_policy,
            "routing_policy": self.routing_policy,
            "discovery_contract_version": self.discovery_contract_version,
            "resource_envelope": {
                "maximum_model_requests": self.maximum_model_requests,
                "maximum_code_executions": self.maximum_code_executions,
                "maximum_external_evaluations": self.maximum_external_evaluations,
                "maximum_review_rounds": self.maximum_review_rounds,
                "maximum_wall_time_seconds": self.maximum_wall_time_seconds,
            },
            "comparison_families": list(self.comparison_families),
            "primary_comparison_family": self.primary_comparison_family,
            "comparison_role": self.comparison_role,
            "output_contract_id": self.output_contract_id,
            "budget_profile_id": self.budget_profile_id,
            "report_profile_id": self.report_profile_id,
        }


# Descriptive report/run naming per condition (first-time-reader friendly).
_REPORT_SLUGS: dict[str, str] = {
    "C0": "deterministic_baseline",
    "C1": "vanilla_gemini",
    "C2": "vanilla_codex_manual",
    "C3": "vanilla_claude_code_manual",
    "C4": "vanilla_deepseek",
    "C5": "antigravity_no_skills_manual",
    "C6": "codex_no_skills_manual",
    "C7": "claude_code_no_skills_manual",
    "C8": "opencode_no_skills_manual",
    "C9": "skills_antigravity_manual",
    "C10": "skills_codex_manual",
    "C11": "skills_claude_code_manual",
    "C12": "skills_opencode_manual",
    "C13": "ai_scientist_gemini",
    "C14": "ai_scientist_deepseek",
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _opencode_fast_model() -> str:
    return _env("OPENCODE_GO_FAST_MODEL", "deepseek-v4-flash")


def build_condition_registry() -> dict[str, ConditionSpec]:
    """Build the registry, resolving model names from the environment.

    Each provider/model anchor is resolved once from its vanilla condition's
    env var and chained as the default for its Skills/AI-Scientist
    counterpart(s), so the strict pairs/trio agree by default and only
    diverge (and get marked confounded) on an explicit override.
    """

    gemini_model = _env("GEMINI_MODEL", "gemini-3.5-flash")
    c1_model = _env("C1_MODEL", gemini_model)
    codex_model = _env("C2_MODEL", "gpt-5.5")
    claude_model = _env("C3_MODEL", "sonnet-5.0")
    deepseek_model = _env("C4_MODEL", _opencode_fast_model())
    # C9 defaults to gemini-3.5-flash (changed from gemini-3.5-pro) so its
    # strict pair C5 and the cross-cutting C1/C13 Gemini comparisons agree
    # by default; C5-C8 anchor to their C9-C12 Skills-unified counterpart's
    # model so each strict pair is confounded only on an explicit override.
    c9_model = _env("C9_MODEL", gemini_model)
    c5_model = _env("C5_MODEL", c9_model)
    c6_model = _env("C6_MODEL", codex_model)
    c7_model = _env("C7_MODEL", claude_model)
    c8_model = _env("C8_MODEL", deepseek_model)

    return {spec.group: spec for spec in (
        ConditionSpec(
            group="C0", condition_id="deterministic_python_baseline",
            label="Deterministic Python baseline",
            provider="none", model=None, harness="deterministic_python",
            algorithm="deterministic_baseline", skills_unified=False,
            branch_search=False, runner="c0_deterministic",
            discovery_controller="deterministic_fixed", maximum_model_requests=0,
            maximum_code_executions=0, maximum_review_rounds=0,
            comparison_families=("deterministic_reference",),
            primary_comparison_family="deterministic_reference",
            comparison_role="baseline_reference",
            output_contract_id="deterministic_v1",
            budget_profile_id="zero_budget",
            report_profile_id="deterministic_reference",
        ),
        ConditionSpec(
            group="C1", condition_id="vanilla_direct_gemini",
            label="Vanilla direct LLM baseline (Gemini)",
            provider="gemini", model=c1_model,
            harness="direct_api", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="vanilla_llm",
            discovery_controller="single_pass", maximum_model_requests=1,
            maximum_code_executions=0, maximum_review_rounds=0,
            comparison_families=("vanilla_model",),
            primary_comparison_family="vanilla_model",
            comparison_role="vanilla_arm",
            output_contract_id="decision_analysis_reporting_v2_vanilla",
            budget_profile_id="vanilla_single_pass",
            report_profile_id="vanilla",
        ),
        ConditionSpec(
            group="C2", condition_id="vanilla_codex_no_skills",
            label="Vanilla LLM baseline (Codex, gpt-5.5)",
            provider="codex", model=codex_model,
            harness="codex", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="manual_harness",
            discovery_controller="single_pass", maximum_model_requests=1,
            maximum_code_executions=0, maximum_review_rounds=0,
            comparison_families=("vanilla_model",),
            primary_comparison_family="vanilla_model",
            comparison_role="vanilla_arm",
            output_contract_id="decision_analysis_reporting_v2_vanilla",
            budget_profile_id="vanilla_single_pass",
            report_profile_id="vanilla",
        ),
        ConditionSpec(
            group="C3", condition_id="vanilla_claude_code_no_skills",
            label="Vanilla LLM baseline (Claude Code, sonnet-5.0)",
            provider="claude_code", model=claude_model,
            harness="claude_code", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="manual_harness",
            discovery_controller="single_pass", maximum_model_requests=1,
            maximum_code_executions=0, maximum_review_rounds=0,
            comparison_families=("vanilla_model",),
            primary_comparison_family="vanilla_model",
            comparison_role="vanilla_arm",
            output_contract_id="decision_analysis_reporting_v2_vanilla",
            budget_profile_id="vanilla_single_pass",
            report_profile_id="vanilla",
        ),
        ConditionSpec(
            group="C4", condition_id="vanilla_direct_deepseek",
            label="Vanilla direct LLM baseline (DeepSeek)",
            provider=_env("C4_PROVIDER", "opencode_go"), model=deepseek_model,
            harness="direct_api", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="vanilla_llm",
            discovery_controller="single_pass", maximum_model_requests=1,
            maximum_code_executions=0, maximum_review_rounds=0,
            comparison_families=("vanilla_model",),
            primary_comparison_family="vanilla_model",
            comparison_role="vanilla_arm",
            output_contract_id="decision_analysis_reporting_v2_vanilla",
            budget_profile_id="vanilla_single_pass",
            report_profile_id="vanilla",
        ),
        ConditionSpec(
            group="C5", condition_id="antigravity_no_skills_coding_agent",
            label="Antigravity coding-agent control",
            provider="antigravity", model=c5_model,
            harness="antigravity", algorithm="native_coding_agent",
            skills_unified=False, branch_search=True,
            runner="coding_agent_no_skills",
            discovery_controller="native_open_ended_agent", search_policy="open_ended",
            comparison_families=("native_agent_model", "skills_ablation_pair"),
            primary_comparison_family="native_agent_model",
            comparison_role="skills_off_arm",
            output_contract_id="decision_analysis_reporting_v2_native_agent",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="open_agentic",
        ),
        ConditionSpec(
            group="C6", condition_id="codex_no_skills_coding_agent",
            label="Codex coding-agent control",
            provider="codex", model=c6_model,
            harness="codex", algorithm="native_coding_agent",
            skills_unified=False, branch_search=True,
            runner="coding_agent_no_skills",
            discovery_controller="native_open_ended_agent", search_policy="open_ended",
            comparison_families=("native_agent_model", "skills_ablation_pair"),
            primary_comparison_family="native_agent_model",
            comparison_role="skills_off_arm",
            output_contract_id="decision_analysis_reporting_v2_native_agent",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="open_agentic",
        ),
        ConditionSpec(
            group="C7", condition_id="claude_code_no_skills_coding_agent",
            label="Claude Code coding-agent control",
            provider="claude_code", model=c7_model,
            harness="claude_code", algorithm="native_coding_agent",
            skills_unified=False, branch_search=True,
            runner="coding_agent_no_skills",
            discovery_controller="native_open_ended_agent", search_policy="open_ended",
            comparison_families=("native_agent_model", "skills_ablation_pair"),
            primary_comparison_family="native_agent_model",
            comparison_role="skills_off_arm",
            output_contract_id="decision_analysis_reporting_v2_native_agent",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="open_agentic",
        ),
        ConditionSpec(
            group="C8", condition_id="opencode_no_skills_coding_agent",
            label="OpenCode coding-agent control",
            provider=_env("C8_PROVIDER", "opencode_go"), model=c8_model,
            harness="opencode", algorithm="native_coding_agent",
            skills_unified=False, branch_search=True,
            runner="coding_agent_no_skills",
            discovery_controller="native_open_ended_agent", search_policy="open_ended",
            comparison_families=("native_agent_model", "skills_ablation_pair"),
            primary_comparison_family="native_agent_model",
            comparison_role="skills_off_arm",
            output_contract_id="decision_analysis_reporting_v2_native_agent",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="open_agentic",
        ),
        ConditionSpec(
            group="C9", condition_id="skills_antigravity",
            label="Skills-Antigravity",
            provider="antigravity",
            model=c9_model,
            harness="antigravity", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
            discovery_controller="persistent_skills_agent", skills_access="enabled",
            search_policy="open_ended_dynamic_skills",
            comparison_families=("skills_agent_model", "skills_ablation_pair"),
            primary_comparison_family="skills_agent_model",
            comparison_role="skills_on_arm",
            output_contract_id="skills_five_objective_v1",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="skills_objectives",
        ),
        ConditionSpec(
            group="C10", condition_id="skills_codex",
            label="Skills-Codex",
            provider="codex", model=_env("C10_MODEL", codex_model),
            harness="codex", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
            discovery_controller="persistent_skills_agent", skills_access="enabled",
            search_policy="open_ended_dynamic_skills",
            comparison_families=("skills_agent_model", "skills_ablation_pair"),
            primary_comparison_family="skills_agent_model",
            comparison_role="skills_on_arm",
            output_contract_id="skills_five_objective_v1",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="skills_objectives",
        ),
        ConditionSpec(
            group="C11", condition_id="skills_claude_code",
            label="Skills-Claude Code",
            provider="claude_code", model=_env("C11_MODEL", claude_model),
            harness="claude_code", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
            discovery_controller="persistent_skills_agent", skills_access="enabled",
            search_policy="open_ended_dynamic_skills",
            comparison_families=("skills_agent_model", "skills_ablation_pair"),
            primary_comparison_family="skills_agent_model",
            comparison_role="skills_on_arm",
            output_contract_id="skills_five_objective_v1",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="skills_objectives",
        ),
        ConditionSpec(
            group="C12", condition_id="skills_opencode",
            label="Skills-OpenCode",
            provider=_env("C12_PROVIDER", "opencode_go"),
            model=_env("C12_MODEL", deepseek_model),
            harness="opencode", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
            discovery_controller="persistent_skills_agent", skills_access="enabled",
            search_policy="open_ended_dynamic_skills",
            comparison_families=("skills_agent_model", "skills_ablation_pair"),
            primary_comparison_family="skills_agent_model",
            comparison_role="skills_on_arm",
            output_contract_id="skills_five_objective_v1",
            budget_profile_id="agentic_default_envelope",
            report_profile_id="skills_objectives",
        ),
        ConditionSpec(
            group="C13", condition_id="ai_scientist_style_gemini",
            label="Traditional AI Scientist-style Gemini",
            provider="gemini", model=_env("C13_MODEL", c1_model),
            harness="direct_api", algorithm="ai_scientist_style",
            skills_unified=False, branch_search=True,
            runner="c13_ai_scientist_gemini",
            discovery_controller="external_python_code_tree", search_policy="progressive_code_tree",
            comparison_families=("ai_scientist_model",),
            primary_comparison_family="ai_scientist_model",
            comparison_role="ai_scientist_arm",
            output_contract_id="ai_scientist_structured_synthesis_v1",
            budget_profile_id="ai_scientist_shared",
            report_profile_id="ai_scientist",
        ),
        ConditionSpec(
            group="C14", condition_id="ai_scientist_style_deepseek",
            label="Traditional AI Scientist-style DeepSeek",
            provider=_env("C14_PROVIDER", "opencode_go"),
            model=_env("C14_MODEL", deepseek_model),
            harness="direct_api", algorithm="ai_scientist_style_large_scale",
            skills_unified=False, branch_search=True,
            runner="c14_ai_scientist_deepseek",
            discovery_controller="external_python_code_tree", search_policy="progressive_code_tree",
            comparison_families=("ai_scientist_model",),
            primary_comparison_family="ai_scientist_model",
            comparison_role="ai_scientist_arm",
            output_contract_id="ai_scientist_structured_synthesis_v1",
            budget_profile_id="ai_scientist_shared",
            report_profile_id="ai_scientist",
        ),
    )}


def _pair_status(
    registry: dict[str, ConditionSpec], group_a: str, group_b: str,
) -> dict[str, Any]:
    a, b = registry[group_a], registry[group_b]
    same_provider = a.provider == b.provider
    same_model = a.model == b.model
    same_harness = a.harness == b.harness
    envelope_fields = (
        "maximum_model_requests", "maximum_code_executions",
        "maximum_external_evaluations", "maximum_review_rounds",
        "maximum_wall_time_seconds",
    )
    same_resource_envelope = all(getattr(a, key) == getattr(b, key) for key in envelope_fields)
    return {
        "same_provider": same_provider,
        "same_model": same_model,
        "same_harness": same_harness,
        "same_resource_envelope": same_resource_envelope,
        "confounded": not (same_provider and same_model),
        group_a.lower(): {"provider": a.provider, "model": a.model, "harness": a.harness},
        group_b.lower(): {"provider": b.provider, "model": b.model, "harness": b.harness},
        "differs_by": "treatment_only" if same_provider and same_model
        else "provider_or_model_mismatch",
        "comparison_limitations": (
            "Provider/model alignment does not remove residual harness, prompting, "
            "or controller implementation differences."
        ),
    }


def c1_c13_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C1 (vanilla direct Gemini) vs C13 (AI Scientist-style Gemini): same
    provider and model, so the delta isolates the full agentic loop over a
    single vanilla pass on the Gemini stack."""

    return _pair_status(registry, "C1", "C13")


def c2_c10_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C2 (vanilla Codex CLI, no Skills) vs C10 (Skills-Codex + branch
    search): same provider and model, so the delta isolates the Skills
    contract and branch search on the same Codex/gpt-5.5 stack."""

    return _pair_status(registry, "C2", "C10")


def c3_c11_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C3 (vanilla Claude Code CLI, no Skills) vs C11 (Skills-Claude Code +
    branch search): same provider and model, isolating the Skills contract
    and branch search on the same Claude Code/sonnet-5.0 stack."""

    return _pair_status(registry, "C3", "C11")


def c4_c12_c14_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C4 (vanilla direct DeepSeek), C12 (Skills-OpenCode + branch search),
    and C14 (AI Scientist-style DeepSeek) must share provider and model;
    otherwise harness/orchestration effects are confounded with model
    effects and E13 must say so. C4 vs C12 isolates the Skills contract over
    a vanilla pass; C4 vs C14 isolates the full AI-Scientist agentic loop;
    C12 vs C14 isolates the interactive Skills harness against the
    direct-API draft-tree loop."""

    c4, c12, c14 = registry["C4"], registry["C12"], registry["C14"]
    same_provider = c4.provider == c12.provider == c14.provider
    same_model = c4.model == c12.model == c14.model
    return {
        "same_provider": same_provider,
        "same_model": same_model,
        "confounded": not (same_provider and same_model),
        "c4": {"provider": c4.provider, "model": c4.model, "harness": c4.harness},
        "c12": {"provider": c12.provider, "model": c12.model, "harness": c12.harness},
        "c14": {"provider": c14.provider, "model": c14.model, "harness": c14.harness},
        "differs_by": "harness_and_orchestration_only" if same_provider and same_model
        else "provider_or_model_mismatch",
    }


def c5_c9_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C5 (Antigravity coding-agent control, no Skills) vs C9
    (Skills-Antigravity + branch search): same provider and model
    (gemini-3.5-flash), so the delta isolates the project-specific Skills
    contract itself, holding the CLI/harness and model fixed."""

    return _pair_status(registry, "C5", "C9")


def c6_c10_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C6 (Codex coding-agent control, no Skills) vs C10 (Skills-Codex +
    branch search): same provider and model (gpt-5.5), isolating the Skills
    contract on the same Codex CLI stack."""

    return _pair_status(registry, "C6", "C10")


def c7_c11_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C7 (Claude Code coding-agent control, no Skills) vs C11
    (Skills-Claude Code + branch search): same provider and model
    (sonnet-5.0), isolating the Skills contract on the same Claude Code CLI
    stack."""

    return _pair_status(registry, "C7", "C11")


def c8_c12_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C8 (OpenCode coding-agent control, no Skills) vs C12 (Skills-OpenCode
    + branch search): same provider and model (deepseek-v4-flash), isolating
    the Skills contract on the same OpenCode CLI stack."""

    return _pair_status(registry, "C8", "C12")


def c9_c13_discovery_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    result = _pair_status(registry, "C9", "C13")
    result["principal_difference"] = "persistent_skills_agent_vs_external_python_code_tree"
    return result


def c12_c14_discovery_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    result = _pair_status(registry, "C12", "C14")
    result["principal_difference"] = "persistent_skills_agent_vs_external_python_code_tree"
    return result
