"""Canonical C0-C13 condition registry.

Single source of truth for condition identity: provider, model, harness,
algorithm family, Skills unification, and branch-search participation.
Model/provider values may be overridden through ``Cx_*`` environment
variables (see ``.env.example``); the strict comparison checks read from here.

Canonical order:

- C0  Deterministic Python baseline
- C1  Vanilla direct LLM baseline — Gemini (no tools, no code, no Skills)
- C2  Vanilla LLM baseline — Codex CLI, no Skills contract (manual harness)
- C3  Vanilla LLM baseline — Claude Code CLI, no Skills contract (manual harness)
- C4  Vanilla direct LLM baseline — DeepSeek via OpenCode Go direct API
- C5  Skills-Antigravity — AGENTS.md + filesystem Skills harness,
      Antigravity CLI (``agy``), Gemini, Skills-unified contract + branch search
- C6  Skills-Codex — AGENTS.md + filesystem Skills harness, Codex CLI
      (``codex``), Skills + branch search
- C7  Skills-Claude Code — AGENTS.md + filesystem Skills harness, Claude Code
      CLI (``claude``), Skills + branch search
- C8  Skills-OpenCode — AGENTS.md + filesystem Skills harness, OpenCode CLI
      (``opencode``), Skills + branch search
- C9  Traditional AI Scientist-style Gemini
- C10 Traditional AI Scientist-style DeepSeek
- C11 Multi-model Shinka-style evolution
- C12 AB-MCTS-style adaptive branching
- C13 Fugu-style dynamic orchestrator
- E13 Cross-condition proposal and agentic-process judge (not a condition)

Four provider/model stacks each pair a vanilla baseline against its
Skills-and/or-agentic-loop counterpart on the *same* provider+model, so the
delta isolates the treatment rather than the model:

- Gemini (gemini-3.5-flash):        C1 (vanilla) vs C9 (AI Scientist)      — strict pair
- Codex (gpt-5.5):                  C2 (vanilla) vs C6 (Skills)            — strict pair
- Claude Code (sonnet-5.0):         C3 (vanilla) vs C7 (Skills)            — strict pair
- OpenCode Go (deepseek-v4-flash):  C4 (vanilla) vs C8 (Skills) vs C10 (AI Scientist) — strict trio

C5 (Antigravity/gemini-3.5-pro) has no vanilla counterpart on the same
provider/model, so it is compared only informationally, alongside C6/C7/C8,
in the C5-C8 Skills-in-harness table (harness+model vary together there) —
never folded into a code-enforced strict-comparison check.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

CONDITION_ORDER: tuple[str, ...] = (
    "C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
    "C12", "C13",
)

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

    @property
    def padded_id(self) -> str:
        """Zero-padded id used for run/report directories (C0 -> C00)."""

        return f"C{int(self.group[1:]):02d}"

    @property
    def report_slug(self) -> str:
        """Self-explanatory report basename, e.g. C09_ai_scientist_gemini."""

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
        }


# Descriptive report/run naming per condition (first-time-reader friendly).
_REPORT_SLUGS: dict[str, str] = {
    "C0": "deterministic_baseline",
    "C1": "vanilla_gemini",
    "C2": "vanilla_codex_manual",
    "C3": "vanilla_claude_code_manual",
    "C4": "vanilla_deepseek",
    "C5": "skills_antigravity_manual",
    "C6": "skills_codex_manual",
    "C7": "skills_claude_code_manual",
    "C8": "skills_opencode_manual",
    "C9": "ai_scientist_gemini",
    "C10": "ai_scientist_deepseek",
    "C11": "shinka_evolution",
    "C12": "ab_mcts",
    "C13": "fugu_router",
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _opencode_roles() -> dict[str, str]:
    return {
        "fast": _env("OPENCODE_GO_FAST_MODEL", "deepseek-v4-flash"),
        "pro": _env("OPENCODE_GO_PRO_MODEL", "deepseek-v4-pro"),
        "code": _env("OPENCODE_GO_CODE_MODEL", "kimi-k2.7-code"),
        "synthesis": _env("OPENCODE_GO_SYNTHESIS_MODEL", "qwen3.7-plus"),
        "judge": _env("OPENCODE_GO_JUDGE_MODEL", "qwen3.7-max"),
    }


def build_condition_registry() -> dict[str, ConditionSpec]:
    """Build the registry, resolving model names from the environment.

    Each provider/model anchor is resolved once from its vanilla condition's
    env var and chained as the default for its Skills/AI-Scientist
    counterpart(s), so the strict pairs/trio agree by default and only
    diverge (and get marked confounded) on an explicit override.
    """

    roles = _opencode_roles()
    gemini_model = _env("GEMINI_MODEL", "gemini-3.5-flash")
    c1_model = _env("C1_MODEL", gemini_model)
    codex_model = _env("C2_MODEL", "gpt-5.5")
    claude_model = _env("C3_MODEL", "sonnet-5.0")
    deepseek_model = _env("C4_MODEL", roles["fast"])
    c13_model = _env("C13_MODEL", roles["synthesis"])

    return {spec.group: spec for spec in (
        ConditionSpec(
            group="C0", condition_id="deterministic_python_baseline",
            label="Deterministic Python baseline",
            provider="none", model=None, harness="deterministic_python",
            algorithm="deterministic_baseline", skills_unified=False,
            branch_search=False, runner="c0_deterministic",
        ),
        ConditionSpec(
            group="C1", condition_id="vanilla_direct_gemini",
            label="Vanilla direct LLM baseline (Gemini)",
            provider="gemini", model=c1_model,
            harness="direct_api", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="vanilla_llm",
        ),
        ConditionSpec(
            group="C2", condition_id="vanilla_codex_no_skills",
            label="Vanilla LLM baseline (Codex, gpt-5.5)",
            provider="codex", model=codex_model,
            harness="codex", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="manual_harness",
        ),
        ConditionSpec(
            group="C3", condition_id="vanilla_claude_code_no_skills",
            label="Vanilla LLM baseline (Claude Code, sonnet-5.0)",
            provider="claude_code", model=claude_model,
            harness="claude_code", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="manual_harness",
        ),
        ConditionSpec(
            group="C4", condition_id="vanilla_direct_deepseek",
            label="Vanilla direct LLM baseline (DeepSeek)",
            provider=_env("C4_PROVIDER", "opencode_go"), model=deepseek_model,
            harness="direct_api", algorithm="vanilla_llm",
            skills_unified=False, branch_search=False, runner="vanilla_llm",
        ),
        ConditionSpec(
            group="C5", condition_id="skills_antigravity",
            label="Skills-Antigravity",
            provider="antigravity",
            model=_env("C5_MODEL", "gemini-3.5-pro"),
            harness="antigravity", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
        ),
        ConditionSpec(
            group="C6", condition_id="skills_codex",
            label="Skills-Codex",
            provider="codex", model=_env("C6_MODEL", codex_model),
            harness="codex", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
        ),
        ConditionSpec(
            group="C7", condition_id="skills_claude_code",
            label="Skills-Claude Code",
            provider="claude_code", model=_env("C7_MODEL", claude_model),
            harness="claude_code", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
        ),
        ConditionSpec(
            group="C8", condition_id="skills_opencode",
            label="Skills-OpenCode",
            provider=_env("C8_PROVIDER", "opencode_go"),
            model=_env("C8_MODEL", deepseek_model),
            harness="opencode", algorithm="skills_branch_search",
            skills_unified=True, branch_search=True,
            runner="agentic_skills_harness", instruction_mode="agents_md_skills",
        ),
        ConditionSpec(
            group="C9", condition_id="ai_scientist_style_gemini",
            label="Traditional AI Scientist-style Gemini",
            provider="gemini", model=_env("C9_MODEL", c1_model),
            harness="direct_api", algorithm="ai_scientist_style",
            skills_unified=False, branch_search=True,
            runner="c9_ai_scientist_gemini",
        ),
        ConditionSpec(
            group="C10", condition_id="ai_scientist_style_deepseek",
            label="Traditional AI Scientist-style DeepSeek",
            provider=_env("C10_PROVIDER", "opencode_go"),
            model=_env("C10_MODEL", deepseek_model),
            harness="direct_api", algorithm="ai_scientist_style_large_scale",
            skills_unified=False, branch_search=True,
            runner="c10_ai_scientist_deepseek",
        ),
        ConditionSpec(
            group="C11", condition_id="multi_model_shinka_evolution",
            label="Multi-model Shinka-style evolution",
            provider="opencode_go", model=roles["fast"],
            harness="direct_api", algorithm="multi_model_evolution",
            skills_unified=False, branch_search=True, runner="c11_evolution",
            optional=True,
            model_roles={
                "mutation": roles["fast"], "critique": roles["pro"],
                "code_repair": roles["code"], "synthesis": roles["synthesis"],
                "judge": roles["judge"],
            },
        ),
        ConditionSpec(
            group="C12", condition_id="ab_mcts_adaptive_branching",
            label="AB-MCTS-style adaptive branching",
            provider="opencode_go", model=roles["fast"],
            harness="direct_api", algorithm="adaptive_branching_mcts",
            skills_unified=False, branch_search=True, runner="c12_ab_mcts",
            optional=True,
            model_roles={
                "expand": _env("C12_EXPAND_MODEL", roles["fast"]),
                "refine": _env("C12_REFINE_MODEL", roles["pro"]),
                "code_repair": _env("C12_CODE_MODEL", roles["code"]),
                "node_evaluation": _env("C12_REVIEW_MODEL", roles["synthesis"]),
                "judge": _env("C12_JUDGE_MODEL", roles["judge"]),
            },
        ),
        ConditionSpec(
            group="C13", condition_id="fugu_dynamic_orchestrator",
            label="Fugu-style dynamic orchestrator",
            provider="opencode_go", model=c13_model,
            harness="direct_api", algorithm="dynamic_multi_agent_orchestrator",
            skills_unified=False, branch_search=True, runner="c13_router",
            optional=True,
            model_roles={
                "router": _env("C13_ROUTER_MODEL", c13_model),
                "fast": _env("C13_FAST_MODEL", roles["fast"]),
                "pro": _env("C13_PRO_MODEL", roles["pro"]),
                "code": _env("C13_CODE_MODEL", roles["code"]),
                "review": _env("C13_REVIEW_MODEL", roles["synthesis"]),
                "judge": _env("C13_JUDGE_MODEL", roles["judge"]),
            },
        ),
    )}


def _pair_status(
    registry: dict[str, ConditionSpec], group_a: str, group_b: str,
) -> dict[str, Any]:
    a, b = registry[group_a], registry[group_b]
    same_provider = a.provider == b.provider
    same_model = a.model == b.model
    return {
        "same_provider": same_provider,
        "same_model": same_model,
        "confounded": not (same_provider and same_model),
        group_a.lower(): {"provider": a.provider, "model": a.model, "harness": a.harness},
        group_b.lower(): {"provider": b.provider, "model": b.model, "harness": b.harness},
        "differs_by": "treatment_only" if same_provider and same_model
        else "provider_or_model_mismatch",
    }


def c1_c9_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C1 (vanilla direct Gemini) vs C9 (AI Scientist-style Gemini): same
    provider and model, so the delta isolates the full agentic loop over a
    single vanilla pass on the Gemini stack."""

    return _pair_status(registry, "C1", "C9")


def c2_c6_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C2 (vanilla Codex CLI, no Skills) vs C6 (Skills-Codex + branch
    search): same provider and model, so the delta isolates the Skills
    contract and branch search on the same Codex/gpt-5.5 stack."""

    return _pair_status(registry, "C2", "C6")


def c3_c7_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C3 (vanilla Claude Code CLI, no Skills) vs C7 (Skills-Claude Code +
    branch search): same provider and model, isolating the Skills contract
    and branch search on the same Claude Code/sonnet-5.0 stack."""

    return _pair_status(registry, "C3", "C7")


def c4_c8_c10_strict_comparison_status(registry: dict[str, ConditionSpec]) -> dict[str, Any]:
    """C4 (vanilla direct DeepSeek), C8 (Skills-OpenCode + branch search),
    and C10 (AI Scientist-style DeepSeek) must share provider and model;
    otherwise harness/orchestration effects are confounded with model
    effects and E13 must say so. C4 vs C8 isolates the Skills contract over
    a vanilla pass; C4 vs C10 isolates the full AI-Scientist agentic loop;
    C8 vs C10 isolates the interactive Skills harness against the
    direct-API draft-tree loop."""

    c4, c8, c10 = registry["C4"], registry["C8"], registry["C10"]
    same_provider = c4.provider == c8.provider == c10.provider
    same_model = c4.model == c8.model == c10.model
    return {
        "same_provider": same_provider,
        "same_model": same_model,
        "confounded": not (same_provider and same_model),
        "c4": {"provider": c4.provider, "model": c4.model, "harness": c4.harness},
        "c8": {"provider": c8.provider, "model": c8.model, "harness": c8.harness},
        "c10": {"provider": c10.provider, "model": c10.model, "harness": c10.harness},
        "differs_by": "harness_and_orchestration_only" if same_provider and same_model
        else "provider_or_model_mismatch",
    }
