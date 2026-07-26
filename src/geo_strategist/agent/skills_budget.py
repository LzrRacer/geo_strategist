"""Explicit, configurable search budgets for the C9-C12 Skills-unified loop.

Defaults are aligned with the AI-Scientist-style C13/C14 budget envelope so
the Skills-unified conditions are not silently given unlimited resources.
Configured and consumed resources are recorded in the run manifest rather
than assumed; if the harness cannot enforce a limit (interactive CLI
sessions cannot be stepped like the direct-API AI-Scientist loop), the
deviation is recorded explicitly rather than pretending parity holds.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


@dataclass(frozen=True)
class SkillsSearchBudget:
    num_drafts: int = 10
    agent_steps: int = 40
    variants_per_branch: int = 6
    max_debug_depth: int = 4
    concurrency: int = 4
    max_revision_rounds: int = 2

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def skills_budget_for(condition_group: str) -> SkillsSearchBudget:
    """Resolve a Skills condition's budget from ``{group}_*`` env vars,
    falling back to the AI-Scientist-aligned defaults."""

    defaults = SkillsSearchBudget()
    return SkillsSearchBudget(
        num_drafts=_env_int(f"{condition_group}_NUM_DRAFTS", defaults.num_drafts),
        agent_steps=_env_int(f"{condition_group}_AGENT_STEPS", defaults.agent_steps),
        variants_per_branch=_env_int(
            f"{condition_group}_VARIANTS_PER_BRANCH", defaults.variants_per_branch),
        max_debug_depth=_env_int(
            f"{condition_group}_MAX_DEBUG_DEPTH", defaults.max_debug_depth),
        concurrency=_env_int(f"{condition_group}_CONCURRENCY", defaults.concurrency),
        max_revision_rounds=_env_int(
            f"{condition_group}_MAX_REVISION_ROUNDS", defaults.max_revision_rounds),
    )


def consumed_resources_from_trace(trace: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize resources actually consumed by a recorded skill trace, for
    comparison against the configured budget in the run manifest."""

    debug_rows = [r for r in trace if str(r.get("skill_id")) == "debug_failed_code"]
    branch_rows = [r for r in trace if str(r.get("skill_id")) == "run_branch_search"]
    revision_rows = [r for r in trace if str(r.get("skill_id")) == "revise_proposal"]
    branches: list[Any] = []
    if branch_rows:
        payload = branch_rows[-1].get("payload") or {}
        branches = payload.get("branches") if isinstance(payload, dict) else None
        branches = branches if isinstance(branches, list) else []
    return {
        "steps_recorded": len(trace),
        "debug_attempts": len(debug_rows),
        "branches_explored": len(branches),
        "revision_rounds": len(revision_rows),
    }


def resource_usage_summary(condition_group: str, trace: list[dict[str, Any]]) -> dict[str, Any]:
    configured = skills_budget_for(condition_group)
    consumed = consumed_resources_from_trace(trace)
    return {
        "condition_group": condition_group,
        "configured_budget": configured.to_dict(),
        "consumed_resources": consumed,
        "within_budget": {
            "agent_steps": consumed["steps_recorded"] <= configured.agent_steps,
            "max_debug_depth": consumed["debug_attempts"] <= configured.max_debug_depth,
            "max_revision_rounds": consumed["revision_rounds"] <= configured.max_revision_rounds,
        },
    }
