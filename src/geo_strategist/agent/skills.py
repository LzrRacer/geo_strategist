"""The canonical Skills-unified contract shared by C5/C6/C7/C8.

Every Skills-unified condition — one manual-harness run per coding-agent CLI
(Antigravity, Codex, Claude Code, OpenCode) — walks the same ten skills, so
skill traces stay comparable across harnesses.
"""

from __future__ import annotations

SKILLS_UNIFIED_CONTRACT: tuple[str, ...] = (
    "inspect_available_data",
    "generate_research_hypotheses",
    "design_evaluation_model",
    "write_experiment_code",
    "execute_generated_code",
    "debug_failed_code",
    "run_branch_search",
    "review_proposal",
    "revise_proposal",
    "write_final_condition_proposal",
)

# Back-compat alias for older imports.
AGENT_SKILLS = SKILLS_UNIFIED_CONTRACT
