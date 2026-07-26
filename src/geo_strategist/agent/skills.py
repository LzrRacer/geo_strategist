"""Dynamic operator library shared by C9/C10/C11/C12.

The tuple is a registry order, not a prescribed execution sequence. Skills
may be selected repeatedly whenever their declared inputs are available.
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
    "expand_decision_regime",
    "compare_search_branches",
    "analyze_branch_divergence",
    "design_robustness_test",
    "run_robustness_ablations",
    "identify_reversal_conditions",
    "review_proposal",
    "revise_proposal",
    "synthesize_decision_portfolios",
    "prioritize_due_diligence",
    "write_final_condition_proposal",
)

# Back-compat alias for older imports.
AGENT_SKILLS = SKILLS_UNIFIED_CONTRACT
