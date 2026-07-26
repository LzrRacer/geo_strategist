"""The five shared branch objectives used by every branch/search condition.

C9, C10, C11, C12, C13, and C14 all search over exactly these five
objectives so their branch structures stay comparable. Each objective maps to
an emphasis profile over the deterministic evaluation-model components, which
gives every condition the same *external* (non-LLM) metric for judging a
candidate slate under that objective — LLM output is steered and scored by
data-grounded metrics, never by another hard-coded ranking.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BranchObjective:
    key: str          # CLI form, e.g. "elderly-demand"
    slug: str         # artifact/python form, e.g. "elderly_demand"
    label: str
    description: str
    direction: str    # maximize | minimize (phrasing only; metrics are higher-is-better)
    component_emphasis: dict[str, float]  # multipliers over engine components


BRANCH_OBJECTIVES: tuple[BranchObjective, ...] = (
    BranchObjective(
        key="elderly-demand",
        slug="elderly_demand",
        label="A: elderly-demand maximization",
        description=(
            "Prioritize municipalities with the highest elderly-care demand "
            "pressure (aging rate and demand scores)."
        ),
        direction="maximize",
        component_emphasis={"aging": 2.0, "demand": 1.6},
    ),
    BranchObjective(
        key="emergency-access",
        slug="emergency_access",
        label="B: emergency-access minimization",
        description=(
            "Minimize emergency-access gaps: prioritize supply-shortage areas "
            "where a new or reorganized facility shortens access."
        ),
        direction="minimize",
        component_emphasis={"supply_shortage": 2.2, "demand": 1.3},
    ),
    BranchObjective(
        key="reorganization-feasibility",
        slug="reorganization_feasibility",
        label="C: reorganization feasibility",
        description=(
            "Favor candidates where reorganization/consolidation is feasible: "
            "existing facility density, land availability, and evidence depth."
        ),
        direction="maximize",
        component_emphasis={"supply_shortage": 1.5, "land": 1.8, "evidence_completeness": 1.4},
    ),
    BranchObjective(
        key="financial-risk",
        slug="financial_risk",
        label="D: financial-risk minimization",
        description=(
            "Minimize financial risk: favorable payback estimates and low "
            "demographic-decline risk."
        ),
        direction="minimize",
        component_emphasis={"financial": 2.2, "demographic_risk": 1.8},
    ),
    BranchObjective(
        key="evidence-completeness",
        slug="evidence_completeness",
        label="E: evidence-completeness maximization",
        description=(
            "Maximize evidence completeness so the slate is auditable: data "
            "coverage and source-traceable facility evidence."
        ),
        direction="maximize",
        component_emphasis={"evidence_completeness": 2.5},
    ),
)

OBJECTIVES_BY_KEY: dict[str, BranchObjective] = {o.key: o for o in BRANCH_OBJECTIVES}
DEFAULT_BRANCH_OBJECTIVES_CSV = ",".join(o.key for o in BRANCH_OBJECTIVES)


def parse_branch_objectives(csv_value: str | None) -> list[BranchObjective]:
    """Parse ``--branch-objectives``; unknown keys raise ValueError."""

    if not csv_value:
        return list(BRANCH_OBJECTIVES)
    objectives: list[BranchObjective] = []
    for raw in csv_value.split(","):
        key = raw.strip()
        if not key:
            continue
        if key not in OBJECTIVES_BY_KEY:
            raise ValueError(
                f"unknown branch objective {key!r}; expected one of "
                f"{sorted(OBJECTIVES_BY_KEY)}"
            )
        objectives.append(OBJECTIVES_BY_KEY[key])
    return objectives or list(BRANCH_OBJECTIVES)


def objective_weights(objective: BranchObjective, base: dict[str, float]) -> dict[str, float]:
    """Apply the objective's emphasis to base component weights, renormalized."""

    weights = {
        name: value * objective.component_emphasis.get(name, 1.0)
        for name, value in base.items()
    }
    total = sum(weights.values()) or 1.0
    return {name: round(value / total, 6) for name, value in weights.items()}
