"""Standard limitations / due-diligence footer for every proposal report.

Cautionary notes are consolidated here instead of being repeated through the
report body: the body presents the proposal, rationale, scores, candidate
comparisons, risks, and next investigation steps, and this single closing
section carries the disclaimers and the run-specific due-diligence items.
"""

from __future__ import annotations

DUE_DILIGENCE_SECTION_TITLE = "## Limitations / Required Due Diligence"

_FIXED_BULLETS: tuple[str, ...] = (
    "This proposal is a decision-support draft for hospital location, hospital "
    "reorganization, and investment evaluation. It is not, by itself, a final "
    "investment decision, healthcare delivery decision, administrative decision, "
    "or regulatory decision.",
    "Addresses, lot numbers, coordinates, land ownership, zoning, buildability, "
    "disaster risk, transport accessibility, permits, and financial actuals must "
    "be verified according to the stated evidence grade.",
    "Items marked as `scenario_assumption`, `model_estimate`, "
    "`unverified_candidate`, or `not_available` require confirmation through "
    "field investigation, public records, expert review, and/or financial audit.",
    "This proposal is an initial draft for comparing candidate sites, "
    "municipalities, and facility-reorganization options.",
    "Before any final decision, due diligence must be performed by specialists "
    "in healthcare policy, legal affairs, real estate, architecture, finance, "
    "regional healthcare planning, and emergency medical services.",
)


def standard_due_diligence_items() -> list[str]:
    """The fixed disclaimer bullets, as a list (for JSON output fields)."""

    return list(_FIXED_BULLETS)


def required_due_diligence_section(items: list[str] | None = None) -> str:
    """Render the standard closing section.

    ``items`` carries run-specific follow-ups: unverified fields, missing data
    sources, provider/API keys that were unavailable, and required field
    investigations. They are appended after the fixed bullets under their own
    subheading so a reader can separate the generic caveats from what this
    specific run could not verify.
    """

    lines = [DUE_DILIGENCE_SECTION_TITLE, ""]
    lines.extend(f"- {bullet}" for bullet in _FIXED_BULLETS)
    dynamic = [item for item in (items or []) if item]
    if dynamic:
        lines.append("")
        lines.append("### Run-specific items requiring verification")
        lines.append("")
        seen: set[str] = set()
        for item in dynamic:
            if item not in seen:
                seen.add(item)
                lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)
