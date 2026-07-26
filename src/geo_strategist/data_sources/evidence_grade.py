"""Shared evidence-grade vocabulary for the site-selection pipeline (S1-S7, E14).

Evidence grades replace the old hard blockers with a graded scale. A concrete
site-level claim (address, coordinate, parcel, financial figure) must carry
one of these grades, and every scenario/model estimate must be labeled as
such. `unverified_candidate` and `rejected_or_blocked` are the only grades
that keep a claim out of a "recommended" decision-support tier; everything
else (including labeled scenario/model estimates) is admissible.
"""

from __future__ import annotations

EVIDENCE_GRADES: tuple[str, ...] = (
    "verified_source",
    "derived_from_verified_source",
    "third_party_estimate",
    "scenario_assumption",
    "model_estimate",
    "unverified_candidate",
    "rejected_or_blocked",
)

# Severity order: lower index = stronger evidence. Used to pick the
# "worst" (least certain) grade across a set of claims.
_SEVERITY_RANK: dict[str, int] = {grade: index for index, grade in enumerate(EVIDENCE_GRADES)}

# Grades that may still support a "recommended" decision-support tier,
# provided every scenario/model estimate is explicitly labeled as such.
RECOMMENDABLE_GRADES: frozenset[str] = frozenset(
    {
        "verified_source",
        "derived_from_verified_source",
        "third_party_estimate",
        "scenario_assumption",
        "model_estimate",
    }
)

NOT_RECOMMENDABLE_GRADES: frozenset[str] = frozenset({"unverified_candidate", "rejected_or_blocked"})


def is_valid_grade(grade: str) -> bool:
    return grade in _SEVERITY_RANK


def is_recommendable_grade(grade: str) -> bool:
    return grade in RECOMMENDABLE_GRADES


def worst_grade(grades: list[str]) -> str:
    """Return the least-certain grade in a list, defaulting to unverified_candidate for an empty list."""

    if not grades:
        return "unverified_candidate"
    return max(grades, key=lambda grade: _SEVERITY_RANK.get(grade, _SEVERITY_RANK["rejected_or_blocked"]))


def all_recommendable(grades: list[str]) -> bool:
    return all(is_recommendable_grade(grade) for grade in grades)
