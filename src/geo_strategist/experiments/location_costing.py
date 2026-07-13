"""Location-specific scenario costing from verified land medians.

The hospital workbook carries prefecture-level model estimates. Candidate
municipalities carry MLIT Reinfolib municipal land medians. This module joins
those two evidence layers with a transparent proportional scaling assumption:
all per-bed cost/payback figures move in proportion to the candidate
municipality's land-price median versus the workbook prefecture median.
"""

from __future__ import annotations

from typing import Any


def _positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None


def location_cost_model(
    facts: dict[str, Any],
    prefecture_cost_model: dict[str, Any],
) -> dict[str, Any]:
    """Return a candidate-specific cost scenario, or unavailable fields.

    The returned estimates are not verified site costings. They are
    ``scenario_assumption`` values derived from verified MLIT municipal land
    medians and workbook ``model_estimate`` prefecture medians.
    """

    municipal_land = _positive_number(facts.get("land_price_median_jpy_per_sqm"))
    workbook_land = _positive_number(
        prefecture_cost_model.get("median_land_price_jpy_per_sqm")
    )
    factor = (
        municipal_land / workbook_land
        if municipal_land is not None and workbook_land is not None
        else None
    )

    result: dict[str, Any] = {
        "municipality_land_price_jpy_per_sqm": municipal_land,
        "workbook_median_land_price_jpy_per_sqm": workbook_land,
        "land_price_sample_count": facts.get("land_price_sample_count"),
        "land_price_year": facts.get("land_price_year"),
        "workbook_sample_size": prefecture_cost_model.get("sample_size"),
        "location_cost_factor": round(factor, 4) if factor is not None else None,
        "evidence_grade": "scenario_assumption" if factor is not None else "not_available",
        "scaling_assumption": (
            "Candidate municipality MLIT land median divided by the workbook "
            "prefecture land median; per-bed construction cost, initial "
            "investment, and payback are scaled proportionally."
        ),
    }
    for source_field, target_field in (
        ("median_construction_cost_per_bed_jpy_mm",
         "estimated_construction_cost_per_bed_jpy_mm"),
        ("median_initial_investment_per_bed_jpy_mm",
         "estimated_initial_investment_per_bed_jpy_mm"),
        ("median_payback_years", "estimated_payback_years"),
    ):
        source_value = _positive_number(prefecture_cost_model.get(source_field))
        result[f"workbook_{source_field}"] = source_value
        result[target_field] = (
            round(source_value * factor, 2)
            if source_value is not None and factor is not None
            else None
        )
    return result
