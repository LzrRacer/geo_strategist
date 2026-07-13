"""Shared report-writing helpers (standard footer, Markdown tables)."""

from geo_strategist.reporting.footer import (
    DUE_DILIGENCE_SECTION_TITLE,
    required_due_diligence_section,
    standard_due_diligence_items,
)
from geo_strategist.reporting.tables import markdown_table

__all__ = [
    "DUE_DILIGENCE_SECTION_TITLE",
    "required_due_diligence_section",
    "standard_due_diligence_items",
    "markdown_table",
]
