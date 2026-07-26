"""Fail-closed stub connector for source categories not configured in this environment.

Used for zoning/land-use, hazard/disaster-risk, transport-accessibility,
and land-parcel/lot-number registries: no licensed dataset or live network
access is available here. This connector always returns an empty record
list plus an explicit issue, so downstream code fails closed rather than
inventing a value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_records(
    repo_root: str | Path = ".",
    *,
    path: str | Path | None = None,
    source_key: str = "unknown_source",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return [], [{
        "issue_code": "source_not_configured_in_this_environment",
        "severity": "info",
        "source_key": source_key,
        "message": (
            f"'{source_key}' has no dataset configured in this environment. "
            "Any dependent field must stay not_available / unverified_candidate."
        ),
    }]
