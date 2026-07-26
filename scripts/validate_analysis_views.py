#!/usr/bin/env python
"""Validate analysis-ready source view artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from geo_strategist.data.validate_views import validate_analysis_views


console = Console()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--require-outputs",
        action="store_true",
        help="Return nonzero if analysis-view outputs are absent.",
    )
    args = parser.parse_args()

    summary = validate_analysis_views(
        repo_root=Path(args.repo_root),
        require_outputs=args.require_outputs,
    )
    console.print(f"Checked outputs: {len(summary.checked_outputs)}")
    console.print(f"Missing outputs: {len(summary.missing_outputs)}")
    console.print(f"Hospital facts: {summary.hospital_fact_count}")
    console.print(f"Population long records: {summary.population_long_count}")
    console.print(f"Population rate records: {summary.population_rate_count}")
    console.print(f"Population geography keys: {summary.population_geography_key_count}")
    console.print(f"Study-area count records: {summary.study_area_population_long_count}")
    console.print(f"Study-area rate records: {summary.study_area_population_rate_count}")
    console.print(f"Study-area geography keys: {summary.study_area_geography_key_count}")
    console.print(f"Study-area issues: {summary.study_area_issue_count}")
    console.print(f"Geography-grain records: {summary.geography_grain_record_count}")
    console.print(f"Geography-grain issues: {summary.geography_grain_issue_count}")
    console.print(f"Population-base records: {summary.population_base_record_count}")
    console.print(f"Population-base issues: {summary.population_base_issue_count}")
    console.print(f"Coverage matrix rows: {summary.coverage_matrix_row_count}")
    console.print(f"Coverage issues: {summary.coverage_issue_count}")
    console.print(f"Age-normalized records: {summary.age_group_record_count}")
    console.print(f"Age-group issues: {summary.age_group_issue_count}")
    console.print(f"Model-input readiness records: {summary.model_input_readiness_record_count}")
    console.print(f"Model-input readiness issues: {summary.model_input_readiness_issue_count}")
    console.print(f"Quality issues: {summary.quality_issue_count}")
    console.print(f"Rate issues: {summary.population_rate_issue_count}")
    console.print(f"Geography issues: {summary.population_geography_issue_count}")
    console.print(f"Warnings: {len(summary.warnings)}")
    if summary.errors:
        console.print("[red]Validation failed:[/red]")
        for error in summary.errors:
            console.print(f"- {error}")
        return 1
    console.print("[green]Analysis views validated.[/green]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
