#!/usr/bin/env python3
"""Phase 13 — workbook-backed deterministic cash-flow substrate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic cash-flow substrate.")
    parser.add_argument(
        "--candidate-actions",
        default=".data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl",
        help="Candidate action JSONL input.",
    )
    parser.add_argument(
        "--candidate-evidence-bundles",
        default=".data/interim/study_area/tokyo_aichi_osaka/candidate_evidence_bundles.jsonl",
        help="Candidate evidence bundle JSONL input.",
    )
    parser.add_argument(
        "--hospital-features",
        default=".data/interim/study_area/tokyo_aichi_osaka/hospital_features.jsonl",
        help="Workbook-derived hospital feature JSONL input.",
    )
    parser.add_argument(
        "--assumptions-config",
        default="configs/cash_flow_assumptions.yaml",
        help="Cash-flow scenario assumption config.",
    )
    args = parser.parse_args()

    from geo_strategist.finance.cash_flow import build_cash_flow_substrate

    result = build_cash_flow_substrate(
        candidate_actions_path=Path(args.candidate_actions),
        candidate_evidence_bundles_path=Path(args.candidate_evidence_bundles),
        hospital_features_path=Path(args.hospital_features),
        assumptions_config_path=Path(args.assumptions_config),
    )

    print(f"Run ID: {result.run_id}")
    print(f"Output: {result.output_dir}")
    print(f"Candidates processed: {result.candidate_count}")
    print(f"Counts by action: {result.counts_by_action}")
    print(f"Computable capex: {result.computable_capex_count}")
    print(f"Computable annual cash flow: {result.computable_annual_cash_flow_count}")
    print(f"Computable payback: {result.computable_payback_count}")
    print(f"Payback status counts: {result.payback_status_counts}")
    print(f"Issue counts by severity: {result.issue_counts_by_severity}")
    print(f"Issue counts by code: {result.issue_counts_by_code}")


if __name__ == "__main__":
    main()
