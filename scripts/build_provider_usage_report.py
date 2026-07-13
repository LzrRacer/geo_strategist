"""Build provider usage artifacts for a condition-proposal run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from geo_strategist.experiments.provider_usage_report import build_provider_usage_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="outputs/condition_proposals/live")
    args = parser.parse_args()

    result = build_provider_usage_report(REPO_ROOT / args.target)
    print(f"provider usage json: {result.summary_json_path}")
    print(f"provider usage csv: {result.calls_csv_path}")
    print(f"provider usage report: {result.markdown_path}")
    print(f"requests={result.total_requests} errors={result.total_errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
