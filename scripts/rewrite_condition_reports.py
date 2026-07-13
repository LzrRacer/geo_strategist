"""Rewrite condition reports from existing condition records — no live calls.

Reconstructs each condition's ``LiveConditionResult`` from
``condition_records.jsonl`` and re-runs the shared report writer, so report
layout/section changes (metric maps, selection funnel, candidate-specific
review comments) reach already-executed conditions without re-running their
live agents. The records file is rewritten afterwards because the qualitative
discussion is regenerated onto each proposal (the comparison judge scores
that structured form).

Usage:
    .venv/bin/python scripts/rewrite_condition_reports.py \
        outputs/condition_proposals/run_stage1 [more_dirs ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def rewrite_reports(stage_dir: Path, registry, data) -> list[str]:
    from geo_strategist.experiments.condition_utils import _read_jsonl, _write_jsonl
    from geo_strategist.experiments.live_common import LiveConditionResult
    from geo_strategist.experiments.live_report import write_condition_report

    records_path = stage_dir / "condition_records.jsonl"
    if not records_path.exists():
        return []
    records = _read_jsonl(records_path)
    rewritten: list[str] = []
    for record in records:
        group = str(record.get("condition_group"))
        spec = registry.get(group)
        if spec is None:
            continue
        result = LiveConditionResult(
            spec=spec,
            execution_mode=str(record.get("execution_mode") or "unknown"),
            comparable_for_e13=bool(record.get("comparable_for_e13")),
            exclusion_reason=record.get("exclusion_reason"),
            proposals=record.get("proposals") or [],
            branch_results=record.get("branch_results") or [],
            generated_code_stats=record.get("generated_code_stats") or {},
            model_call_summary=record.get("model_call_summary") or {},
            due_diligence=record.get("required_due_diligence") or [],
            narrative_sections=record.get("narrative_sections") or {},
            artifacts={Path(path).name: path
                       for path in record.get("source_artifacts") or []},
            failure_notes=record.get("failure_notes") or [],
            steps_run=int(record.get("steps_run") or 0),
        )
        report_path = write_condition_report(spec, result, stage_dir, data=data)
        record["proposal_report_path"] = str(report_path)
        rewritten.append(group)

    _write_jsonl(records_path, records)
    return rewritten


def main() -> int:
    stage_dirs = [Path(arg) for arg in sys.argv[1:]]
    if not stage_dirs:
        print("usage: rewrite_condition_reports.py <stage_dir> [...]",
              file=sys.stderr)
        return 2

    from geo_strategist.experiments.condition_registry import build_condition_registry
    from geo_strategist.experiments.deterministic_evaluation_engine import load_data_bundle

    # One registry/bundle serves every stage dir (both are repo-level state).
    registry = build_condition_registry()
    data = load_data_bundle(REPO_ROOT)
    for stage_dir in stage_dirs:
        if not stage_dir.is_absolute():
            stage_dir = REPO_ROOT / stage_dir
        groups = rewrite_reports(stage_dir, registry, data)
        print(f"{stage_dir}: rewrote {len(groups)} report(s) ({', '.join(groups)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
