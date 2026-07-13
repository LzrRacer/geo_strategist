"""Assemble the final live artifact set from staged condition runs.

Merges ``condition_records.jsonl`` from the staged run directories (later
stages override earlier records per condition), copies each condition's
report, figures, and ``runs/<CNN>`` trace directory into the canonical
target (default ``outputs/condition_proposals/live``), re-runs the
condition-comparison judge over the merged records, regenerates the manual
harness prompts + README, and writes ``README.md`` / ``artifact_index.md`` /
``run_manifest.json`` / ``condition_outputs_summary.json``.

Usage:
    .venv/bin/python scripts/rebuild_live_artifacts.py \
        [--target outputs/condition_proposals/live] [stage_dir ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_STAGES = [
    "outputs/condition_proposals/run_stage1",
    "outputs/condition_proposals/run_stage2",
    "outputs/condition_proposals/run_stage3",
    "outputs/condition_proposals/run_manual",
]

_LIVE_README = """\
# Live condition-comparison artifact set

Final aggregated results of the C0-C13 hospital location / reorganization
condition track. Everything here is generated; raw data stays under
`.data/` and source code under `src/`.

- `reports/` — one proposal report per condition
  (`C08_ai_scientist_deepseek.md`, ...) plus
  `condition_comparison_report.md`, the cross-condition judge report.
- `figures/` — all PNG figures referenced by the reports.
- `runs/C00..C11/` — per-condition traces: journals, generated code,
  sandboxes, redacted model-call traces, model-call summaries.
- `manual_harness/` — handoff prompts and instructions for the C3/C4/C5/C6
  conditions that require an interactive coding-agent session.
- `condition_records.jsonl` — machine-readable per-condition records.
- `condition_judge_scores.jsonl` — 18-dimension judge scores per condition.
- `condition_judge_manifest.json` — judge run metadata and result groups.
- `condition_outputs_summary.json` / `run_manifest.json` — run metadata.
- `artifact_index.md` — table of everything above.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stages", nargs="*", default=None)
    parser.add_argument("--target", default="outputs/condition_proposals/live")
    args = parser.parse_args()

    stage_dirs = [REPO_ROOT / s for s in (args.stages or DEFAULT_STAGES)]
    target = REPO_ROOT / args.target
    target.mkdir(parents=True, exist_ok=True)
    (target / "figures").mkdir(exist_ok=True)
    (target / "reports").mkdir(exist_ok=True)
    (target / "runs").mkdir(exist_ok=True)

    from geo_strategist.experiments.condition_output_contract import merge_condition_record
    from geo_strategist.experiments.condition_registry import (
        CONDITION_ORDER,
        build_condition_registry,
    )
    from geo_strategist.experiments.condition_utils import _read_jsonl, _write_jsonl

    records: dict[str, dict] = {}
    record_source: dict[str, Path] = {}
    # Later stages override earlier ones; the target dir merges last so
    # ingested manual results beat stage waiting placeholders (shared
    # precedence rule in merge_condition_record).
    for stage in [*stage_dirs, target]:
        for record in _read_jsonl(stage / "condition_records.jsonl"):
            group = str(record.get("condition_group"))
            merged = merge_condition_record(records.get(group), record)
            if merged is not records.get(group):
                record_source[group] = stage
            records[group] = merged

    registry = build_condition_registry()
    ordered_groups = [g for g in CONDITION_ORDER if g in records]

    for group in ordered_groups:
        stage = record_source[group]
        spec = registry[group]
        report = stage / "reports" / f"{spec.report_slug}.md"
        if report.exists() and stage.resolve() != target.resolve():
            shutil.copyfile(report, target / "reports" / report.name)
        if report.exists():
            records[group]["proposal_report_path"] = str(target / "reports" / report.name)
        elif records[group].get("proposals"):
            print(f"WARNING: {group}: no report at {report}; the merged record "
                  "keeps a stale proposal_report_path — regenerate the stage "
                  "with scripts/rewrite_condition_reports.py", file=sys.stderr)
        figures = stage / "figures"
        if figures.exists() and stage.resolve() != target.resolve():
            for figure in figures.glob(f"{spec.report_slug}_*.png"):
                shutil.copyfile(figure, target / "figures" / figure.name)
        run_dir = stage / "runs" / spec.padded_id
        if run_dir.exists() and stage.resolve() != target.resolve():
            destination = target / "runs" / spec.padded_id
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(run_dir, destination)
        records[group]["source_stage_dir"] = str(stage)

    _write_jsonl(target / "condition_records.jsonl",
                 [records[group] for group in ordered_groups])

    from geo_strategist.harnesses.prompts import (
        write_manual_harness_prompts,
        write_manual_harness_readme,
    )

    write_manual_harness_prompts(REPO_ROOT, output_dir=target)
    write_manual_harness_readme(target)

    from geo_strategist.experiments.condition_comparison_judge import (
        run_condition_comparison_judge,
    )
    from geo_strategist.experiments.provider_usage_report import (
        build_provider_usage_report,
    )

    judge = run_condition_comparison_judge(REPO_ROOT, proposals_dir=target)
    provider_usage = build_provider_usage_report(target)

    generated_at = datetime.now(timezone.utc).isoformat()
    summaries = [
        {
            "condition_group": group,
            "label": records[group].get("label"),
            "provider": records[group].get("provider"),
            "model": records[group].get("model"),
            "harness": records[group].get("harness"),
            "execution_mode": records[group].get("execution_mode"),
            "comparable_for_e13": records[group].get("comparable_for_e13"),
            "eligible_for_judge": records[group].get("eligible_for_judge"),
            "exclusion_reason": records[group].get("exclusion_reason"),
            "proposal_count": len(records[group].get("proposals") or []),
            "source_stage_dir": str(record_source[group]),
            "report_path": records[group].get("proposal_report_path"),
        }
        for group in ordered_groups
    ]
    (target / "condition_outputs_summary.json").write_text(
        json.dumps({
            "run_id": str(uuid.uuid4()),
            "generated_at": generated_at,
            "aggregated_from": [str(s) for s in stage_dirs],
            "conditions": summaries,
            "judge_report_path": judge.comparison_report_path,
            "judge_live_status": judge.live_judge_status,
            "provider_usage_report_path": provider_usage.markdown_path,
            "provider_usage_summary_path": provider_usage.summary_json_path,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "run_manifest.json").write_text(
        json.dumps({
            "run_id": str(uuid.uuid4()),
            "generated_at": generated_at,
            "output_dir": str(target),
            "conditions_run": ordered_groups,
            "aggregated_from": [str(s) for s in stage_dirs],
            "artifacts": {
                "condition_records": str(target / "condition_records.jsonl"),
                "condition_judge_scores": str(target / "condition_judge_scores.jsonl"),
                "condition_comparison_report": judge.comparison_report_path,
                "provider_usage_summary": provider_usage.summary_json_path,
                "provider_usage_csv": provider_usage.calls_csv_path,
                "provider_usage_report": provider_usage.markdown_path,
                **{f"{g}_report": records[g].get("proposal_report_path")
                   for g in ordered_groups},
            },
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Artifact Index (aggregated live run)",
        "",
        "| condition | execution mode | comparable | report | run traces |",
        "| --- | --- | --- | --- | --- |",
    ]
    for summary in summaries:
        group = summary["condition_group"]
        spec = registry[group]
        lines.append(
            f"| {group} | {summary['execution_mode']} | {summary['comparable_for_e13']} "
            f"| `reports/{spec.report_slug}.md` | `runs/{spec.padded_id}/` |")
    lines.extend([
        "",
        "- Judge report: `reports/condition_comparison_report.md`; scores: "
        "`condition_judge_scores.jsonl`; manifest: `condition_judge_manifest.json`.",
        "- Provider/API usage report: `reports/provider_usage_report.md`; "
        "machine-readable summary: `reports/provider_usage_summary.json`; "
        "CSV table: `reports/provider_usage_by_condition.csv`.",
        "- Manual harness prompts + instructions: `manual_harness/`.",
        "",
    ])
    (target / "artifact_index.md").write_text("\n".join(lines), encoding="utf-8")
    (target / "README.md").write_text(_LIVE_README, encoding="utf-8")

    print(f"aggregated {len(ordered_groups)} conditions -> {target}")
    print(f"judge: {judge.comparison_report_path} (live judge: {judge.live_judge_status})")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
