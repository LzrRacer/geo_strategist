"""Rewrite condition reports from existing condition records — no live calls.

Reconstructs each condition's ``LiveConditionResult`` from
``condition_records.jsonl`` and re-runs the shared report writer, so report
layout/section changes (metric maps, selection funnel, candidate-specific
review comments) reach already-executed conditions without re-running their
live agents. The records file is rewritten afterwards because the qualitative
discussion is regenerated onto each proposal (the comparison judge scores
that structured form).

For Vanilla LLM conditions (``spec.algorithm == "vanilla_llm"``, C1-C4) the
shared post-hoc candidate-review/author-response fields
(``candidate_review_packets``, ``candidate_review_threads``,
``candidate_qualitative_assessments``, ``candidate_assessment_reviews``,
``candidate_deliberation_summary``) and any candidate-review artifact
references are explicitly cleared before the report is rewritten, even if an
older record still carries them from before this exclusion existed — the
regenerated report and record must never expose stale candidate-review
content for a Vanilla condition. Every other (non-Vanilla) condition's
candidate-review fields are preserved unchanged.

Usage:
    .venv/bin/python scripts/rewrite_condition_reports.py \
        outputs/condition_proposals/run_stage1 [more_dirs ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Artifact-name substrings that identify the shared post-hoc candidate-review
# / author-response augmentation (as opposed to any condition's own
# internally-implemented review step) — never left attached to a Vanilla
# condition's reconstructed result.
_CANDIDATE_REVIEW_ARTIFACT_NAME_MARKERS: tuple[str, ...] = (
    "candidate_review_packets",
    "candidate_review_threads",
    "candidate_review_summary",
    "candidate_review_severity",
    "candidate_review_coverage",
    "candidate_author_response_status",
    "candidate_residual_risk",
)


def _is_candidate_review_artifact(name: str) -> bool:
    return any(marker in name for marker in _CANDIDATE_REVIEW_ARTIFACT_NAME_MARKERS)


def _resolve_artifact_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def rewrite_reports(stage_dir: Path, registry, data) -> list[str]:
    from geo_strategist.experiments.condition_utils import _read_jsonl, _write_jsonl
    from geo_strategist.experiments.decision_analysis import (
        DecisionAnalysisBundle,
        bundle_from_condition_result,
        load_decision_analysis_bundle,
    )
    from geo_strategist.experiments.live_common import LiveConditionResult
    from geo_strategist.experiments.live_report import write_condition_report
    from geo_strategist.experiments.run_condition_proposals import _write_code_manifest

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
        is_vanilla = spec.algorithm == "vanilla_llm"

        source_artifacts = list(record.get("source_artifacts") or [])
        if is_vanilla:
            source_artifacts = [
                path for path in source_artifacts
                if not _is_candidate_review_artifact(Path(path).name)]
        artifacts = {Path(path).name: path for path in source_artifacts}

        if is_vanilla:
            candidate_review_packets: list = []
            candidate_qualitative_assessments: list = []
            candidate_assessment_reviews: list = []
            candidate_deliberation_summary: dict = {}
            candidate_review_threads: list = []
        else:
            candidate_review_packets = record.get("candidate_review_packets") or []
            candidate_qualitative_assessments = (
                record.get("candidate_qualitative_assessments") or [])
            candidate_assessment_reviews = record.get("candidate_assessment_reviews") or []
            candidate_deliberation_summary = record.get("candidate_deliberation_summary") or {}
            # condition_records.jsonl never persists candidate_review_threads
            # (only the packets derived from them) — recover it from the raw
            # JSONL artifact so a non-Vanilla rewrite loses no information.
            candidate_review_threads = []
            threads_artifact = artifacts.get("candidate_review_threads.jsonl")
            if threads_artifact:
                threads_path = _resolve_artifact_path(threads_artifact)
                if threads_path.exists():
                    candidate_review_threads = _read_jsonl(threads_path)

        embedded_bundle = record.get("decision_analysis_bundle")
        analysis_bundle = None
        if isinstance(embedded_bundle, dict):
            analysis_bundle = DecisionAnalysisBundle.model_validate(embedded_bundle)
        if analysis_bundle is None:
            source = next((path for path in source_artifacts
                           if Path(path).name == "decision_analysis_bundle_v1.json"), None)
            source = source or next((path for path in source_artifacts
                                     if Path(path).name == "manual_result.json"), None)
            if source:
                analysis_bundle = load_decision_analysis_bundle(
                    _resolve_artifact_path(source), repo_root=REPO_ROOT,
                    condition_id=spec.condition_id)

        ranked_rows = record.get("ranked_rows") or [
            {"rank": row.get("rank"), "candidate_id": row.get("candidate_id"),
             "municipality": row.get("municipality"), "prefecture": row.get("prefecture"),
             "action_type": row.get("action_type"), "composite_score": row.get("composite_score")}
            for row in record.get("proposals") or []
        ]
        if analysis_bundle is None:
            analysis_bundle = bundle_from_condition_result(
                condition_group=group, condition_id=spec.condition_id,
                execution_mode=str(record.get("execution_mode") or "unknown"),
                comparable=bool(record.get("comparable_for_e13")),
                ranked_candidates=ranked_rows,
                branch_results=record.get("branch_results") or [],
                review_rows=record.get("review_rows") or [],
                narrative_sections=record.get("narrative_sections") or {},
                artifacts=artifacts, branch_search=spec.branch_search,
                candidate_universe_size=len(data.candidates),
                raw_search_nodes=record.get("search_nodes") or [],
                robustness_results=record.get("robustness_results") or [],
                compatibility_migration=True,
            )
        normalized_path = stage_dir / "runs" / spec.padded_id / "decision_analysis_bundle_v1.json"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized_path.write_text(
            analysis_bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
        artifacts["decision_analysis_bundle_v1.json"] = str(normalized_path)

        result = LiveConditionResult(
            spec=spec,
            execution_mode=str(record.get("execution_mode") or "unknown"),
            comparable_for_e13=bool(record.get("comparable_for_e13")),
            exclusion_reason=record.get("exclusion_reason"),
            proposals=record.get("proposals") or [],
            ranked_rows=ranked_rows,
            branch_results=analysis_bundle.branch_results,
            search_nodes=[row.model_dump(mode="json") for row in analysis_bundle.search_nodes],
            generated_code_stats=record.get("generated_code_stats") or {},
            model_call_summary=record.get("model_call_summary") or {},
            due_diligence=record.get("required_due_diligence") or [],
            narrative_sections=record.get("narrative_sections") or {},
            artifacts=artifacts,
            failure_notes=record.get("failure_notes") or [],
            steps_run=int(record.get("steps_run") or 0),
            robustness_results=[row.model_dump(mode="json")
                                for row in analysis_bundle.robustness_analysis],
            analysis_bundle=analysis_bundle,
            candidate_review_packets=candidate_review_packets,
            candidate_review_threads=candidate_review_threads,
            candidate_qualitative_assessments=candidate_qualitative_assessments,
            candidate_assessment_reviews=candidate_assessment_reviews,
            candidate_deliberation_summary=candidate_deliberation_summary,
        )
        # A report-only migration must repair the operational manifest too.
        # Otherwise a legacy manifest can continue to claim branch search ran
        # while retaining the pre-normalization empty branch result list.
        _write_code_manifest(
            spec, result, stage_dir / "runs" / spec.padded_id, REPO_ROOT)
        # write_condition_report -> build_qualitative_site_discussions
        # overwrites proposal["qualitative_discussion"] in place for every
        # proposal, so any candidate-review text embedded by a previous
        # rewrite is regenerated fresh here (with include_candidate_review
        # gated on spec.algorithm inside write_condition_report itself).
        report_path = write_condition_report(spec, result, stage_dir, data=data)
        record["proposal_report_path"] = str(report_path)
        record["proposals"] = result.proposals
        record["source_artifacts"] = sorted(artifacts.values())
        record["branch_results"] = analysis_bundle.branch_results
        record["search_nodes"] = [row.model_dump(mode="json") for row in analysis_bundle.search_nodes]
        record["decision_analysis_bundle"] = analysis_bundle.model_dump(mode="json")
        if is_vanilla:
            record["candidate_review_packets"] = []
            record["candidate_qualitative_assessments"] = []
            record["candidate_assessment_reviews"] = []
            record["candidate_deliberation_summary"] = {}
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
