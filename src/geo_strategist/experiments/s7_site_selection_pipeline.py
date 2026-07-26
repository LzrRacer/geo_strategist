"""S7 iterative end-to-end site-selection/investment-proposal pipeline.

Wires S2 (candidate generation) -> S3 (feature engineering) -> S4 (tree
search) -> S5 (proposal report) -> E14 (proposal-quality review) into a
single bounded loop:

    candidate generation -> feature engineering -> tree search
      -> proposal report -> AI-Scientist-style review
      -> revision instructions -> expanded search
      -> final ranked proposal report

Each round re-runs S4/S5/E14 over the same real candidate pool. If a round
produces revision requests and rounds remain, the next round tightens
`min_evidence_score` deterministically (never invents new data) and re-runs
search/report/review; the loop stops early once a round produces no
revision requests or the round budget is exhausted.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_strategist.experiments.site_selection_review_judge import run_e14_site_selection_proposal_judge
from geo_strategist.experiments.s2_candidate_site_generation import run_s2_candidate_site_generation
from geo_strategist.experiments.s3_site_feature_engineering import run_s3_site_feature_engineering
from geo_strategist.experiments.s4_site_tree_search import run_s4_site_tree_search
from geo_strategist.experiments.s5_site_selection_proposal_report import run_s5_site_selection_proposal_report


OUTPUT_ROOT = Path(".runs/experiments/s7_site_selection_pipeline")
MIN_EVIDENCE_SCORE_STEP = 0.1


@dataclass(frozen=True)
class S7Result:
    run_id: str
    output_dir: Path
    rounds_run: int
    final_proposal_count: int
    final_revision_request_count: int
    output_paths: dict[str, str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_s7_site_selection_pipeline(
    repo_root: str | Path = ".",
    *,
    mode: str = "public_dataset_mode",
    manual_source_path: str | Path | None = None,
    beam_width: int = 5,
    max_search_depth: int = 4,
    top_k_sites: int = 10,
    max_review_rounds: int = 2,
    min_evidence_score: float = 0.0,
    allow_scenario_assumptions: bool = True,
    require_parcel_id: bool = False,
    allow_unverified_candidates_for_draft: bool = True,
    output_root: str | Path | None = None,
) -> S7Result:
    repo_root = Path(repo_root).resolve()
    run_id = str(uuid.uuid4())
    out_root = Path(output_root) if output_root else repo_root / OUTPUT_ROOT
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / run_id
    generated_at = _now_iso()

    s2_result = run_s2_candidate_site_generation(
        repo_root, mode=mode, manual_source_path=manual_source_path, output_root=out_dir / "s2",
    )
    if not allow_unverified_candidates_for_draft:
        # Drop candidates that have neither a source-traceable address nor a
        # matched financial anchor; they cannot support any recommendation
        # tier in this environment, so excluding them keeps the draft pool
        # focused on candidates that can plausibly advance.
        candidates_path = Path(s2_result.output_paths["candidate_site_records"])
        rows = [json.loads(line) for line in candidates_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        filtered = [row for row in rows if row.get("address") or row.get("anchor_master_id")]
        candidates_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in filtered) + ("\n" if filtered else ""),
            encoding="utf-8",
        )

    s3_result = run_s3_site_feature_engineering(repo_root, s2_run_dir=s2_result.output_dir, output_root=out_dir / "s3")

    rounds: list[dict[str, Any]] = []
    current_min_evidence_score = min_evidence_score
    s4_result = None
    s5_result = None
    e14_result = None
    for round_index in range(1, max(1, max_review_rounds) + 1):
        s4_result = run_s4_site_tree_search(
            repo_root,
            s2_run_dir=s2_result.output_dir,
            s3_run_dir=s3_result.output_dir,
            beam_width=beam_width,
            max_depth=max_search_depth,
            top_k_sites=top_k_sites,
            min_evidence_score=current_min_evidence_score,
            output_root=out_dir / f"s4_round_{round_index}",
        )
        s5_result = run_s5_site_selection_proposal_report(
            repo_root,
            s4_run_dir=s4_result.output_dir,
            top_k_sites=top_k_sites,
            allow_scenario_assumptions=allow_scenario_assumptions,
            require_parcel_id=require_parcel_id,
            output_root=out_dir / f"s5_round_{round_index}",
        )
        e14_result = run_e14_site_selection_proposal_judge(
            repo_root,
            s5_run_dir=s5_result.output_dir,
            output_root=out_dir / f"e14_round_{round_index}",
        )
        e14_report = _read_json(Path(e14_result.output_paths["report_json"]))
        revision_count = e14_report.get("revision_request_count", 0)
        rounds.append({
            "round_index": round_index,
            "min_evidence_score_used": current_min_evidence_score,
            "s4_run_dir": str(s4_result.output_dir.relative_to(repo_root)),
            "s5_run_dir": str(s5_result.output_dir.relative_to(repo_root)),
            "e14_run_dir": str(e14_result.output_dir.relative_to(repo_root)),
            "proposal_count": s5_result.proposal_count,
            "revision_request_count": revision_count,
        })
        if revision_count == 0:
            break
        current_min_evidence_score = min(1.0, current_min_evidence_score + MIN_EVIDENCE_SCORE_STEP)

    output_paths = {
        "manifest": str(out_dir / "s7_manifest.json"),
        "report_json": str(out_dir / "s7_report.json"),
        "report_markdown": str(out_dir / "s7_report.md"),
    }
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "mode": mode,
        "beam_width": beam_width,
        "max_search_depth": max_search_depth,
        "top_k_sites": top_k_sites,
        "max_review_rounds": max_review_rounds,
        "allow_scenario_assumptions": allow_scenario_assumptions,
        "require_parcel_id": require_parcel_id,
        "allow_unverified_candidates_for_draft": allow_unverified_candidates_for_draft,
        "rounds": rounds,
        "rounds_run": len(rounds),
        "final_s2_run_dir": str(s2_result.output_dir.relative_to(repo_root)),
        "final_s3_run_dir": str(s3_result.output_dir.relative_to(repo_root)),
        "final_s4_run_dir": rounds[-1]["s4_run_dir"] if rounds else None,
        "final_s5_run_dir": rounds[-1]["s5_run_dir"] if rounds else None,
        "final_e14_run_dir": rounds[-1]["e14_run_dir"] if rounds else None,
        "final_proposal_count": rounds[-1]["proposal_count"] if rounds else 0,
        "final_revision_request_count": rounds[-1]["revision_request_count"] if rounds else 0,
    }
    manifest = {
        "run_id": run_id,
        "stage": "s7_site_selection_pipeline",
        "output_artifacts": {key: str(Path(path).relative_to(repo_root)) for key, path in output_paths.items()},
    }

    _write_json(Path(output_paths["manifest"]), manifest)
    _write_json(Path(output_paths["report_json"]), report)
    Path(output_paths["report_markdown"]).write_text(
        "\n".join([
            "# S7 Iterative Site-Selection Pipeline",
            "",
            f"Run ID: `{run_id}`",
            f"Rounds run: {len(rounds)} (of max {max_review_rounds})",
            f"Final proposal count: {report['final_proposal_count']}",
            f"Final revision requests remaining: {report['final_revision_request_count']}",
            "",
            "This is a research decision-support pipeline. Final proposals are not a",
            "certified investment recommendation; human expert due diligence is required.",
            "",
        ]),
        encoding="utf-8",
    )

    return S7Result(
        run_id=run_id,
        output_dir=out_dir,
        rounds_run=len(rounds),
        final_proposal_count=report["final_proposal_count"],
        final_revision_request_count=report["final_revision_request_count"],
        output_paths=output_paths,
    )
