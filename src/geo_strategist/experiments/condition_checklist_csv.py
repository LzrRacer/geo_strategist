"""CSV exports for the checklist-based condition-comparison judge.

Pure functions over the exact dict shapes persisted to
``condition_deterministic_checklist.jsonl`` / ``condition_llm_checklist.jsonl``
/ ``condition_judge_scores.jsonl`` by ``condition_comparison_judge.py``, so
these CSVs are fully reproducible offline (no LLM/CLI calls) — see
``rebuild_checklist_csvs``.

The primary score export retains the compatibility filename
``condition_agentic_reasoning_scores.csv`` (five category scores 0-5 plus
``llm_points_total`` 0-25 — the report-visible decision-analysis-quality
rubric conditions are ranked on). The deterministic
checklist is reported separately, in its own matrix CSV and as auxiliary
columns on ``condition_checklist_totals.csv`` — never combined into a single
0-50 "total"; that combined figure was retired precisely because it could be
mistaken for the primary score.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from geo_strategist.experiments.checklist_judge_panel import CATEGORY_SCORE_FIELDS

_CATEGORY_SCORE_FIELDS: tuple[str, ...] = tuple(CATEGORY_SCORE_FIELDS.values())


def write_agentic_reasoning_scores_csv(
    path: Path,
    scored_rows: list[dict[str, Any]],
    *,
    llm_max: int,
) -> Path:
    """The primary qualitative score export: one row per comparable
    condition with its five category scores (0-5 each) and
    ``llm_points_total`` (0-``llm_max``). A condition whose report was
    unavailable to the judge gets blank score fields — never an invented 0."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "condition", *_CATEGORY_SCORE_FIELDS, "llm_points_total", "llm_max",
            "process_score", "decision_support_score",
        ])
        for row in scored_rows:
            if not row.get("comparable"):
                continue
            category_scores = row.get("category_scores") or {}
            writer.writerow([
                row["condition_group"],
                *[category_scores.get(field, "") for field in _CATEGORY_SCORE_FIELDS],
                row.get("llm_points_total", ""), llm_max,
                row.get("process_score", ""), row.get("decision_support_score", ""),
            ])
    return path


def write_llm_checklist_matrix_csv(path: Path, llm_checklist_rows: list[dict[str, Any]]) -> Path:
    """One row per condition x question: each judge's yes/no/
    yes_invalid_evidence/unavailable answer plus yes_count/panel_size/point."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "question_id", "category", "gpt", "claude", "gemini",
                          "deepseek", "yes_count", "panel_size", "point"])
        for row in llm_checklist_rows:
            group = row["condition_group"]
            panel_size = len(row.get("judge_status") or {})
            for question in row["questions"]:
                judges = question.get("judges") or {}
                writer.writerow([
                    group, question["question_id"], question["category"],
                    judges.get("gpt", "unavailable"), judges.get("claude", "unavailable"),
                    judges.get("gemini", "unavailable"), judges.get("deepseek", "unavailable"),
                    question.get("yes_count", 0), panel_size, question.get("point", ""),
                ])
    return path


def write_deterministic_checklist_matrix_csv(
    path: Path, deterministic_checklist_rows: list[dict[str, Any]],
) -> Path:
    """One row per condition x deterministic checklist item (auxiliary)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["condition", "item_id", "category", "status", "detail"])
        for row in deterministic_checklist_rows:
            group = row["condition_group"]
            for item in row["items"]:
                writer.writerow([group, item["item_id"], item["category"], item["status"], item["detail"]])
    return path


def write_checklist_totals_csv(
    path: Path,
    scored_rows: list[dict[str, Any]],
    *,
    llm_max: int,
    deterministic_max: int,
) -> Path:
    """Backward-compatible totals export: the same five category scores and
    ``llm_points_total`` as ``condition_agentic_reasoning_scores.csv`` (the
    primary score), plus the deterministic checklist's ``deterministic_passed``
    /``deterministic_applicable``/``deterministic_max`` as clearly separate,
    auxiliary columns. There is deliberately no combined "0-50 total" column
    — that figure was retired because it could be mistaken for the primary
    ranking metric, which is ``llm_points_total`` alone."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "condition", *_CATEGORY_SCORE_FIELDS, "llm_points_total", "llm_max",
            "process_score", "decision_support_score",
            "deterministic_passed_auxiliary", "deterministic_applicable_auxiliary",
            "deterministic_max_auxiliary",
        ])
        for row in scored_rows:
            if not row.get("comparable"):
                continue
            category_scores = row.get("category_scores") or {}
            det_summary = row.get("deterministic_checklist") or {}
            writer.writerow([
                row["condition_group"],
                *[category_scores.get(field, "") for field in _CATEGORY_SCORE_FIELDS],
                row.get("llm_points_total", ""), llm_max,
                row.get("process_score", ""), row.get("decision_support_score", ""),
                row.get("deterministic_passed", ""), det_summary.get("applicable_items", ""),
                deterministic_max,
            ])
    return path


def rebuild_checklist_csvs(proposals_dir: str | Path) -> dict[str, Path]:
    """Regenerate all four checklist CSVs from already-persisted JSONL
    artifacts under ``proposals_dir`` — no LLM/CLI calls, safe to run after
    editing report/CSV code without re-running the live judge."""

    from geo_strategist.experiments.condition_utils import _read_jsonl

    proposals_path = Path(proposals_dir)
    reports_dir = proposals_path / "reports"
    deterministic_rows = _read_jsonl(proposals_path / "condition_deterministic_checklist.jsonl")
    llm_rows = _read_jsonl(proposals_path / "condition_llm_checklist.jsonl")
    scored_rows = _read_jsonl(proposals_path / "condition_judge_scores.jsonl")

    llm_max = llm_rows[0]["summary"].get("llm_max", 25) if llm_rows else 25
    deterministic_max = len(deterministic_rows[0]["items"]) if deterministic_rows else 25

    return {
        "agentic_reasoning_scores": write_agentic_reasoning_scores_csv(
            reports_dir / "condition_agentic_reasoning_scores.csv", scored_rows, llm_max=llm_max),
        "llm_checklist_matrix": write_llm_checklist_matrix_csv(
            reports_dir / "condition_llm_checklist_matrix.csv", llm_rows),
        "deterministic_checklist_matrix": write_deterministic_checklist_matrix_csv(
            reports_dir / "condition_deterministic_checklist_matrix.csv", deterministic_rows),
        "checklist_totals": write_checklist_totals_csv(
            reports_dir / "condition_checklist_totals.csv", scored_rows,
            llm_max=llm_max, deterministic_max=deterministic_max),
    }
