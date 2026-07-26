"""Summary report for pre-demand study-area population-base artifacts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.geography_grain import (
    GeographyGrain,
    PopulationBaseRole,
    StudyAreaPopulationBaseRecord,
)
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json


class PopulationBaseReportResult(BaseModel):
    """Result of population-base report generation."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    candidate_model_input_rows: int = 0
    context_prefecture_total_rows: int = 0
    unknown_geography_grain_rows: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def build_population_base_report(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> PopulationBaseReportResult:
    """Write JSON and Markdown summaries for pre-demand population-base rows."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    input_path = root / outputs["population_base"]
    payloads = _iter_jsonl(input_path)
    records = [StudyAreaPopulationBaseRecord.model_validate(payload) for payload in payloads]
    output_paths = {
        "json": outputs["population_base_report_json"],
        "markdown": outputs["population_base_report_markdown"],
    }
    if not records:
        return PopulationBaseReportResult(input_found=False, output_paths=output_paths)

    rows_by_prefecture = Counter(record.matched_target_prefecture for record in records)
    rows_by_grain = Counter(record.geography_grain.value for record in records)
    rows_by_value_kind = Counter(record.value_kind.value for record in records)
    rows_by_role = Counter(record.population_base_role.value for record in records)
    missing_reclassified = sum(
        1
        for record in records
        if record.geography_grain is GeographyGrain.PREFECTURE_TOTAL
        and not record.raw_municipality
    )
    candidate_rows = rows_by_role[PopulationBaseRole.MODEL_INPUT_CANDIDATE.value]
    context_rows = rows_by_role[PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL.value]
    unknown_rows = rows_by_grain[GeographyGrain.UNKNOWN.value]
    report = {
        "study_area_id": study_area.study_area_id,
        "target_prefectures": study_area.target_prefectures,
        "rows_total": len(records),
        "rows_by_target_prefecture": dict(rows_by_prefecture),
        "rows_by_geography_grain": dict(rows_by_grain),
        "rows_by_value_kind": dict(rows_by_value_kind),
        "rows_by_population_base_role": dict(rows_by_role),
        "missing_municipality_rows_reclassified_as_prefecture_total": missing_reclassified,
        "unknown_geography_grain_rows": unknown_rows,
        "candidate_model_input_row_count": candidate_rows,
        "context_prefecture_total_row_count": context_rows,
    }
    write_json(root / outputs["population_base_report_json"], report)
    markdown_path = root / outputs["population_base_report_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# Population Base Report",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Total rows: {len(records)}",
                f"- Candidate model-input rows: {candidate_rows}",
                f"- Context prefecture-total rows: {context_rows}",
                f"- Unknown geography-grain rows: {unknown_rows}",
                f"- Missing municipality rows reclassified as prefecture_total: {missing_reclassified}",
                "- This report summarizes pre-demand rows only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return PopulationBaseReportResult(
        input_found=input_path.exists(),
        records_read=len(records),
        candidate_model_input_rows=candidate_rows,
        context_prefecture_total_rows=context_rows,
        unknown_geography_grain_rows=unknown_rows,
        output_paths=output_paths,
    )
