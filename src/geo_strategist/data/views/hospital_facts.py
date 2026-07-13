"""Build conservative hospital workbook facts from normalized records."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import AnalysisViewManifest, HospitalWorkbookFact
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.views.common import read_normalized_jsonl, write_json, write_jsonl


class HospitalFactsResult(BaseModel):
    """Result of building hospital workbook facts."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    records_written: int = 0
    warnings: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_config() -> dict:
    with Path("configs/analysis_views.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _summary(result: HospitalFactsResult) -> str:
    return (
        "# Hospital Workbook Facts Summary\n\n"
        f"- Input found: {result.input_found}\n"
        f"- Records read: {result.records_read}\n"
        f"- Facts written: {result.records_written}\n"
        f"- Warnings: {len(result.warnings)}\n"
    )


def build_hospital_facts(repo_root: str | Path = ".") -> HospitalFactsResult:
    """Build hospital workbook facts without interpreting business meaning."""

    root = Path(repo_root).resolve()
    config = _load_config()
    inputs = config["inputs"]
    outputs = config["outputs"]
    input_path = root / inputs["hospital_normalized_records"]
    normalized_records = read_normalized_jsonl(input_path)
    warnings: list[str] = []
    facts: list[HospitalWorkbookFact] = []

    for record in normalized_records:
        facts.append(
            HospitalWorkbookFact(
                fact_id=f"hospital_fact:{record.record_id}",
                source_record_ids=[record.record_id],
                source_file_path=record.source_file_path,
                source_file_hash=record.source_file_hash,
                source_sheet=record.source_sheet,
                field_name=record.normalized_field_name,
                value=record.normalized_value,
                value_type=record.value_type,
                unit=record.unit,
                source_row_number=record.source_row_number,
                source_column_number=record.source_column_number,
                original_column=record.original_column,
                original_header=record.original_header,
                provenance=record.provenance,
            )
        )

    output_files = [
        Path(outputs["hospital_facts"]),
        Path(outputs["hospital_manifest"]),
        Path(outputs["hospital_summary"]),
    ]
    manifest = AnalysisViewManifest(
        run_id=f"hospital_facts:{now_utc().isoformat()}",
        view_name="hospital_workbook_facts",
        generated_at=now_utc(),
        input_files=[Path(inputs["hospital_normalized_records"])],
        output_files=output_files,
        record_counts={
            "normalized_records_read": len(normalized_records),
            "facts_written": len(facts),
        },
        warnings=warnings,
        unresolved_mapping_count=0,
    )
    output_paths = {
        "facts": outputs["hospital_facts"],
        "manifest": outputs["hospital_manifest"],
        "summary": outputs["hospital_summary"],
    }
    result = HospitalFactsResult(
        input_found=input_path.exists(),
        records_read=len(normalized_records),
        records_written=len(facts),
        warnings=warnings,
        output_paths=output_paths,
    )
    write_jsonl(root / outputs["hospital_facts"], facts)
    write_json(root / outputs["hospital_manifest"], manifest)
    summary_path = root / outputs["hospital_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary(result), encoding="utf-8")
    return result
