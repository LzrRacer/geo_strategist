"""Deterministic extraction mapping review utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import MappingStatus, NormalizationManifest, now_utc


DEFAULT_OUTPUTS = {
    "review_json": Path(".cache/review/extraction_mapping_review.json"),
    "review_markdown": Path(".cache/review/extraction_mapping_review.md"),
    "manual_candidates": Path(".cache/review/manual_mapping_candidates.yaml"),
}

REPORT_PATHS = {
    "hospital_workbook": Path(".cache/normalization/hospital_workbook_mapping_report.json"),
    "population": Path(".cache/normalization/population_mapping_report.json"),
}

MANIFEST_PATHS = {
    "hospital_workbook": Path(".cache/normalization/hospital_workbook_manifest.json"),
    "population": Path(".cache/normalization/population_manifest.json"),
}


class MappingReviewResult(BaseModel):
    """Summary of deterministic extraction mapping review."""

    model_config = ConfigDict(extra="forbid")

    source_table_count: int = 0
    normalized_record_count: int = 0
    inferred_mapping_count: int = 0
    unresolved_mapping_count: int = 0
    warning_count: int = 0
    manual_review_tables: list[dict[str, Any]] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _manual_candidate(source_name: str, table: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source_name,
        "source_table_id": table.get("table_id"),
        "source_file_path": table.get("workbook_path"),
        "sheet_name": table.get("sheet_name"),
        "dimensions": {
            "rows": table.get("row_count"),
            "columns": table.get("column_count"),
        },
        "detected_header_row": table.get("detected_header_row"),
        "detected_headers": table.get("column_names", []),
        "reason_unresolved": mapping.get("warnings") or table.get("warnings") or [
            "Mapping status is unresolved."
        ],
        "manual_decision": {
            "status": "",
            "header_row": "",
            "value_columns": [],
            "label_columns": [],
            "year_columns": [],
            "age_columns": [],
            "notes": "",
        },
    }


def _markdown(result: MappingReviewResult) -> str:
    lines = [
        "# Extraction Mapping Review",
        "",
        f"- Source tables: {result.source_table_count}",
        f"- Normalized records: {result.normalized_record_count}",
        f"- Inferred mappings: {result.inferred_mapping_count}",
        f"- Unresolved mappings: {result.unresolved_mapping_count}",
        f"- Warnings: {result.warning_count}",
        "",
        "| Source | Sheet | Rows | Columns | Reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for item in result.manual_review_tables:
        reasons = item.get("reason_unresolved") or []
        reason = "; ".join(str(value) for value in reasons)
        lines.append(
            f"| {item.get('source')} | {item.get('sheet_name')} | "
            f"{item.get('dimensions', {}).get('rows', '')} | "
            f"{item.get('dimensions', {}).get('columns', '')} | {reason} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def review_extraction_mappings(repo_root: str | Path = ".") -> MappingReviewResult:
    """Review mapping reports and write deterministic review artifacts."""

    root = Path(repo_root).resolve()
    review_payload: dict[str, Any] = {
        "generated_at": now_utc().isoformat(),
        "sources": {},
        "manual_review_tables": [],
        "summary": {},
    }
    result = MappingReviewResult()

    for source_name, report_rel in REPORT_PATHS.items():
        report = _load_json(root / report_rel)
        manifest_payload = _load_json(root / MANIFEST_PATHS[source_name])
        if report is None:
            continue
        manifest = (
            NormalizationManifest.model_validate(manifest_payload)
            if manifest_payload is not None
            else None
        )
        tables = {table["table_id"]: table for table in report.get("source_tables", [])}
        mappings = report.get("mappings", [])
        inferred = [mapping for mapping in mappings if mapping.get("status") == MappingStatus.INFERRED]
        unresolved = [
            mapping for mapping in mappings if mapping.get("status") == MappingStatus.UNRESOLVED
        ]
        warnings = manifest.warnings if manifest else report.get("summary", {}).get("warnings", [])

        result.source_table_count += int(report.get("summary", {}).get("source_table_count", 0))
        result.normalized_record_count += int(
            report.get("summary", {}).get("normalized_record_count", 0)
        )
        result.inferred_mapping_count += len(inferred)
        result.unresolved_mapping_count += len(unresolved)
        result.warning_count += len(warnings)

        manual_tables = [
            _manual_candidate(source_name, tables.get(mapping["source_table_id"], {}), mapping)
            for mapping in unresolved
        ]
        result.manual_review_tables.extend(manual_tables)
        review_payload["sources"][source_name] = {
            "source_table_count": report.get("summary", {}).get("source_table_count", 0),
            "normalized_record_count": report.get("summary", {}).get(
                "normalized_record_count", 0
            ),
            "inferred_mapping_count": len(inferred),
            "unresolved_mapping_count": len(unresolved),
            "warnings": warnings,
        }
        review_payload["manual_review_tables"].extend(manual_tables)

    review_payload["summary"] = {
        "source_table_count": result.source_table_count,
        "normalized_record_count": result.normalized_record_count,
        "inferred_mapping_count": result.inferred_mapping_count,
        "unresolved_mapping_count": result.unresolved_mapping_count,
        "warning_count": result.warning_count,
    }

    outputs = {
        label: root / relative
        for label, relative in DEFAULT_OUTPUTS.items()
    }
    _write_json(outputs["review_json"], review_payload)
    outputs["review_markdown"].parent.mkdir(parents=True, exist_ok=True)
    outputs["review_markdown"].write_text(_markdown(result), encoding="utf-8")
    outputs["manual_candidates"].parent.mkdir(parents=True, exist_ok=True)
    outputs["manual_candidates"].write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "generated_at": review_payload["generated_at"],
                "note": "Suggestion file only; do not copy into committed configs without manual review.",
                "unresolved_source_tables": result.manual_review_tables,
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    result.output_paths = {
        label: str(path.relative_to(root))
        for label, path in outputs.items()
    }
    return result
