"""Build a long-form population rate view from normalized records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import AnalysisViewManifest, IssueSeverity, SourceQualityIssue
from geo_strategist.data.normalization import NormalizedRecord, ValueType, now_utc
from geo_strategist.data.population_views import (
    PopulationRateRecord,
    PopulationValueKind,
    normalize_label_text,
)
from geo_strategist.data.provenance import ProvenanceRecord
from geo_strategist.data.views.common import first_label, write_json, write_jsonl


class PopulationRatesResult(BaseModel):
    """Result of building population rates long view."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    records_written: int = 0
    quality_issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_config() -> dict[str, Any]:
    with Path("configs/analysis_views.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _iter_payloads(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not path.exists():
        return payloads
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payloads.append(json.loads(line))
    return payloads


def _quality_issue(
    payload: dict[str, Any],
    issue_type: str,
    message: str,
    recommended_action: str,
    severity: IssueSeverity = IssueSeverity.WARNING,
) -> SourceQualityIssue:
    record_id = str(payload.get("record_id") or "unknown")
    source_file = payload.get("source_file_path")
    return SourceQualityIssue(
        issue_id=f"population_rates:{issue_type}:{record_id}",
        severity=severity,
        source_file=Path(source_file) if source_file else None,
        sheet=payload.get("source_sheet"),
        issue_type=issue_type,
        message=message,
        recommended_action=recommended_action,
    )


def _detect_rate_unit(payload: dict[str, Any]) -> str | None:
    unit = payload.get("unit")
    if isinstance(unit, str) and unit.strip():
        return unit.strip()
    header = " ".join(
        str(part)
        for part in (
            payload.get("original_header"),
            payload.get("normalized_field_name"),
        )
        if part
    )
    if any(marker in header for marker in ("％", "%", "割合", "rate", "Rate")):
        return "percent"
    return None


def _payload_to_rate_record(
    payload: dict[str, Any],
) -> tuple[PopulationRateRecord | None, list[SourceQualityIssue]]:
    issues: list[SourceQualityIssue] = []
    if payload.get("value_role") != "population_rate":
        return None, issues

    record_id = str(payload.get("record_id") or "").strip()
    if not record_id:
        issues.append(
            _quality_issue(
                payload,
                "missing_source_record_id",
                "Population rate record lacks a source record ID.",
                "Regenerate normalized records with source record IDs.",
                IssueSeverity.ERROR,
            )
        )

    source_file_hash = str(payload.get("source_file_hash") or "").strip()
    if not source_file_hash:
        issues.append(
            _quality_issue(
                payload,
                "missing_source_hash",
                "Population rate record lacks a source file hash.",
                "Regenerate normalized records with source file hashes.",
                IssueSeverity.ERROR,
            )
        )

    provenance = payload.get("provenance") or []
    if not provenance:
        issues.append(
            _quality_issue(
                payload,
                "missing_provenance",
                "Population rate record lacks provenance.",
                "Regenerate normalized records with provenance.",
                IssueSeverity.ERROR,
            )
        )

    geography_labels = payload.get("geography_labels") or {}
    time_labels = payload.get("time_labels") or {}
    age_labels = payload.get("age_labels") or {}

    prefecture = first_label(geography_labels, ("prefecture", "都道府県"))
    municipality = first_label(geography_labels, ("municipality", "市区町村", "市町村"))
    year = first_label(time_labels, ("year", "年度", "年次", "調査年", "基準年"))
    age_group = first_label(age_labels, ("age", "年齢", "歳", "年齢階級"))

    if not (prefecture or municipality):
        issues.append(
            _quality_issue(
                payload,
                "missing_geography",
                "Population rate record lacks geography labels.",
                "Confirm geography labels before using this row for analysis.",
            )
        )
    if not year:
        issues.append(
            _quality_issue(
                payload,
                "missing_time_label",
                "Population rate record lacks time labels.",
                "Confirm time labels before using this row for analysis.",
            )
        )

    raw_value = payload.get("normalized_value")
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        issues.append(
            _quality_issue(
                payload,
                "non_numeric_value",
                "Population rate value is not numeric.",
                "Use only numeric normalized population rate records.",
                IssueSeverity.ERROR,
            )
        )

    unit = _detect_rate_unit(payload)
    if not unit:
        issues.append(
            _quality_issue(
                payload,
                "invalid_rate_unit",
                "Population rate record lacks a detectable percent-like unit.",
                "Preserve percent/rate semantics from the source headers.",
            )
        )
    elif unit not in {"percent", "%", "％", "rate"}:
        issues.append(
            _quality_issue(
                payload,
                "invalid_rate_unit",
                f"Population rate record uses unsupported unit {unit!r}.",
                "Use percent-like units only for population rate rows.",
            )
        )

    if issues:
        return None, issues

    year_value: int | str | None = year
    if isinstance(year, str) and year.isdigit():
        year_value = int(year)

    return (
        PopulationRateRecord(
            record_id=record_id,
            source_record_ids=[record_id],
            source_file_path=Path(str(payload.get("source_file_path"))) if payload.get("source_file_path") else None,
            source_sheet=str(payload.get("source_sheet") or ""),
            geography_labels={str(k): str(v) for k, v in geography_labels.items()},
            time_labels={str(k): str(v) for k, v in time_labels.items()},
            age_labels={str(k): str(v) for k, v in age_labels.items()},
            source_file_hash=source_file_hash,
            prefecture=normalize_label_text(prefecture),
            municipality=normalize_label_text(municipality),
            age_group=normalize_label_text(age_group),
            year=year_value,
            rate_value=float(raw_value),
            unit=unit,
            provenance=[ProvenanceRecord.model_validate(item) for item in provenance],
            value_kind=PopulationValueKind.RATE,
        ),
        issues,
    )


def _summary(result: PopulationRatesResult) -> str:
    return (
        "# Population Rates Long View Summary\n\n"
        f"- Input found: {result.input_found}\n"
        f"- Records read: {result.records_read}\n"
        f"- Rate records written: {result.records_written}\n"
        f"- Quality issues: {result.quality_issue_count}\n"
        f"- Warnings: {len(result.warnings)}\n"
    )


def build_population_rates_long(repo_root: str | Path = ".") -> PopulationRatesResult:
    """Build a long-form population rate view without converting to counts."""

    root = Path(repo_root).resolve()
    config = _load_config()
    inputs = config["inputs"]
    outputs = config["outputs"]
    input_path = root / inputs["population_normalized_records"]
    payloads = _iter_payloads(input_path)
    warnings: list[str] = []
    rows: list[PopulationRateRecord] = []
    issues: list[SourceQualityIssue] = []

    for payload in payloads:
        row, row_issues = _payload_to_rate_record(payload)
        issues.extend(row_issues)
        if row is not None:
            rows.append(row)

    output_files = [
        Path(outputs["population_rates_long"]),
        Path(outputs["population_rates_manifest"]),
        Path(outputs["population_rates_summary"]),
        Path(outputs["population_rates_quality_issues"]),
    ]
    manifest = AnalysisViewManifest(
        run_id=f"population_rates:{now_utc().isoformat()}",
        view_name="population_rates_long",
        generated_at=now_utc(),
        input_files=[Path(inputs["population_normalized_records"])],
        output_files=output_files,
        record_counts={
            "normalized_records_read": len(payloads),
            "population_rate_records_written": len(rows),
            "quality_issues": len(issues),
        },
        warnings=warnings,
        unresolved_mapping_count=0,
    )
    output_paths = {
        "population_rates_long": outputs["population_rates_long"],
        "manifest": outputs["population_rates_manifest"],
        "summary": outputs["population_rates_summary"],
        "quality_issues": outputs["population_rates_quality_issues"],
    }
    result = PopulationRatesResult(
        input_found=input_path.exists(),
        records_read=len(payloads),
        records_written=len(rows),
        quality_issue_count=len(issues),
        warnings=warnings,
        output_paths=output_paths,
    )
    write_jsonl(root / outputs["population_rates_long"], rows)
    write_jsonl(root / outputs["population_rates_quality_issues"], issues)
    write_json(root / outputs["population_rates_manifest"], manifest)
    summary_path = root / outputs["population_rates_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_summary(result), encoding="utf-8")
    return result
