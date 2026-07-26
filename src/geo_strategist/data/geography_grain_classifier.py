"""Deterministic geography-grain classification for study-area population rows."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import IssueSeverity
from geo_strategist.data.geography_grain import (
    GeographyGrain,
    GeographyGrainIssue,
    GeographyGrainRecord,
    StudyAreaPopulationBaseManifest,
)
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.population_views import normalize_label_text
from geo_strategist.data.study_area import StudyAreaPopulationRecord
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class GeographyGrainClassifierResult(BaseModel):
    """Result of geography-grain classification."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    count_rows_read: int = 0
    rate_rows_read: int = 0
    records_written: int = 0
    issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
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


def _municipality_label_from_geography_labels(labels: dict[str, str]) -> str | None:
    for key, value in labels.items():
        normalized_key = normalize_label_text(key) or ""
        if any(token in normalized_key.lower() for token in ("municipality", "市区町村", "市町村")):
            return normalize_label_text(value)
    return None


def _has_prefecture_label(labels: dict[str, str]) -> bool:
    for key, value in labels.items():
        normalized_key = normalize_label_text(key) or ""
        if any(token in normalized_key.lower() for token in ("prefecture", "都道府県")):
            return bool(normalize_label_text(value))
    return False


def classify_geography_grain_for_record(
    record: StudyAreaPopulationRecord,
) -> tuple[GeographyGrain, list[str]]:
    """Classify one target-scope population row by deterministic label evidence."""

    municipality = normalize_label_text(record.raw_municipality) or normalize_label_text(record.municipality)
    if municipality:
        return GeographyGrain.MUNICIPALITY, ["non_empty_municipality_label"]

    municipality_label = _municipality_label_from_geography_labels(record.geography_labels)
    has_prefecture_label = _has_prefecture_label(record.geography_labels)
    if (
        record.matched_target_prefecture
        and not municipality
        and municipality_label is None
        and has_prefecture_label
    ):
        return GeographyGrain.PREFECTURE_TOTAL, [
            "matched_target_prefecture_present",
            "raw_municipality_missing",
            "municipality_label_absent_or_empty",
            "prefecture_label_present",
        ]

    return GeographyGrain.UNKNOWN, ["insufficient_deterministic_geography_grain_evidence"]


def _grain_record(record: StudyAreaPopulationRecord) -> GeographyGrainRecord:
    grain, evidence = classify_geography_grain_for_record(record)
    municipality = normalize_label_text(record.raw_municipality) or normalize_label_text(record.municipality)
    return GeographyGrainRecord(
        record_id=f"geography_grain:{record.record_id}:{record.value_kind.value}",
        source_record_ids=record.source_record_ids,
        source_file_path=record.source_file_path,
        source_sheet=record.source_sheet,
        source_file_hash=record.source_file_hash,
        provenance=record.provenance,
        study_area_id=record.study_area_id,
        matched_target_prefecture=record.matched_target_prefecture or "",
        raw_prefecture=record.raw_prefecture,
        raw_municipality=record.raw_municipality,
        municipality=municipality,
        geography_labels=record.geography_labels,
        year=record.year,
        age_group=record.age_group,
        population_value=record.population_value,
        rate_value=record.rate_value,
        unit=record.unit,
        value_kind=record.value_kind,
        geography_grain=grain,
        classification_evidence=evidence,
    )


def _issue(record: GeographyGrainRecord) -> GeographyGrainIssue | None:
    if record.geography_grain is GeographyGrain.UNKNOWN:
        return GeographyGrainIssue(
            issue_id=f"geography_grain:unknown:{record.record_id}",
            severity=IssueSeverity.WARNING,
            issue_type="geography_grain_unknown",
            message="Target-scope row lacks enough deterministic evidence for municipality or prefecture-total grain.",
            study_area_id=record.study_area_id,
            matched_target_prefecture=record.matched_target_prefecture,
            raw_municipality=record.raw_municipality,
            municipality=record.municipality,
            geography_grain=record.geography_grain,
            year=record.year,
            age_group=record.age_group,
            value_kind=record.value_kind,
            source_record_ids=record.source_record_ids,
            source_file_hash=record.source_file_hash,
            recommended_action="Review source geography labels; do not infer municipality names.",
        )
    return None


def classify_geography_grain(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> GeographyGrainClassifierResult:
    """Classify target-scope population rows into deterministic geography grains."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    count_path = root / outputs["study_area_population_long"]
    rate_path = root / outputs["study_area_population_rates_long"]
    count_payloads = _iter_jsonl(count_path)
    rate_payloads = _iter_jsonl(rate_path)
    count_records = [StudyAreaPopulationRecord.model_validate(payload) for payload in count_payloads]
    rate_records = [StudyAreaPopulationRecord.model_validate(payload) for payload in rate_payloads]

    grain_records = [_grain_record(record) for record in count_records + rate_records]
    issues = [issue for record in grain_records if (issue := _issue(record)) is not None]
    source_hashes = sorted({record.source_file_hash for record in grain_records if record.source_file_hash})
    output_paths = {
        "geography_grain_records": outputs["geography_grain_records"],
        "manifest": outputs["geography_grain_manifest"],
        "summary": outputs["geography_grain_summary"],
        "issues": outputs["geography_grain_issues"],
    }
    if not grain_records:
        return GeographyGrainClassifierResult(input_found=False, output_paths=output_paths)

    grain_counts = Counter(record.geography_grain.value for record in grain_records)
    value_kind_counts = Counter(record.value_kind.value for record in grain_records)
    issue_counts = Counter(issue.issue_type for issue in issues)
    manifest = StudyAreaPopulationBaseManifest(
        run_id=f"geography_grain:{study_area.study_area_id}:{now_utc().isoformat()}",
        generated_at=now_utc(),
        study_area_id=study_area.study_area_id,
        target_prefectures=study_area.target_prefectures,
        input_files=[
            Path(outputs["study_area_population_long"]),
            Path(outputs["study_area_population_rates_long"]),
        ],
        output_files=[
            Path(outputs["geography_grain_records"]),
            Path(outputs["geography_grain_manifest"]),
            Path(outputs["geography_grain_summary"]),
            Path(outputs["geography_grain_issues"]),
        ],
        source_file_hashes=source_hashes,
        record_counts={
            "count_rows_read": len(count_records),
            "rate_rows_read": len(rate_records),
            "geography_grain_records_written": len(grain_records),
            **{f"grain_{key}": value for key, value in grain_counts.items()},
            **{f"value_kind_{key}": value for key, value in value_kind_counts.items()},
        },
        issue_counts=dict(issue_counts),
        warnings=[],
    )
    write_jsonl(root / outputs["geography_grain_records"], grain_records)
    write_jsonl(root / outputs["geography_grain_issues"], issues)
    write_json(root / outputs["geography_grain_manifest"], manifest)
    summary_path = root / outputs["geography_grain_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Geography Grain Summary",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Count rows read: {len(count_records)}",
                f"- Rate rows read: {len(rate_records)}",
                f"- Geography grain records: {len(grain_records)}",
                f"- Municipality rows: {grain_counts[GeographyGrain.MUNICIPALITY.value]}",
                f"- Prefecture-total rows: {grain_counts[GeographyGrain.PREFECTURE_TOTAL.value]}",
                f"- Unknown rows: {grain_counts[GeographyGrain.UNKNOWN.value]}",
                "- No municipality names are inferred.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return GeographyGrainClassifierResult(
        input_found=count_path.exists() or rate_path.exists(),
        count_rows_read=len(count_records),
        rate_rows_read=len(rate_records),
        records_written=len(grain_records),
        issue_count=len(issues),
        warnings=[],
        output_paths=output_paths,
    )
