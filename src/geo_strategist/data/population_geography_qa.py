"""Deterministic QA for population geography keys and units."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.normalization import now_utc
from geo_strategist.data.population_views import (
    PopulationGeographyIssue,
    PopulationGeographyKey,
    PopulationQAManifest,
    PopulationQASeverity,
    PopulationValueKind,
    normalize_label_text,
)
from geo_strategist.data.provenance import ProvenanceRecord
from geo_strategist.data.views.common import write_json, write_jsonl


class PopulationGeographyQAResult(BaseModel):
    """Result of population geography QA."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    rate_records_read: int = 0
    keys_written: int = 0
    issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def _load_config() -> dict[str, Any]:
    with Path("configs/population_qa.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _iter_payloads(path: Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    if not path.exists():
        return payloads
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payloads.append(json.loads(line))
    return payloads


def _first_label(labels: dict[str, Any] | None, candidates: tuple[str, ...]) -> str | None:
    if not labels:
        return None
    for key, value in labels.items():
        if value is None:
            continue
        normalized_key = normalize_label_text(str(key)) or ""
        if any(candidate.lower() in normalized_key.lower() for candidate in candidates):
            normalized_value = normalize_label_text(str(value))
            if normalized_value:
                return normalized_value
    return None


def _normalize_geography_row(payload: dict[str, Any]) -> tuple[PopulationGeographyKey | None, list[PopulationGeographyIssue]]:
    issues: list[PopulationGeographyIssue] = []
    source_record_id = str(payload.get("record_id") or "").strip()
    source_record_ids = [source_record_id] if source_record_id else []
    source_file_hash = str(payload.get("source_file_hash") or "").strip()
    provenance_payload = payload.get("provenance") or []

    raw_labels = payload.get("geography_labels") or {}
    raw_prefecture = _first_label(raw_labels, ("prefecture", "都道府県"))
    raw_municipality = _first_label(raw_labels, ("municipality", "市区町村", "市町村"))
    normalized_prefecture = normalize_label_text(payload.get("prefecture"))
    normalized_municipality = normalize_label_text(payload.get("municipality"))
    year = payload.get("year")
    age_group = normalize_label_text(payload.get("age_group"))

    if not source_record_ids:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_source_record_id:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.ERROR,
                issue_type="missing_source_record_id",
                message="Population view row lacks a source record ID.",
                source_record_ids=[],
                source_file_hash=source_file_hash or None,
                recommended_action="Regenerate the view with source record IDs.",
            )
        )
    if not source_file_hash:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_source_hash:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.ERROR,
                issue_type="missing_source_hash",
                message="Population view row lacks a source file hash.",
                source_record_ids=source_record_ids,
                source_file_hash=None,
                recommended_action="Regenerate the view with source file hashes.",
            )
        )
    if not provenance_payload:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_provenance:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.ERROR,
                issue_type="missing_provenance",
                message="Population view row lacks provenance.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Regenerate the view with provenance.",
            )
        )

    if not (raw_prefecture or raw_municipality):
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_geography:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.WARNING,
                issue_type="missing_geography",
                message="Population row lacks geography labels.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Confirm prefecture or municipality labels before analysis.",
            )
        )
    if normalized_prefecture is None:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_prefecture:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.WARNING,
                issue_type="missing_prefecture",
                message="Population row lacks a prefecture label.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Confirm prefecture labels before analysis.",
            )
        )
    if normalized_municipality is None:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_municipality:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.WARNING,
                issue_type="missing_municipality",
                message="Population row lacks a municipality label.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Confirm municipality labels before analysis.",
            )
        )
    if year in (None, ""):
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_year:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.WARNING,
                issue_type="missing_year",
                message="Population row lacks a year label.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Confirm year labels before analysis.",
            )
        )
    if age_group is None:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:missing_age_label:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.WARNING,
                issue_type="missing_age_label",
                message="Population row lacks an age label.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Confirm age labels before analysis.",
            )
        )

    value_kind_raw = str(payload.get("value_kind") or "").strip()
    if value_kind_raw == PopulationValueKind.COUNT.value:
        value_kind = PopulationValueKind.COUNT
    elif value_kind_raw == PopulationValueKind.RATE.value:
        value_kind = PopulationValueKind.RATE
    else:
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:mixed_count_rate_semantics:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.ERROR,
                issue_type="mixed_count_rate_semantics",
                message="Population row does not state explicit count or rate semantics.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Regenerate the view with an explicit value_kind field.",
            )
        )
        value_kind = PopulationValueKind.COUNT

    value_field = "population_value" if value_kind is PopulationValueKind.COUNT else "rate_value"
    raw_value = payload.get(value_field)
    if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:non_numeric_value:{source_record_id or 'unknown'}",
                severity=PopulationQASeverity.ERROR,
                issue_type="non_numeric_value",
                message="Population row has a non-numeric value.",
                source_record_ids=source_record_ids,
                source_file_hash=source_file_hash or None,
                recommended_action="Preserve numeric source values only.",
            )
        )

    unit = normalize_label_text(payload.get("unit"))
    if value_kind is PopulationValueKind.RATE:
        if unit not in {"percent", "%", "％", "rate"}:
            issues.append(
                PopulationGeographyIssue(
                    issue_id=f"population_geo:invalid_rate_unit:{source_record_id or 'unknown'}",
                    severity=PopulationQASeverity.WARNING,
                    issue_type="invalid_rate_unit",
                    message="Population rate row lacks a percent-like unit.",
                    source_record_ids=source_record_ids,
                    source_file_hash=source_file_hash or None,
                    recommended_action="Preserve rate units as percent-like values.",
                )
            )
    elif unit is not None:
        allowed_count_units = {"people", "person", "persons", "人", "count", "counts"}
        if unit in {"percent", "%", "％", "rate"} or unit not in allowed_count_units:
            issues.append(
                PopulationGeographyIssue(
                    issue_id=f"population_geo:invalid_count_unit:{source_record_id or 'unknown'}",
                    severity=PopulationQASeverity.WARNING,
                    issue_type="invalid_count_unit",
                    message="Population count row uses an unsupported unit.",
                    source_record_ids=source_record_ids,
                    source_file_hash=source_file_hash or None,
                    recommended_action="Keep count rows on count-like units only.",
                )
            )

    if issues and not source_record_ids:
        return None, issues

    if not (raw_prefecture or raw_municipality):
        return None, issues

    if not source_file_hash or not provenance_payload:
        return None, issues

    provenance = [ProvenanceRecord.model_validate(item) for item in provenance_payload]
    key = PopulationGeographyKey(
        key_id=f"population_key:{source_record_id}:{value_kind.value}",
        source_record_ids=source_record_ids,
        source_file_path=Path(str(payload.get("source_file_path"))) if payload.get("source_file_path") else None,
        source_file_hash=source_file_hash,
        raw_prefecture_label=raw_prefecture,
        raw_municipality_label=raw_municipality,
        normalized_prefecture_label=normalized_prefecture,
        normalized_municipality_label=normalized_municipality,
        year=year,
        age_group=age_group,
        value_kind=value_kind,
        provenance=provenance,
    )
    return key, issues


def _duplicate_key(counter_key: tuple[Any, ...]) -> str:
    prefecture, municipality, year, age_group, value_kind = counter_key
    return (
        f"{prefecture or '∅'}|{municipality or '∅'}|{year or '∅'}|"
        f"{age_group or '∅'}|{value_kind}"
    )


def build_population_geography_qa(repo_root: str | Path = ".") -> PopulationGeographyQAResult:
    """Build deterministic QA summaries for population geography keys."""

    root = Path(repo_root).resolve()
    config = _load_config()
    inputs = config["inputs"]
    outputs = config["outputs"]
    long_path = root / inputs["population_long"]
    rate_path = root / inputs["population_rates_long"]
    long_payloads = _iter_payloads(long_path)
    rate_payloads = _iter_payloads(rate_path)
    warnings: list[str] = []
    keys: list[PopulationGeographyKey] = []
    issues: list[PopulationGeographyIssue] = []
    geography_key_counts: Counter[tuple[Any, ...]] = Counter()

    for payload in long_payloads + rate_payloads:
        key, row_issues = _normalize_geography_row(payload)
        issues.extend(row_issues)
        if key is not None:
            keys.append(key)
            geography_key_counts[
                (
                    key.normalized_prefecture_label,
                    key.normalized_municipality_label,
                    key.year,
                    key.age_group,
                    key.value_kind.value,
                )
            ] += 1

    for counter_key, count in geography_key_counts.items():
        if count <= 1:
            continue
        prefecture, municipality, year, age_group, value_kind = counter_key
        issues.append(
            PopulationGeographyIssue(
                issue_id=f"population_geo:duplicate:{_duplicate_key(counter_key)}",
                severity=PopulationQASeverity.WARNING,
                issue_type="duplicate_population_key",
                message=(
                    "Duplicate population geography key observed "
                    f"for {count} records."
                ),
                source_record_ids=[],
                source_file_hash=None,
                recommended_action="Deduplicate the normalized view rows before using them in analysis.",
            )
        )

    issue_counts = Counter(issue.issue_type for issue in issues)
    record_counts = {
        "population_long_records_read": len(long_payloads),
        "population_rate_records_read": len(rate_payloads),
        "geography_keys_written": len(keys),
        "issues_written": len(issues),
    }
    output_files = [
        Path(outputs["population_geography_keys"]),
        Path(outputs["population_qa_manifest"]),
        Path(outputs["population_qa_summary"]),
        Path(outputs["population_geography_issues"]),
    ]
    manifest = PopulationQAManifest(
        run_id=f"population_geography_qa:{now_utc().isoformat()}",
        generated_at=now_utc(),
        input_files=[Path(inputs["population_long"]), Path(inputs["population_rates_long"])],
        output_files=output_files,
        record_counts=record_counts,
        issue_counts=dict(issue_counts),
        warnings=warnings,
    )
    output_paths = {
        "population_geography_keys": outputs["population_geography_keys"],
        "manifest": outputs["population_qa_manifest"],
        "summary": outputs["population_qa_summary"],
        "issues": outputs["population_geography_issues"],
    }
    result = PopulationGeographyQAResult(
        input_found=long_path.exists() or rate_path.exists(),
        records_read=len(long_payloads),
        rate_records_read=len(rate_payloads),
        keys_written=len(keys),
        issue_count=len(issues),
        warnings=warnings,
        output_paths=output_paths,
    )
    write_jsonl(root / outputs["population_geography_keys"], keys)
    write_jsonl(root / outputs["population_geography_issues"], issues)
    write_json(root / outputs["population_qa_manifest"], manifest)
    summary_path = root / outputs["population_qa_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Population Geography QA Summary",
                "",
                f"- Long records read: {len(long_payloads)}",
                f"- Rate records read: {len(rate_payloads)}",
                f"- Geography keys written: {len(keys)}",
                f"- Issues written: {len(issues)}",
                f"- Duplicate key groups: {sum(1 for count in geography_key_counts.values() if count > 1)}",
                f"- Warnings: {len(warnings)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result
