"""Target-scoped geography QA for the configured study area."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import IssueSeverity
from geo_strategist.data.population_views import PopulationValueKind, normalize_label_text
from geo_strategist.data.study_area import (
    StudyAreaGeographyIssue,
    StudyAreaPopulationRecord,
    StudyAreaScopeStatus,
)
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class StudyAreaGeographyQAResult(BaseModel):
    """Result of target-scoped geography QA."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    count_rows_read: int = 0
    rate_rows_read: int = 0
    geography_keys_written: int = 0
    issue_count: int = 0
    duplicate_key_count: int = 0
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


def _issue(
    issue_type: str,
    message: str,
    record: StudyAreaPopulationRecord,
    severity: IssueSeverity = IssueSeverity.WARNING,
    recommended_action: str = "Review target-scope geography labels before demand modeling.",
) -> StudyAreaGeographyIssue:
    return StudyAreaGeographyIssue(
        issue_id=(
            f"study_area_geo:{issue_type}:{record.record_id}:"
            f"{record.value_kind.value}"
        ),
        severity=severity,
        issue_type=issue_type,
        message=message,
        study_area_id=record.study_area_id,
        scope_status=record.scope_status,
        target_prefecture=record.matched_target_prefecture,
        raw_prefecture=record.raw_prefecture,
        raw_municipality=record.raw_municipality,
        year=record.year,
        age_group=record.age_group,
        value_kind=record.value_kind,
        source_record_ids=record.source_record_ids,
        source_file_hash=record.source_file_hash,
        recommended_action=recommended_action,
    )


def _geography_key(record: StudyAreaPopulationRecord) -> dict[str, Any]:
    return {
        "key_id": f"study_area_key:{record.study_area_id}:{record.record_id}:{record.value_kind.value}",
        "study_area_id": record.study_area_id,
        "target_prefecture": record.matched_target_prefecture,
        "raw_prefecture": record.raw_prefecture,
        "raw_municipality": record.raw_municipality,
        "normalized_municipality": normalize_label_text(record.raw_municipality),
        "year": record.year,
        "age_group": record.age_group,
        "value_kind": record.value_kind.value,
        "source_record_ids": record.source_record_ids,
        "source_file_hash": record.source_file_hash,
        "provenance": [item.model_dump(mode="json") for item in record.provenance],
    }


def _duplicate_key(record: StudyAreaPopulationRecord) -> tuple[Any, ...]:
    return (
        record.matched_target_prefecture,
        normalize_label_text(record.raw_municipality),
        record.year,
        record.age_group,
        record.value_kind.value,
    )


def _duplicate_key_label(key: tuple[Any, ...]) -> str:
    target, municipality, year, age_group, value_kind = key
    return (
        f"{target or 'missing'}|{municipality or 'missing'}|"
        f"{year or 'missing'}|{age_group or 'missing'}|{value_kind}"
    )


def _validate_target_record(record: StudyAreaPopulationRecord) -> list[StudyAreaGeographyIssue]:
    issues: list[StudyAreaGeographyIssue] = []
    if record.scope_status is not StudyAreaScopeStatus.IN_SCOPE:
        issues.append(
            _issue(
                "missing_target_prefecture",
                "Target geography QA input includes a row that is not marked in scope.",
                record,
                severity=IssueSeverity.ERROR,
                recommended_action="Regenerate study-area filtering before target geography QA.",
            )
        )
    if not record.matched_target_prefecture:
        issues.append(
            _issue(
                "missing_target_prefecture",
                "In-scope target row lacks a matched target prefecture.",
                record,
                severity=IssueSeverity.ERROR,
                recommended_action="Regenerate study-area filtering with configured aliases.",
            )
        )
    if not record.raw_municipality:
        issues.append(
            _issue(
                "missing_municipality_in_target_scope",
                "Target-scope population row lacks a municipality label.",
                record,
                recommended_action="Review source labels; do not infer or invent municipalities.",
            )
        )
    if record.year in (None, ""):
        issues.append(
            _issue(
                "missing_year",
                "Target-scope population row lacks a year label.",
                record,
            )
        )
    if not normalize_label_text(record.age_group):
        issues.append(
            _issue(
                "missing_age_label",
                "Target-scope population row lacks an age-group label.",
                record,
            )
        )
    if record.value_kind is PopulationValueKind.COUNT and record.rate_value is not None:
        issues.append(
            _issue(
                "mixed_count_rate_semantics_in_target_scope",
                "Count row contains a rate value.",
                record,
                severity=IssueSeverity.ERROR,
                recommended_action="Keep count and rate target-scope rows in separate outputs.",
            )
        )
    if record.value_kind is PopulationValueKind.RATE and record.population_value is not None:
        issues.append(
            _issue(
                "mixed_count_rate_semantics_in_target_scope",
                "Rate row contains a count value.",
                record,
                severity=IssueSeverity.ERROR,
                recommended_action="Keep count and rate target-scope rows in separate outputs.",
            )
        )
    return issues


def run_study_area_geography_qa(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> StudyAreaGeographyQAResult:
    """Run deterministic geography QA over filtered target-scope rows."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    count_path = root / outputs["study_area_population_long"]
    rate_path = root / outputs["study_area_population_rates_long"]
    count_payloads = _iter_jsonl(count_path)
    rate_payloads = _iter_jsonl(rate_path)

    count_records = [StudyAreaPopulationRecord.model_validate(payload) for payload in count_payloads]
    rate_records = [StudyAreaPopulationRecord.model_validate(payload) for payload in rate_payloads]
    records = count_records + rate_records

    issues: list[StudyAreaGeographyIssue] = []
    keys: list[dict[str, Any]] = []
    count_rows_by_target_prefecture: Counter[str] = Counter()
    rate_rows_by_target_prefecture: Counter[str] = Counter()
    missing_municipality_by_target_prefecture: Counter[str] = Counter()
    duplicate_counts: Counter[tuple[Any, ...]] = Counter()
    duplicate_record_ids: defaultdict[tuple[Any, ...], list[str]] = defaultdict(list)

    for record in records:
        issues.extend(_validate_target_record(record))
        keys.append(_geography_key(record))
        target = record.matched_target_prefecture or "missing"
        if record.value_kind is PopulationValueKind.COUNT:
            count_rows_by_target_prefecture[target] += 1
        elif record.value_kind is PopulationValueKind.RATE:
            rate_rows_by_target_prefecture[target] += 1
        if not record.raw_municipality:
            missing_municipality_by_target_prefecture[target] += 1
        key = _duplicate_key(record)
        duplicate_counts[key] += 1
        duplicate_record_ids[key].extend(record.source_record_ids)

    duplicate_key_count = 0
    for key, count in duplicate_counts.items():
        if count <= 1:
            continue
        duplicate_key_count += 1
        target, municipality, year, age_group, value_kind = key
        issues.append(
            StudyAreaGeographyIssue(
                issue_id=f"study_area_geo:duplicate:{_duplicate_key_label(key)}",
                severity=IssueSeverity.WARNING,
                issue_type="duplicate_target_geography_key",
                message=f"Duplicate target-scope geography key observed for {count} records.",
                study_area_id=study_area.study_area_id,
                scope_status=StudyAreaScopeStatus.IN_SCOPE,
                target_prefecture=target,
                raw_prefecture=target,
                raw_municipality=municipality,
                year=year,
                age_group=age_group,
                value_kind=PopulationValueKind(value_kind),
                source_record_ids=duplicate_record_ids[key],
                source_file_hash=None,
                recommended_action="Deduplicate target-scope geography keys before demand modeling.",
            )
        )

    issue_counts = Counter(issue.issue_type for issue in issues)
    summary = {
        "study_area_id": study_area.study_area_id,
        "target_prefectures": study_area.target_prefectures,
        "count_rows_by_target_prefecture": dict(count_rows_by_target_prefecture),
        "rate_rows_by_target_prefecture": dict(rate_rows_by_target_prefecture),
        "missing_municipality_count_by_target_prefecture": dict(
            missing_municipality_by_target_prefecture
        ),
        "missing_year_count": issue_counts.get("missing_year", 0),
        "missing_age_group_count": issue_counts.get("missing_age_label", 0),
        "duplicate_target_geography_key_count": duplicate_key_count,
        "count_rate_separation": {
            "count_rows_only_in_count_output": all(
                record.value_kind is PopulationValueKind.COUNT for record in count_records
            ),
            "rate_rows_only_in_rate_output": all(
                record.value_kind is PopulationValueKind.RATE for record in rate_records
            ),
            "mixed_count_rate_semantics_issue_count": issue_counts.get(
                "mixed_count_rate_semantics_in_target_scope", 0
            ),
        },
        "issue_counts": dict(issue_counts),
        "geography_keys_written": len(keys),
    }
    write_json(root / outputs["geography_qa_json"], summary)
    write_jsonl(root / outputs["geography_issues"], issues)
    write_jsonl(root / outputs["study_area_geography_keys"], keys)
    markdown_path = root / outputs["geography_qa_markdown"]
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        "\n".join(
            [
                "# Study Area Geography QA",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Count rows read: {len(count_records)}",
                f"- Rate rows read: {len(rate_records)}",
                f"- Geography keys written: {len(keys)}",
                f"- Missing municipality issues: {issue_counts.get('missing_municipality_in_target_scope', 0)}",
                f"- Missing year issues: {issue_counts.get('missing_year', 0)}",
                f"- Missing age-group issues: {issue_counts.get('missing_age_label', 0)}",
                f"- Duplicate target geography keys: {duplicate_key_count}",
                "- This QA does not geocode or infer municipality names.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths = {
        "geography_qa_json": outputs["geography_qa_json"],
        "geography_qa_markdown": outputs["geography_qa_markdown"],
        "geography_issues": outputs["geography_issues"],
        "geography_keys": outputs["study_area_geography_keys"],
    }
    return StudyAreaGeographyQAResult(
        input_found=count_path.exists() or rate_path.exists(),
        count_rows_read=len(count_records),
        rate_rows_read=len(rate_records),
        geography_keys_written=len(keys),
        issue_count=len(issues),
        duplicate_key_count=duplicate_key_count,
        warnings=[],
        output_paths=output_paths,
    )
