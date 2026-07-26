"""Deterministic age-group normalization for population-base rows."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.age_groups import (
    AgeGroupAlias,
    AgeGroupKind,
    AgeGroupMatchStatus,
    AgeGroupNormalizedRecord,
    AgeGroupQAIssue,
    AgeGroupQAManifest,
    normalize_age_label,
)
from geo_strategist.data.geography_grain import StudyAreaPopulationBaseRecord
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.population_base_coverage import CoverageSeverity
from geo_strategist.data.population_views import PopulationValueKind
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class AgeGroupNormalizationResult(BaseModel):
    """Result of deterministic age-group normalization."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    records_written: int = 0
    issue_count: int = 0
    unknown_age_group_count: int = 0
    missing_age_group_count: int = 0
    duplicate_normalized_key_count: int = 0
    conflicting_value_count: int = 0
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


def load_age_group_aliases(path: str | Path = "configs/age_groups.yaml") -> list[AgeGroupAlias]:
    """Load configured age-group aliases."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        AgeGroupAlias(
            age_group_id=item["id"],
            canonical_label=item["canonical_label"],
            kind=item["kind"],
            sort_key=item["sort_key"],
            aliases=item["aliases"],
        )
        for item in payload.get("age_groups", [])
    ]


def build_age_alias_lookup(aliases: list[AgeGroupAlias]) -> dict[str, list[AgeGroupAlias]]:
    """Build normalized alias lookup, preserving ambiguity if configured."""

    lookup: dict[str, list[AgeGroupAlias]] = defaultdict(list)
    for item in aliases:
        for label in [item.canonical_label, *item.aliases]:
            key = normalize_age_label(label)
            if key:
                lookup[key].append(item)
    return dict(lookup)


def match_age_group(
    raw_age_group: str | None,
    aliases: list[AgeGroupAlias],
) -> tuple[AgeGroupMatchStatus, AgeGroupAlias | None, bool]:
    """Match a raw age-group label against configured aliases."""

    normalized = normalize_age_label(raw_age_group)
    if normalized is None:
        return AgeGroupMatchStatus.MISSING, None, False
    matches = build_age_alias_lookup(aliases).get(normalized, [])
    unique = {match.age_group_id: match for match in matches}
    if len(unique) > 1:
        return AgeGroupMatchStatus.UNKNOWN, None, True
    if not unique:
        return AgeGroupMatchStatus.UNKNOWN, None, False
    match = next(iter(unique.values()))
    raw_direct = " ".join(str(raw_age_group).split())
    status = (
        AgeGroupMatchStatus.CANONICAL
        if raw_direct == match.canonical_label
        else AgeGroupMatchStatus.ALIAS
    )
    return status, match, False


def _value(record: AgeGroupNormalizedRecord) -> float | None:
    if record.value_kind is PopulationValueKind.COUNT:
        return record.population_value
    return record.rate_value


def _normalized_key(record: AgeGroupNormalizedRecord) -> tuple[Any, ...]:
    return (
        record.matched_target_prefecture,
        record.municipality,
        record.geography_grain.value,
        record.population_base_role.value,
        str(record.year) if record.year is not None else None,
        record.canonical_age_group_id,
        record.value_kind.value,
    )


def _key_label(key: tuple[Any, ...]) -> str:
    return "|".join(str(value) if value is not None else "missing" for value in key)


def _issue(
    issue_type: str,
    severity: CoverageSeverity,
    message: str,
    record: AgeGroupNormalizedRecord,
    recommended_action: str,
    normalized_key: str | None = None,
    source_record_ids: list[str] | None = None,
    source_file_hashes: list[str] | None = None,
) -> AgeGroupQAIssue:
    return AgeGroupQAIssue(
        issue_id=f"age_group:{issue_type}:{normalized_key or record.record_id}",
        severity=severity,
        issue_type=issue_type,
        message=message,
        study_area_id=record.study_area_id,
        normalized_key=normalized_key,
        raw_age_group=record.raw_age_group,
        canonical_age_group_id=record.canonical_age_group_id,
        canonical_age_group_label=record.canonical_age_group_label,
        value_kind=record.value_kind,
        geography_grain=record.geography_grain,
        population_base_role=record.population_base_role,
        source_record_ids=source_record_ids or record.source_record_ids,
        source_file_hashes=source_file_hashes
        or ([record.source_file_hash] if record.source_file_hash else []),
        recommended_action=recommended_action,
    )


def _normalized_record(
    record: StudyAreaPopulationBaseRecord,
    aliases: list[AgeGroupAlias],
) -> tuple[AgeGroupNormalizedRecord, AgeGroupQAIssue | None]:
    status, match, ambiguous = match_age_group(record.age_group, aliases)
    issue: AgeGroupQAIssue | None = None
    normalized = AgeGroupNormalizedRecord(
        record_id=f"age_group:{record.base_record_id}",
        base_record_id=record.base_record_id,
        source_record_ids=record.source_record_ids,
        source_file_path=record.source_file_path,
        source_sheet=record.source_sheet,
        source_file_hash=record.source_file_hash,
        provenance=record.provenance,
        study_area_id=record.study_area_id,
        matched_target_prefecture=record.matched_target_prefecture,
        raw_municipality=record.raw_municipality,
        municipality=record.municipality,
        year=record.year,
        raw_age_group=record.age_group,
        canonical_age_group_id=match.age_group_id if match else None,
        canonical_age_group_label=match.canonical_label if match else None,
        age_group_match_status=status,
        age_group_kind=match.kind if match else AgeGroupKind.UNKNOWN,
        age_sort_key=match.sort_key if match else None,
        population_value=record.population_value,
        rate_value=record.rate_value,
        unit=record.unit,
        value_kind=record.value_kind,
        geography_grain=record.geography_grain,
        population_base_role=record.population_base_role,
    )
    if ambiguous:
        issue = _issue(
            "ambiguous_age_group_alias",
            CoverageSeverity.ERROR,
            "Age-group alias maps to multiple canonical groups.",
            normalized,
            "Fix committed age-group aliases before feature engineering.",
        )
    elif status is AgeGroupMatchStatus.MISSING:
        issue = _issue(
            "missing_age_group",
            CoverageSeverity.WARNING,
            "Population-base row lacks an age-group label.",
            normalized,
            "Review source labels; do not infer age groups.",
        )
    elif status is AgeGroupMatchStatus.UNKNOWN:
        issue = _issue(
            "unknown_age_group",
            CoverageSeverity.WARNING,
            "Age-group label does not match configured aliases.",
            normalized,
            "Add a committed alias only if source evidence supports it.",
        )
    elif not record.source_record_ids or not record.source_file_hash or not record.provenance:
        issue = _issue(
            "age_group_missing_source_traceability",
            CoverageSeverity.ERROR,
            "Age-normalized row lacks source traceability.",
            normalized,
            "Regenerate upstream population-base rows with source record IDs, hashes, and provenance.",
        )
    return normalized, issue


def normalize_age_groups(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
    age_group_config_path: str | Path = "configs/age_groups.yaml",
) -> AgeGroupNormalizationResult:
    """Normalize age-group labels on population-base rows."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    input_path = root / outputs["population_base"]
    aliases = load_age_group_aliases(root / age_group_config_path)
    payloads = _iter_jsonl(input_path)
    base_records = [StudyAreaPopulationBaseRecord.model_validate(payload) for payload in payloads]
    output_paths = {
        "records": outputs["population_base_age_normalized"],
        "manifest": outputs["age_group_qa_manifest"],
        "summary": outputs["age_group_qa_summary"],
        "issues": outputs["age_group_qa_issues"],
    }
    if not base_records:
        return AgeGroupNormalizationResult(input_found=False, output_paths=output_paths)

    normalized_records: list[AgeGroupNormalizedRecord] = []
    issues: list[AgeGroupQAIssue] = []
    for record in base_records:
        normalized, issue = _normalized_record(record, aliases)
        normalized_records.append(normalized)
        if issue is not None:
            issues.append(issue)

    key_to_records: defaultdict[tuple[Any, ...], list[AgeGroupNormalizedRecord]] = defaultdict(list)
    for record in normalized_records:
        if record.canonical_age_group_id:
            key_to_records[_normalized_key(record)].append(record)
    for key, rows in key_to_records.items():
        if len(rows) <= 1:
            continue
        values = {_value(row) for row in rows}
        source_ids = sorted({source_id for row in rows for source_id in row.source_record_ids})
        hashes = sorted({row.source_file_hash for row in rows if row.source_file_hash})
        label = _key_label(key)
        first = rows[0]
        if len(values) > 1:
            issues.append(
                _issue(
                    "age_group_conflicting_values",
                    CoverageSeverity.ERROR,
                    "Age-normalized rows share a normalized key but have conflicting values.",
                    first,
                    "Review source rows before demand-feature engineering; do not aggregate conflicting values.",
                    normalized_key=label,
                    source_record_ids=source_ids,
                    source_file_hashes=hashes,
                )
            )
        else:
            issues.append(
                _issue(
                    "age_group_duplicate_normalized_key",
                    CoverageSeverity.WARNING,
                    "Age-normalized rows share the same normalized key.",
                    first,
                    "Review duplicate normalized rows before demand-feature engineering.",
                    normalized_key=label,
                    source_record_ids=source_ids,
                    source_file_hashes=hashes,
                )
            )

    match_counts = Counter(record.age_group_match_status.value for record in normalized_records)
    kind_counts = Counter(record.age_group_kind.value for record in normalized_records)
    issue_counts = Counter(issue.issue_type for issue in issues)
    manifest = AgeGroupQAManifest(
        run_id=f"age_group_normalization:{study_area.study_area_id}:{now_utc().isoformat()}",
        generated_at=now_utc(),
        study_area_id=study_area.study_area_id,
        input_files=[Path(outputs["population_base"]), Path(age_group_config_path)],
        output_files=[
            Path(outputs["population_base_age_normalized"]),
            Path(outputs["age_group_qa_manifest"]),
            Path(outputs["age_group_qa_summary"]),
            Path(outputs["age_group_qa_issues"]),
        ],
        record_counts={
            "population_base_records_read": len(base_records),
            "age_normalized_records_written": len(normalized_records),
            **{f"match_status_{key}": value for key, value in match_counts.items()},
            **{f"age_group_kind_{key}": value for key, value in kind_counts.items()},
        },
        issue_counts=dict(issue_counts),
        warnings=[],
    )
    write_jsonl(root / outputs["population_base_age_normalized"], normalized_records)
    write_jsonl(root / outputs["age_group_qa_issues"], issues)
    write_json(root / outputs["age_group_qa_manifest"], manifest)
    summary_path = root / outputs["age_group_qa_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Age Group QA Summary",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Population-base rows read: {len(base_records)}",
                f"- Age-normalized records written: {len(normalized_records)}",
                f"- Missing age groups: {issue_counts.get('missing_age_group', 0)}",
                f"- Unknown age groups: {issue_counts.get('unknown_age_group', 0)}",
                f"- Duplicate normalized keys: {issue_counts.get('age_group_duplicate_normalized_key', 0)}",
                f"- Conflicting values: {issue_counts.get('age_group_conflicting_values', 0)}",
                "- This is label canonicalization only and does not calculate demand.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return AgeGroupNormalizationResult(
        input_found=input_path.exists(),
        records_read=len(base_records),
        records_written=len(normalized_records),
        issue_count=len(issues),
        unknown_age_group_count=issue_counts.get("unknown_age_group", 0),
        missing_age_group_count=issue_counts.get("missing_age_group", 0),
        duplicate_normalized_key_count=issue_counts.get("age_group_duplicate_normalized_key", 0),
        conflicting_value_count=issue_counts.get("age_group_conflicting_values", 0),
        warnings=[],
        output_paths=output_paths,
    )
