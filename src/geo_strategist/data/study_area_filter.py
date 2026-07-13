"""Deterministic filtering of population views into a configured study area."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.analysis_views import IssueSeverity
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.population_views import PopulationValueKind, normalize_label_text
from geo_strategist.data.study_area import (
    StudyArea,
    StudyAreaGeographyIssue,
    StudyAreaManifest,
    StudyAreaPopulationRecord,
    StudyAreaScopeStatus,
)
from geo_strategist.data.views.common import write_json, write_jsonl


class StudyAreaFilterResult(BaseModel):
    """Result of deterministic study-area population filtering."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    long_records_read: int = 0
    rate_records_read: int = 0
    long_records_written: int = 0
    rate_records_written: int = 0
    outside_scope_rows: int = 0
    scope_unknown_rows: int = 0
    scope_issue_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)


def load_study_area_config(path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml") -> tuple[StudyArea, dict[str, Any]]:
    """Load the configured study area and raw YAML payload."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    study_payload = payload["study_area"]
    study_area = StudyArea(
        study_area_id=study_payload["id"],
        target_prefectures=study_payload["target_prefectures"],
        aliases=study_payload["aliases"],
        scope_policy=study_payload.get("scope_policy") or {},
    )
    return study_area, payload


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


def _alias_key(value: str | None) -> str | None:
    normalized = normalize_label_text(value)
    if normalized is None:
        return None
    return normalized.casefold()


def build_prefecture_alias_map(study_area: StudyArea) -> dict[str, str]:
    """Build a deterministic lookup from configured aliases to target labels."""

    aliases: dict[str, str] = {}
    for target in study_area.target_prefectures:
        for value in [target, *study_area.aliases.get(target, [])]:
            key = _alias_key(value)
            if key:
                aliases[key] = target
    return aliases


def match_target_prefecture(raw_prefecture: str | None, study_area: StudyArea) -> str | None:
    """Return the configured target prefecture for a raw label, if any."""

    key = _alias_key(raw_prefecture)
    if key is None:
        return None
    return build_prefecture_alias_map(study_area).get(key)


def _is_unknown_prefecture_label(raw_prefecture: str | None, study_area: StudyArea) -> bool:
    key = _alias_key(raw_prefecture)
    if key is None:
        return True
    configured = study_area.scope_policy.get("unknown_prefecture_aliases") or []
    unknown_keys = {_alias_key(str(value)) for value in configured}
    return key in {value for value in unknown_keys if value}


def _first_label(labels: dict[str, Any] | None, candidates: tuple[str, ...]) -> str | None:
    if not labels:
        return None
    for key, value in labels.items():
        if value is None:
            continue
        normalized_key = normalize_label_text(str(key)) or ""
        if any(candidate.lower() in normalized_key.lower() for candidate in candidates):
            return normalize_label_text(str(value))
    return None


def _raw_prefecture(payload: dict[str, Any]) -> str | None:
    return normalize_label_text(payload.get("prefecture")) or _first_label(
        payload.get("geography_labels"), ("prefecture", "都道府県")
    )


def _raw_municipality(payload: dict[str, Any]) -> str | None:
    return normalize_label_text(payload.get("municipality")) or _first_label(
        payload.get("geography_labels"), ("municipality", "市区町村", "市町村")
    )


def classify_scope(payload: dict[str, Any], study_area: StudyArea) -> tuple[StudyAreaScopeStatus, str | None, str | None]:
    """Classify a source payload into the configured study-area scope."""

    raw_prefecture = _raw_prefecture(payload)
    matched_target = match_target_prefecture(raw_prefecture, study_area)
    if matched_target:
        return StudyAreaScopeStatus.IN_SCOPE, raw_prefecture, matched_target
    if _is_unknown_prefecture_label(raw_prefecture, study_area):
        return StudyAreaScopeStatus.SCOPE_UNKNOWN, raw_prefecture, None
    if raw_prefecture:
        return StudyAreaScopeStatus.OUTSIDE_SCOPE, raw_prefecture, None
    return StudyAreaScopeStatus.SCOPE_UNKNOWN, raw_prefecture, None


def _value_kind(payload: dict[str, Any]) -> PopulationValueKind:
    return PopulationValueKind(str(payload.get("value_kind") or ""))


def _study_area_row(
    payload: dict[str, Any],
    study_area: StudyArea,
    status: StudyAreaScopeStatus,
    raw_prefecture: str | None,
    matched_target: str | None,
) -> StudyAreaPopulationRecord:
    value_kind = _value_kind(payload)
    return StudyAreaPopulationRecord(
        record_id=str(payload.get("record_id") or ""),
        source_record_ids=list(payload.get("source_record_ids") or []),
        source_file_path=Path(str(payload["source_file_path"])) if payload.get("source_file_path") else None,
        source_sheet=payload.get("source_sheet"),
        source_file_hash=str(payload.get("source_file_hash") or ""),
        provenance=payload.get("provenance") or [],
        study_area_id=study_area.study_area_id,
        scope_status=status,
        raw_prefecture=raw_prefecture,
        matched_target_prefecture=matched_target,
        raw_municipality=_raw_municipality(payload),
        prefecture=payload.get("prefecture"),
        municipality=payload.get("municipality"),
        year=payload.get("year"),
        age_group=payload.get("age_group"),
        geography_labels=payload.get("geography_labels") or {},
        time_labels=payload.get("time_labels") or {},
        age_labels=payload.get("age_labels") or {},
        population_value=payload.get("population_value") if value_kind is PopulationValueKind.COUNT else None,
        rate_value=payload.get("rate_value") if value_kind is PopulationValueKind.RATE else None,
        unit=payload.get("unit"),
        value_kind=value_kind,
    )


def _scope_issue(
    payload: dict[str, Any],
    study_area: StudyArea,
    status: StudyAreaScopeStatus,
    raw_prefecture: str | None,
    matched_target: str | None,
    value_kind: PopulationValueKind,
) -> StudyAreaGeographyIssue:
    source_ids = list(payload.get("source_record_ids") or [])
    source_record = str(payload.get("record_id") or (source_ids[0] if source_ids else "unknown"))
    if status is StudyAreaScopeStatus.OUTSIDE_SCOPE:
        issue_type = "outside_scope_prefecture"
        severity = IssueSeverity.INFO
        message = "Population row is outside the configured Tokyo/Aichi/Osaka study area."
        action = "Keep the row in all-source views; exclude it from this target study area."
    else:
        issue_type = "scope_unknown_prefecture"
        severity = IssueSeverity.WARNING
        message = "Population row lacks a prefecture label and cannot be scoped."
        action = "Review source geography labels before using the row in target-scope analysis."
    return StudyAreaGeographyIssue(
        issue_id=f"study_area_scope:{issue_type}:{source_record}:{value_kind.value}",
        severity=severity,
        issue_type=issue_type,
        message=message,
        study_area_id=study_area.study_area_id,
        scope_status=status,
        target_prefecture=matched_target,
        raw_prefecture=raw_prefecture,
        raw_municipality=_raw_municipality(payload),
        year=payload.get("year"),
        age_group=payload.get("age_group"),
        value_kind=value_kind,
        source_record_ids=source_ids,
        source_file_hash=payload.get("source_file_hash"),
        recommended_action=action,
    )


def _process_payloads(
    payloads: list[dict[str, Any]],
    study_area: StudyArea,
) -> tuple[list[StudyAreaPopulationRecord], list[StudyAreaGeographyIssue], Counter[str], Counter[str], set[str]]:
    in_scope_rows: list[StudyAreaPopulationRecord] = []
    scope_issues: list[StudyAreaGeographyIssue] = []
    status_counts: Counter[str] = Counter()
    prefecture_status_counts: Counter[str] = Counter()
    source_hashes: set[str] = set()

    for payload in payloads:
        if payload.get("source_file_hash"):
            source_hashes.add(str(payload["source_file_hash"]))
        value_kind = _value_kind(payload)
        status, raw_prefecture, matched_target = classify_scope(payload, study_area)
        status_counts[status.value] += 1
        prefecture_status_counts[f"{raw_prefecture or 'missing'}|{status.value}"] += 1
        if status is StudyAreaScopeStatus.IN_SCOPE:
            in_scope_rows.append(
                _study_area_row(payload, study_area, status, raw_prefecture, matched_target)
            )
        else:
            scope_issues.append(
                _scope_issue(payload, study_area, status, raw_prefecture, matched_target, value_kind)
            )
    return in_scope_rows, scope_issues, status_counts, prefecture_status_counts, source_hashes


def filter_study_area_population(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> StudyAreaFilterResult:
    """Filter all-source population views to the configured target prefectures."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    inputs = config["inputs"]
    outputs = config["outputs"]
    long_path = root / inputs["population_long"]
    rate_path = root / inputs["population_rates_long"]
    long_payloads = _iter_jsonl(long_path)
    rate_payloads = _iter_jsonl(rate_path)

    (
        long_rows,
        long_issues,
        long_status_counts,
        long_prefecture_status_counts,
        long_hashes,
    ) = _process_payloads(long_payloads, study_area)
    (
        rate_rows,
        rate_issues,
        rate_status_counts,
        rate_prefecture_status_counts,
        rate_hashes,
    ) = _process_payloads(rate_payloads, study_area)

    scope_issues = long_issues + rate_issues
    status_counts = long_status_counts + rate_status_counts
    prefecture_status_counts = long_prefecture_status_counts + rate_prefecture_status_counts
    source_hashes = sorted(long_hashes | rate_hashes)
    if (long_payloads or rate_payloads) and not source_hashes:
        raise ValueError("study-area filtering requires source_file_hash on input rows")
    if not long_payloads and not rate_payloads:
        output_paths = {
            "population_long": outputs["study_area_population_long"],
            "population_rates_long": outputs["study_area_population_rates_long"],
            "manifest": outputs["study_area_manifest"],
            "summary": outputs["study_area_summary"],
            "scope_issues": outputs["study_area_scope_issues"],
            "outside_scope_counts": outputs["outside_scope_counts"],
        }
        return StudyAreaFilterResult(input_found=False, output_paths=output_paths)
    issue_counts = Counter(issue.issue_type for issue in scope_issues)
    record_counts = {
        "population_long_records_read": len(long_payloads),
        "population_rate_records_read": len(rate_payloads),
        "population_long_records_written": len(long_rows),
        "population_rate_records_written": len(rate_rows),
        "in_scope_rows": status_counts[StudyAreaScopeStatus.IN_SCOPE.value],
        "outside_scope_rows": status_counts[StudyAreaScopeStatus.OUTSIDE_SCOPE.value],
        "scope_unknown_rows": status_counts[StudyAreaScopeStatus.SCOPE_UNKNOWN.value],
        "scope_issues_written": len(scope_issues),
    }
    output_files = [
        Path(outputs["study_area_population_long"]),
        Path(outputs["study_area_population_rates_long"]),
        Path(outputs["study_area_manifest"]),
        Path(outputs["study_area_summary"]),
        Path(outputs["study_area_scope_issues"]),
        Path(outputs["outside_scope_counts"]),
    ]
    manifest = StudyAreaManifest(
        run_id=f"study_area_filter:{study_area.study_area_id}:{now_utc().isoformat()}",
        generated_at=now_utc(),
        study_area_id=study_area.study_area_id,
        target_prefectures=study_area.target_prefectures,
        input_files=[Path(inputs["population_long"]), Path(inputs["population_rates_long"])],
        output_files=output_files,
        source_file_hashes=source_hashes,
        record_counts=record_counts,
        issue_counts=dict(issue_counts),
        warnings=[],
    )
    outside_scope_counts = {
        "study_area_id": study_area.study_area_id,
        "target_prefectures": study_area.target_prefectures,
        "counts_by_status": dict(status_counts),
        "counts_by_prefecture_status": dict(sorted(prefecture_status_counts.items())),
        "outside_scope_rows": status_counts[StudyAreaScopeStatus.OUTSIDE_SCOPE.value],
        "scope_unknown_rows": status_counts[StudyAreaScopeStatus.SCOPE_UNKNOWN.value],
    }

    write_jsonl(root / outputs["study_area_population_long"], long_rows)
    write_jsonl(root / outputs["study_area_population_rates_long"], rate_rows)
    write_jsonl(root / outputs["study_area_scope_issues"], scope_issues)
    write_json(root / outputs["study_area_manifest"], manifest)
    write_json(root / outputs["outside_scope_counts"], outside_scope_counts)
    summary_path = root / outputs["study_area_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Study Area Scope Summary",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Target prefectures: {', '.join(study_area.target_prefectures)}",
                f"- Count rows read: {len(long_payloads)}",
                f"- Rate rows read: {len(rate_payloads)}",
                f"- Count rows in scope: {len(long_rows)}",
                f"- Rate rows in scope: {len(rate_rows)}",
                f"- Outside-scope rows: {status_counts[StudyAreaScopeStatus.OUTSIDE_SCOPE.value]}",
                f"- Scope-unknown rows: {status_counts[StudyAreaScopeStatus.SCOPE_UNKNOWN.value]}",
                "- Outside-scope prefectures are summarized for scope control and are not source-data errors.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_paths = {
        "population_long": outputs["study_area_population_long"],
        "population_rates_long": outputs["study_area_population_rates_long"],
        "manifest": outputs["study_area_manifest"],
        "summary": outputs["study_area_summary"],
        "scope_issues": outputs["study_area_scope_issues"],
        "outside_scope_counts": outputs["outside_scope_counts"],
    }
    return StudyAreaFilterResult(
        input_found=long_path.exists() or rate_path.exists(),
        long_records_read=len(long_payloads),
        rate_records_read=len(rate_payloads),
        long_records_written=len(long_rows),
        rate_records_written=len(rate_rows),
        outside_scope_rows=status_counts[StudyAreaScopeStatus.OUTSIDE_SCOPE.value],
        scope_unknown_rows=status_counts[StudyAreaScopeStatus.SCOPE_UNKNOWN.value],
        scope_issue_count=len(scope_issues),
        warnings=[],
        output_paths=output_paths,
    )
