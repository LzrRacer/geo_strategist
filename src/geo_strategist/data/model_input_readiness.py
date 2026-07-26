"""Model-input readiness QA gate for age-normalized population base."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError

from geo_strategist.data.age_groups import AgeGroupNormalizedRecord
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.population_base_coverage import CoverageSeverity
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class ReadinessRole(str, Enum):
    """Readiness classification role for a population-base row."""

    MODEL_INPUT = "model_input"
    CONTEXT = "context"


class ReadinessStatus(str, Enum):
    """Model-input readiness status for a population-base row."""

    READY = "ready"
    CONTEXT_ONLY = "context_only"
    BLOCKED = "blocked"


class ModelInputReadinessRecord(BaseModel):
    """Readiness classification for one age-normalized population-base row."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    matched_target_prefecture: str = Field(min_length=1)
    municipality: str | None = None
    year: int | str | None = None
    canonical_age_group_id: str | None = None
    canonical_age_group_label: str | None = None
    value_kind: str = Field(min_length=1)
    geography_grain: str = Field(min_length=1)
    population_base_role: str = Field(min_length=1)
    readiness_role: ReadinessRole
    readiness_status: ReadinessStatus
    readiness_key: str = Field(min_length=1)
    blocking_issue_codes: list[str] = Field(default_factory=list)


class ModelInputReadinessIssue(BaseModel):
    """Issue discovered during model-input readiness classification."""

    model_config = ConfigDict(extra="forbid")

    issue_id: str = Field(min_length=1)
    severity: CoverageSeverity
    issue_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    study_area_id: str = Field(min_length=1)
    readiness_key: str | None = None
    matched_target_prefecture: str | None = None
    municipality: str | None = None
    year: str | None = None
    canonical_age_group_id: str | None = None
    value_kind: str | None = None
    geography_grain: str | None = None
    population_base_role: str | None = None
    source_record_ids: list[str] = Field(default_factory=list)
    recommended_action: str = Field(min_length=1)


class ModelInputReadinessResult(BaseModel):
    """Result of the model-input readiness QA gate."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    readiness_records_written: int = 0
    ready_model_input_rows: int = 0
    context_rows: int = 0
    blocked_rows: int = 0
    issue_count: int = 0
    blocking_error_count: int = 0
    output_paths: dict[str, str] = Field(default_factory=dict)


_BLOCKING_MESSAGES: dict[str, str] = {
    "unknown_geography_grain": "Record has unknown geography grain and cannot be classified for model input.",
    "unknown_age_group": "Record has unknown or missing canonical age group.",
    "prefecture_scope_violation": "Record prefecture is not in the target study area.",
    "municipality_model_input_missing_identifier": "Municipality-grain model-input row is missing its municipality identifier.",
    "duplicate_readiness_key_conflict": "Duplicate readiness key with conflicting values detected.",
    "unexpected_role_grain_combination": "Record has an unexpected role/grain combination.",
}

_NON_BLOCKING_UPSTREAM_CODES: frozenset[str] = frozenset(
    {"missing_rate_for_count_combination", "missing_count_for_rate_combination"}
)


def _required_input_paths(
    root: Path, config_path: Path, outputs: dict[str, str]
) -> dict[str, Path]:
    return {
        "population_base_age_normalized": root / outputs["population_base_age_normalized"],
        "age_group_coverage_report": root / outputs["age_group_coverage_report_json"],
        "age_group_qa_issues": root / outputs["age_group_qa_issues"],
        "population_base_coverage_report": root / outputs["population_base_coverage_report_json"],
        "population_base_coverage_issues": root / outputs["population_base_coverage_issues"],
        "population_base_report": root / outputs["population_base_report_json"],
        "study_area_config": config_path,
        "age_group_config": root / "configs/age_groups.yaml",
        "population_workbook_mapping_config": root / "configs/source_mappings/population_workbooks.yaml",
    }


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _load_yaml_object(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return payload


def _readiness_key(record: dict[str, Any]) -> str:
    prefecture = record.get("matched_target_prefecture") or ""
    municipality = record.get("municipality") or ""
    grain = record.get("geography_grain") or ""
    base_role = record.get("population_base_role") or ""
    year = str(record.get("year") or "")
    age_id = record.get("canonical_age_group_id") or ""
    vk = record.get("value_kind") or ""
    return f"{prefecture}:{municipality}:{grain}:{base_role}:{year}:{age_id}:{vk}"


def _record_value(record: dict[str, Any]) -> float | None:
    if record.get("value_kind") == "count":
        return record.get("population_value")
    if record.get("value_kind") == "rate":
        return record.get("rate_value")
    return None


def _make_issue(
    issue_seq: int,
    issue_code: str,
    severity: CoverageSeverity,
    message: str,
    study_area_id: str,
    recommended_action: str,
    readiness_key: str | None = None,
    record: dict[str, Any] | None = None,
    source_record_ids: list[str] | None = None,
) -> ModelInputReadinessIssue:
    record = record or {}
    return ModelInputReadinessIssue(
        issue_id=f"model_input_readiness:{issue_code}:{readiness_key or issue_seq}:{issue_seq}",
        severity=severity,
        issue_code=issue_code,
        message=message,
        study_area_id=study_area_id,
        readiness_key=readiness_key,
        matched_target_prefecture=record.get("matched_target_prefecture"),
        municipality=record.get("municipality"),
        year=str(record.get("year") or "") if record.get("year") is not None else None,
        canonical_age_group_id=record.get("canonical_age_group_id"),
        value_kind=record.get("value_kind"),
        geography_grain=record.get("geography_grain"),
        population_base_role=record.get("population_base_role"),
        source_record_ids=source_record_ids or list(record.get("source_record_ids", [])),
        recommended_action=recommended_action,
    )


def _write_blocked_result(
    root: Path,
    outputs: dict[str, str],
    output_paths: dict[str, str],
    study_area_id: str,
    target_prefectures: list[str],
    input_paths: dict[str, Path],
    issues: list[ModelInputReadinessIssue],
    records_read: int = 0,
) -> ModelInputReadinessResult:
    write_jsonl(root / outputs["model_input_readiness"], [])
    write_jsonl(root / outputs["model_input_ready_population_base"], [])
    write_jsonl(root / outputs["model_input_context_population_base"], [])
    write_jsonl(root / outputs["model_input_readiness_issues"], issues)

    severity_counts = Counter(i.severity.value for i in issues)
    code_counts = Counter(i.issue_code for i in issues)
    blocking_error_count = severity_counts.get("error", 0)
    now = now_utc()
    manifest = {
        "run_id": f"model_input_readiness:{study_area_id}:{now.isoformat()}",
        "generated_at": now.isoformat(),
        "study_area_id": study_area_id,
        "input_files": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in input_paths.values()
        ],
        "output_files": list(output_paths.values()),
        "record_counts": {
            "records_read": records_read,
            "readiness_records_written": 0,
            "ready_model_input_rows": 0,
            "context_rows": 0,
            "blocked_rows": records_read,
        },
        "issue_counts_by_severity": dict(severity_counts),
        "issue_counts_by_code": dict(code_counts),
        "warnings": [],
    }
    write_json(root / outputs["model_input_readiness_manifest"], manifest)

    report = {
        "study_area_id": study_area_id,
        "target_prefectures": sorted(target_prefectures),
        "records_read": records_read,
        "readiness_records_written": 0,
        "ready_model_input_rows": 0,
        "context_rows": 0,
        "blocked_rows": records_read,
        "issue_count": len(issues),
        "issue_counts_by_severity": dict(severity_counts),
        "issue_counts_by_code": dict(code_counts),
        "blocking_error_count": blocking_error_count,
        "model_input_readiness_passed": False,
        "expected_counts": {
            "total_records": 13167,
            "ready_model_input_rows": 12978,
            "context_rows": 189,
        },
        "count_deviations": {
            "total_records": records_read - 13167,
            "ready_model_input_rows": -12978,
            "context_rows": -189,
        },
        "count_deviation_issues": [
            "total_records",
            "ready_model_input_rows",
            "context_rows",
        ],
    }
    write_json(root / outputs["model_input_readiness_report_json"], report)

    summary_lines = [
        "# Model Input Readiness Summary",
        "",
        f"- Study area: {study_area_id}",
        f"- Records read: {records_read}",
        "- Ready model-input rows: 0",
        "- Context prefecture-total rows: 0",
        f"- Blocked rows: {records_read}",
        f"- Total issues: {len(issues)}",
        f"- Blocking errors: {blocking_error_count}",
        "- Model-input readiness gate: **FAILED**",
        "- This report does not calculate demand.",
    ]
    summary_path = root / outputs["model_input_readiness_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Model Input Readiness Report",
        "",
        f"- Study area: {study_area_id}",
        f"- Target prefectures: {', '.join(sorted(target_prefectures))}",
        f"- Records read: {records_read}",
        "- Ready model-input rows: 0",
        "- Context prefecture-total rows: 0",
        f"- Blocked rows: {records_read}",
        "",
        "## Issues",
        "",
        f"- Total issues: {len(issues)}",
        f"- Errors (blocking): {blocking_error_count}",
        f"- Warnings: {severity_counts.get('warning', 0)}",
        f"- Info: {severity_counts.get('info', 0)}",
        "",
        "## Verdict",
        "",
        "Model-input readiness gate: **FAILED**",
        "",
        "This report does not calculate demand, land scores, or cash flows.",
    ]
    report_md_path = root / outputs["model_input_readiness_report_markdown"]
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return ModelInputReadinessResult(
        input_found=False,
        records_read=records_read,
        readiness_records_written=0,
        ready_model_input_rows=0,
        context_rows=0,
        blocked_rows=records_read,
        issue_count=len(issues),
        blocking_error_count=blocking_error_count,
        output_paths=output_paths,
    )


def _classify(
    record: dict[str, Any],
    target_prefectures: frozenset[str],
) -> tuple[ReadinessRole, ReadinessStatus, list[str]]:
    """Return (role, status, blocking_codes)."""
    blocking: list[str] = []
    grain = record.get("geography_grain", "")
    base_role = record.get("population_base_role", "")
    prefecture = record.get("matched_target_prefecture") or ""

    if prefecture not in target_prefectures:
        blocking.append("prefecture_scope_violation")

    if grain == "unknown":
        blocking.append("unknown_geography_grain")

    match_status = record.get("age_group_match_status", "")
    if match_status in {"unknown", "missing"} or record.get("canonical_age_group_id") is None:
        blocking.append("unknown_age_group")

    if grain == "municipality" and not record.get("municipality"):
        blocking.append("municipality_model_input_missing_identifier")

    if blocking:
        role = ReadinessRole.CONTEXT if base_role == "context_prefecture_total" else ReadinessRole.MODEL_INPUT
        return role, ReadinessStatus.BLOCKED, blocking

    if grain == "municipality" and base_role == "model_input_candidate":
        return ReadinessRole.MODEL_INPUT, ReadinessStatus.READY, []
    if grain == "prefecture_total" and base_role == "context_prefecture_total":
        return ReadinessRole.CONTEXT, ReadinessStatus.CONTEXT_ONLY, []

    blocking.append("unexpected_role_grain_combination")
    return ReadinessRole.MODEL_INPUT, ReadinessStatus.BLOCKED, blocking


def run_model_input_readiness(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
) -> ModelInputReadinessResult:
    """Run model-input readiness QA gate on age-normalized population-base rows."""

    root = Path(repo_root).resolve()
    resolved_config_path = root / config_path
    study_area, config = load_study_area_config(resolved_config_path)
    outputs = config["outputs"]

    output_paths = {k: outputs[k] for k in (
        "model_input_readiness",
        "model_input_ready_population_base",
        "model_input_context_population_base",
        "model_input_readiness_manifest",
        "model_input_readiness_issues",
        "model_input_readiness_summary",
        "model_input_readiness_report_json",
        "model_input_readiness_report_markdown",
    )}

    input_paths = _required_input_paths(root, resolved_config_path, outputs)
    missing_inputs = [(label, path) for label, path in input_paths.items() if not path.exists()]
    if missing_inputs:
        issues = [
            _make_issue(
                issue_seq=index,
                issue_code="missing_required_input_artifact",
                severity=CoverageSeverity.ERROR,
                message=f"Required upstream artifact is missing: {label}.",
                study_area_id=study_area.study_area_id,
                recommended_action="Run the prerequisite deterministic population-base, coverage, and age-group QA steps.",
                readiness_key=label,
            )
            for index, (label, _path) in enumerate(missing_inputs, start=1)
        ]
        return _write_blocked_result(
            root,
            outputs,
            output_paths,
            study_area.study_area_id,
            study_area.target_prefectures,
            input_paths,
            issues,
        )

    target_prefectures = frozenset(study_area.target_prefectures)
    age_normalized_path = input_paths["population_base_age_normalized"]
    coverage_issues_path = input_paths["population_base_coverage_issues"]
    age_issues_path = input_paths["age_group_qa_issues"]

    try:
        _load_json_object(input_paths["age_group_coverage_report"])
        _load_json_object(input_paths["population_base_coverage_report"])
        _load_json_object(input_paths["population_base_report"])
        _load_yaml_object(input_paths["age_group_config"])
        _load_yaml_object(input_paths["population_workbook_mapping_config"])
        parsed_rows = _iter_jsonl(age_normalized_path)
        upstream_coverage_issues = _iter_jsonl(coverage_issues_path)
        upstream_age_issues = _iter_jsonl(age_issues_path)
    except (OSError, ValueError) as exc:
        issue = _make_issue(
            issue_seq=1,
            issue_code="invalid_required_input_artifact",
            severity=CoverageSeverity.ERROR,
            message=str(exc),
            study_area_id=study_area.study_area_id,
            recommended_action="Regenerate or repair the invalid prerequisite artifact before model-input readiness QA.",
        )
        return _write_blocked_result(
            root,
            outputs,
            output_paths,
            study_area.study_area_id,
            study_area.target_prefectures,
            input_paths,
            [issue],
        )

    valid_rows: list[dict[str, Any]] = []
    structural_issues: list[ModelInputReadinessIssue] = []
    issue_seq = 0
    records_read_total = len(parsed_rows)
    for index, row in enumerate(parsed_rows, start=1):
        try:
            record = AgeGroupNormalizedRecord.model_validate(row)
        except ValidationError as exc:
            issue_seq += 1
            structural_issues.append(
                _make_issue(
                    issue_seq=issue_seq,
                    issue_code="invalid_age_normalized_record",
                    severity=CoverageSeverity.ERROR,
                    message=(
                        f"Age-normalized input row {index} failed schema validation: "
                        f"{exc.errors()[0]['msg']}"
                    ),
                    study_area_id=study_area.study_area_id,
                    recommended_action=(
                        "Regenerate age-normalized population base with required traceability, "
                        "geography, age-group, and value-kind fields."
                    ),
                    readiness_key=row.get("record_id") or f"row:{index}",
                    record=row,
                )
            )
            continue
        valid_rows.append(record.model_dump(mode="json"))

    upstream_rows = valid_rows

    # Pass 1: compute keys for duplicate detection
    key_entries: dict[str, list[tuple[str, float | None]]] = defaultdict(list)
    for row in upstream_rows:
        key = _readiness_key(row)
        value = _record_value(row)
        key_entries[key].append((row.get("record_id", ""), value))

    duplicate_keys: set[str] = set()
    conflicting_keys: set[str] = set()
    for key, entries in key_entries.items():
        if len(entries) > 1:
            duplicate_keys.add(key)
            if len({e[1] for e in entries}) > 1:
                conflicting_keys.add(key)

    # Pass 2: classify records
    readiness_records: list[ModelInputReadinessRecord] = []
    ready_rows: list[dict[str, Any]] = []
    context_rows: list[dict[str, Any]] = []
    issues: list[ModelInputReadinessIssue] = list(structural_issues)

    for row in upstream_rows:
        role, status, blocking = _classify(row, target_prefectures)
        key = _readiness_key(row)

        if key in conflicting_keys and status is not ReadinessStatus.BLOCKED:
            blocking = list(blocking) + ["duplicate_readiness_key_conflict"]
            status = ReadinessStatus.BLOCKED

        readiness_records.append(ModelInputReadinessRecord(
            record_id=row.get("record_id", ""),
            study_area_id=row.get("study_area_id", study_area.study_area_id),
            matched_target_prefecture=row.get("matched_target_prefecture", ""),
            municipality=row.get("municipality"),
            year=row.get("year"),
            canonical_age_group_id=row.get("canonical_age_group_id"),
            canonical_age_group_label=row.get("canonical_age_group_label"),
            value_kind=row.get("value_kind", ""),
            geography_grain=row.get("geography_grain", ""),
            population_base_role=row.get("population_base_role", ""),
            readiness_role=role,
            readiness_status=status,
            readiness_key=key,
            blocking_issue_codes=blocking,
        ))

        if status is ReadinessStatus.READY:
            ready_rows.append(row)
        elif status is ReadinessStatus.CONTEXT_ONLY:
            context_rows.append(row)

        for code in blocking:
            issue_seq += 1
            issues.append(_make_issue(
                issue_seq=issue_seq,
                issue_code=code,
                severity=CoverageSeverity.ERROR,
                message=_BLOCKING_MESSAGES.get(code, f"Blocking issue: {code}."),
                study_area_id=study_area.study_area_id,
                recommended_action="Review and correct the upstream data or classification.",
                readiness_key=key,
                record=row,
            ))

    # Warn on non-conflicting duplicate keys
    for key in sorted(duplicate_keys - conflicting_keys):
        issue_seq += 1
        issues.append(ModelInputReadinessIssue(
            issue_id=f"model_input_readiness:duplicate_readiness_key:{key}:{issue_seq}",
            severity=CoverageSeverity.WARNING,
            issue_code="duplicate_readiness_key",
            message=f"Duplicate readiness key with consistent values: {key}",
            study_area_id=study_area.study_area_id,
            readiness_key=key,
            recommended_action="Review upstream records for unexpected duplicates.",
        ))

    # Propagate non-blocking upstream coverage issues
    for upstream_issue in upstream_coverage_issues:
        code = upstream_issue.get("issue_type", "")
        if code not in _NON_BLOCKING_UPSTREAM_CODES:
            continue
        issue_seq += 1
        severity = CoverageSeverity(upstream_issue.get("severity", "info"))
        issues.append(ModelInputReadinessIssue(
            issue_id=f"model_input_readiness:upstream:{upstream_issue.get('issue_id', str(issue_seq))}",
            severity=severity,
            issue_code=code,
            message=upstream_issue.get("message", "Upstream coverage gap."),
            study_area_id=study_area.study_area_id,
            matched_target_prefecture=upstream_issue.get("target_prefecture"),
            year=str(upstream_issue.get("year") or ""),
            canonical_age_group_id=upstream_issue.get("age_group"),
            geography_grain=upstream_issue.get("geography_grain"),
            population_base_role=upstream_issue.get("population_base_role"),
            recommended_action=upstream_issue.get(
                "recommended_action",
                "Keep count/rate semantics separate; review only if counterpart coverage is required later.",
            ),
        ))

    # Propagate non-blocking upstream age-group issues
    for upstream_issue in upstream_age_issues:
        code = upstream_issue.get("issue_type", "")
        severity_str = upstream_issue.get("severity", "info")
        if severity_str == "error":
            continue
        issue_seq += 1
        severity = CoverageSeverity(severity_str)
        issues.append(ModelInputReadinessIssue(
            issue_id=f"model_input_readiness:upstream_age:{upstream_issue.get('issue_id', str(issue_seq))}",
            severity=severity,
            issue_code=code or "upstream_age_group_issue",
            message=upstream_issue.get("message", "Upstream age-group issue."),
            study_area_id=study_area.study_area_id,
            recommended_action=upstream_issue.get("recommended_action", "Review upstream age-group QA."),
        ))

    # Write outputs
    write_jsonl(root / outputs["model_input_readiness"], readiness_records)
    write_jsonl(root / outputs["model_input_ready_population_base"], ready_rows)
    write_jsonl(root / outputs["model_input_context_population_base"], context_rows)
    write_jsonl(root / outputs["model_input_readiness_issues"], issues)

    # Counts and report
    records_read = records_read_total
    ready_count = len(ready_rows)
    context_count = len(context_rows)
    blocked_count = len(structural_issues) + sum(
        1 for r in readiness_records if r.readiness_status is ReadinessStatus.BLOCKED
    )
    severity_counts = Counter(i.severity.value for i in issues)
    code_counts = Counter(i.issue_code for i in issues)
    blocking_error_count = severity_counts.get("error", 0)

    now = now_utc()
    manifest = {
        "run_id": f"model_input_readiness:{study_area.study_area_id}:{now.isoformat()}",
        "generated_at": now.isoformat(),
        "study_area_id": study_area.study_area_id,
        "input_files": [
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            for path in input_paths.values()
        ],
        "output_files": list(output_paths.values()),
        "record_counts": {
            "records_read": records_read,
            "readiness_records_written": len(readiness_records),
            "ready_model_input_rows": ready_count,
            "context_rows": context_count,
            "blocked_rows": blocked_count,
        },
        "issue_counts_by_severity": dict(severity_counts),
        "issue_counts_by_code": dict(code_counts),
        "warnings": [],
    }
    write_json(root / outputs["model_input_readiness_manifest"], manifest)

    _EXPECTED_TOTAL = 13167
    _EXPECTED_READY = 12978
    _EXPECTED_CONTEXT = 189

    deviations: dict[str, int] = {
        "total_records": records_read - _EXPECTED_TOTAL,
        "ready_model_input_rows": ready_count - _EXPECTED_READY,
        "context_rows": context_count - _EXPECTED_CONTEXT,
    }

    report = {
        "study_area_id": study_area.study_area_id,
        "target_prefectures": sorted(target_prefectures),
        "records_read": records_read,
        "readiness_records_written": len(readiness_records),
        "ready_model_input_rows": ready_count,
        "context_rows": context_count,
        "blocked_rows": blocked_count,
        "issue_count": len(issues),
        "issue_counts_by_severity": dict(severity_counts),
        "issue_counts_by_code": dict(code_counts),
        "blocking_error_count": blocking_error_count,
        "model_input_readiness_passed": blocking_error_count == 0,
        "expected_counts": {
            "total_records": _EXPECTED_TOTAL,
            "ready_model_input_rows": _EXPECTED_READY,
            "context_rows": _EXPECTED_CONTEXT,
        },
        "count_deviations": deviations,
        "count_deviation_issues": [
            k for k, v in deviations.items() if v != 0
        ],
    }
    write_json(root / outputs["model_input_readiness_report_json"], report)

    passed_str = "PASSED" if blocking_error_count == 0 else "FAILED"
    deviation_notes = [
        f"  - {k}: {'+' if v > 0 else ''}{v}" for k, v in deviations.items() if v != 0
    ]

    summary_lines = [
        "# Model Input Readiness Summary",
        "",
        f"- Study area: {study_area.study_area_id}",
        f"- Records read: {records_read}",
        f"- Ready model-input rows: {ready_count}",
        f"- Context prefecture-total rows: {context_count}",
        f"- Blocked rows: {blocked_count}",
        f"- Total issues: {len(issues)}",
        f"- Blocking errors: {blocking_error_count}",
        f"- Model-input readiness gate: **{passed_str}**",
    ]
    if deviation_notes:
        summary_lines += ["- Count deviations from expected:"] + deviation_notes
    summary_lines.append("- This report does not calculate demand.")
    summary_path = root / outputs["model_input_readiness_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    report_lines = [
        "# Model Input Readiness Report",
        "",
        f"- Study area: {study_area.study_area_id}",
        f"- Target prefectures: {', '.join(sorted(target_prefectures))}",
        f"- Records read: {records_read} (expected {_EXPECTED_TOTAL:,})",
        f"- Ready model-input rows: {ready_count} (expected {_EXPECTED_READY:,})",
        f"- Context prefecture-total rows: {context_count} (expected {_EXPECTED_CONTEXT:,})",
        f"- Blocked rows: {blocked_count}",
        "",
        "## Issues",
        "",
        f"- Total issues: {len(issues)}",
        f"- Errors (blocking): {blocking_error_count}",
        f"- Warnings: {severity_counts.get('warning', 0)}",
        f"- Info: {severity_counts.get('info', 0)}",
        "",
        "## Verdict",
        "",
        f"Model-input readiness gate: **{passed_str}**",
        "",
        "This report does not calculate demand, land scores, or cash flows.",
    ]
    report_md_path = root / outputs["model_input_readiness_report_markdown"]
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return ModelInputReadinessResult(
        input_found=True,
        records_read=records_read,
        readiness_records_written=len(readiness_records),
        ready_model_input_rows=ready_count,
        context_rows=context_count,
        blocked_rows=blocked_count,
        issue_count=len(issues),
        blocking_error_count=blocking_error_count,
        output_paths=output_paths,
    )
