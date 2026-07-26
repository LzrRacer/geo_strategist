"""Pre-demand coverage QA for study-area population-base rows."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.geography_grain import (
    GeographyGrain,
    PopulationBaseRole,
    StudyAreaPopulationBaseRecord,
)
from geo_strategist.data.normalization import now_utc
from geo_strategist.data.population_base_coverage import (
    CoverageAxis,
    CoverageIssueType,
    CoverageSeverity,
    PopulationBaseCoverageIssue,
    PopulationBaseCoverageManifest,
    PopulationBaseCoverageMatrix,
)
from geo_strategist.data.population_views import PopulationValueKind, normalize_label_text
from geo_strategist.data.study_area_filter import load_study_area_config
from geo_strategist.data.views.common import write_json, write_jsonl


class PopulationBaseCoverageQAResult(BaseModel):
    """Result of population-base coverage QA."""

    model_config = ConfigDict(extra="forbid")

    input_found: bool
    records_read: int = 0
    matrix_rows_written: int = 0
    issue_count: int = 0
    model_blocking_error_count: int = 0
    duplicate_key_count: int = 0
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


def _load_mapping_expectations(path: Path) -> tuple[list[str], list[str], list[dict[str, object]]]:
    if not path.exists():
        return [], [], []
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    years: set[str] = set()
    age_groups: set[str] = set()
    evidence: list[dict[str, object]] = []
    for override in payload.get("manual_overrides", []):
        if override.get("status") != "manual_verified":
            continue
        override_years = sorted(str(value) for value in (override.get("year_by_column") or {}).values())
        age_group = normalize_label_text(override.get("age_label"))
        years.update(override_years)
        if age_group:
            age_groups.add(age_group)
        evidence.append(
            {
                "config_path": str(path),
                "override_id": override.get("id"),
                "years": override_years,
                "age_group": age_group,
                "evidence_text": override.get("evidence_text"),
            }
        )
    return sorted(years), sorted(age_groups), evidence


def _coverage_key(record: StudyAreaPopulationBaseRecord) -> tuple[Any, ...]:
    return (
        record.matched_target_prefecture,
        record.municipality,
        record.geography_grain.value,
        record.population_base_role.value,
        str(record.year) if record.year is not None else None,
        record.age_group,
        record.value_kind.value,
    )


def _coverage_key_label(key: tuple[Any, ...]) -> str:
    return "|".join(str(value) if value is not None else "missing" for value in key)


def _value(record: StudyAreaPopulationBaseRecord) -> float | None:
    if record.value_kind is PopulationValueKind.COUNT:
        return record.population_value
    return record.rate_value


def _issue(
    issue_type: CoverageIssueType,
    severity: CoverageSeverity,
    message: str,
    study_area_id: str,
    recommended_action: str,
    coverage_key: str | None = None,
    record: StudyAreaPopulationBaseRecord | None = None,
    target_prefecture: str | None = None,
    year: int | str | None = None,
    age_group: str | None = None,
    value_kind: PopulationValueKind | None = None,
    geography_grain: GeographyGrain | None = None,
    population_base_role: PopulationBaseRole | None = None,
    source_record_ids: list[str] | None = None,
    source_file_hashes: list[str] | None = None,
) -> PopulationBaseCoverageIssue:
    if record is not None:
        target_prefecture = record.matched_target_prefecture
        year = record.year
        age_group = record.age_group
        value_kind = record.value_kind
        geography_grain = record.geography_grain
        population_base_role = record.population_base_role
        source_record_ids = record.source_record_ids
        source_file_hashes = [record.source_file_hash] if record.source_file_hash else []
    issue_key = coverage_key or ":".join(
        str(value)
        for value in [
            issue_type.value,
            target_prefecture,
            year,
            age_group,
            value_kind.value if value_kind else None,
            geography_grain.value if geography_grain else None,
            population_base_role.value if population_base_role else None,
        ]
    )
    return PopulationBaseCoverageIssue(
        issue_id=f"population_base_coverage:{issue_key}",
        severity=severity,
        issue_type=issue_type,
        message=message,
        study_area_id=study_area_id,
        coverage_key=coverage_key,
        target_prefecture=target_prefecture,
        year=year,
        age_group=age_group,
        value_kind=value_kind,
        geography_grain=geography_grain,
        population_base_role=population_base_role,
        source_record_ids=source_record_ids or [],
        source_file_hashes=source_file_hashes or [],
        recommended_action=recommended_action,
    )


def _matrix_row(
    study_area_id: str,
    coverage_axis: CoverageAxis,
    axis_values: dict[str, str | int | None],
    row_count: int,
) -> PopulationBaseCoverageMatrix:
    return PopulationBaseCoverageMatrix(
        matrix_id=f"{coverage_axis.value}:{'|'.join(str(v) for v in axis_values.values())}",
        study_area_id=study_area_id,
        coverage_axis=coverage_axis,
        axis_values=axis_values,
        target_prefecture=axis_values.get("target_prefecture"),
        year=axis_values.get("year"),
        age_group=axis_values.get("age_group"),
        value_kind=axis_values.get("value_kind"),
        geography_grain=axis_values.get("geography_grain"),
        population_base_role=axis_values.get("population_base_role"),
        row_count=row_count,
    )


def _build_matrices(
    study_area_id: str,
    records: list[StudyAreaPopulationBaseRecord],
) -> list[PopulationBaseCoverageMatrix]:
    matrix_specs: list[tuple[CoverageAxis, tuple[str, ...], Counter[tuple[Any, ...]]]] = [
        (
            CoverageAxis.POPULATION_BASE_ROLE,
            (
                "target_prefecture",
                "year",
                "age_group",
                "value_kind",
                "geography_grain",
                "population_base_role",
            ),
            Counter(
                (
                    row.matched_target_prefecture,
                    str(row.year) if row.year is not None else None,
                    row.age_group,
                    row.value_kind.value,
                    row.geography_grain.value,
                    row.population_base_role.value,
                )
                for row in records
            ),
        ),
        (
            CoverageAxis.POPULATION_BASE_ROLE,
            ("target_prefecture", "year", "age_group", "value_kind"),
            Counter(
                (
                    row.matched_target_prefecture,
                    str(row.year) if row.year is not None else None,
                    row.age_group,
                    row.value_kind.value,
                )
                for row in records
                if row.population_base_role is PopulationBaseRole.MODEL_INPUT_CANDIDATE
            ),
        ),
        (
            CoverageAxis.POPULATION_BASE_ROLE,
            ("target_prefecture", "year", "age_group", "value_kind"),
            Counter(
                (
                    row.matched_target_prefecture,
                    str(row.year) if row.year is not None else None,
                    row.age_group,
                    row.value_kind.value,
                )
                for row in records
                if row.population_base_role is PopulationBaseRole.CONTEXT_PREFECTURE_TOTAL
            ),
        ),
        (
            CoverageAxis.VALUE_KIND,
            ("target_prefecture", "year", "age_group", "geography_grain"),
            Counter(
                (
                    row.matched_target_prefecture,
                    str(row.year) if row.year is not None else None,
                    row.age_group,
                    row.geography_grain.value,
                )
                for row in records
                if row.value_kind is PopulationValueKind.COUNT
            ),
        ),
        (
            CoverageAxis.VALUE_KIND,
            ("target_prefecture", "year", "age_group", "geography_grain"),
            Counter(
                (
                    row.matched_target_prefecture,
                    str(row.year) if row.year is not None else None,
                    row.age_group,
                    row.geography_grain.value,
                )
                for row in records
                if row.value_kind is PopulationValueKind.RATE
            ),
        ),
    ]
    rows: list[PopulationBaseCoverageMatrix] = []
    for axis, keys, counter in matrix_specs:
        for values, count in sorted(counter.items()):
            axis_values = dict(zip(keys, values, strict=True))
            rows.append(_matrix_row(study_area_id, axis, axis_values, count))
    return rows


def _detect_issues(
    study_area_id: str,
    target_prefectures: list[str],
    records: list[StudyAreaPopulationBaseRecord],
    expected_years: list[str],
    expected_age_groups: list[str],
) -> list[PopulationBaseCoverageIssue]:
    issues: list[PopulationBaseCoverageIssue] = []
    observed_years = {str(row.year) for row in records if row.year not in (None, "")}
    observed_age_groups = {row.age_group for row in records if row.age_group}
    for year in expected_years:
        if year not in observed_years:
            issues.append(
                _issue(
                    CoverageIssueType.MISSING_EXPECTED_YEAR,
                    CoverageSeverity.WARNING,
                    "Expected year is absent from population-base rows.",
                    study_area_id,
                    "Review source mappings or regenerate population-base rows.",
                    year=year,
                )
            )
    for age_group in expected_age_groups:
        if age_group not in observed_age_groups:
            issues.append(
                _issue(
                    CoverageIssueType.MISSING_EXPECTED_AGE_GROUP,
                    CoverageSeverity.WARNING,
                    "Expected age group is absent from population-base rows.",
                    study_area_id,
                    "Review source mappings or regenerate population-base rows.",
                    age_group=age_group,
                )
            )

    observed_pref_year_age = {
        (row.matched_target_prefecture, str(row.year), row.age_group)
        for row in records
        if row.year not in (None, "") and row.age_group
    }
    for prefecture in target_prefectures:
        for year in expected_years or sorted(observed_years):
            for age_group in expected_age_groups or sorted(observed_age_groups):
                if (prefecture, year, age_group) not in observed_pref_year_age:
                    issues.append(
                        _issue(
                            CoverageIssueType.MISSING_PREFECTURE_YEAR_AGE_COMBINATION,
                            CoverageSeverity.WARNING,
                            "Expected prefecture/year/age-group combination is absent.",
                            study_area_id,
                            "Review coverage before demand-feature engineering; do not impute values.",
                            target_prefecture=prefecture,
                            year=year,
                            age_group=age_group,
                        )
                    )

    key_to_records: defaultdict[tuple[Any, ...], list[StudyAreaPopulationBaseRecord]] = defaultdict(list)
    for record in records:
        key_to_records[_coverage_key(record)].append(record)
        if record.geography_grain not in set(GeographyGrain):
            issues.append(
                _issue(
                    CoverageIssueType.UNEXPECTED_GEOGRAPHY_GRAIN,
                    CoverageSeverity.ERROR,
                    "Population-base row has unexpected geography grain.",
                    study_area_id,
                    "Regenerate geography-grain classification.",
                    record=record,
                )
            )
        if record.value_kind not in set(PopulationValueKind):
            issues.append(
                _issue(
                    CoverageIssueType.UNEXPECTED_VALUE_KIND,
                    CoverageSeverity.ERROR,
                    "Population-base row has unexpected value kind.",
                    study_area_id,
                    "Regenerate population-base rows with explicit count/rate semantics.",
                    record=record,
                )
            )
        if not record.source_record_ids or not record.source_file_hash or not record.provenance:
            issues.append(
                _issue(
                    CoverageIssueType.MISSING_SOURCE_TRACEABILITY,
                    CoverageSeverity.ERROR,
                    "Population-base row lacks source traceability.",
                    study_area_id,
                    "Regenerate population-base rows with source record IDs, hashes, and provenance.",
                    coverage_key=_coverage_key_label(_coverage_key(record)),
                    record=record,
                )
            )

    for key, rows in key_to_records.items():
        if len(rows) <= 1:
            continue
        values = {_value(row) for row in rows}
        source_ids = sorted({source_id for row in rows for source_id in row.source_record_ids})
        hashes = sorted({row.source_file_hash for row in rows if row.source_file_hash})
        label = _coverage_key_label(key)
        if len(values) > 1:
            issues.append(
                _issue(
                    CoverageIssueType.CONFLICTING_VALUES_FOR_SAME_KEY,
                    CoverageSeverity.ERROR,
                    "Population-base rows share a key but have conflicting values.",
                    study_area_id,
                    "Review source rows before modeling; do not aggregate conflicting values.",
                    coverage_key=label,
                    target_prefecture=key[0],
                    year=key[4],
                    age_group=key[5],
                    value_kind=PopulationValueKind(key[6]),
                    geography_grain=GeographyGrain(key[2]),
                    population_base_role=PopulationBaseRole(key[3]),
                    source_record_ids=source_ids,
                    source_file_hashes=hashes,
                )
            )
        else:
            issues.append(
                _issue(
                    CoverageIssueType.DUPLICATE_POPULATION_BASE_KEY,
                    CoverageSeverity.WARNING,
                    "Population-base rows share the same deterministic key.",
                    study_area_id,
                    "Review duplicate source rows before demand-feature engineering.",
                    coverage_key=label,
                    target_prefecture=key[0],
                    year=key[4],
                    age_group=key[5],
                    value_kind=PopulationValueKind(key[6]),
                    geography_grain=GeographyGrain(key[2]),
                    population_base_role=PopulationBaseRole(key[3]),
                    source_record_ids=source_ids,
                    source_file_hashes=hashes,
                )
            )

    combo_kinds: defaultdict[tuple[Any, ...], set[str]] = defaultdict(set)
    for record in records:
        combo = (
            record.matched_target_prefecture,
            str(record.year) if record.year is not None else None,
            record.age_group,
            record.geography_grain.value,
            record.population_base_role.value,
        )
        combo_kinds[combo].add(record.value_kind.value)
    for combo, kinds in combo_kinds.items():
        prefecture, year, age_group, grain, role = combo
        if "count" in kinds and "rate" not in kinds:
            issues.append(
                _issue(
                    CoverageIssueType.MISSING_RATE_FOR_COUNT_COMBINATION,
                    CoverageSeverity.INFO,
                    "A count combination has no matching rate row.",
                    study_area_id,
                    "Keep count/rate semantics separate; review only if rate coverage is required later.",
                    target_prefecture=prefecture,
                    year=year,
                    age_group=age_group,
                    geography_grain=GeographyGrain(grain),
                    population_base_role=PopulationBaseRole(role),
                )
            )
        if "rate" in kinds and "count" not in kinds:
            issues.append(
                _issue(
                    CoverageIssueType.MISSING_COUNT_FOR_RATE_COMBINATION,
                    CoverageSeverity.WARNING,
                    "A rate combination has no matching count row.",
                    study_area_id,
                    "Review count coverage before demand-feature engineering; do not convert rates to counts.",
                    target_prefecture=prefecture,
                    year=year,
                    age_group=age_group,
                    geography_grain=GeographyGrain(grain),
                    population_base_role=PopulationBaseRole(role),
                )
            )
    return issues


def _write_matrix_csv(path: Path, rows: list[PopulationBaseCoverageMatrix]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "matrix_id",
        "coverage_axis",
        "target_prefecture",
        "year",
        "age_group",
        "value_kind",
        "geography_grain",
        "population_base_role",
        "row_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = row.model_dump(mode="json")
            writer.writerow({key: payload.get(key) for key in fieldnames})


def run_population_base_coverage_qa(
    repo_root: str | Path = ".",
    config_path: str | Path = "configs/study_area_tokyo_aichi_osaka.yaml",
    mapping_config_path: str | Path = "configs/source_mappings/population_workbooks.yaml",
) -> PopulationBaseCoverageQAResult:
    """Run pre-demand coverage QA over population-base rows."""

    root = Path(repo_root).resolve()
    study_area, config = load_study_area_config(root / config_path)
    outputs = config["outputs"]
    population_base_path = root / outputs["population_base"]
    payloads = _iter_jsonl(population_base_path)
    records = [StudyAreaPopulationBaseRecord.model_validate(payload) for payload in payloads]
    output_paths = {
        "matrix_json": outputs["population_base_coverage_matrix_json"],
        "matrix_csv": outputs["population_base_coverage_matrix_csv"],
        "issues": outputs["population_base_coverage_issues"],
        "manifest": outputs["population_base_coverage_manifest"],
        "summary": outputs["population_base_coverage_summary"],
    }
    if not records:
        return PopulationBaseCoverageQAResult(input_found=False, output_paths=output_paths)

    expected_years, expected_age_groups, expectation_sources = _load_mapping_expectations(
        root / mapping_config_path
    )
    if not expected_years:
        expected_years = sorted({str(row.year) for row in records if row.year not in (None, "")})
        expectation_sources.append({"source": "observed_population_base_rows", "field": "year"})
    if not expected_age_groups:
        expected_age_groups = sorted({row.age_group for row in records if row.age_group})
        expectation_sources.append({"source": "observed_population_base_rows", "field": "age_group"})

    matrix_rows = _build_matrices(study_area.study_area_id, records)
    issues = _detect_issues(
        study_area.study_area_id,
        study_area.target_prefectures,
        records,
        expected_years,
        expected_age_groups,
    )
    issue_counts = Counter(issue.issue_type.value for issue in issues)
    severity_counts = Counter(issue.severity.value for issue in issues)
    matrix_payload = {
        "study_area_id": study_area.study_area_id,
        "target_prefectures": study_area.target_prefectures,
        "expected_years": expected_years,
        "expected_age_groups": expected_age_groups,
        "matrix_rows": [row.model_dump(mode="json") for row in matrix_rows],
    }
    manifest = PopulationBaseCoverageManifest(
        run_id=f"population_base_coverage:{study_area.study_area_id}:{now_utc().isoformat()}",
        generated_at=now_utc(),
        study_area_id=study_area.study_area_id,
        target_prefectures=study_area.target_prefectures,
        input_files=[
            Path(outputs["population_base"]),
            Path(outputs["population_base_municipality"]),
            Path(outputs["population_base_prefecture_total"]),
            Path(outputs["geography_grain_records"]),
            Path(outputs["population_base_manifest"]),
            Path(outputs["population_base_report_json"]),
            Path(mapping_config_path),
        ],
        output_files=[
            Path(outputs["population_base_coverage_matrix_json"]),
            Path(outputs["population_base_coverage_matrix_csv"]),
            Path(outputs["population_base_coverage_issues"]),
            Path(outputs["population_base_coverage_manifest"]),
            Path(outputs["population_base_coverage_summary"]),
        ],
        record_counts={
            "population_base_records_read": len(records),
            "coverage_matrix_rows_written": len(matrix_rows),
            "coverage_issues_written": len(issues),
            "model_blocking_errors": severity_counts[CoverageSeverity.ERROR.value],
        },
        issue_counts=dict(issue_counts),
        expectation_sources=expectation_sources,
        warnings=[],
    )
    write_json(root / outputs["population_base_coverage_matrix_json"], matrix_payload)
    _write_matrix_csv(root / outputs["population_base_coverage_matrix_csv"], matrix_rows)
    write_jsonl(root / outputs["population_base_coverage_issues"], issues)
    write_json(root / outputs["population_base_coverage_manifest"], manifest)
    summary_path = root / outputs["population_base_coverage_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        "\n".join(
            [
                "# Population Base Coverage QA",
                "",
                f"- Study area: {study_area.study_area_id}",
                f"- Population-base rows read: {len(records)}",
                f"- Coverage matrix rows: {len(matrix_rows)}",
                f"- Coverage issues: {len(issues)}",
                f"- Model-blocking errors: {severity_counts[CoverageSeverity.ERROR.value]}",
                f"- Expected years: {', '.join(expected_years)}",
                f"- Expected age groups: {', '.join(expected_age_groups)}",
                "- This QA does not calculate demand or impute missing rows.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return PopulationBaseCoverageQAResult(
        input_found=population_base_path.exists(),
        records_read=len(records),
        matrix_rows_written=len(matrix_rows),
        issue_count=len(issues),
        model_blocking_error_count=severity_counts[CoverageSeverity.ERROR.value],
        duplicate_key_count=issue_counts[CoverageIssueType.DUPLICATE_POPULATION_BASE_KEY.value],
        conflicting_value_count=issue_counts[CoverageIssueType.CONFLICTING_VALUES_FOR_SAME_KEY.value],
        warnings=[],
        output_paths=output_paths,
    )
