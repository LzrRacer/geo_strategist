"""Validation for deterministic normalized output artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from geo_strategist.data.normalization import NormalizationManifest, NormalizedRecord, SourceTable


class NormalizedValidationSummary(BaseModel):
    """Validation summary for normalized outputs."""

    model_config = ConfigDict(extra="forbid")

    checked_outputs: list[str] = Field(default_factory=list)
    missing_outputs: list[str] = Field(default_factory=list)
    record_count: int = 0
    source_table_count: int = 0
    unresolved_mapping_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return whether validation passed."""

        return not self.errors


def _load_normalization_config() -> dict:
    with Path("configs/normalization.yaml").open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _iter_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return records


def _validate_manifest(path: Path, summary: NormalizedValidationSummary) -> None:
    manifest = NormalizationManifest.model_validate_json(path.read_text(encoding="utf-8"))
    summary.checked_outputs.append(str(path))
    summary.unresolved_mapping_count += manifest.unresolved_mapping_count
    summary.warnings.extend(manifest.warnings)


def _validate_records(path: Path, summary: NormalizedValidationSummary) -> None:
    for payload in _iter_jsonl(path):
        record = NormalizedRecord.model_validate(payload)
        summary.record_count += 1
        if record.value_type.value == "number" and not record.provenance:
            summary.errors.append(f"{path}: numeric record lacks provenance")
    summary.checked_outputs.append(str(path))


def _validate_source_tables(path: Path, summary: NormalizedValidationSummary) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload:
        SourceTable.model_validate(item)
        summary.source_table_count += 1
    summary.checked_outputs.append(str(path))


def validate_normalized_outputs(
    repo_root: str | Path = ".",
    require_outputs: bool = False,
) -> NormalizedValidationSummary:
    """Validate normalized records, source tables, and manifests when present."""

    root = Path(repo_root).resolve()
    config = _load_normalization_config()
    paths = {
        "hospital_records": Path(config["paths"]["hospital_records"]),
        "population_records": Path(config["paths"]["population_records"]),
        "hospital_manifest": Path(config["paths"]["hospital_manifest"]),
        "population_manifest": Path(config["paths"]["population_manifest"]),
        "hospital_source_tables": Path(
            ".data/interim/normalized/hospital_workbook/source_tables.json"
        ),
        "population_source_tables": Path(".data/interim/normalized/population/source_tables.json"),
    }
    summary = NormalizedValidationSummary()

    for name, relative_path in paths.items():
        path = root / relative_path
        if not path.exists():
            summary.missing_outputs.append(str(relative_path))
            if require_outputs:
                summary.errors.append(f"Missing required normalized output: {relative_path}")
            continue
        try:
            if name.endswith("_records"):
                _validate_records(path, summary)
            elif name.endswith("_manifest"):
                _validate_manifest(path, summary)
            elif name.endswith("_source_tables"):
                _validate_source_tables(path, summary)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            summary.errors.append(f"{relative_path}: {exc}")

    return summary
