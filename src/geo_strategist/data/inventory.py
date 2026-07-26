"""Local data inventory utilities.

The inventory records file metadata only. It never reads `.env` files and never
prints or stores workbook cell contents.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from geo_strategist.data.provenance import ProvenanceRecord, SourceKind, SourceRef


CANONICAL_MANUAL_ROOT = Path(".data/manual")
LEGACY_MANUAL_ROOT = Path("data/manual")
DEFAULT_INVENTORY_JSON = Path(".cache/inventory/local_data_inventory.json")
DEFAULT_PROVENANCE_JSONL = Path(".cache/inventory/local_data_provenance.jsonl")

WORKBOOK_SUFFIXES = {".xlsx", ".xlsm", ".xls"}
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".cache",
    ".data/api_raw",
    ".runs",
    ".scratch",
    ".junk",
    "__pycache__",
}


class InventoryFile(BaseModel):
    """Metadata for one local data file."""

    model_config = ConfigDict(extra="forbid")

    path: str
    source_root: str
    category: str
    file_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    modified_at: datetime


class InventoryResult(BaseModel):
    """Result of scanning local manual data roots."""

    model_config = ConfigDict(extra="forbid")

    scanned_root: str | None
    files: list[InventoryFile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    output_path: str | None = None
    provenance_path: str | None = None


def sha256_file(path: str | Path) -> str:
    """Return a SHA256 digest for a file without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_file_type(path: str | Path) -> str:
    """Infer a conservative file type from the extension."""

    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return "excel_workbook"
    if suffix == ".xls":
        return "legacy_excel_workbook"
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return "json"
    return suffix.removeprefix(".") or "unknown"


def categorize_file(path: Path) -> str:
    """Classify known local data files by path shape, not contents."""

    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "hospital_cf_workbook" in parts and path.suffix.lower() in WORKBOOK_SUFFIXES:
        return "hospital_cashflow_workbook"
    if "population_data" in parts and path.suffix.lower() in WORKBOOK_SUFFIXES:
        return "population_workbook"
    if path.suffix.lower() in WORKBOOK_SUFFIXES:
        return "workbook"
    return "manual_file"


def _select_manual_root(repo_root: Path, include_legacy_fallback: bool) -> tuple[Path | None, list[str]]:
    canonical = repo_root / CANONICAL_MANUAL_ROOT
    legacy = repo_root / LEGACY_MANUAL_ROOT
    warnings: list[str] = []

    if canonical.exists():
        return canonical, warnings

    if include_legacy_fallback and legacy.exists():
        warnings.append(
            "Using legacy data/manual fallback; canonical .data/manual was not found."
        )
        return legacy, warnings

    return None, ["No local manual data root found."]


def _should_skip(path: Path) -> bool:
    if path.name == ".env":
        return True
    parts = set(path.parts)
    return any(name in parts for name in SKIP_DIR_NAMES)


def _iter_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _should_skip(current / dirname)
        ]
        for filename in filenames:
            path = current / filename
            if not _should_skip(path):
                files.append(path)
    return sorted(files)


def _source_id_for(relative_path: Path, digest: str) -> str:
    safe_path = "_".join(part.replace(" ", "_") for part in relative_path.parts)
    return f"local_manual:{safe_path}:{digest[:12]}"


def provenance_for_inventory_file(file: InventoryFile) -> ProvenanceRecord:
    """Create a provenance record for one inventory entry."""

    source_ref = SourceRef(
        source_id=_source_id_for(Path(file.path), file.sha256),
        kind=SourceKind.MANUAL_FILE,
        path=Path(file.path),
        sha256=file.sha256,
        notes="Local manual data file inventoried by metadata only.",
    )
    return ProvenanceRecord(
        provenance_id=f"inventory:{source_ref.source_id}",
        source_ref=source_ref,
        claim="Local manual data file exists and was hashed.",
        locator=file.path,
    )


def inventory_local_data(
    repo_root: str | Path = ".",
    output_path: str | Path = DEFAULT_INVENTORY_JSON,
    provenance_path: str | Path | None = DEFAULT_PROVENANCE_JSONL,
    include_legacy_fallback: bool = True,
) -> InventoryResult:
    """Scan local manual data files and write an inventory JSON artifact."""

    root = Path(repo_root).resolve()
    manual_root, warnings = _select_manual_root(root, include_legacy_fallback)
    files: list[InventoryFile] = []

    if manual_root is not None:
        for path in _iter_files(manual_root):
            stat = path.stat()
            relative = path.relative_to(root)
            files.append(
                InventoryFile(
                    path=str(relative),
                    source_root=str(manual_root.relative_to(root)),
                    category=categorize_file(relative),
                    file_type=infer_file_type(path),
                    size_bytes=stat.st_size,
                    sha256=sha256_file(path),
                    modified_at=datetime.fromtimestamp(stat.st_mtime, UTC),
                )
            )

    result = InventoryResult(
        scanned_root=str(manual_root.relative_to(root)) if manual_root else None,
        files=files,
        warnings=warnings,
    )

    output = root / output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    result.output_path = str(output.relative_to(root))
    output.write_text(
        json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if provenance_path is not None:
        provenance_output = root / provenance_path
        provenance_output.parent.mkdir(parents=True, exist_ok=True)
        records = [provenance_for_inventory_file(file) for file in files]
        provenance_output.write_text(
            "\n".join(record.model_dump_json() for record in records)
            + ("\n" if records else ""),
            encoding="utf-8",
        )
        result.provenance_path = str(provenance_output.relative_to(root))

        output.write_text(
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return result
