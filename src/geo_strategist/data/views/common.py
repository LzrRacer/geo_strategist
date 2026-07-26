"""Shared helpers for analysis-ready source views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from geo_strategist.data.normalization import NormalizedRecord


def read_normalized_jsonl(path: str | Path) -> list[NormalizedRecord]:
    """Read normalized records from JSONL."""

    records: list[NormalizedRecord] = []
    file_path = Path(path)
    if not file_path.exists():
        return records
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(NormalizedRecord.model_validate_json(line))
    return records


def write_jsonl(path: str | Path, rows: Iterable[object]) -> None:
    """Write Pydantic rows or JSON-serializable rows as JSONL."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in rows:
        if hasattr(row, "model_dump_json"):
            payload = row.model_dump(mode="json", exclude_none=False, exclude_defaults=False)
            lines.append(json.dumps(payload, ensure_ascii=False))
        else:
            lines.append(json.dumps(row, ensure_ascii=False))
    file_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_json(path: str | Path, payload: object) -> None:
    """Write JSON with stable formatting."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_label(labels: dict[str, str], keywords: tuple[str, ...]) -> str | None:
    """Return first label value whose key contains one of the keywords."""

    for key, value in labels.items():
        lowered = key.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return value
    return None
