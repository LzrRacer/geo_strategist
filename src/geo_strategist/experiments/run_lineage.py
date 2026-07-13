"""Run lineage registry and validation/report index utilities."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_ROOT = Path(".runs/registry")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stage_from_path(path: Path) -> str:
    parts = path.parts
    if "experiments" in parts:
        idx = parts.index("experiments")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.name


def build_run_lineage_index(repo_root: str | Path = ".", *, canonical_stage: str | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    runs_root = root / ".runs"
    records: list[dict[str, Any]] = []
    if runs_root.exists():
        for path in sorted(runs_root.rglob("*")):
            if path.name not in {
                "s2_manifest.json", "s3_manifest.json", "s4_manifest.json", "s5_manifest.json",
                "s7_manifest.json", "e10a_manifest.json", "e11_manifest.json",
                "e13_manifest.json", "condition_judge_manifest.json",
                "e14_manifest.json", "provider_run_manifest.json",
            }:
                continue
            manifest = _read_json(path)
            run_dir = path.parent
            stage = manifest.get("stage") or _stage_from_path(path)
            output_paths = manifest.get("output_artifacts") or {}
            outputs = []
            for label, rel in output_paths.items():
                out_path = root / rel if not Path(rel).is_absolute() else Path(rel)
                outputs.append({"label": label, "path": str(out_path.relative_to(root)) if out_path.exists() else str(rel), "sha256": file_hash(out_path)})
            input_paths = []
            for key, value in manifest.items():
                if key.startswith("input_") and value:
                    p = root / str(value)
                    input_paths.append({"label": key, "path": str(value), "sha256": file_hash(p)})
            records.append({
                "lineage_record_id": f"run_lineage:{uuid.uuid5(uuid.NAMESPACE_URL, str(path.relative_to(root)))}",
                "run_id": manifest.get("run_id") or run_dir.name,
                "run_stage": stage,
                "run_dir": str(run_dir.relative_to(root)),
                "input_paths": input_paths,
                "output_paths": outputs,
                "provider_used": manifest.get("provider") or manifest.get("provider_family") or "deterministic_offline",
                "fallback_used": bool(manifest.get("fallback_used") or manifest.get("manual_transcript_ingestion_status")),
                "fallback_reason": manifest.get("fallback_reason"),
                "manual_transcript_ingestion_status": manifest.get("manual_transcript_ingestion_status", "not_applicable"),
                "blocking_issue_count": int(manifest.get("blocking_issue_count", 0) or 0),
                "warning_count": int(manifest.get("warning_count", 0) or 0),
                "timestamp": manifest.get("generated_at") or manifest.get("timestamp") or _now_iso(),
                "canonical_latest_successful": False,
            })
    latest = None
    filtered = [row for row in records if canonical_stage is None or row["run_stage"] == canonical_stage]
    if filtered:
        latest = sorted(filtered, key=lambda row: row["timestamp"], reverse=True)[0]
        latest["canonical_latest_successful"] = True

    registry_root = root / REGISTRY_ROOT
    index = {
        "generated_at": _now_iso(),
        "record_count": len(records),
        "records": records,
        "latest_successful_run": latest,
        "failed_automated_e4_vs_manual_fallback_note": (
            "Automated E4 provider timeouts and manually ingested fallback proposal sets are indexed separately when present; "
            "manual fallback must carry source/provenance metadata and does not certify recommendation quality."
        ),
    }
    _write_json(registry_root / "run_lineage_registry.json", index)
    _write_json(registry_root / "validation_report_index.json", index)
    _write_json(registry_root / "latest_successful_run.json", latest or {})
    return index
