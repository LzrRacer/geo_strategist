"""Municipality master keys for study-area joins."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/study_area_tokyo_aichi_osaka.yaml")
DEFAULT_OUTPUT = Path(".data/interim/study_area/tokyo_aichi_osaka/municipality_master_records.jsonl")


def normalize_municipality_name(name: str | None) -> str:
    text = str(name or "").strip()
    return text.replace("　", "").replace(" ", "")


def stable_municipality_key(prefecture: str, municipality: str) -> str:
    return f"{normalize_municipality_name(prefecture)}::{normalize_municipality_name(municipality)}"


def build_municipality_master(
    repo_root: str | Path = ".",
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> list[dict[str, Any]]:
    root = Path(repo_root).resolve()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    study_area = cfg.get("study_area", cfg)
    outputs = cfg.get("outputs", {})
    target_prefectures = study_area.get("target_prefectures", cfg.get("target_prefectures", []))
    municipalities_by_prefecture: dict[str, set[str]] = {
        pref: set(study_area.get("municipalities_by_prefecture", {}).get(pref, []))
        for pref in target_prefectures
    }
    for rel in (
        outputs.get("population_base_municipality"),
        outputs.get("study_area_geography_keys"),
        outputs.get("municipality_feature_base"),
    ):
        if not rel:
            continue
        path = root / rel
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pref = row.get("prefecture")
            muni = row.get("municipality")
            if pref in municipalities_by_prefecture and muni:
                municipalities_by_prefecture[pref].add(muni)
    rows: list[dict[str, Any]] = []
    for pref in target_prefectures:
        municipalities = sorted(municipalities_by_prefecture.get(pref, []))
        for muni in municipalities:
            key = stable_municipality_key(pref, muni)
            rows.append({
                "municipality_master_id": f"muni_master:{uuid.uuid5(uuid.NAMESPACE_URL, key)}",
                "national_local_government_code": None,
                "prefecture": pref,
                "municipality": muni,
                "stable_join_key": key,
                "normalized_prefecture": normalize_municipality_name(pref),
                "normalized_municipality": normalize_municipality_name(muni),
                "municipality_name_variants": sorted({muni, normalize_municipality_name(muni)}),
                "old_municipality_names": [],
                "is_tokyo_ward": pref == "東京都" and muni.endswith("区"),
                "designated_city": muni.split("市")[0] + "市" if "市" in muni and muni.endswith("区") else None,
                "provenance": {
                    "source_artifact": str(DEFAULT_CONFIG),
                    "source_field": "target_prefectures,municipalities_by_prefecture",
                },
            })
    out = Path(output_path)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
    return rows
