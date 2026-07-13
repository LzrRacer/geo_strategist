"""S2 exact-site candidate generation.

Generates real candidate-site records for the site-selection pipeline. It
never invents an address, coordinate, or parcel ID: concrete site geometry
comes only from `healthcare_facility_connector` (a real geocoded facility
pull) or stays `not_available`. Financial/land figures come from
`financial_workbook_connector`. Demand-only "build" hypotheses stay at
municipality level with no address/parcel claim, matching this repo's prior
policy that unverified exact-site geometry cannot be claimed.

Three generation modes are supported:

- `public_dataset_mode`: derive candidates from the real local datasets
  already registered in S1 (existing-facility financial anchors + top
  demand municipalities). There is no live public API access in this
  environment, so this mode operates on the locally cached, source-cited
  datasets rather than a live fetch.
- `manual_source_mode`: ingest a manually curated CSV/JSON file of candidate
  sites with source refs supplied by the caller.
- `database_mode`: query a configured local database table. No database is
  configured in this environment, so this mode fails closed with an
  explicit issue.
"""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from geo_strategist.data_sources.connectors import (
    financial_workbook_connector,
    healthcare_facility_connector,
    population_demand_connector,
)
from geo_strategist.data.facility_taxonomy import resolve_workbook_hospitals, write_entity_resolution_report
from geo_strategist.data_sources.evidence_grade import worst_grade
from geo_strategist.data_sources.geo_utils import haversine_distance_km


OUTPUT_ROOT = Path(".runs/experiments/s2_candidate_site_generation")
NEARBY_RADIUS_KM = 3.0
DEFAULT_TOP_N_BUILD_MUNICIPALITIES = 10


@dataclass(frozen=True)
class S2Result:
    run_id: str
    output_dir: Path
    candidate_count: int
    output_paths: dict[str, str]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, canonical)}"


def _match_facility_by_resolution(
    financial: dict[str, Any],
    resolution_by_master_id: dict[str, dict[str, Any]],
    facility_by_supply_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    resolution = resolution_by_master_id.get(str(financial.get("master_id")))
    if not resolution:
        return None, None
    supply_id = resolution.get("matched_supply_record_id")
    return (facility_by_supply_id.get(str(supply_id)) if supply_id else None), resolution


def _nearby_facilities(
    latitude: float | None,
    longitude: float | None,
    facility_records: list[dict[str, Any]],
    *,
    exclude_facility_name: str | None,
    radius_km: float = NEARBY_RADIUS_KM,
) -> list[dict[str, Any]]:
    if latitude is None or longitude is None:
        return []
    nearby = []
    for record in facility_records:
        if record.get("facility_name") == exclude_facility_name:
            continue
        lat, lon = record.get("latitude"), record.get("longitude")
        if lat is None or lon is None:
            continue
        distance = haversine_distance_km(latitude, longitude, lat, lon)
        if distance <= radius_km:
            nearby.append({
                "facility_name": record.get("facility_name"),
                "distance_km": round(distance, 3),
                "is_hospital_like": bool(record.get("is_hospital_like")),
            })
    return sorted(nearby, key=lambda row: row["distance_km"])


def _infer_action_type(financial_row: dict[str, Any], nearby_hospital_count: int) -> str:
    if financial_row.get("payback_flag") == "Payback厳しい" and nearby_hospital_count >= 3:
        return "consolidate"
    if nearby_hospital_count >= 5:
        return "relocate"
    return "rebuild"


def _existing_facility_candidates(
    repo_root: Path,
    financial_records: list[dict[str, Any]],
    facility_records: list[dict[str, Any]],
    entity_resolution_rows: list[dict[str, Any]],
    generated_at: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    resolution_by_master_id = {str(row.get("master_id")): row for row in entity_resolution_rows if row.get("master_id")}
    facility_by_supply_id = {str(row.get("supply_record_id")): row for row in facility_records if row.get("supply_record_id")}
    for financial in financial_records:
        hospital_name = financial.get("hospital_name")
        matched, resolution = _match_facility_by_resolution(financial, resolution_by_master_id, facility_by_supply_id)
        source_refs = [
            {
                "source_artifact": financial["source_artifact"],
                "source_record_id": financial["source_record_id"],
                "source_field": "hospital_name,prefecture,financial_model",
                "source_url": financial.get("source_url"),
            }
        ]
        blocking_issues: list[str] = []
        component_grades = [financial.get("financial_evidence_grade", "model_estimate")]
        if financial.get("beds_evidence_grade"):
            component_grades.append(financial["beds_evidence_grade"])

        if matched:
            address = matched.get("address")
            latitude = matched.get("latitude")
            longitude = matched.get("longitude")
            municipality = matched.get("municipality")
            address_evidence_grade = "verified_source"
            source_refs.append({
                "source_artifact": matched["source_artifact"],
                "source_record_id": matched.get("supply_record_id", hospital_name),
                "source_field": "address,latitude,longitude,municipality",
                "source_url": matched.get("source_url"),
            })
            component_grades.append("verified_source")
        else:
            address = None
            latitude = None
            longitude = None
            municipality = None
            address_evidence_grade = "unverified_candidate"
            blocking_issues.append("address_not_matched_to_geocoded_facility_record")
            component_grades.append("unverified_candidate")
        if resolution:
            source_refs.append({
                "source_artifact": "s2_hospital_entity_resolution_report.json",
                "source_record_id": resolution["resolution_id"],
                "source_field": "match_confidence,match_method,human_review_required",
                "source_url": None,
            })

        nearby = _nearby_facilities(latitude, longitude, facility_records, exclude_facility_name=hospital_name)
        nearby_hospital_count = sum(1 for row in nearby if row["is_hospital_like"])
        action_type = _infer_action_type(financial, nearby_hospital_count)

        blocking_issues.append("parcel_id_not_available_no_licensed_registry")
        blocking_issues.append("zoning_not_available_no_dataset_configured")
        blocking_issues.append("nearest_transport_access_not_available_no_dataset_configured")
        component_grades.append("unverified_candidate")  # parcel/zoning/transport gap

        candidate = {
            "candidate_site_id": _stable_id("candidate_site", {"master_id": financial["master_id"], "action": "existing_facility"}),
            "action_type": action_type,
            "anchor_master_id": financial["master_id"],
            "facility_name": hospital_name,
            "prefecture": financial.get("prefecture"),
            "municipality": municipality,
            "address": address,
            "latitude": latitude,
            "longitude": longitude,
            "parcel_id": None,
            "parcel_id_not_available_reason": "no_licensed_parcel_lot_number_registry_configured",
            "site_area_m2": financial.get("land_area_sqm"),
            "zoning_or_land_use": None,
            "nearest_transport_access": None,
            "catchment_area_definition": "municipality_administrative_boundary_proxy",
            "nearby_existing_facilities": nearby,
            "entity_resolution": {
                "matched_supply_record_id": resolution.get("matched_supply_record_id") if resolution else None,
                "match_confidence": resolution.get("match_confidence") if resolution else 0.0,
                "match_method": resolution.get("match_method") if resolution else "no_resolution_record",
                "human_review_required": resolution.get("human_review_required") if resolution else True,
                "taxonomy": resolution.get("taxonomy") if resolution else None,
            },
            "source_refs": source_refs,
            "site_evidence_grade": worst_grade(component_grades),
            "address_evidence_grade": address_evidence_grade,
            "blocking_issues": blocking_issues,
            "generation_mode": "public_dataset_mode",
            "generated_at": generated_at,
        }
        candidates.append(candidate)
    return candidates


def _build_candidates_by_demand(
    population_records: list[dict[str, Any]],
    existing_municipalities: set[str],
    top_n: int,
    generated_at: str,
) -> list[dict[str, Any]]:
    ranked = sorted(
        (row for row in population_records if row.get("municipality") not in existing_municipalities),
        key=lambda row: (row.get("elderly_ratio_65_plus") or 0.0),
        reverse=True,
    )[:top_n]
    candidates: list[dict[str, Any]] = []
    for row in ranked:
        candidate = {
            "candidate_site_id": _stable_id("candidate_site", {"municipality": row["municipality"], "action": "build"}),
            "action_type": "build",
            "anchor_master_id": None,
            "facility_name": None,
            "prefecture": row.get("prefecture"),
            "municipality": row.get("municipality"),
            "address": None,
            "latitude": None,
            "longitude": None,
            "parcel_id": None,
            "parcel_id_not_available_reason": "no_licensed_parcel_lot_number_registry_configured",
            "site_area_m2": None,
            "zoning_or_land_use": None,
            "nearest_transport_access": None,
            "catchment_area_definition": "municipality_administrative_boundary_proxy",
            "nearby_existing_facilities": [],
            "source_refs": [{
                "source_artifact": row["source_artifact"],
                "source_record_id": row["source_record_ids"][0] if row.get("source_record_ids") else row["municipality"],
                "source_field": "population_total,population_age_65_plus,elderly_ratio_65_plus",
                "source_url": None,
            }],
            "site_evidence_grade": "unverified_candidate",
            "address_evidence_grade": "unverified_candidate",
            "blocking_issues": [
                "exact_site_not_identified_municipality_level_only",
                "parcel_id_not_available_no_licensed_registry",
                "zoning_not_available_no_dataset_configured",
                "nearest_transport_access_not_available_no_dataset_configured",
            ],
            "generation_mode": "public_dataset_mode",
            "generated_at": generated_at,
        }
        candidates.append(candidate)
    return candidates


def _manual_source_candidates(manual_source_path: Path | None, generated_at: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if manual_source_path is None:
        return [], []
    if not manual_source_path.exists():
        return [], [{
            "issue_code": "manual_source_file_missing",
            "severity": "error",
            "message": f"Manual candidate source file not found: {manual_source_path}",
        }]
    rows: list[dict[str, Any]]
    if manual_source_path.suffix.lower() == ".json":
        rows = json.loads(manual_source_path.read_text(encoding="utf-8"))
    elif manual_source_path.suffix.lower() == ".csv":
        with manual_source_path.open("r", encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    else:
        return [], [{
            "issue_code": "manual_source_file_unsupported_extension",
            "severity": "error",
            "message": f"Only .csv and .json manual candidate files are supported: {manual_source_path}",
        }]

    candidates: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not row.get("source_artifact") or not row.get("source_record_id"):
            issues.append({
                "issue_code": "manual_candidate_missing_source_refs",
                "severity": "warning",
                "row_index": index,
            })
            continue
        candidates.append({
            "candidate_site_id": row.get("candidate_site_id") or _stable_id("candidate_site", {"manual_row": index, "path": str(manual_source_path)}),
            "action_type": row.get("action_type", "defer"),
            "anchor_master_id": row.get("anchor_master_id"),
            "facility_name": row.get("facility_name"),
            "prefecture": row.get("prefecture"),
            "municipality": row.get("municipality"),
            "address": row.get("address"),
            "latitude": float(row["latitude"]) if row.get("latitude") not in (None, "") else None,
            "longitude": float(row["longitude"]) if row.get("longitude") not in (None, "") else None,
            "parcel_id": row.get("parcel_id") or None,
            "parcel_id_not_available_reason": row.get("parcel_id_not_available_reason") or ("manual_source_did_not_supply_parcel_id" if not row.get("parcel_id") else None),
            "site_area_m2": float(row["site_area_m2"]) if row.get("site_area_m2") not in (None, "") else None,
            "zoning_or_land_use": row.get("zoning_or_land_use") or None,
            "nearest_transport_access": row.get("nearest_transport_access") or None,
            "catchment_area_definition": row.get("catchment_area_definition") or "manual_source_definition_not_supplied",
            "nearby_existing_facilities": [],
            "source_refs": [{
                "source_artifact": row["source_artifact"],
                "source_record_id": row["source_record_id"],
                "source_field": row.get("source_field", "manual_curated_candidate"),
                "source_url": row.get("source_url"),
            }],
            "site_evidence_grade": row.get("site_evidence_grade", "third_party_estimate"),
            "address_evidence_grade": row.get("address_evidence_grade") or (row.get("site_evidence_grade", "third_party_estimate") if row.get("address") else "unverified_candidate"),
            "blocking_issues": json.loads(row["blocking_issues"]) if isinstance(row.get("blocking_issues"), str) and row.get("blocking_issues", "").startswith("[") else (row.get("blocking_issues") or []),
            "generation_mode": "manual_source_mode",
            "generated_at": generated_at,
        })
    return candidates, issues


def _database_mode_candidates() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # No local database is configured in this environment; fail closed.
    return [], [{
        "issue_code": "database_mode_not_configured",
        "severity": "info",
        "message": "No local database table is configured for candidate generation in this environment.",
    }]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def run_s2_candidate_site_generation(
    repo_root: str | Path = ".",
    *,
    mode: str = "public_dataset_mode",
    top_n_build_municipalities: int = DEFAULT_TOP_N_BUILD_MUNICIPALITIES,
    manual_source_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> S2Result:
    repo_root = Path(repo_root).resolve()
    run_id = str(uuid.uuid4())
    out_root = Path(output_root) if output_root else repo_root / OUTPUT_ROOT
    if not out_root.is_absolute():
        out_root = repo_root / out_root
    out_dir = out_root / run_id
    generated_at = _now_iso()

    issues: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    entity_resolution_rows: list[dict[str, Any]] = []

    if mode == "public_dataset_mode":
        financial_records, financial_issues = financial_workbook_connector.load_records(repo_root)
        facility_records, facility_issues = healthcare_facility_connector.load_records(repo_root)
        population_records, population_issues = population_demand_connector.load_records(repo_root)
        issues.extend(financial_issues)
        issues.extend(facility_issues)
        issues.extend(population_issues)

        entity_resolution_rows = resolve_workbook_hospitals(financial_records, facility_records)
        existing_candidates = _existing_facility_candidates(
            repo_root, financial_records, facility_records, entity_resolution_rows, generated_at
        )
        existing_municipalities = {row["municipality"] for row in existing_candidates if row.get("municipality")}
        build_candidates = _build_candidates_by_demand(population_records, existing_municipalities, top_n_build_municipalities, generated_at)
        candidates = existing_candidates + build_candidates
    elif mode == "manual_source_mode":
        manual_path = Path(manual_source_path) if manual_source_path else None
        if manual_path and not manual_path.is_absolute():
            manual_path = repo_root / manual_path
        candidates, manual_issues = _manual_source_candidates(manual_path, generated_at)
        issues.extend(manual_issues)
    elif mode == "database_mode":
        candidates, db_issues = _database_mode_candidates()
        issues.extend(db_issues)
    else:
        issues.append({"issue_code": "unknown_generation_mode", "severity": "error", "mode": mode})

    output_paths = {
        "manifest": str(out_dir / "s2_manifest.json"),
        "candidate_site_records": str(out_dir / "s2_candidate_site_records.jsonl"),
        "issues": str(out_dir / "s2_issues.jsonl"),
        "entity_resolution_report": str(out_dir / "s2_hospital_entity_resolution_report.json"),
        "report_json": str(out_dir / "s2_report.json"),
        "report_markdown": str(out_dir / "s2_report.md"),
    }
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "mode": mode,
        "candidate_count": len(candidates),
        "candidates_with_verified_address": sum(1 for row in candidates if row.get("address")),
        "candidates_without_address": sum(1 for row in candidates if not row.get("address")),
        "action_type_counts": {
            action: sum(1 for row in candidates if row["action_type"] == action)
            for action in sorted({row["action_type"] for row in candidates})
        },
        "issue_count": len(issues),
        "entity_resolution_record_count": len(entity_resolution_rows),
        "entity_resolution_human_review_required_count": sum(1 for row in entity_resolution_rows if row.get("human_review_required")),
    }
    manifest = {
        "run_id": run_id,
        "stage": "s2_candidate_site_generation",
        "mode": mode,
        "generated_at": generated_at,
        "output_artifacts": {key: str(Path(path).relative_to(repo_root)) for key, path in output_paths.items()},
    }

    _write_json(Path(output_paths["manifest"]), manifest)
    _write_jsonl(Path(output_paths["candidate_site_records"]), candidates)
    _write_jsonl(Path(output_paths["issues"]), issues)
    write_entity_resolution_report(Path(output_paths["entity_resolution_report"]), entity_resolution_rows)
    _write_json(Path(output_paths["report_json"]), report)
    Path(output_paths["report_markdown"]).write_text(
        "\n".join([
            "# S2 Candidate Site Generation",
            "",
            f"Run ID: `{run_id}`",
            f"Mode: `{mode}`",
            f"Candidates generated: {len(candidates)}",
            f"Candidates with a source-traceable address: {report['candidates_with_verified_address']}",
            f"Candidates without an address (municipality-level only): {report['candidates_without_address']}",
            "",
            "No address, coordinate, or parcel ID was invented. Concrete site",
            "geometry comes only from the S1-registered geocoded facility source;",
            "everything else stays `not_available` with an explicit blocking issue.",
            "",
        ]),
        encoding="utf-8",
    )

    return S2Result(
        run_id=run_id,
        output_dir=out_dir,
        candidate_count=len(candidates),
        output_paths=output_paths,
    )
