"""Source-traceable concrete facility evidence loading."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from geo_strategist.agent.schemas import SourceEvidenceRef


HOSPITAL_FEATURES_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/hospital_features.jsonl")
HEALTHCARE_SUPPLY_RECORDS_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/healthcare_supply_records.jsonl")


@dataclass(frozen=True)
class FacilityEvidenceRecord:
    facility_record_id: str
    facility_name: str | None
    facility_address: str | None
    latitude: float | None
    longitude: float | None
    prefecture: str
    municipality: str
    source_artifact: str
    source_record_id: str
    source_fields: dict[str, str]
    evidence_status: str = "source_traceable"
    missing_fields: list[str] | None = None
    blocked_fields: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return f"{prefix}:{uuid.uuid5(uuid.NAMESPACE_URL, canonical)}"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _valid_lat_lon(latitude: float | None, longitude: float | None) -> bool:
    return latitude is not None and longitude is not None and -90 <= latitude <= 90 and -180 <= longitude <= 180


def _source_record_id(row: dict[str, Any]) -> str:
    return str(row.get("supply_record_id") or row.get("feature_id") or row.get("master_id") or row.get("source_record_id") or "")


def _record_is_usable(record: FacilityEvidenceRecord) -> bool:
    return (
        _present(record.facility_name)
        and _present(record.prefecture)
        and record.prefecture != "not_available"
        and _present(record.municipality)
        and record.municipality != "not_available"
        and "facility_name" in record.source_fields
    )


def _evidence_strength(record: FacilityEvidenceRecord) -> tuple[int, int, int, int, str]:
    has_coordinates = int(record.latitude is not None and record.longitude is not None and "coordinates" in record.source_fields)
    has_address = int(_present(record.facility_address) and "facility_address" in record.source_fields)
    has_name = int(_present(record.facility_name) and "facility_name" in record.source_fields)
    source_rich_artifact = int(record.source_artifact == str(HEALTHCARE_SUPPLY_RECORDS_PATH))
    return (has_coordinates, has_address, has_name, source_rich_artifact, record.source_artifact)


def _load_healthcare_supply_records(root: Path) -> tuple[list[FacilityEvidenceRecord], list[dict[str, Any]]]:
    """Load source-traceable facility records with addresses and coordinates."""

    path = root / HEALTHCARE_SUPPLY_RECORDS_PATH
    if not path.exists():
        return [], [{
            "issue_code": "healthcare_supply_records_missing",
            "severity": "warning",
            "message": "Healthcare supply records are missing; address and coordinate concrete evidence are unavailable.",
            "evidence_ref": str(HEALTHCARE_SUPPLY_RECORDS_PATH),
        }]

    records: list[FacilityEvidenceRecord] = []
    issues: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        source_record_id = _source_record_id(row)
        missing_fields: list[str] = []
        blocked_fields: list[str] = []
        source_fields: dict[str, str] = {}

        name = str(row.get("facility_name")) if _present(row.get("facility_name")) else None
        address = str(row.get("address")) if _present(row.get("address")) else None
        prefecture = str(row.get("prefecture")) if _present(row.get("prefecture")) else "not_available"
        municipality = str(row.get("municipality")) if _present(row.get("municipality")) else "not_available"
        latitude = _num(row.get("latitude"))
        longitude = _num(row.get("longitude"))

        if not source_record_id:
            missing_fields.append("source_record_id")
        if not name:
            missing_fields.append("facility_name")
        elif source_record_id:
            source_fields["facility_name"] = "facility_name"
        else:
            blocked_fields.append("facility_name")

        if not address:
            missing_fields.append("facility_address")
        elif source_record_id:
            source_fields["facility_address"] = "address"
        else:
            blocked_fields.append("facility_address")
            address = None

        if row.get("latitude") in (None, ""):
            missing_fields.append("latitude")
        if row.get("longitude") in (None, ""):
            missing_fields.append("longitude")
        if latitude is None or longitude is None:
            latitude = None
            longitude = None
        elif _valid_lat_lon(latitude, longitude) and source_record_id:
            source_fields["latitude"] = "latitude"
            source_fields["longitude"] = "longitude"
            source_fields["coordinates"] = "latitude,longitude"
        else:
            blocked_fields.append("target_coordinates")
            latitude = None
            longitude = None

        if not _present(row.get("prefecture")):
            missing_fields.append("prefecture")
        elif source_record_id:
            source_fields["prefecture"] = "prefecture"
        if not _present(row.get("municipality")):
            missing_fields.append("municipality")
        elif source_record_id:
            source_fields["municipality"] = "municipality"

        if not name:
            issues.append({
                "issue_code": "facility_record_missing_name",
                "severity": "warning",
                "message": "Facility evidence row lacked a facility name and cannot support concrete proposals.",
                "evidence_ref": str(HEALTHCARE_SUPPLY_RECORDS_PATH),
            })

        payload = {
            "source_artifact": str(HEALTHCARE_SUPPLY_RECORDS_PATH),
            "source_record_id": source_record_id,
            "facility_name": name,
            "facility_address": address,
            "latitude": latitude,
            "longitude": longitude,
        }
        records.append(FacilityEvidenceRecord(
            facility_record_id=_stable_id("facility_evidence", payload),
            facility_name=name,
            facility_address=address,
            latitude=latitude,
            longitude=longitude,
            prefecture=prefecture,
            municipality=municipality,
            source_artifact=str(HEALTHCARE_SUPPLY_RECORDS_PATH),
            source_record_id=source_record_id,
            source_fields=source_fields,
            missing_fields=sorted(set(missing_fields)),
            blocked_fields=sorted(set(blocked_fields)),
        ))
    return records, issues


def _load_hospital_feature_name_records(root: Path) -> tuple[list[FacilityEvidenceRecord], list[dict[str, Any]]]:
    """Load fallback name-only records from the model hospital feature view."""

    path = root / HOSPITAL_FEATURES_PATH
    if not path.exists():
        return [], [{
            "issue_code": "facility_evidence_missing",
            "severity": "warning",
            "message": "Hospital feature analysis view is missing; fallback concrete facility names are unavailable.",
            "evidence_ref": str(HOSPITAL_FEATURES_PATH),
        }]

    records: list[FacilityEvidenceRecord] = []
    issues: list[dict[str, Any]] = []
    for row in _read_jsonl(path):
        name = row.get("hospital_name")
        source_fact_ids = row.get("source_fact_ids") or []
        feature_id = str(row.get("feature_id") or row.get("master_id") or "")
        if not name or not feature_id or not source_fact_ids:
            issues.append({
                "issue_code": "facility_name_without_required_trace",
                "severity": "warning",
                "message": "Facility row lacked name, feature id, or source fact ids and was not used.",
                "evidence_ref": str(HOSPITAL_FEATURES_PATH),
            })
            continue
        payload = {
            "source_artifact": str(HOSPITAL_FEATURES_PATH),
            "source_record_id": feature_id,
            "facility_name": name,
        }
        records.append(FacilityEvidenceRecord(
            facility_record_id=_stable_id("facility_evidence", payload),
            facility_name=str(name),
            facility_address=None,
            latitude=None,
            longitude=None,
            prefecture=str(row.get("prefecture") or "not_available"),
            municipality=str(row.get("municipality") or "not_available"),
            source_artifact=str(HOSPITAL_FEATURES_PATH),
            source_record_id=feature_id,
            source_fields={
                "facility_name": "hospital_name",
                "prefecture": "prefecture",
                "municipality": "municipality",
                "source_fact_ids": "source_fact_ids",
            },
            missing_fields=sorted({"facility_address", "latitude", "longitude", "municipality"}),
            blocked_fields=[],
        ))
    return records, issues


def load_facility_evidence_records(repo_root: str | Path = ".") -> tuple[list[FacilityEvidenceRecord], list[dict[str, Any]]]:
    """Load facility evidence from existing analysis views without inventing fields."""

    root = Path(repo_root)
    issues: list[dict[str, Any]] = []
    records: list[FacilityEvidenceRecord] = []
    supply_records, supply_issues = _load_healthcare_supply_records(root)
    fallback_records, fallback_issues = _load_hospital_feature_name_records(root)
    issues.extend(supply_issues)
    issues.extend(fallback_issues)
    records.extend(supply_records)
    records.extend(fallback_records)
    if not records:
        issues.append({
            "issue_code": "no_source_traceable_facility_records",
            "severity": "warning",
            "message": "No source-traceable facility records were available for concrete proposals.",
            "evidence_ref": str(HOSPITAL_FEATURES_PATH),
        })
    return sorted(records, key=lambda rec: (rec.prefecture, rec.municipality, rec.facility_name or "", rec.source_record_id)), issues


def facility_evidence_inventory_report(records: list[FacilityEvidenceRecord]) -> dict[str, Any]:
    """Summarize field-level concrete evidence availability."""

    usable_for_name = records_usable_for_concrete_proposals(records)
    usable_for_address = [
        rec for rec in usable_for_name
        if _present(rec.facility_address) and "facility_address" in rec.source_fields
    ]
    usable_for_coordinates = [
        rec for rec in usable_for_name
        if rec.latitude is not None
        and rec.longitude is not None
        and "coordinates" in rec.source_fields
    ]
    return {
        "facility_evidence_records_loaded": len(records),
        "facility_records_with_name": sum(1 for rec in records if _present(rec.facility_name) and "facility_name" in rec.source_fields),
        "facility_records_with_address": sum(1 for rec in records if _present(rec.facility_address) and "facility_address" in rec.source_fields),
        "facility_records_with_coordinates": sum(1 for rec in records if rec.latitude is not None and rec.longitude is not None and "coordinates" in rec.source_fields),
        "facility_records_usable_for_concrete_name": len(usable_for_name),
        "facility_records_usable_for_concrete_address": len(usable_for_address),
        "facility_records_usable_for_coordinates": len(usable_for_coordinates),
        "facility_evidence_source_artifacts": sorted({rec.source_artifact for rec in records}),
    }


def records_usable_for_concrete_proposals(records: list[FacilityEvidenceRecord]) -> list[FacilityEvidenceRecord]:
    best_by_identity: dict[tuple[str, str, str], FacilityEvidenceRecord] = {}
    for rec in records:
        if not _record_is_usable(rec):
            continue
        key = (rec.prefecture, rec.municipality, str(rec.facility_name))
        current = best_by_identity.get(key)
        if current is None or _evidence_strength(rec) > _evidence_strength(current):
            best_by_identity[key] = rec
    return sorted(best_by_identity.values(), key=lambda rec: (rec.prefecture, rec.municipality, rec.facility_name or "", rec.source_record_id))


def evidence_refs_for_facility(record: FacilityEvidenceRecord) -> list[SourceEvidenceRef]:
    refs = [
        SourceEvidenceRef(
            field_name="target_facility_name",
            source_artifact=record.source_artifact,
            source_record_id=record.source_record_id,
            source_field=record.source_fields["facility_name"],
        )
    ] if "facility_name" in record.source_fields and record.facility_name else []
    if record.facility_address and "facility_address" in record.source_fields:
        refs.append(SourceEvidenceRef(
            field_name="target_facility_address",
            source_artifact=record.source_artifact,
            source_record_id=record.source_record_id,
            source_field=record.source_fields.get("facility_address", "facility_address"),
        ))
    if record.latitude is not None and record.longitude is not None and "coordinates" in record.source_fields:
        refs.append(SourceEvidenceRef(
            field_name="target_coordinates",
            source_artifact=record.source_artifact,
            source_record_id=record.source_record_id,
            source_field=record.source_fields.get("coordinates", "coordinates"),
        ))
    return refs
