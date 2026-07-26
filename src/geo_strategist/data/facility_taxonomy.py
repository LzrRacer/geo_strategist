"""Facility taxonomy and workbook-to-geocoded entity resolution.

The classifier is deliberately evidence-combining rather than a single
Japanese substring check. It uses Yahoo/category text, facility names,
workbook rows, and available source fields. When the evidence is weak, the
record remains unknown/ambiguous and carries a human-review flag.
"""

from __future__ import annotations

import json
import re
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


FACILITY_TYPES = {
    "hospital",
    "clinic",
    "emergency_hospital",
    "tertiary_emergency_center",
    "psychiatric_hospital",
    "long_term_care_chronic_care_hospital",
    "dpc_acute_care_hospital",
    "unknown_ambiguous",
}

_CLINIC_MARKERS = ("クリニック", "診療所", "医院", "内科", "外科")
_HOSPITAL_MARKERS = ("病院", "医療センター", "メディカルセンター", "hospital", "medical center")
_EMERGENCY_MARKERS = ("救急", "ER", "emergency", "救命")
_TERTIARY_MARKERS = ("救命救急センター", "高度救命", "tertiary emergency", "critical care center")
_PSYCH_MARKERS = ("精神", "こころ", "psychiatric", "mental")
_CHRONIC_MARKERS = ("療養", "慢性期", "リハビリ", "rehabilitation", "long-term", "chronic")
_ACUTE_MARKERS = ("急性期", "DPC", "特定機能", "acute")
_NON_HOSPITAL_MARKERS = ("動物病院", "獣医", "薬局", "歯科")


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").lower())


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in markers)


def classify_facility(row: dict[str, Any], workbook_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classify a facility and return evidence-rich taxonomy metadata."""

    name = str(row.get("facility_name") or row.get("hospital_name") or "")
    category = str(row.get("facility_category") or "")
    description = str(row.get("description") or row.get("model_archetype") or "")
    workbook_name = str((workbook_row or {}).get("hospital_name") or "")
    text = " ".join([name, category, description, workbook_name])
    norm = _norm_text(text)
    evidence: list[dict[str, Any]] = []

    def add(marker_group: str, source_field: str, confidence: float) -> None:
        evidence.append({
            "marker_group": marker_group,
            "source_field": source_field,
            "confidence_contribution": confidence,
        })

    if _contains_any(norm, _NON_HOSPITAL_MARKERS):
        return {
            "facility_type": "unknown_ambiguous",
            "taxonomy_confidence": 0.35,
            "taxonomy_method": "non_hospital_exclusion_marker",
            "taxonomy_evidence": [{"marker_group": "non_hospital_exclusion", "source_field": "name_or_category"}],
            "human_review_required": True,
        }

    facility_type = "unknown_ambiguous"
    confidence = 0.30
    method = "ambiguous_text_evidence"

    if _contains_any(norm, _TERTIARY_MARKERS):
        facility_type = "tertiary_emergency_center"
        confidence = 0.90
        method = "tertiary_emergency_marker"
        add("tertiary_emergency", "name_category_description", confidence)
    elif _contains_any(norm, _EMERGENCY_MARKERS) and _contains_any(norm, _HOSPITAL_MARKERS):
        facility_type = "emergency_hospital"
        confidence = 0.82
        method = "emergency_and_hospital_markers"
        add("emergency_hospital", "name_category_description", confidence)
    elif _contains_any(norm, _PSYCH_MARKERS) and _contains_any(norm, _HOSPITAL_MARKERS):
        facility_type = "psychiatric_hospital"
        confidence = 0.78
        method = "psychiatric_and_hospital_markers"
        add("psychiatric_hospital", "name_category_description", confidence)
    elif _contains_any(norm, _CHRONIC_MARKERS) and _contains_any(norm, _HOSPITAL_MARKERS):
        facility_type = "long_term_care_chronic_care_hospital"
        confidence = 0.74
        method = "chronic_care_and_hospital_markers"
        add("chronic_care_hospital", "name_category_description", confidence)
    elif _contains_any(norm, _ACUTE_MARKERS) and _contains_any(norm, _HOSPITAL_MARKERS):
        facility_type = "dpc_acute_care_hospital"
        confidence = 0.72
        method = "acute_care_marker"
        add("acute_care_hospital", "name_category_description", confidence)
    elif _contains_any(norm, _HOSPITAL_MARKERS):
        facility_type = "hospital"
        confidence = 0.68
        method = "hospital_markers"
        add("hospital", "name_category_description", confidence)
    elif _contains_any(norm, _CLINIC_MARKERS):
        facility_type = "clinic"
        confidence = 0.66
        method = "clinic_markers"
        add("clinic", "name_category_description", confidence)

    if workbook_row:
        # The workbook is a trusted fallback source for major hospitals.
        if facility_type in {"hospital", "unknown_ambiguous"}:
            facility_type = "hospital"
        confidence = min(0.98, max(confidence, 0.78))
        method = f"{method}+workbook_fallback"
        add("trusted_workbook_hospital_list", "hospital_master_68", 0.78)

    return {
        "facility_type": facility_type,
        "taxonomy_confidence": round(confidence, 4),
        "taxonomy_method": method,
        "taxonomy_evidence": evidence,
        "human_review_required": confidence < 0.70 or facility_type == "unknown_ambiguous",
    }


def _name_similarity(left: str | None, right: str | None) -> float:
    left_norm = _norm_text(left)
    right_norm = _norm_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    if left_norm in right_norm or right_norm in left_norm:
        return 0.92
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def _address_score(workbook_row: dict[str, Any], facility_row: dict[str, Any]) -> float:
    score = 0.0
    if workbook_row.get("prefecture") and workbook_row.get("prefecture") == facility_row.get("prefecture"):
        score += 0.10
    if workbook_row.get("municipality") and workbook_row.get("municipality") == facility_row.get("municipality"):
        score += 0.12
    w_addr = _norm_text(workbook_row.get("address"))
    f_addr = _norm_text(facility_row.get("address"))
    if w_addr and f_addr and (w_addr in f_addr or f_addr in w_addr):
        score += 0.18
    if workbook_row.get("phone") and workbook_row.get("phone") == facility_row.get("phone"):
        score += 0.20
    return min(score, 0.35)


def resolve_workbook_hospitals(
    workbook_records: list[dict[str, Any]],
    facility_records: list[dict[str, Any]],
    *,
    low_confidence_threshold: float = 0.78,
) -> list[dict[str, Any]]:
    """Match workbook hospitals to geocoded facility records with confidence."""

    rows: list[dict[str, Any]] = []
    for workbook in workbook_records:
        best: tuple[float, dict[str, Any] | None, str] = (0.0, None, "unmatched")
        for facility in facility_records:
            name_score = _name_similarity(workbook.get("hospital_name"), facility.get("facility_name"))
            address_score = _address_score(workbook, facility)
            category_score = 0.08 if classify_facility(facility).get("facility_type") in {
                "hospital",
                "emergency_hospital",
                "tertiary_emergency_center",
                "psychiatric_hospital",
                "long_term_care_chronic_care_hospital",
                "dpc_acute_care_hospital",
            } else 0.0
            confidence = min(1.0, round((name_score * 0.72) + address_score + category_score, 4))
            method = "exact_name" if name_score == 1.0 else "fuzzy_name_address_category"
            if confidence > best[0]:
                best = (confidence, facility, method)

        confidence, match, method = best
        if confidence < low_confidence_threshold:
            match = None
            method = "no_high_confidence_match"
        rows.append({
            "resolution_id": f"hospital_entity_resolution:{uuid.uuid5(uuid.NAMESPACE_URL, str(workbook.get('master_id') or workbook.get('hospital_name')))}",
            "master_id": workbook.get("master_id"),
            "workbook_hospital_name": workbook.get("hospital_name"),
            "matched_supply_record_id": match.get("supply_record_id") if match else None,
            "matched_facility_name": match.get("facility_name") if match else None,
            "matched_address": match.get("address") if match else None,
            "matched_latitude": match.get("latitude") if match else None,
            "matched_longitude": match.get("longitude") if match else None,
            "matched_municipality": match.get("municipality") if match else None,
            "match_confidence": confidence,
            "match_method": method if match else "no_candidate_match",
            "human_review_required": confidence < low_confidence_threshold,
            "workbook_source_artifact": workbook.get("source_artifact"),
            "geocoded_source_artifact": match.get("source_artifact") if match else None,
            "taxonomy": classify_facility(match or workbook, workbook),
        })
    return rows


def write_entity_resolution_report(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "record_count": len(rows),
        "matched_count": sum(1 for row in rows if row.get("matched_supply_record_id")),
        "human_review_required_count": sum(1 for row in rows if row.get("human_review_required")),
        "low_confidence_threshold": 0.78,
    }
    path.write_text(json.dumps({"summary": summary, "records": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
