"""Distance, density, and catchment metrics for site-selection experiments."""

from __future__ import annotations

from collections import Counter
from typing import Any

from geo_strategist.data.facility_taxonomy import classify_facility
from geo_strategist.data_sources.geo_utils import haversine_distance_km


MAJOR_HOSPITAL_BED_THRESHOLD = 200.0
LOCAL_RADIUS_KM = 3.0
MEDICAL_CATCHMENT_RADIUS_KM = 10.0
OUTPATIENT_CATCHMENT_RADIUS_KM = 5.0
AMBULANCE_CATCHMENT_RADIUS_KM = 15.0


def _hospital_rows(facility_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in facility_records:
        taxonomy = row.get("taxonomy") or classify_facility(row)
        ftype = taxonomy.get("facility_type")
        if ftype in {
            "hospital",
            "emergency_hospital",
            "tertiary_emergency_center",
            "psychiatric_hospital",
            "long_term_care_chronic_care_hospital",
            "dpc_acute_care_hospital",
        }:
            enriched = dict(row)
            enriched["taxonomy"] = taxonomy
            rows.append(enriched)
    return rows


def _distance(candidate: dict[str, Any], facility: dict[str, Any]) -> float | None:
    if candidate.get("latitude") is None or candidate.get("longitude") is None:
        return None
    if facility.get("latitude") is None or facility.get("longitude") is None:
        return None
    return haversine_distance_km(candidate["latitude"], candidate["longitude"], facility["latitude"], facility["longitude"])


def _counts_within(distances: list[tuple[float, dict[str, Any]]], radius_km: float) -> int:
    return sum(1 for dist, _ in distances if dist <= radius_km)


def build_site_catchment_metrics(
    candidate: dict[str, Any],
    facility_records: list[dict[str, Any]],
    workbook_records: list[dict[str, Any]] | None = None,
    population_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute transparent proxy metrics from available geocoded data."""

    workbook_records = workbook_records or []
    hospitals = _hospital_rows(facility_records)
    distances = sorted(
        ((dist, facility) for facility in hospitals if (dist := _distance(candidate, facility)) is not None),
        key=lambda row: row[0],
    )

    workbook_pref_rows = [row for row in workbook_records if row.get("prefecture") == candidate.get("prefecture")]
    workbook_muni_rows = [row for row in workbook_records if row.get("municipality") and row.get("municipality") == candidate.get("municipality")]
    workbook_major_rows = [
        row for row in workbook_pref_rows
        if (row.get("official_beds_total") or row.get("beds_used_in_model") or 0) >= MAJOR_HOSPITAL_BED_THRESHOLD
    ]

    emergency_distances = [
        (dist, row) for dist, row in distances
        if (row.get("taxonomy") or {}).get("facility_type") in {"emergency_hospital", "tertiary_emergency_center"}
    ]
    major_distances = [
        (dist, row) for dist, row in distances
        if (row.get("official_beds_total") or row.get("beds_used_in_model") or 0) >= MAJOR_HOSPITAL_BED_THRESHOLD
        or row.get("facility_name") in {w.get("hospital_name") for w in workbook_major_rows}
    ]

    muni_counts = Counter((row.get("prefecture"), row.get("municipality")) for row in hospitals)
    same_muni_supply = muni_counts.get((candidate.get("prefecture"), candidate.get("municipality")), 0)
    neighboring_supply = sum(
        count for (pref, muni), count in muni_counts.items()
        if pref == candidate.get("prefecture") and muni != candidate.get("municipality")
    )
    local_count = _counts_within(distances, LOCAL_RADIUS_KM)
    medical_count = _counts_within(distances, MEDICAL_CATCHMENT_RADIUS_KM)
    outpatient_count = _counts_within(distances, OUTPATIENT_CATCHMENT_RADIUS_KM)
    ambulance_count = _counts_within(emergency_distances, AMBULANCE_CATCHMENT_RADIUS_KM)
    population = population_row.get("population_total") if population_row else None
    density_per_100k = round((same_muni_supply / population) * 100000, 4) if population else None

    coordinate_proxy = candidate.get("latitude") is not None and candidate.get("longitude") is not None
    evidence_status = "distance_proxy_from_geocoded_facility_records" if coordinate_proxy else "municipality_supply_proxy_no_candidate_coordinates"
    uncertainty = "medium" if coordinate_proxy else "high"

    competition_intensity = "high" if local_count >= 5 or medical_count >= 10 else "medium" if local_count >= 2 or medical_count >= 4 else "low"
    underserved_signal = bool(
        (density_per_100k is not None and density_per_100k < 3.0)
        or (coordinate_proxy and (not distances or (distances[0][0] > MEDICAL_CATCHMENT_RADIUS_KM)))
        or (not coordinate_proxy and same_muni_supply == 0)
    )

    return {
        "evidence_status": evidence_status,
        "distance_method": "haversine_geographic_distance_proxy_km",
        "uncertainty_level": uncertainty,
        "hospital_density_same_municipality_count": same_muni_supply,
        "hospital_density_per_100k_population": density_per_100k,
        "local_hospital_count_3km": local_count if coordinate_proxy else None,
        "medical_catchment_hospital_count_10km": medical_count if coordinate_proxy else None,
        "outpatient_catchment_facility_count_5km": outpatient_count if coordinate_proxy else None,
        "ambulance_emergency_catchment_count_15km": ambulance_count if coordinate_proxy else None,
        "distance_to_nearest_hospital_km": round(distances[0][0], 4) if distances else None,
        "distance_to_nearest_major_hospital_km": round(major_distances[0][0], 4) if major_distances else None,
        "distance_to_nearest_emergency_hospital_km": round(emergency_distances[0][0], 4) if emergency_distances else None,
        "neighboring_municipality_hospital_supply_count_prefecture_proxy": neighboring_supply,
        "workbook_fallback_hospital_count_prefecture": len(workbook_pref_rows),
        "workbook_fallback_major_hospital_count_prefecture": len(workbook_major_rows),
        "workbook_fallback_hospital_count_municipality": len(workbook_muni_rows),
        "competition_intensity": competition_intensity,
        "underserved_area_signal": underserved_signal,
        "confidence_score": 0.72 if coordinate_proxy and hospitals else 0.45 if same_muni_supply or workbook_pref_rows else 0.25,
        "limitations": [
            "Travel-time data is unavailable; geographic distance is a proxy.",
            "Workbook fallback is trusted for major hospital coverage but may lack coordinates or municipality fields.",
        ],
    }
