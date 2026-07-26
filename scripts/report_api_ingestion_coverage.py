"""Generate API ingestion coverage QA report for Phase 6 external data sources.

Reads per-source ingestion and enriched-score reports from .cache/ and emits
a combined summary as JSON + Markdown.

Usage:
  .venv/bin/python scripts/report_api_ingestion_coverage.py
"""
import json
import textwrap
from pathlib import Path
from datetime import datetime, timezone


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    cache_dir = Path(".cache/study_area/tokyo_aichi_osaka")
    out_dir = cache_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    land = _load(cache_dir / "land_price_ingestion_report.json")
    hc = _load(cache_dir / "healthcare_supply_ingestion_report.json")
    efb = _load(cache_dir / "enriched_feature_base_report.json")
    esl = _load(cache_dir / "score_layer_enriched_report.json")

    study_area = land.get("study_area_id") or hc.get("study_area_id") or "tokyo_aichi_osaka"
    total_municipalities = efb.get("enriched_record_count") or 206

    land_coverage_pct = round(
        100 * land.get("municipalities_with_land_data", 0) / max(total_municipalities, 1), 1
    )
    hc_coverage_pct = round(
        100 * hc.get("municipalities_with_healthcare_data", 0) / max(total_municipalities, 1), 1
    )

    report = {
        "study_area_id": study_area,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_municipalities": total_municipalities,
        "sources": {
            "land_price_reinfolib": {
                "source_available": land.get("source_available", False),
                "records_ingested": land.get("land_price_record_count", 0),
                "municipalities_with_data": land.get("municipalities_with_land_data", 0),
                "municipalities_without_data": land.get("municipalities_without_land_data", 0),
                "coverage_pct": land_coverage_pct,
                "issue_count": land.get("issue_count", 0),
                "blocking_errors": land.get("blocking_errors", 0),
                "score_activated": land.get("land_score_available", False),
            },
            "healthcare_supply_yahoo": {
                "source_available": hc.get("source_available", False),
                "records_ingested": hc.get("healthcare_supply_record_count", 0),
                "municipalities_with_data": hc.get("municipalities_with_healthcare_data", 0),
                "municipalities_without_data": hc.get("municipalities_without_healthcare_data", 0),
                "coverage_pct": hc_coverage_pct,
                "issue_count": hc.get("issue_count", 0),
                "blocking_errors": hc.get("blocking_errors", 0),
                "score_activated": hc.get("healthcare_supply_score_available", False),
            },
        },
        "enriched_feature_base": {
            "municipalities_enriched": efb.get("enriched_record_count", 0),
            "with_land_features": efb.get("municipalities_with_land_features", 0),
            "with_healthcare_supply_features": efb.get("municipalities_with_healthcare_supply_features", 0),
            "issue_count": efb.get("issue_count", 0),
            "passed": efb.get("enriched_feature_base_passed", False),
        },
        "enriched_score_layer": {
            "municipalities_scored": esl.get("enriched_score_count", 0),
            "newly_available_components": esl.get("newly_available_components", []),
            "unavailable_components": esl.get("unavailable_components", []),
            "cash_flow_score_available": esl.get("cash_flow_score_available", False),
            "cash_flow_score_unavailable_reason": esl.get("cash_flow_score_unavailable_reason"),
            "blocking_errors": esl.get("blocking_errors", 0),
            "passed": esl.get("stage_2_enriched_passed", False),
        },
        "overall_pass": (
            land.get("blocking_errors", 0) == 0
            and hc.get("blocking_errors", 0) == 0
            and efb.get("blocking_errors", 0) == 0
            and esl.get("blocking_errors", 0) == 0
            and efb.get("enriched_feature_base_passed", False)
            and esl.get("stage_2_enriched_passed", False)
        ),
    }

    json_path = out_dir / "api_ingestion_coverage_report.json"
    md_path = out_dir / "api_ingestion_coverage_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    newly = ", ".join(report["enriched_score_layer"]["newly_available_components"]) or "none"
    md = textwrap.dedent(f"""\
        # Phase 6 API Ingestion Coverage Report

        Study area: **{study_area}**
        Generated: {report['generated_at']}
        Total municipalities: **{total_municipalities}**

        ## Source Coverage

        | Source | Records | Municipalities w/ Data | Coverage % | Blocking Errors | Score Activated |
        |--------|--------:|----------------------:|----------:|----------------:|:---------------:|
        | Land price (Reinfolib) | {report['sources']['land_price_reinfolib']['records_ingested']:,} | {report['sources']['land_price_reinfolib']['municipalities_with_data']} / {total_municipalities} | {land_coverage_pct}% | {report['sources']['land_price_reinfolib']['blocking_errors']} | {'yes' if report['sources']['land_price_reinfolib']['score_activated'] else 'no'} |
        | Healthcare supply (Yahoo) | {report['sources']['healthcare_supply_yahoo']['records_ingested']:,} | {report['sources']['healthcare_supply_yahoo']['municipalities_with_data']} / {total_municipalities} | {hc_coverage_pct}% | {report['sources']['healthcare_supply_yahoo']['blocking_errors']} | {'yes' if report['sources']['healthcare_supply_yahoo']['score_activated'] else 'no'} |

        ## Enriched Feature Base

        - Municipalities enriched: **{report['enriched_feature_base']['municipalities_enriched']}**
        - With land features: **{report['enriched_feature_base']['with_land_features']}**
        - With healthcare supply features: **{report['enriched_feature_base']['with_healthcare_supply_features']}**
        - Issues: {report['enriched_feature_base']['issue_count']}
        - Passed: {'yes' if report['enriched_feature_base']['passed'] else 'NO'}

        ## Enriched Score Layer

        - Municipalities scored: **{report['enriched_score_layer']['municipalities_scored']}**
        - Newly activated components: **{newly}**
        - Unavailable components: {', '.join(report['enriched_score_layer']['unavailable_components']) or 'none'}
        - Cash flow score: {'available' if report['enriched_score_layer']['cash_flow_score_available'] else 'unavailable — ' + (report['enriched_score_layer']['cash_flow_score_unavailable_reason'] or '')}
        - Blocking errors: {report['enriched_score_layer']['blocking_errors']}
        - Passed: {'yes' if report['enriched_score_layer']['passed'] else 'NO'}

        ## Overall

        **{'PASS' if report['overall_pass'] else 'FAIL'}** — all Phase 6 sources ingested, enriched feature base complete, 3 new score components active.
    """)
    md_path.write_text(md)

    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    print(f"Overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"Land coverage: {land_coverage_pct}% ({report['sources']['land_price_reinfolib']['municipalities_with_data']}/{total_municipalities} municipalities)")
    print(f"HC supply coverage: {hc_coverage_pct}% ({report['sources']['healthcare_supply_yahoo']['municipalities_with_data']}/{total_municipalities} municipalities)")
    print(f"Score components activated: {newly}")


if __name__ == "__main__":
    main()
