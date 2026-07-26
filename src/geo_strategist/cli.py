"""Command line interface for the Geo Strategist scaffold."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

import typer
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from geo_strategist.data.inventory import inventory_local_data
from geo_strategist.data.age_group_coverage_report import build_age_group_coverage_report
from geo_strategist.data.age_group_normalizer import normalize_age_groups
from geo_strategist.data.feature_substrate import run_feature_substrate
from geo_strategist.data.model_input_readiness import run_model_input_readiness
from geo_strategist.data.score_layer import run_score_layer
from geo_strategist.data.land_price_ingestion import run_land_price_ingestion
from geo_strategist.data.healthcare_supply_ingestion import run_healthcare_supply_ingestion
from geo_strategist.data.enriched_feature_base import run_enriched_feature_base
from geo_strategist.data.enriched_score_layer import run_enriched_score_layer
from geo_strategist.data.candidate_generation import run_candidate_generation
from geo_strategist.data.geography_grain_classifier import classify_geography_grain
from geo_strategist.data.mapping_review import review_extraction_mappings
from geo_strategist.data.normalizers.hospital_workbook import normalize_hospital_workbook
from geo_strategist.data.normalizers.population import normalize_population_data
from geo_strategist.data.population import inspect_population_data
from geo_strategist.data.population_mapping_diagnostics import diagnose_population_mappings
from geo_strategist.data.population_geography_qa import build_population_geography_qa
from geo_strategist.data.population_base import build_population_base
from geo_strategist.data.population_base_coverage_qa import run_population_base_coverage_qa
from geo_strategist.data.population_base_coverage_report import (
    build_population_base_coverage_report,
)
from geo_strategist.data.population_base_report import build_population_base_report
from geo_strategist.data.study_area_filter import filter_study_area_population
from geo_strategist.data.study_area_geography_qa import run_study_area_geography_qa
from geo_strategist.data.validate_views import validate_analysis_views
from geo_strategist.data.validate_normalized import validate_normalized_outputs
from geo_strategist.data.views.hospital_facts import build_hospital_facts
from geo_strategist.data.views.population_long import build_population_long
from geo_strategist.data.views.population_rates import build_population_rates_long
from geo_strategist.data.workbook import inspect_hospital_workbook
from geo_strategist.evaluation.rubric import load_rubric
from geo_strategist.settings import load_settings


console = Console()
app = typer.Typer(
    help="Geo Strategist scaffold utilities.",
    no_args_is_help=True,
)


def _package_version() -> str:
    try:
        return version("geo-strategist")
    except PackageNotFoundError:
        return "0.1.0"


def _version_callback(value: bool) -> None:
    if value:
        console.print(_package_version())
        raise typer.Exit()


def _load_repo_dotenv() -> None:
    """Load local runtime secrets for CLI commands without printing them."""

    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path, override=False)


@app.callback()
def main(
    version_flag: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show package version and exit.",
    ),
) -> None:
    """Geo Strategist scaffold entry point."""

    _load_repo_dotenv()


@app.command("settings")
def show_settings(
    env_file: Optional[Path] = typer.Option(
        None,
        "--env-file",
        help="Optional dotenv file to read intentionally.",
    ),
) -> None:
    """Show non-secret runtime settings."""

    settings = load_settings(env_file=env_file)
    safe_values = {
        "cache_dir": str(settings.cache_dir),
        "data_dir": str(settings.data_dir),
        "runs_dir": str(settings.runs_dir),
        "log_level": settings.log_level,
        "openrouter_base_url": settings.openrouter_base_url,
        "default_judge_model": settings.default_judge_model,
        "default_openrouter_model": settings.default_openrouter_model,
    }
    console.print(yaml.safe_dump(safe_values, sort_keys=True))


@app.command("doctor")
def doctor() -> None:
    """Report scaffold status without contacting external services."""

    console.print("Geo Strategist scaffold is importable.")


@app.command("validate-config")
def validate_config() -> None:
    """Validate config files and local JSON contract files."""

    errors: list[str] = []
    config_paths = sorted(Path("configs").rglob("*.yaml"))
    contract_paths = sorted(Path("data/contracts").glob("*.schema.json"))
    catalog_path = Path("data/source_catalog.yaml")

    for path in config_paths:
        try:
            with path.open("r", encoding="utf-8") as file:
                yaml.safe_load(file)
        except Exception as exc:  # pragma: no cover - defensive CLI path
            errors.append(f"{path}: {exc}")

    if catalog_path.exists():
        try:
            with catalog_path.open("r", encoding="utf-8") as file:
                yaml.safe_load(file)
        except Exception as exc:  # pragma: no cover - defensive CLI path
            errors.append(f"{catalog_path}: {exc}")

    for path in contract_paths:
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
            for key in ("$schema", "title", "type"):
                if key not in schema:
                    errors.append(f"{path}: missing {key}")
        except Exception as exc:  # pragma: no cover - defensive CLI path
            errors.append(f"{path}: {exc}")

    try:
        load_rubric()
    except Exception as exc:  # pragma: no cover - defensive CLI path
        errors.append(f"configs/evaluation_rubric.yaml: {exc}")

    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)

    console.print(
        f"Validated {len(config_paths)} config files and {len(contract_paths)} JSON schemas."
    )


@app.command("inventory-data")
def inventory_data() -> None:
    """Inventory local manual data files by metadata only."""

    result = inventory_local_data()
    for warning in result.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Scanned root: {result.scanned_root or 'none'}")
    console.print(f"Files inventoried: {len(result.files)}")
    console.print(f"Inventory JSON: {result.output_path}")
    if result.provenance_path:
        console.print(f"Provenance JSONL: {result.provenance_path}")


@app.command("inspect-workbook")
def inspect_workbook(
    require_data: bool = typer.Option(
        False,
        "--require-data",
        help="Return nonzero if the hospital workbook is missing.",
    ),
) -> None:
    """Inspect the hospital workbook structure."""

    profile = inspect_hospital_workbook()
    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Workbook: {profile.path or 'not found'}")
    console.print(f"Sheets: {len(profile.sheet_names)}")
    console.print(f"Profile JSON: {profile.output_json}")
    console.print(f"Profile Markdown: {profile.output_markdown}")
    if require_data and not profile.found:
        raise typer.Exit(1)


@app.command("inspect-population")
def inspect_population(
    require_data: bool = typer.Option(
        False,
        "--require-data",
        help="Return nonzero if population workbooks are missing.",
    ),
) -> None:
    """Inspect population workbook structures."""

    profile = inspect_population_data()
    for warning in profile.warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")
    console.print(f"Population directory: {profile.source_directory or 'none'}")
    console.print(f"Workbook files: {len(profile.files)}")
    console.print(f"Profile JSON: {profile.output_json}")
    console.print(f"Profile Markdown: {profile.output_markdown}")
    if require_data and not profile.found:
        raise typer.Exit(1)


@app.command("normalize-hospital-workbook")
def normalize_hospital_workbook_command(
    require_data: bool = typer.Option(
        False,
        "--require-data",
        help="Return nonzero if the hospital workbook is missing.",
    ),
) -> None:
    """Normalize the hospital workbook into intermediate records."""

    result = normalize_hospital_workbook()
    console.print(f"Input found: {result.found}")
    console.print(f"Source tables: {len(result.source_tables)}")
    console.print(f"Normalized records: {len(result.normalized_records)}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if require_data and not result.found:
        raise typer.Exit(1)


@app.command("normalize-population")
def normalize_population_command(
    require_data: bool = typer.Option(
        False,
        "--require-data",
        help="Return nonzero if population workbooks are missing.",
    ),
) -> None:
    """Normalize population workbooks into intermediate records."""

    result = normalize_population_data()
    console.print(f"Input found: {result.found}")
    console.print(f"Source tables: {len(result.source_tables)}")
    console.print(f"Normalized records: {len(result.normalized_records)}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if require_data and not result.found:
        raise typer.Exit(1)


@app.command("validate-normalized")
def validate_normalized(
    require_outputs: bool = typer.Option(
        False,
        "--require-outputs",
        help="Return nonzero if normalized outputs are absent.",
    ),
) -> None:
    """Validate normalized output artifacts."""

    summary = validate_normalized_outputs(require_outputs=require_outputs)
    console.print(f"Checked outputs: {len(summary.checked_outputs)}")
    console.print(f"Missing outputs: {len(summary.missing_outputs)}")
    console.print(f"Source tables: {summary.source_table_count}")
    console.print(f"Normalized records: {summary.record_count}")
    console.print(f"Unresolved mappings: {summary.unresolved_mapping_count}")
    console.print(f"Warnings: {len(summary.warnings)}")
    if summary.errors:
        for error in summary.errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)


@app.command("review-mappings")
def review_mappings() -> None:
    """Review extraction mappings and prepare manual mapping candidates."""

    result = review_extraction_mappings()
    console.print(f"Source tables: {result.source_table_count}")
    console.print(f"Normalized records: {result.normalized_record_count}")
    console.print(f"Inferred mappings: {result.inferred_mapping_count}")
    console.print(f"Unresolved mappings: {result.unresolved_mapping_count}")
    console.print(f"Warnings: {result.warning_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("diagnose-population-mappings")
def diagnose_population_mappings_command() -> None:
    """Diagnose unresolved population mapping evidence."""

    result = diagnose_population_mappings()
    console.print(f"Source tables: {result.source_table_count}")
    console.print(f"Diagnostics: {result.diagnostics_count}")
    console.print(f"Unresolved mappings: {result.unresolved_mapping_count}")
    console.print(f"Quarantined issues: {result.quarantined_issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("build-hospital-facts")
def build_hospital_facts_command() -> None:
    """Build conservative hospital workbook facts."""

    result = build_hospital_facts()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("build-population-long")
def build_population_long_command() -> None:
    """Build conservative population long view."""

    result = build_population_long()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Quality issues: {result.quality_issue_count}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("build-population-rates")
def build_population_rates_command() -> None:
    """Build population rate view from normalized records."""

    result = build_population_rates_long()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Quality issues: {result.quality_issue_count}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("qa-population-geography")
def qa_population_geography_command() -> None:
    """Build deterministic geography QA for population views."""

    result = build_population_geography_qa()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Long records read: {result.records_read}")
    console.print(f"Rate records read: {result.rate_records_read}")
    console.print(f"Geography keys written: {result.keys_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Warnings: {len(result.warnings)}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("filter-study-area-population")
def filter_study_area_population_command() -> None:
    """Filter population views to the configured Tokyo/Aichi/Osaka study area."""

    result = filter_study_area_population()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Count rows read: {result.long_records_read}")
    console.print(f"Rate rows read: {result.rate_records_read}")
    console.print(f"Count rows written: {result.long_records_written}")
    console.print(f"Rate rows written: {result.rate_records_written}")
    console.print(f"Outside-scope rows: {result.outside_scope_rows}")
    console.print(f"Scope-unknown rows: {result.scope_unknown_rows}")
    console.print(f"Scope issues: {result.scope_issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("qa-study-area-geography")
def qa_study_area_geography_command() -> None:
    """Run deterministic target-scoped geography QA for Tokyo/Aichi/Osaka."""

    result = run_study_area_geography_qa()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Count rows read: {result.count_rows_read}")
    console.print(f"Rate rows read: {result.rate_rows_read}")
    console.print(f"Geography keys written: {result.geography_keys_written}")
    console.print(f"Duplicate target geography keys: {result.duplicate_key_count}")
    console.print(f"Issues: {result.issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("classify-geography-grain")
def classify_geography_grain_command() -> None:
    """Classify target-scope population rows by deterministic geography grain."""

    result = classify_geography_grain()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Count rows read: {result.count_rows_read}")
    console.print(f"Rate rows read: {result.rate_rows_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Issues: {result.issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("build-population-base")
def build_population_base_command() -> None:
    """Build pre-demand population-base views from geography-grain rows."""

    result = build_population_base()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Municipality records: {result.municipality_records_written}")
    console.print(f"Prefecture-total records: {result.prefecture_total_records_written}")
    console.print(f"Excluded records: {result.excluded_records}")
    console.print(f"Issues: {result.issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("report-population-base")
def report_population_base_command() -> None:
    """Report summary counts for pre-demand population-base views."""

    result = build_population_base_report()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Candidate model-input rows: {result.candidate_model_input_rows}")
    console.print(f"Context prefecture-total rows: {result.context_prefecture_total_rows}")
    console.print(f"Unknown geography-grain rows: {result.unknown_geography_grain_rows}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("qa-population-base-coverage")
def qa_population_base_coverage_command() -> None:
    """Run pre-demand coverage QA for population-base rows."""

    result = run_population_base_coverage_qa()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Matrix rows written: {result.matrix_rows_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Model-blocking errors: {result.model_blocking_error_count}")
    console.print(f"Duplicate keys: {result.duplicate_key_count}")
    console.print(f"Conflicting values: {result.conflicting_value_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("report-population-base-coverage")
def report_population_base_coverage_command() -> None:
    """Report summary counts for population-base coverage QA."""

    result = build_population_base_coverage_report()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Matrix rows read: {result.matrix_rows_read}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Model-blocking errors: {result.model_blocking_error_count}")
    console.print(f"Duplicate keys: {result.duplicate_key_count}")
    console.print(f"Conflicting values: {result.conflicting_value_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("normalize-age-groups")
def normalize_age_groups_command() -> None:
    """Normalize population-base age-group labels deterministically."""

    result = normalize_age_groups()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Records written: {result.records_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Unknown age groups: {result.unknown_age_group_count}")
    console.print(f"Missing age groups: {result.missing_age_group_count}")
    console.print(f"Duplicate normalized keys: {result.duplicate_normalized_key_count}")
    console.print(f"Conflicting values: {result.conflicting_value_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("qa-model-input-readiness")
def qa_model_input_readiness_command() -> None:
    """Run model-input readiness QA gate on age-normalized population-base rows."""

    result = run_model_input_readiness()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Readiness records written: {result.readiness_records_written}")
    console.print(f"Ready model-input rows: {result.ready_model_input_rows}")
    console.print(f"Context rows: {result.context_rows}")
    console.print(f"Blocked rows: {result.blocked_rows}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Blocking errors: {result.blocking_error_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count:
        console.print("Model-input readiness gate: FAILED")
        raise typer.Exit(1)


@app.command("report-age-group-coverage")
def report_age_group_coverage_command() -> None:
    """Report coverage for age-group-normalized population-base rows."""

    result = build_age_group_coverage_report()
    console.print(f"Input found: {result.input_found}")
    console.print(f"Records read: {result.records_read}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Unknown age groups: {result.unknown_age_group_count}")
    console.print(f"Missing age groups: {result.missing_age_group_count}")
    console.print(f"Duplicate normalized keys: {result.duplicate_normalized_key_count}")
    console.print(f"Conflicting values: {result.conflicting_value_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("build-feature-substrate")
def build_feature_substrate_command() -> None:
    """Build deterministic population and hospital feature substrate."""

    result = run_feature_substrate()
    console.print(f"Input found: {result.input_found}")
    if not result.input_found:
        raise typer.Exit(1)
    console.print(f"Population features written: {result.population_features_written}")
    console.print(f"Hospital features written: {result.hospital_features_written}")
    console.print(f"Municipality feature base written: {result.municipality_feature_base_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Blocking errors: {result.blocking_error_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count > 0:
        raise typer.Exit(1)


@app.command("build-score-layer")
def build_score_layer_command() -> None:
    """Build deterministic municipality score layer from feature substrate."""

    result = run_score_layer()
    console.print(f"Stage 1 passed: {result.stage_1_passed}")
    if not result.stage_1_passed:
        console.print("[red]Stage 1 has blocking errors.[/red]")
        raise typer.Exit(1)
    console.print(f"Scores written: {result.scores_written}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Blocking errors: {result.blocking_error_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count > 0:
        raise typer.Exit(1)


@app.command("ingest-land-prices")
def ingest_land_prices_command() -> None:
    """Ingest real land-price data from MLIT Reinfolib (graceful unavailable if no key)."""

    result = run_land_price_ingestion()
    console.print(f"Study area: {result.study_area_id}")
    console.print(f"Source available: {result.source_available}")
    console.print(f"Records ingested: {result.records_ingested}")
    console.print(f"Municipality features written: {result.municipality_features_written}")
    console.print(f"Issues: {result.issue_count}")
    if not result.source_available:
        console.print(
            "[yellow]Note:[/yellow] Set REINFOLIB_API_KEY to enable real land-price data."
        )
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count > 0:
        raise typer.Exit(1)


@app.command("ingest-healthcare-supply")
def ingest_healthcare_supply_command() -> None:
    """Ingest real healthcare supply data from Yahoo Local Search (graceful unavailable if no key)."""

    result = run_healthcare_supply_ingestion()
    console.print(f"Study area: {result.study_area_id}")
    console.print(f"Source available: {result.source_available}")
    console.print(f"Records ingested: {result.records_ingested}")
    console.print(f"Municipality features written: {result.municipality_features_written}")
    console.print(f"Issues: {result.issue_count}")
    if not result.source_available:
        console.print(
            "[yellow]Note:[/yellow] Set YAHOO_CLIENT_ID to enable real healthcare supply data."
        )
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count > 0:
        raise typer.Exit(1)


@app.command("build-enriched-feature-base")
def build_enriched_feature_base_command() -> None:
    """Build enriched municipality feature base joining land and healthcare supply."""

    result = run_enriched_feature_base()
    console.print(f"Study area: {result.study_area_id}")
    console.print(f"Enriched records written: {result.records_written}")
    console.print(f"Issues: {result.issue_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")


@app.command("build-enriched-score-layer")
def build_enriched_score_layer_command() -> None:
    """Build enriched municipality score layer from enriched feature base."""

    result = run_enriched_score_layer()
    console.print(f"Study area: {result.study_area_id}")
    console.print(f"Scores written: {result.scores_written}")
    console.print(f"Newly available components: {result.newly_available_components or ['(none)']}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Blocking errors: {result.blocking_error_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count > 0:
        raise typer.Exit(1)


@app.command("generate-candidate-actions")
def generate_candidate_actions_command() -> None:
    """Generate deterministic municipality/action candidates (E0, Phase 7)."""

    result = run_candidate_generation()
    console.print(f"Study area: {result.study_area_id}")
    console.print(f"Candidates written: {result.candidates_written}")
    console.print(f"Evidence bundles written: {result.evidence_bundles_written}")
    for action, count in result.candidate_counts_by_action.items():
        console.print(f"  {action}: {count}")
    console.print(f"Municipalities evaluated: {result.municipalities_evaluated}")
    console.print(f"Municipalities with no action: {result.municipalities_with_no_action}")
    console.print(f"Issues: {result.issue_count}")
    console.print(f"Blocking errors: {result.blocking_error_count}")
    for label, path in result.output_paths.items():
        console.print(f"{label}: {path}")
    if result.blocking_error_count > 0:
        raise typer.Exit(1)


@app.command("validate-views")
def validate_views(
    require_outputs: bool = typer.Option(
        False,
        "--require-outputs",
        help="Return nonzero if analysis-view outputs are absent.",
    ),
) -> None:
    """Validate analysis-ready source view artifacts."""

    summary = validate_analysis_views(require_outputs=require_outputs)
    console.print(f"Checked outputs: {len(summary.checked_outputs)}")
    console.print(f"Missing outputs: {len(summary.missing_outputs)}")
    console.print(f"Hospital facts: {summary.hospital_fact_count}")
    console.print(f"Population long records: {summary.population_long_count}")
    console.print(f"Population rate records: {summary.population_rate_count}")
    console.print(f"Population geography keys: {summary.population_geography_key_count}")
    console.print(f"Study-area count records: {summary.study_area_population_long_count}")
    console.print(f"Study-area rate records: {summary.study_area_population_rate_count}")
    console.print(f"Study-area geography keys: {summary.study_area_geography_key_count}")
    console.print(f"Study-area issues: {summary.study_area_issue_count}")
    console.print(f"Geography-grain records: {summary.geography_grain_record_count}")
    console.print(f"Population-base records: {summary.population_base_record_count}")
    console.print(f"Coverage matrix rows: {summary.coverage_matrix_row_count}")
    console.print(f"Coverage issues: {summary.coverage_issue_count}")
    console.print(f"Age-normalized records: {summary.age_group_record_count}")
    console.print(f"Age-group issues: {summary.age_group_issue_count}")
    console.print(f"Quality issues: {summary.quality_issue_count}")
    console.print(f"Rate issues: {summary.population_rate_issue_count}")
    console.print(f"Geography issues: {summary.population_geography_issue_count}")
    console.print(f"Warnings: {len(summary.warnings)}")
    if summary.errors:
        for error in summary.errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)


@app.command("check-live-agent-providers")
def check_live_agent_providers_command() -> None:
    """Probe every live provider and harness; write outputs/provider_preflight/."""

    from geo_strategist.providers.preflight import run_provider_preflight

    result = run_provider_preflight(Path("."))
    table = Table(title="Provider preflight")
    table.add_column("provider")
    table.add_column("model")
    table.add_column("endpoint")
    table.add_column("status")
    for row in result.provider_status:
        table.add_row(
            str(row.get("provider")), str(row.get("model")),
            str(row.get("endpoint_category")), str(row.get("status")),
        )
    console.print(table)
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Primary live providers OK: {result.all_primary_ok}")
    if not result.all_primary_ok:
        raise typer.Exit(1)


@app.command("run-condition-proposals")
def run_condition_proposals_command(
    conditions: str = typer.Option(
        "C0,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14", "--conditions",
        help="Comma-separated condition groups to run (canonical C0-C14).",
    ),
    output_dir: Path = typer.Option(
        Path("outputs/condition_proposals/live"), "--output-dir",
    ),
    top_k_sites: int = typer.Option(5, "--top-k-sites"),
    max_review_rounds: int = typer.Option(2, "--max-review-rounds"),
    require_live_agents: bool = typer.Option(
        True, "--require-live-agents/--allow-non-live-agents",
        help="Never replace failed live-agent runs with deterministic rankings.",
    ),
    disable_deterministic_fallback_for_comparison: bool = typer.Option(
        True,
        "--disable-deterministic-fallback-for-comparison/--allow-deterministic-fallback-for-debug",
        help="Skip even debug-only deterministic fallback reports.",
    ),
    branch_objectives: Optional[str] = typer.Option(
        None, "--branch-objectives",
        help="Comma-separated branch objectives (default: the five shared objectives).",
    ),
    skip_e13: bool = typer.Option(False, "--skip-judge", "--skip-e13"),
    manual_result: Optional[Path] = typer.Option(
        None, "--manual-result",
        help="Path to a manual-harness manual_result.json to ingest for the requested condition.",
    ),
    candidate_deliberation: bool = typer.Option(
        True, "--candidate-deliberation/--no-candidate-deliberation",
        help="Run the shared candidate-level critical-reviewer deliberation "
             "pipeline after each eligible condition's base slate. Never runs "
             "for the deterministic baseline (C0) or any Vanilla LLM "
             "condition (C1-C4, algorithm == \"vanilla_llm\") — those remain "
             "true single-pass baselines even when this flag is on.",
    ),
    allow_candidate_replacement: bool = typer.Option(
        False, "--allow-candidate-replacement",
        help="Let the deliberation pipeline swap a candidate with >=2 blocking "
             "replace/reject findings for a reviewed replacement (off by default).",
    ),
    auto_agentic_harness: bool = typer.Option(
        False, "--auto-agentic-harness/--no-auto-agentic-harness",
        help="Opt in to non-interactive CLI execution for C2/C3/C5-C8/C9-C12 adapters that support it.",
    ),
) -> None:
    """Generate a proposal report per C0-C14 condition plus the E13 comparison."""

    from geo_strategist.experiments.run_condition_proposals import run_condition_proposals

    result = run_condition_proposals(
        Path("."),
        conditions=[item.strip() for item in conditions.split(",") if item.strip()],
        output_dir=output_dir,
        top_k_sites=top_k_sites,
        max_review_rounds=max_review_rounds,
        require_live_agents=require_live_agents,
        disable_deterministic_fallback_for_comparison=disable_deterministic_fallback_for_comparison,
        branch_objectives_csv=branch_objectives,
        run_e13=not skip_e13,
        manual_result_path=manual_result,
        enable_candidate_deliberation=candidate_deliberation,
        allow_candidate_replacement=allow_candidate_replacement,
        auto_agentic_harness=auto_agentic_harness,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Conditions run: {', '.join(result.conditions_run)}")
    console.print(f"Condition comparison report: {result.e13_report_path}")
    console.print(f"Summary: {result.summary_path}")


@app.command("write-manual-harness-prompts")
def write_manual_harness_prompts_command(
    output_dir: Path = typer.Option(
        Path("outputs/condition_proposals/live"), "--output-dir",
    ),
) -> None:
    """Write C2/C3/C5-C8 handoff prompts and C9-C12 Skills launcher files,
    plus the manual_harness/README.md index (always regenerated from the
    current prompt-generation source, never left stale)."""

    from geo_strategist.harnesses.prompts import (
        write_manual_harness_prompts,
        write_manual_harness_readme,
    )

    root = Path(".").resolve()
    live_dir = output_dir if output_dir.is_absolute() else root / output_dir
    written = write_manual_harness_prompts(root, output_dir=live_dir)
    for condition, path in written.items():
        console.print(f"{condition}: {path}")
    readme_path = write_manual_harness_readme(live_dir)
    console.print(f"README: {readme_path}")


@app.command("run-condition-comparison-judge")
@app.command("run-e13-llm-workflow-judge", hidden=True, deprecated=True)
def run_condition_comparison_judge_command(
    proposals_dir: Optional[Path] = typer.Option(
        None, "--proposals-dir",
        help="Orchestrator output dir with condition_records.jsonl "
             "(default: outputs/condition_proposals/live).",
    ),
    allow_live_judge: bool = typer.Option(
        True, "--allow-live-judge/--structural-judge-only",
        help="Run the LLM checklist judge (the primary qualitative score, "
             "0-25) in addition to the always-on deterministic checklist "
             "(auxiliary, never combined into the primary score). Makes "
             "live judge CLI/API calls by default; pass "
             "--structural-judge-only (or set E13_DISABLE_LIVE_JUDGE=1) for "
             "a fully offline run with only the deterministic checklist.",
    ),
) -> None:
    """Run the E13 condition-comparison judge across available C0-C14 records.

    The primary score is the report-visible decision-analysis-quality checklist:
    each comparable condition's sanitized final Markdown report only (never
    provider/model/algorithm metadata) is scored 0-5 on each of five
    categories (0-25 total, ``llm_points_total`` — the ranking metric). The
    25-item deterministic document checklist is reported separately as
    auxiliary structural validation.
    """

    from geo_strategist.experiments.condition_comparison_judge import run_condition_comparison_judge

    result = run_condition_comparison_judge(
        Path("."), proposals_dir=proposals_dir, allow_live_judge=allow_live_judge,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Scored conditions: {result.scored_condition_count}")
    console.print(f"Live judge enabled: {result.live_judge_enabled}")
    console.print(f"Comparison report: {result.comparison_report_path}")


@app.command("validate-c5-result")
def validate_c5_result_command(
    result_path: Path = typer.Argument(
        Path("outputs/condition_proposals/live/runs/C05/manual_result.json"),
        help="Path to the manual_result.json to validate.",
    ),
    expected_condition_group: str = typer.Option(
        "C5", "--expected-condition-group",
        help="Reused for C6/C7/C8 by passing the matching group.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the machine-readable report instead of the human summary.",
    ),
) -> None:
    """Validate a C5-C8 coding-agent-control manual_result.json.

    Checks JSON validity, condition_group, candidate-id/count/uniqueness
    against candidate_actions.jsonl, the full per-candidate
    qualitative_discussion contract, rationale length, evidence/fabrication
    markers (asserted travel times, invented land/construction/regulatory
    facts, unqualified currency claims), presence of generated_code/, and
    signals of C0 deterministic-baseline substitution. Exits nonzero on any
    error or C0-substitution flag so a repair loop can act on the result.
    """

    from geo_strategist.harnesses.c5_result_validator import validate_c5_result

    report = validate_c5_result(
        Path(result_path), repo_root=Path("."), expected_condition_group=expected_condition_group,
    )
    if json_output:
        console.print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        console.print(report.human_report())
    if not report.ok:
        raise typer.Exit(1)


@app.command("validate-skills-result")
def validate_skills_result_command(
    result_path: Path = typer.Argument(
        Path("outputs/condition_proposals/live/runs/C09/manual_result.json"),
        help="Path to the manual_result.json to validate.",
    ),
    expected_condition_group: str = typer.Option(
        "C9", "--expected-condition-group",
        help="Reused for C10/C11/C12 by passing the matching group.",
    ),
    json_output: bool = typer.Option(
        False, "--json", help="Print the machine-readable report instead of the human summary.",
    ),
) -> None:
    """Validate a C9-C12 Skills-unified manual_result.json.

    Runs the same ranked_candidates/qualitative_discussion/evidence/
    C0-substitution checks as validate-c5-result, plus the Skills-unified
    skill_trace contract (hypothesis completeness, branch lineage, debug
    depth, reviewer artifacts, final-synthesis traceability) via the same
    real trace validator used at ingestion. Only fundamental (unsupported-
    claim) trace issues are hard errors; other trace-shape/lifecycle
    differences are reported as deviations, not failures. Exits nonzero on
    any error or C0-substitution flag so a repair loop can act on the result.
    """

    from geo_strategist.harnesses.skills_result_validator import validate_skills_agent_result

    report = validate_skills_agent_result(
        Path(result_path), repo_root=Path("."), expected_condition_group=expected_condition_group,
    )
    if json_output:
        console.print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    else:
        console.print(report.human_report())
    if not report.ok:
        raise typer.Exit(1)


@app.command("antigravity-preflight")
def antigravity_preflight_command(
    model: Optional[str] = typer.Option(
        None, "--model", help="Registry model string to resolve (default: C5_MODEL/C9_MODEL from the registry).",
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Bounded smoke test that a full C5 Antigravity run is actually possible.

    Checks the agy binary, version, model-slug resolution, a trivial
    --print round trip, a real repository file read, and a test-directory
    write — before committing to a long unattended run. Does not exercise
    tool-approval behavior (headless mode has none; see
    harnesses/antigravity_support.py).
    """

    from geo_strategist.experiments.condition_registry import build_condition_registry
    from geo_strategist.harnesses.antigravity_support import run_antigravity_preflight

    configured_model = model
    if configured_model is None:
        configured_model = build_condition_registry()["C5"].model
    result = run_antigravity_preflight(Path(".").resolve(), configured_model=configured_model)
    if json_output:
        console.print(json.dumps({
            "ok": result.ok, "checks": result.checks, "notes": result.notes,
            "resolved_model": result.resolved_model, "detail": result.detail,
        }, ensure_ascii=False, indent=2))
    else:
        console.print(f"Preflight: {'PASS' if result.ok else 'FAIL'} ({result.detail})")
        for name, passed in result.checks.items():
            console.print(f"  [{'x' if passed else ' '}] {name}")
        for note in result.notes:
            console.print(f"  note: {note}")
    if not result.ok:
        raise typer.Exit(1)


@app.command("build-run-lineage-index")
def build_run_lineage_index_command() -> None:
    """Build validation report index, run lineage registry, and latest marker."""

    from geo_strategist.experiments.run_lineage import build_run_lineage_index

    index = build_run_lineage_index(Path("."))
    console.print(f"Lineage records: {index['record_count']}")
    console.print("Registry: .runs/registry/run_lineage_registry.json")


@app.command("build-municipality-master")
def build_municipality_master_command() -> None:
    """Build municipality master records with stable join keys."""

    from geo_strategist.data.municipality_master import build_municipality_master

    rows = build_municipality_master(Path("."))
    console.print(f"Municipality master records: {len(rows)}")


@app.command("validate-site-sources")
def validate_site_sources_command() -> None:
    """Validate the S1 site-selection source registry (fail-closed by design)."""

    from geo_strategist.data_sources.source_registry import run_source_registry_validation

    result = run_source_registry_validation(Path("."))
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Connected sources: {result.connected_count}")
    console.print(f"Not-configured sources: {result.not_configured_count}")
    console.print(f"Missing/error sources: {result.missing_or_error_count}")


@app.command("generate-site-candidates")
def generate_site_candidates_command(
    area: str = typer.Option("Tokyo,Aichi,Osaka", "--area", help="Informational only; the study area is fixed by configs/study_area_tokyo_aichi_osaka.yaml."),
    mode: str = typer.Option("public_dataset_mode", "--mode", help="public_dataset_mode | manual_source_mode | database_mode"),
    top_n: int = typer.Option(10, "--top-n", help="Top-N municipalities to consider for greenfield build candidates."),
    manual_source_path: Optional[Path] = typer.Option(None, "--manual-source-path"),
) -> None:
    """S2: generate real, source-traceable candidate sites."""

    from geo_strategist.experiments.s2_candidate_site_generation import run_s2_candidate_site_generation

    result = run_s2_candidate_site_generation(
        Path("."), mode=mode, top_n_build_municipalities=top_n, manual_source_path=manual_source_path,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Candidates generated: {result.candidate_count}")


@app.command("engineer-site-features")
def engineer_site_features_command() -> None:
    """S3: compute demand/supply/land/financial/risk features for each candidate site."""

    from geo_strategist.experiments.s3_site_feature_engineering import run_s3_site_feature_engineering

    result = run_s3_site_feature_engineering(Path("."))
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Feature records: {result.feature_record_count}")


@app.command("run-site-tree-search")
def run_site_tree_search_command(
    beam_width: int = typer.Option(5, "--beam-width"),
    max_depth: int = typer.Option(4, "--max-depth"),
    top_k_sites: int = typer.Option(10, "--top-k-sites"),
    min_evidence_score: float = typer.Option(0.0, "--min-evidence-score"),
) -> None:
    """S4: agentic tree search over real candidate sites and financial scenarios."""

    from geo_strategist.experiments.s4_site_tree_search import run_s4_site_tree_search

    result = run_s4_site_tree_search(
        Path("."), beam_width=beam_width, max_depth=max_depth, top_k_sites=top_k_sites, min_evidence_score=min_evidence_score,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Nodes: {result.node_count}")
    console.print(f"Selected top-k sites: {result.selected_count}")


@app.command("write-site-selection-proposals")
def write_site_selection_proposals_command(
    top_k_sites: Optional[int] = typer.Option(None, "--top-k-sites"),
    allow_scenario_assumptions: bool = typer.Option(True, "--allow-scenario-assumptions/--no-allow-scenario-assumptions"),
    require_parcel_id: bool = typer.Option(False, "--require-parcel-id"),
) -> None:
    """S5: write decision-support site-selection/investment proposal reports."""

    from geo_strategist.experiments.s5_site_selection_proposal_report import run_s5_site_selection_proposal_report

    result = run_s5_site_selection_proposal_report(
        Path("."), top_k_sites=top_k_sites, allow_scenario_assumptions=allow_scenario_assumptions, require_parcel_id=require_parcel_id,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Proposals written: {result.proposal_count}")


@app.command("run-e14-site-selection-proposal-judge")
def run_e14_site_selection_proposal_judge_command() -> None:
    """E14: AI-Scientist-style review board for site-selection/investment proposals."""

    from geo_strategist.experiments.site_selection_review_judge import run_e14_site_selection_proposal_judge

    result = run_e14_site_selection_proposal_judge(Path("."))
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Proposals reviewed: {result.proposal_count}")
    console.print(f"Reviews written: {result.review_count}")


@app.command("run-site-selection-end-to-end")
def run_site_selection_end_to_end_command(
    area: str = typer.Option("Tokyo,Aichi,Osaka", "--area"),
    mode: str = typer.Option("public_dataset_mode", "--mode"),
    beam_width: int = typer.Option(5, "--beam-width"),
    max_depth: int = typer.Option(4, "--max-depth"),
    top_k_sites: int = typer.Option(5, "--top-k-sites"),
    max_review_rounds: int = typer.Option(2, "--max-review-rounds"),
    min_evidence_score: float = typer.Option(0.0, "--min-evidence-score"),
    allow_scenario_assumptions: bool = typer.Option(True, "--allow-scenario-assumptions/--no-allow-scenario-assumptions"),
    require_parcel_id: bool = typer.Option(False, "--require-parcel-id"),
    allow_unverified_candidates_for_draft: bool = typer.Option(True, "--allow-unverified-candidates-for-draft/--no-allow-unverified-candidates-for-draft"),
) -> None:
    """S2-S7/E14: run the full site-selection/investment decision-support pipeline."""

    from geo_strategist.experiments.s7_site_selection_pipeline import run_s7_site_selection_pipeline

    result = run_s7_site_selection_pipeline(
        Path("."),
        mode=mode,
        beam_width=beam_width,
        max_search_depth=max_depth,
        top_k_sites=top_k_sites,
        max_review_rounds=max_review_rounds,
        min_evidence_score=min_evidence_score,
        allow_scenario_assumptions=allow_scenario_assumptions,
        require_parcel_id=require_parcel_id,
        allow_unverified_candidates_for_draft=allow_unverified_candidates_for_draft,
    )
    console.print(f"Run ID: {result.run_id}")
    console.print(f"Output dir: {result.output_dir}")
    console.print(f"Rounds run: {result.rounds_run}")
    console.print(f"Final proposal count: {result.final_proposal_count}")
    console.print(f"Final revision requests remaining: {result.final_revision_request_count}")


@app.command("show-paths")
def show_paths() -> None:
    """Show canonical local-only paths used by this phase."""

    table = Table(title="Geo Strategist Paths")
    table.add_column("Purpose")
    table.add_column("Path")
    table.add_row("Manual inputs", ".data/manual")
    table.add_row("Hospital workbook", str(Path(".data/manual/hospital_cf_workbook")))
    table.add_row("Population workbooks", str(Path(".data/manual/population_data")))
    table.add_row("API raw responses", ".data/api_raw")
    table.add_row("Normalized interim", ".data/interim/normalized")
    table.add_row("Analysis views", ".data/interim/views")
    table.add_row("Population QA", ".data/interim/qa")
    table.add_row("Study area views", ".data/interim/study_area/tokyo_aichi_osaka")
    table.add_row("Inventory cache", ".cache/inventory")
    table.add_row("Inspection cache", ".cache/inspection")
    table.add_row("Normalization cache", ".cache/normalization")
    table.add_row("Review cache", ".cache/review")
    table.add_row("View cache", ".cache/views")
    table.add_row("QA cache", ".cache/qa")
    table.add_row("Study area cache", ".cache/study_area/tokyo_aichi_osaka")
    table.add_row("Run outputs", ".runs")
    table.add_row("Legacy fallback", "data/manual")
    console.print(table)


if __name__ == "__main__":
    app()
