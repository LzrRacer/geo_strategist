"""Workbook-backed deterministic cash-flow substrate.

This module builds reusable economic features for candidate actions. It uses
existing candidate/evidence artifacts, workbook-derived hospital finance
features, and explicit scenario assumptions. It does not generate proposals,
select exact sites, call APIs, or issue final recommendations.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import yaml


ALLOWED_ACTIONS = ("build", "reorganize", "consolidate")
RUN_ROOT = Path(".runs/experiments/cash_flow_substrate")
DEFAULT_CONFIG_PATH = Path("configs/cash_flow_assumptions.yaml")
DEFAULT_CANDIDATES_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl")
DEFAULT_BUNDLES_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/candidate_evidence_bundles.jsonl")
DEFAULT_HOSPITAL_FEATURES_PATH = Path(".data/interim/study_area/tokyo_aichi_osaka/hospital_features.jsonl")
PHASE_12_LIMITATION = (
    "Phase 12 inspected e-Stat 医療施設調査 tables 0003026802, 0003027893, "
    "and 0003027909; none had an area dimension or target municipality JIS "
    "codes, so official municipality-level e-Stat medical-facility counts are "
    "not currently available from those inspected tables."
)


@dataclass(frozen=True)
class CashFlowIssue:
    issue_id: str
    severity: str
    issue_code: str
    message: str
    candidate_id: str | None = None
    candidate_action: str | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NumericTrace:
    source_type: str
    source_artifact: str
    source_field: str
    formula_id: str
    input_fields: tuple[str, ...]
    scenario_name: str
    value_basis: str
    generated_at: str
    source_file_hash: str | None = None


@dataclass(frozen=True)
class CashFlowComponent:
    component_id: str
    candidate_id: str
    candidate_action: str
    scenario_name: str
    component_name: str
    value: float | None
    unit: str
    provenance: NumericTrace
    issue_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CashFlowSensitivityRecord:
    sensitivity_id: str
    candidate_id: str
    candidate_action: str
    prefecture: str
    municipality: str
    scenario_name: str
    capex_multiplier: float
    revenue_multiplier: float
    expense_multiplier: float
    total_initial_investment_jpy_mm: float | None
    annual_net_cash_flow_jpy_mm: float | None
    payback_years: float | None
    payback_status: str
    issue_codes: tuple[str, ...]


@dataclass(frozen=True)
class CashFlowRecord:
    record_id: str
    candidate_id: str
    study_area_id: str
    prefecture: str
    municipality: str
    candidate_action: str
    reference_prefecture: str
    reference_bed_scale: float | None
    estimated_required_beds: float | None
    land_price_jpy_per_sqm: float | None
    land_price_source_year: int | None
    base_total_initial_investment_jpy_mm: float | None
    base_annual_revenue_jpy_mm: float | None
    base_annual_operating_expense_jpy_mm: float | None
    base_annual_net_cash_flow_jpy_mm: float | None
    base_payback_years: float | None
    payback_status: str
    computable_capex: bool
    computable_annual_cash_flow: bool
    computable_payback: bool
    issue_codes: tuple[str, ...]
    source_artifact_refs: tuple[str, ...]
    generated_at: str


@dataclass(frozen=True)
class PrefectureReferenceAssumption:
    prefecture: str
    source_type: str
    source_artifact: str
    source_file_hash: str | None
    source_record_count: int
    reference_beds: float | None
    construction_cost_per_bed_jpy_mm: float | None
    equipment_it_per_bed_jpy_mm: float | None
    land_area_sqm_per_bed: float | None
    working_capital_per_bed_jpy_mm: float | None
    annual_revenue_per_bed_jpy_mm: float | None
    annual_cash_expense_per_bed_jpy_mm: float | None
    contingency_rate: float | None
    scenario_label: str = "workbook_derived_reference"


@dataclass(frozen=True)
class CashFlowRunResult:
    run_id: str
    output_dir: str
    candidate_count: int
    counts_by_action: dict[str, int]
    computable_capex_count: int
    computable_annual_cash_flow_count: int
    computable_payback_count: int
    payback_status_counts: dict[str, int]
    issue_counts_by_severity: dict[str, int]
    issue_counts_by_code: dict[str, int]
    output_paths: dict[str, str]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_to_jsonable(row), ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    n = _num(numerator)
    d = _num(denominator)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _median(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    return float(median(clean)) if clean else None


def _issue(
    severity: str,
    issue_code: str,
    message: str,
    candidate_id: str | None = None,
    candidate_action: str | None = None,
    context: dict[str, Any] | None = None,
) -> CashFlowIssue:
    return CashFlowIssue(
        issue_id=str(uuid.uuid5(
            uuid.NAMESPACE_URL,
            json.dumps(
                {
                    "severity": severity,
                    "issue_code": issue_code,
                    "candidate_id": candidate_id,
                    "candidate_action": candidate_action,
                    "context": context or {},
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )),
        severity=severity,
        issue_code=issue_code,
        message=message,
        candidate_id=candidate_id,
        candidate_action=candidate_action,
        context=context or {},
    )


def build_prefecture_reference_assumptions(
    hospital_features: list[dict[str, Any]],
    source_artifact: str,
) -> dict[str, PrefectureReferenceAssumption]:
    """Build source-backed reference ratios from workbook-derived hospital rows."""
    by_pref: dict[str, list[dict[str, Any]]] = {}
    for row in hospital_features:
        pref = row.get("prefecture")
        if not pref:
            continue
        by_pref.setdefault(str(pref), []).append(row)

    refs: dict[str, PrefectureReferenceAssumption] = {}
    for pref, rows in sorted(by_pref.items()):
        source_hashes = sorted({str(r.get("source_file_hash")) for r in rows if r.get("source_file_hash")})
        refs[pref] = PrefectureReferenceAssumption(
            prefecture=pref,
            source_type="hospital_workbook_derived_feature",
            source_artifact=source_artifact,
            source_file_hash=source_hashes[0] if len(source_hashes) == 1 else None,
            source_record_count=len(rows),
            reference_beds=_median([_num(r.get("beds_used_in_model")) for r in rows]),
            construction_cost_per_bed_jpy_mm=_median([
                _ratio(r.get("construction_cost_jpy_mm"), r.get("beds_used_in_model"))
                for r in rows
            ]),
            equipment_it_per_bed_jpy_mm=_median([
                _ratio(r.get("equipment_it_jpy_mm"), r.get("beds_used_in_model"))
                for r in rows
            ]),
            land_area_sqm_per_bed=_median([
                _ratio(r.get("land_area_sqm"), r.get("beds_used_in_model"))
                for r in rows
            ]),
            working_capital_per_bed_jpy_mm=_median([
                _ratio(r.get("working_capital_jpy_mm"), r.get("beds_used_in_model"))
                for r in rows
            ]),
            annual_revenue_per_bed_jpy_mm=_median([
                _ratio(r.get("estimated_revenue_jpy_mm"), r.get("beds_used_in_model"))
                for r in rows
            ]),
            annual_cash_expense_per_bed_jpy_mm=_median([
                _ratio(r.get("cash_expenses_jpy_mm"), r.get("beds_used_in_model"))
                for r in rows
            ]),
            contingency_rate=_median([
                _ratio(
                    r.get("contingency_jpy_mm"),
                    (_num(r.get("construction_cost_jpy_mm")) or 0.0)
                    + (_num(r.get("equipment_it_jpy_mm")) or 0.0),
                )
                for r in rows
            ]),
        )
    return refs


def _trace(
    *,
    source_type: str,
    source_artifact: str,
    source_field: str,
    formula_id: str,
    input_fields: tuple[str, ...],
    scenario_name: str,
    value_basis: str,
    generated_at: str,
    source_file_hash: str | None = None,
) -> NumericTrace:
    return NumericTrace(
        source_type=source_type,
        source_artifact=source_artifact,
        source_field=source_field,
        formula_id=formula_id,
        input_fields=input_fields,
        scenario_name=scenario_name,
        value_basis=value_basis,
        generated_at=generated_at,
        source_file_hash=source_file_hash,
    )


def _component(
    candidate_id: str,
    action: str,
    scenario: str,
    name: str,
    value: float | None,
    unit: str,
    provenance: NumericTrace,
    issue_codes: tuple[str, ...] = (),
) -> CashFlowComponent:
    return CashFlowComponent(
        component_id=f"cf_component:{candidate_id}:{scenario}:{name}",
        candidate_id=candidate_id,
        candidate_action=action,
        scenario_name=scenario,
        component_name=name,
        value=value,
        unit=unit,
        provenance=provenance,
        issue_codes=issue_codes,
    )


def classify_payback(
    total_initial_investment_jpy_mm: float | None,
    annual_net_cash_flow_jpy_mm: float | None,
    target_years: float,
    near_target_multiplier: float,
) -> tuple[float | None, str]:
    if (
        total_initial_investment_jpy_mm is None
        or annual_net_cash_flow_jpy_mm is None
        or annual_net_cash_flow_jpy_mm <= 0
    ):
        return None, "not_computable"
    years = total_initial_investment_jpy_mm / annual_net_cash_flow_jpy_mm
    if years <= target_years:
        return years, "meets_target"
    if years <= target_years * near_target_multiplier:
        return years, "near_target"
    return years, "misses_target"


def _scenario_value(config: dict[str, Any], scenario: str, field_name: str) -> float:
    return float(config["sensitivity_scenarios"][scenario][field_name])


def _action_factor(config: dict[str, Any], action: str) -> float:
    return float(config["action_scope_factors"][action]["value"])


def _candidate_land_price(bundle: dict[str, Any]) -> tuple[float | None, int | None, str | None]:
    land = bundle.get("land_features_summary", {})
    price = _num(land.get("land_price_median"))
    year = land.get("land_price_latest_year")
    return price, int(year) if year is not None else None, land.get("land_price_unit")


def _calculate_candidate(
    candidate: dict[str, Any],
    bundle: dict[str, Any] | None,
    ref: PrefectureReferenceAssumption | None,
    config: dict[str, Any],
    generated_at: str,
    input_paths: dict[str, str],
) -> tuple[CashFlowRecord | None, list[CashFlowComponent], list[CashFlowSensitivityRecord], list[CashFlowIssue]]:
    candidate_id = str(candidate.get("candidate_id", ""))
    action = str(candidate.get("candidate_action", ""))
    issues: list[CashFlowIssue] = []
    components: list[CashFlowComponent] = []
    sensitivities: list[CashFlowSensitivityRecord] = []

    if action not in ALLOWED_ACTIONS:
        issues.append(_issue(
            "error",
            "unknown_candidate_action",
            f"Candidate action '{action}' is not allowed.",
            candidate_id=candidate_id or None,
            candidate_action=action or None,
            context={"allowed_candidate_actions": list(ALLOWED_ACTIONS)},
        ))
        return None, components, sensitivities, issues

    if bundle is None:
        issues.append(_issue(
            "error",
            "candidate_evidence_bundle_missing",
            "Candidate has no matching evidence bundle.",
            candidate_id=candidate_id,
            candidate_action=action,
        ))
        return None, components, sensitivities, issues

    pref = str(candidate.get("prefecture", ""))
    muni = str(candidate.get("municipality", ""))
    if ref is None:
        issues.append(_issue(
            "error",
            "prefecture_workbook_reference_missing",
            "No workbook-derived finance reference exists for the candidate prefecture.",
            candidate_id=candidate_id,
            candidate_action=action,
            context={"prefecture": pref},
        ))
        return None, components, sensitivities, issues

    factor = _action_factor(config, action)
    required_beds = ref.reference_beds * factor if ref.reference_beds is not None else None
    if required_beds is None:
        issues.append(_issue(
            "error",
            "required_beds_not_computable",
            "Required bed scale could not be computed from workbook reference beds and action scope factor.",
            candidate_id=candidate_id,
            candidate_action=action,
        ))

    land_price, land_year, land_unit = _candidate_land_price(bundle)
    if action == "build" and land_price is None:
        issues.append(_issue(
            "warning",
            "land_price_missing_for_build",
            "Build land acquisition cost is not computable because candidate land price is unavailable.",
            candidate_id=candidate_id,
            candidate_action=action,
        ))
    if land_unit not in (None, "JPY/m2"):
        issues.append(_issue(
            "warning",
            "unexpected_land_price_unit",
            f"Land price unit is '{land_unit}', expected JPY/m2.",
            candidate_id=candidate_id,
            candidate_action=action,
        ))

    target_years = float(config["payback_status"]["target_payback_years"]["value"])
    near_mult = float(config["payback_status"]["near_target_multiplier"]["value"])
    source_refs = tuple(str(x) for x in bundle.get("source_artifact_refs", []))
    candidate_issue_codes: list[str] = []
    base_values: dict[str, float | None] = {}

    for scenario in sorted(config["sensitivity_scenarios"].keys()):
        capex_mult = _scenario_value(config, scenario, "capex_multiplier")
        revenue_mult = _scenario_value(config, scenario, "revenue_multiplier")
        expense_mult = _scenario_value(config, scenario, "expense_multiplier")

        construction = (
            required_beds * ref.construction_cost_per_bed_jpy_mm * capex_mult
            if action == "build"
            and required_beds is not None
            and ref.construction_cost_per_bed_jpy_mm is not None
            else None
        )
        equipment = (
            required_beds * ref.equipment_it_per_bed_jpy_mm * capex_mult
            if required_beds is not None and ref.equipment_it_per_bed_jpy_mm is not None
            else None
        )
        land_area = (
            required_beds * ref.land_area_sqm_per_bed
            if action == "build"
            and required_beds is not None
            and ref.land_area_sqm_per_bed is not None
            else None
        )
        land_cost = (
            land_area * land_price / 1_000_000.0 * capex_mult
            if action == "build" and land_area is not None and land_price is not None
            else None
        )
        working_capital = (
            required_beds * ref.working_capital_per_bed_jpy_mm
            if required_beds is not None and ref.working_capital_per_bed_jpy_mm is not None
            else None
        )
        contingency_base = sum(v for v in (construction, equipment) if v is not None)
        contingency = (
            contingency_base * ref.contingency_rate
            if ref.contingency_rate is not None and contingency_base > 0
            else None
        )
        capex_inputs = [equipment, working_capital, contingency]
        if action == "build":
            capex_inputs.extend([construction, land_cost])
        total_initial = sum(v for v in capex_inputs if v is not None)
        if not capex_inputs or any(v is None for v in capex_inputs):
            total_initial_value: float | None = None
            issues.append(_issue(
                "warning",
                "total_initial_investment_not_computable",
                "Total initial investment is null because at least one required capex component is unavailable.",
                candidate_id=candidate_id,
                candidate_action=action,
                context={"scenario_name": scenario},
            ))
        else:
            total_initial_value = total_initial

        revenue = (
            required_beds * ref.annual_revenue_per_bed_jpy_mm * revenue_mult
            if required_beds is not None and ref.annual_revenue_per_bed_jpy_mm is not None
            else None
        )
        expense = (
            required_beds * ref.annual_cash_expense_per_bed_jpy_mm * expense_mult
            if required_beds is not None and ref.annual_cash_expense_per_bed_jpy_mm is not None
            else None
        )
        net_cf = revenue - expense if revenue is not None and expense is not None else None
        payback_years, payback_status = classify_payback(
            total_initial_value,
            net_cf,
            target_years=target_years,
            near_target_multiplier=near_mult,
        )
        scenario_issue_codes = tuple(i.issue_code for i in issues if i.candidate_id == candidate_id)

        ref_artifact = ref.source_artifact
        ref_hash = ref.source_file_hash
        components.extend([
            _component(candidate_id, action, scenario, "estimated_required_beds", required_beds, "beds", _trace(
                source_type="scenario_assumption_and_workbook_reference",
                source_artifact=f"{ref_artifact}; {DEFAULT_CONFIG_PATH}",
                source_field="reference_beds; action_scope_factors",
                formula_id="required_beds = reference_beds * action_scope_factor",
                input_fields=("reference_beds", "action_scope_factor"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "construction_capex_jpy_mm", construction, "JPY_mm", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="construction_cost_jpy_mm / beds_used_in_model",
                formula_id="construction_capex = required_beds * construction_cost_per_bed * capex_multiplier",
                input_fields=("required_beds", "construction_cost_per_bed_jpy_mm", "capex_multiplier"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "equipment_it_capex_jpy_mm", equipment, "JPY_mm", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="equipment_it_jpy_mm / beds_used_in_model",
                formula_id="equipment_it_capex = required_beds * equipment_it_per_bed * capex_multiplier",
                input_fields=("required_beds", "equipment_it_per_bed_jpy_mm", "capex_multiplier"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "land_area_requirement_sqm", land_area, "sqm", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="land_area_sqm / beds_used_in_model",
                formula_id="land_area = required_beds * land_area_sqm_per_bed",
                input_fields=("required_beds", "land_area_sqm_per_bed"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "land_acquisition_cost_jpy_mm", land_cost, "JPY_mm", _trace(
                source_type="candidate_evidence_and_workbook_reference",
                source_artifact=f"{input_paths['candidate_evidence_bundles']}; {ref_artifact}",
                source_field="land_price_median; land_area_sqm_per_bed",
                formula_id="land_cost = land_area * land_price_jpy_per_sqm / 1_000_000 * capex_multiplier",
                input_fields=("land_area_requirement_sqm", "land_price_median", "capex_multiplier"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "working_capital_jpy_mm", working_capital, "JPY_mm", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="working_capital_jpy_mm / beds_used_in_model",
                formula_id="working_capital = required_beds * working_capital_per_bed",
                input_fields=("required_beds", "working_capital_per_bed_jpy_mm"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "contingency_jpy_mm", contingency, "JPY_mm", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="contingency_jpy_mm / (construction_cost_jpy_mm + equipment_it_jpy_mm)",
                formula_id="contingency = (construction_capex + equipment_it_capex) * contingency_rate",
                input_fields=("construction_capex_jpy_mm", "equipment_it_capex_jpy_mm", "contingency_rate"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "total_initial_investment_jpy_mm", total_initial_value, "JPY_mm", _trace(
                source_type="derived",
                source_artifact=f"{ref_artifact}; {input_paths['candidate_evidence_bundles']}",
                source_field="computed_capex_components",
                formula_id="total_initial_investment = sum(required capex components)",
                input_fields=("construction_capex_jpy_mm", "equipment_it_capex_jpy_mm", "land_acquisition_cost_jpy_mm", "working_capital_jpy_mm", "contingency_jpy_mm"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "annual_demand_linked_revenue_jpy_mm", revenue, "JPY_mm_per_year", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="estimated_revenue_jpy_mm / beds_used_in_model",
                formula_id="annual_revenue = required_beds * annual_revenue_per_bed * revenue_multiplier",
                input_fields=("required_beds", "annual_revenue_per_bed_jpy_mm", "revenue_multiplier"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "annual_demand_linked_operating_expense_jpy_mm", expense, "JPY_mm_per_year", _trace(
                source_type="workbook_derived_reference",
                source_artifact=ref_artifact,
                source_field="cash_expenses_jpy_mm / beds_used_in_model",
                formula_id="annual_expense = required_beds * annual_cash_expense_per_bed * expense_multiplier",
                input_fields=("required_beds", "annual_cash_expense_per_bed_jpy_mm", "expense_multiplier"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "annual_net_cash_flow_jpy_mm", net_cf, "JPY_mm_per_year", _trace(
                source_type="derived",
                source_artifact=ref_artifact,
                source_field="annual_demand_linked_revenue_jpy_mm; annual_demand_linked_operating_expense_jpy_mm",
                formula_id="annual_net_cash_flow = annual_revenue - annual_operating_expense",
                input_fields=("annual_demand_linked_revenue_jpy_mm", "annual_demand_linked_operating_expense_jpy_mm"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
            _component(candidate_id, action, scenario, "payback_years", payback_years, "years", _trace(
                source_type="derived",
                source_artifact=f"{ref_artifact}; {DEFAULT_CONFIG_PATH}",
                source_field="total_initial_investment_jpy_mm; annual_net_cash_flow_jpy_mm",
                formula_id="payback_years = total_initial_investment / annual_net_cash_flow",
                input_fields=("total_initial_investment_jpy_mm", "annual_net_cash_flow_jpy_mm"),
                scenario_name=scenario,
                value_basis="derived",
                generated_at=generated_at,
                source_file_hash=ref_hash,
            )),
        ])

        sensitivities.append(CashFlowSensitivityRecord(
            sensitivity_id=f"cf_sensitivity:{candidate_id}:{scenario}",
            candidate_id=candidate_id,
            candidate_action=action,
            prefecture=pref,
            municipality=muni,
            scenario_name=scenario,
            capex_multiplier=capex_mult,
            revenue_multiplier=revenue_mult,
            expense_multiplier=expense_mult,
            total_initial_investment_jpy_mm=total_initial_value,
            annual_net_cash_flow_jpy_mm=net_cf,
            payback_years=payback_years,
            payback_status=payback_status,
            issue_codes=scenario_issue_codes,
        ))

        if scenario == "base":
            base_values = {
                "total_initial": total_initial_value,
                "revenue": revenue,
                "expense": expense,
                "net_cf": net_cf,
                "payback_years": payback_years,
                "payback_status": payback_status,
            }
            candidate_issue_codes = list(scenario_issue_codes)

    record = CashFlowRecord(
        record_id=f"cf_record:{candidate_id}",
        candidate_id=candidate_id,
        study_area_id=str(candidate.get("study_area_id", "")),
        prefecture=pref,
        municipality=muni,
        candidate_action=action,
        reference_prefecture=ref.prefecture,
        reference_bed_scale=ref.reference_beds,
        estimated_required_beds=required_beds,
        land_price_jpy_per_sqm=land_price,
        land_price_source_year=land_year,
        base_total_initial_investment_jpy_mm=base_values.get("total_initial"),
        base_annual_revenue_jpy_mm=base_values.get("revenue"),
        base_annual_operating_expense_jpy_mm=base_values.get("expense"),
        base_annual_net_cash_flow_jpy_mm=base_values.get("net_cf"),
        base_payback_years=base_values.get("payback_years"),
        payback_status=str(base_values.get("payback_status", "not_computable")),
        computable_capex=base_values.get("total_initial") is not None,
        computable_annual_cash_flow=base_values.get("net_cf") is not None,
        computable_payback=base_values.get("payback_years") is not None,
        issue_codes=tuple(sorted(set(candidate_issue_codes))),
        source_artifact_refs=source_refs,
        generated_at=generated_at,
    )
    return record, components, sensitivities, issues


def _assumptions_payload(
    config: dict[str, Any],
    references: dict[str, PrefectureReferenceAssumption],
) -> dict[str, Any]:
    return {
        "config_assumptions": config,
        "workbook_derived_prefecture_references": [
            asdict(ref) for ref in references.values()
        ],
        "limitations": [
            "Workbook-derived references are screening inputs, not investment recommendations.",
            PHASE_12_LIMITATION,
        ],
    }


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cash-Flow Substrate Report",
        "",
        f"Run ID: `{report['run_id']}`",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Candidates processed | {report['candidate_count']} |",
        f"| Computable capex | {report['computable_capex_count']} |",
        f"| Computable annual cash flow | {report['computable_annual_cash_flow_count']} |",
        f"| Computable payback | {report['computable_payback_count']} |",
        "",
        "## Counts By Action",
        "",
    ]
    for action, count in sorted(report["counts_by_action"].items()):
        lines.append(f"- `{action}`: {count}")
    lines += ["", "## Payback Status", ""]
    for status, count in sorted(report["payback_status_counts"].items()):
        lines.append(f"- `{status}`: {count}")
    lines += ["", "## Issues", ""]
    if report["issue_counts_by_code"]:
        for code, count in sorted(report["issue_counts_by_code"].items()):
            lines.append(f"- `{code}`: {count}")
    else:
        lines.append("_No issues._")
    lines += [
        "",
        "## Top Computable Payback Candidates",
        "",
    ]
    top = report.get("top_candidates_by_payback", [])
    if top:
        lines += ["| candidate_id | action | payback_years | status |", "|--------------|--------|---------------|--------|"]
        for row in top:
            lines.append(
                f"| `{row['candidate_id']}` | {row['candidate_action']} | {row['base_payback_years']:.4f} | {row['payback_status']} |"
            )
    else:
        lines.append("_No candidates have computable payback._")
    lines += [
        "",
        "## Assumptions",
        "",
        "- Monetary and bed-scale ratios are derived from workbook-backed hospital features.",
        "- Scenario/action multipliers come from `configs/cash_flow_assumptions.yaml` and are labeled scenario assumptions.",
        "",
        "## Limitations",
        "",
        f"- {PHASE_12_LIMITATION}",
        "- This is a deterministic substrate for later proposal generation; it is not a final recommendation.",
        "- No LLM proposal generation, reviewer agents, tree search, exact parcel/site selection, or final hospital recommendations were implemented.",
        "",
    ]
    return "\n".join(lines)


def build_cash_flow_substrate(
    *,
    candidate_actions_path: Path = DEFAULT_CANDIDATES_PATH,
    candidate_evidence_bundles_path: Path = DEFAULT_BUNDLES_PATH,
    hospital_features_path: Path = DEFAULT_HOSPITAL_FEATURES_PATH,
    assumptions_config_path: Path = DEFAULT_CONFIG_PATH,
    run_root: Path = RUN_ROOT,
    run_id: str | None = None,
    generated_at: str | None = None,
) -> CashFlowRunResult:
    run_id = run_id or str(uuid.uuid4())
    generated_at = generated_at or now_utc_iso()
    out_dir = run_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _load_yaml(assumptions_config_path)
    candidates = _read_jsonl(candidate_actions_path)
    bundles = _read_jsonl(candidate_evidence_bundles_path)
    hospital_features = _read_jsonl(hospital_features_path)
    bundle_by_id = {str(b.get("candidate_id")): b for b in bundles}
    references = build_prefecture_reference_assumptions(
        hospital_features,
        source_artifact=str(hospital_features_path),
    )

    input_paths = {
        "candidate_actions": str(candidate_actions_path),
        "candidate_evidence_bundles": str(candidate_evidence_bundles_path),
        "hospital_features": str(hospital_features_path),
        "assumptions_config": str(assumptions_config_path),
    }

    records: list[CashFlowRecord] = []
    components: list[CashFlowComponent] = []
    sensitivities: list[CashFlowSensitivityRecord] = []
    issues: list[CashFlowIssue] = []

    configured_actions = tuple(config.get("scope", {}).get("allowed_candidate_actions", []))
    if configured_actions != ALLOWED_ACTIONS:
        issues.append(_issue(
            "error",
            "allowed_candidate_actions_changed",
            "Cash-flow assumptions config must preserve allowed actions exactly: build, reorganize, consolidate.",
            context={"configured_actions": list(configured_actions), "allowed_actions": list(ALLOWED_ACTIONS)},
        ))

    for candidate in sorted(candidates, key=lambda r: str(r.get("candidate_id", ""))):
        ref = references.get(str(candidate.get("prefecture", "")))
        record, cpts, sens, cand_issues = _calculate_candidate(
            candidate,
            bundle_by_id.get(str(candidate.get("candidate_id", ""))),
            ref,
            config,
            generated_at,
            input_paths,
        )
        if record is not None:
            records.append(record)
        components.extend(cpts)
        sensitivities.extend(sens)
        issues.extend(cand_issues)

    counts_by_action = {action: 0 for action in ALLOWED_ACTIONS}
    for rec in records:
        counts_by_action[rec.candidate_action] = counts_by_action.get(rec.candidate_action, 0) + 1
    payback_status_counts = dict(sorted(Counter(rec.payback_status for rec in records).items()))
    issue_counts_by_severity = dict(sorted(Counter(issue.severity for issue in issues).items()))
    issue_counts_by_code = dict(sorted(Counter(issue.issue_code for issue in issues).items()))

    computable_capex_count = sum(1 for rec in records if rec.computable_capex)
    computable_annual_cash_flow_count = sum(1 for rec in records if rec.computable_annual_cash_flow)
    computable_payback_count = sum(1 for rec in records if rec.computable_payback)
    top_candidates = [
        {
            "candidate_id": rec.candidate_id,
            "candidate_action": rec.candidate_action,
            "prefecture": rec.prefecture,
            "municipality": rec.municipality,
            "base_payback_years": rec.base_payback_years,
            "payback_status": rec.payback_status,
        }
        for rec in sorted(
            [r for r in records if r.base_payback_years is not None],
            key=lambda r: (r.base_payback_years or 10**9, r.candidate_id),
        )[:10]
    ]

    output_paths = {
        "cash_flow_manifest": str(out_dir / "cash_flow_manifest.json"),
        "cash_flow_inputs_manifest": str(out_dir / "cash_flow_inputs_manifest.json"),
        "cash_flow_assumptions": str(out_dir / "cash_flow_assumptions.json"),
        "cash_flow_records": str(out_dir / "cash_flow_records.jsonl"),
        "cash_flow_components": str(out_dir / "cash_flow_components.jsonl"),
        "cash_flow_sensitivity": str(out_dir / "cash_flow_sensitivity.jsonl"),
        "cash_flow_issues": str(out_dir / "cash_flow_issues.jsonl"),
        "cash_flow_report_json": str(out_dir / "cash_flow_report.json"),
        "cash_flow_report_md": str(out_dir / "cash_flow_report.md"),
    }

    manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "output_dir": str(out_dir),
        "candidate_count": len(records),
        "input_candidate_count": len(candidates),
        "counts_by_action": counts_by_action,
        "computable_capex_count": computable_capex_count,
        "computable_annual_cash_flow_count": computable_annual_cash_flow_count,
        "computable_payback_count": computable_payback_count,
        "payback_status_counts": payback_status_counts,
        "issue_counts_by_severity": issue_counts_by_severity,
        "issue_counts_by_code": issue_counts_by_code,
        "output_paths": output_paths,
        "limitations": [
            PHASE_12_LIMITATION,
            "No final hospital recommendations are generated.",
        ],
    }
    inputs_manifest = {
        "run_id": run_id,
        "generated_at": generated_at,
        "input_paths": input_paths,
        "input_record_counts": {
            "candidate_actions": len(candidates),
            "candidate_evidence_bundles": len(bundles),
            "hospital_features": len(hospital_features),
            "prefecture_references": len(references),
        },
    }
    assumptions_payload = _assumptions_payload(config, references)
    report = {
        "run_id": run_id,
        "generated_at": generated_at,
        "candidate_count": len(records),
        "counts_by_action": counts_by_action,
        "computable_capex_count": computable_capex_count,
        "computable_annual_cash_flow_count": computable_annual_cash_flow_count,
        "computable_payback_count": computable_payback_count,
        "payback_status_counts": payback_status_counts,
        "issue_counts_by_severity": issue_counts_by_severity,
        "issue_counts_by_code": issue_counts_by_code,
        "assumption_prefecture_count": len(references),
        "sensitivity_scenarios": sorted(config.get("sensitivity_scenarios", {}).keys()),
        "top_candidates_by_payback": top_candidates,
        "phase_12_limitation": PHASE_12_LIMITATION,
        "not_final_recommendations": True,
    }

    _write_json(out_dir / "cash_flow_manifest.json", manifest)
    _write_json(out_dir / "cash_flow_inputs_manifest.json", inputs_manifest)
    _write_json(out_dir / "cash_flow_assumptions.json", assumptions_payload)
    _write_jsonl(out_dir / "cash_flow_records.jsonl", records)
    _write_jsonl(out_dir / "cash_flow_components.jsonl", components)
    _write_jsonl(out_dir / "cash_flow_sensitivity.jsonl", sensitivities)
    _write_jsonl(out_dir / "cash_flow_issues.jsonl", issues)
    _write_json(out_dir / "cash_flow_report.json", report)
    (out_dir / "cash_flow_report.md").write_text(_report_markdown(report), encoding="utf-8")

    return CashFlowRunResult(
        run_id=run_id,
        output_dir=str(out_dir),
        candidate_count=len(records),
        counts_by_action=counts_by_action,
        computable_capex_count=computable_capex_count,
        computable_annual_cash_flow_count=computable_annual_cash_flow_count,
        computable_payback_count=computable_payback_count,
        payback_status_counts=payback_status_counts,
        issue_counts_by_severity=issue_counts_by_severity,
        issue_counts_by_code=issue_counts_by_code,
        output_paths=output_paths,
    )
