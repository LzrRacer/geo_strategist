"""Standalone validator for a C5-C8 coding-agent-control ``manual_result.json``.

Built for C5 (Antigravity) but reusable for C6/C7/C8 (same
``coding_agent_no_skills`` family, same output contract) via
``expected_condition_group``. Runs the same checks whether invoked by a
human, by the CLI (``geo_strategist.cli validate-c5-result``), or by the
agent itself as part of its own write -> validate -> repair loop (see the
"When you are done" section of the generated C5 prompt in
``harnesses/prompts.py``, which names this exact command).

This does not duplicate the full ingestion pipeline in
``experiments/run_condition_proposals.py`` (candidate-universe rebuild,
evidence-graded proposal construction, skill-trace contract checks) — it is
a lighter, faster pre-ingestion self-check an agent can run repeatedly
during its own repair loop before the orchestrator ever sees the file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CANDIDATE_ACTIONS_PATH = Path(
    ".data/interim/study_area/tokyo_aichi_osaka/candidate_actions.jsonl")
DEFAULT_C0_RANKED_CANDIDATES_PATH = Path(
    "outputs/condition_proposals/live/runs/C00/ranked_candidates.jsonl")

_REQUIRED_QUALITATIVE_KEYS: tuple[str, ...] = (
    "regional", "population", "demand_supply", "access",
    "cost_financial", "preferred_action", "review_comments",
)

_MIN_RANKED_CANDIDATES = 5
_RATIONALE_WORD_CONTRACT = 60
_RATIONALE_WORD_HARD_LIMIT = 150

# Prohibited per the "explicitly prohibit C0 ranking substitution" rules:
# rationale text that names the deterministic baseline as the reason for a
# ranking decision is a strong signal the agent copied it rather than
# building its own evaluation.
_C0_SUBSTITUTION_TEXT_MARKERS: tuple[str, ...] = (
    "c0 score", "c0 composite", "c0 ranking", "c0 baseline ranking",
    "c0 final slate", "deterministic_evaluation_engine", "priority_score",
    "deterministic composite score", "deterministic baseline ranking",
)

_TRAVEL_TIME_RE = re.compile(
    r"\b\d+\s*(?:minute|min|hour|hr)s?\b[^.]{0,40}"
    r"(?:drive|driving|travel|away|to reach|by (?:car|ambulance|bus|train))",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"[¥$€]\s?[\d,]+|\d+\s*(?:億|万)円")
# Matches rank_candidates(..., DEFAULT_WEIGHTS) / rank_candidates(..., weights=DEFAULT_WEIGHTS)
# — a direct, unmodified passthrough of the C0 default weights — but not
# rank_candidates(bundle, weights) where `weights` was itself derived from
# DEFAULT_WEIGHTS earlier (e.g. via objective_weights(objective,
# DEFAULT_WEIGHTS)), which is legitimate parameterized reuse.
_RANK_CANDIDATES_WITH_DEFAULT_WEIGHTS_RE = re.compile(
    r"rank_candidates\([^)]*\bDEFAULT_WEIGHTS\b[^)]*\)")
_INVENTED_FACT_MARKERS: tuple[str, ...] = (
    "land parcel", "acquired for", "regulatory approval has been",
    "zoning has been approved", "construction has begun", "renovation is complete",
)
_UNCERTAINTY_QUALIFIERS: tuple[str, ...] = (
    "unconfirmed", "not_available", "model_estimate", "scenario_assumption",
    "unverified", "requires verification", "cash-flow", "cash flow", "workbook",
)


@dataclass
class ValidationIssue:
    code: str
    message: str
    candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "candidate_id": self.candidate_id}


@dataclass
class C5ValidationReport:
    ok: bool
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    c0_substitution_flags: list[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "c0_substitution_flags": [issue.to_dict() for issue in self.c0_substitution_flags],
        }

    def human_report(self) -> str:
        lines = [f"C5 result validation: {'PASS' if self.ok else 'FAIL'}"]
        if self.errors:
            lines.append(f"\n{len(self.errors)} error(s):")
            lines.extend(f"  - [{issue.code}] {issue.message}" for issue in self.errors)
        if self.c0_substitution_flags:
            lines.append(f"\n{len(self.c0_substitution_flags)} possible C0-substitution flag(s):")
            lines.extend(f"  - [{issue.code}] {issue.message}" for issue in self.c0_substitution_flags)
        if self.warnings:
            lines.append(f"\n{len(self.warnings)} warning(s):")
            lines.extend(f"  - [{issue.code}] {issue.message}" for issue in self.warnings)
        if not self.errors and not self.c0_substitution_flags and not self.warnings:
            lines.append("No issues found.")
        return "\n".join(lines)


def _load_known_candidate_ids(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate_id = row.get("candidate_id") if isinstance(row, dict) else None
            if candidate_id:
                ids.add(str(candidate_id))
    return ids


def _load_ordered_candidate_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    ordered: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            candidate_id = row.get("candidate_id") if isinstance(row, dict) else None
            if candidate_id:
                ordered.append(str(candidate_id))
    return ordered


def _validate_ranked_candidates_and_evidence(
    payload: dict[str, Any],
    *,
    repo_root: Path,
    candidate_actions_path: Path | None,
    c0_ranked_candidates_path: Path | None,
    run_dir: Path,
    require_generated_code: bool = True,
    extra_independent_analysis_evidence: bool = False,
) -> tuple[list[ValidationIssue], list[ValidationIssue], list[ValidationIssue]]:
    """Shared ranked_candidates/qualitative_discussion/evidence/
    C0-substitution checks — the C5-C8 and C9-C12 output contracts are
    identical for this part of the payload; only the skill_trace
    requirement (Skills-unified only) differs, and is checked separately by
    ``harnesses/skills_result_validator.py``.

    ``require_generated_code`` is a hard error when True (C5-C8: the
    executed analysis code IS the only evidence of independent work).
    ``extra_independent_analysis_evidence`` lets a caller supply an
    additional signal (C9-C12: a non-empty ``skill_trace``) that, on its
    own, is enough to avoid the "identical to C0 with no independent
    method" flag even when ``generated_code/`` is absent or empty.
    """

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    c0_flags: list[ValidationIssue] = []

    ranked = payload.get("ranked_candidates")
    if not isinstance(ranked, list) or not ranked:
        errors.append(ValidationIssue("no_ranked_candidates", "ranked_candidates is missing or empty"))
        ranked = []

    candidate_actions_path = candidate_actions_path or (repo_root / DEFAULT_CANDIDATE_ACTIONS_PATH)
    known_ids = _load_known_candidate_ids(candidate_actions_path)
    if len(ranked) < _MIN_RANKED_CANDIDATES:
        if known_ids is None or len(known_ids) >= _MIN_RANKED_CANDIDATES:
            errors.append(ValidationIssue(
                "too_few_candidates",
                f"only {len(ranked)} ranked candidates; at least {_MIN_RANKED_CANDIDATES} "
                "are required when the candidate universe permits"))
        else:
            warnings.append(ValidationIssue(
                "too_few_candidates_small_universe",
                f"only {len(ranked)} ranked candidates, but the candidate universe itself "
                f"has only {len(known_ids)}"))

    seen_ids: set[str] = set()
    for index, row in enumerate(ranked):
        if not isinstance(row, dict):
            errors.append(ValidationIssue(
                "invalid_candidate_row", f"ranked_candidates[{index}] is not an object"))
            continue
        candidate_id = str(row.get("candidate_id") or "")
        label = candidate_id or f"ranked_candidates[{index}]"
        if not candidate_id:
            errors.append(ValidationIssue(
                "missing_candidate_id", "a ranked candidate has no candidate_id", label))
            continue
        if candidate_id in seen_ids:
            errors.append(ValidationIssue(
                "duplicate_candidate_id", f"candidate_id {candidate_id!r} appears more than once",
                candidate_id))
        seen_ids.add(candidate_id)
        if known_ids is not None and candidate_id not in known_ids:
            errors.append(ValidationIssue(
                "unknown_candidate_id",
                f"candidate_id {candidate_id!r} not found in {candidate_actions_path}", candidate_id))

        rationale = str(row.get("rationale") or "")
        if not rationale.strip():
            errors.append(ValidationIssue("missing_rationale", "rationale is empty", candidate_id))
        else:
            word_count = len(rationale.split())
            if word_count > _RATIONALE_WORD_HARD_LIMIT:
                errors.append(ValidationIssue(
                    "rationale_too_long",
                    f"rationale is {word_count} words, over the {_RATIONALE_WORD_HARD_LIMIT}-word "
                    f"hard limit (contract asks for <= {_RATIONALE_WORD_CONTRACT} words)", candidate_id))
            elif word_count > _RATIONALE_WORD_CONTRACT:
                warnings.append(ValidationIssue(
                    "rationale_over_contract",
                    f"rationale is {word_count} words, over the "
                    f"{_RATIONALE_WORD_CONTRACT}-word contract", candidate_id))
            lowered_rationale = rationale.lower()
            for marker in _C0_SUBSTITUTION_TEXT_MARKERS:
                if marker in lowered_rationale:
                    c0_flags.append(ValidationIssue(
                        "c0_reference_in_rationale",
                        f"rationale references {marker!r}, which reads as a deterministic-baseline "
                        "substitution rather than the agent's own analysis", candidate_id))

        discussion = row.get("qualitative_discussion")
        if not isinstance(discussion, dict):
            errors.append(ValidationIssue(
                "missing_qualitative_discussion",
                "qualitative_discussion is missing or not an object", candidate_id))
            discussion = {}
        for key in _REQUIRED_QUALITATIVE_KEYS:
            value = discussion.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(ValidationIssue(
                    "missing_qualitative_field",
                    f"qualitative_discussion.{key} is missing or empty", candidate_id))
                continue
            lowered_value = value.lower()
            if key == "access" and _TRAVEL_TIME_RE.search(value):
                errors.append(ValidationIssue(
                    "asserted_travel_time",
                    "qualitative_discussion.access appears to assert a concrete travel time "
                    "without travel-time evidence in this dataset (distance is only a proxy)",
                    candidate_id))
            if key == "cost_financial":
                has_qualifier = any(marker in lowered_value for marker in _UNCERTAINTY_QUALIFIERS)
                for match in _CURRENCY_RE.findall(value):
                    if not has_qualifier:
                        warnings.append(ValidationIssue(
                            "unsupported_currency_claim",
                            "qualitative_discussion.cost_financial asserts a currency amount "
                            f"({match.strip()}) without an evidence/estimate qualifier", candidate_id))
            for marker in _INVENTED_FACT_MARKERS:
                if marker in lowered_value:
                    errors.append(ValidationIssue(
                        "invented_fact_marker",
                        f"qualitative_discussion.{key} contains {marker!r}, which reads as an "
                        "asserted land/acquisition/construction/regulatory fact rather than an "
                        "unconfirmed due-diligence item", candidate_id))

    generated_code_dir = run_dir / "generated_code"
    has_generated_code = generated_code_dir.is_dir() and any(generated_code_dir.iterdir())
    if require_generated_code and not has_generated_code:
        errors.append(ValidationIssue(
            "missing_generated_code",
            f"{generated_code_dir} does not exist or is empty; the executed analysis code must "
            "be kept alongside the result"))

    # C0-substitution detection beyond rationale text: exact final-slate
    # identity with the deterministic baseline and no independent analysis
    # artifacts, or generated code that imports/reads the baseline directly.
    c0_ranked_candidates_path = c0_ranked_candidates_path or (
        repo_root / DEFAULT_C0_RANKED_CANDIDATES_PATH)
    c0_ids = _load_ordered_candidate_ids(c0_ranked_candidates_path)
    c5_ids = [str(row.get("candidate_id")) for row in ranked if isinstance(row, dict)]
    has_independent_analysis = has_generated_code or extra_independent_analysis_evidence
    if c0_ids and c5_ids and c5_ids == c0_ids[: len(c5_ids)] and not has_independent_analysis:
        c0_flags.append(ValidationIssue(
            "identical_to_c0_slate",
            "final ranked_candidates order is identical to the C0 deterministic baseline and no "
            "independent analysis artifacts exist to support an independent method"))

    if generated_code_dir.is_dir():
        for code_file in sorted(generated_code_dir.rglob("*.py")):
            try:
                text = code_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            relative = code_file.relative_to(run_dir)
            # Importing deterministic_evaluation_engine at all is not, by
            # itself, a violation: load_data_bundle is a shared raw-data
            # loader, and rank_candidates is a generic weighted-scoring
            # utility the agent may legitimately call with its OWN derived
            # weight vectors (e.g. one set of weights per branch
            # objective) — that is parameterized reuse of a library
            # function, not "using C0's ranking as the final method". Only
            # flag the two patterns that actually reproduce C0's own
            # output: calling the full C0 pipeline entry point directly, or
            # calling rank_candidates with the literal, unmodified
            # DEFAULT_WEIGHTS constant as its weights argument.
            if "build_deterministic_outcome(" in text:
                c0_flags.append(ValidationIssue(
                    "calls_deterministic_engine_pipeline",
                    f"{relative} calls build_deterministic_outcome(), the full C0 pipeline "
                    "entry point, directly"))
            if _RANK_CANDIDATES_WITH_DEFAULT_WEIGHTS_RE.search(text):
                c0_flags.append(ValidationIssue(
                    "rank_candidates_with_default_weights",
                    f"{relative} calls rank_candidates(...) with the literal, unmodified "
                    "DEFAULT_WEIGHTS constant — this reproduces C0's exact composite rather "
                    "than the agent's own weighting"))
            if re.search(r"C00[/\\]ranked_candidates", text):
                c0_flags.append(ValidationIssue(
                    "reads_c0_ranked_output",
                    f"{relative} reads C0's ranked_candidates output directly"))

    if not payload.get("ranked_candidates") and not payload.get("skill_trace"):
        warnings.append(ValidationIssue(
            "no_agent_created_analysis_artifact",
            "no skill_trace and no ranked_candidates recorded; there is no evidence of an "
            "agent-created evaluation method"))

    return errors, warnings, c0_flags


def validate_c5_result(
    result_path: Path,
    *,
    repo_root: Path | None = None,
    candidate_actions_path: Path | None = None,
    c0_ranked_candidates_path: Path | None = None,
    run_dir: Path | None = None,
    expected_condition_group: str = "C5",
) -> C5ValidationReport:
    """Validate one coding-agent-control ``manual_result.json``.

    ``run_dir`` defaults to ``result_path.parent`` and is where
    ``generated_code/`` is expected to live.
    """

    repo_root = (repo_root or Path(".")).resolve()
    run_dir = run_dir or result_path.parent

    if not result_path.exists():
        return C5ValidationReport(
            ok=False, errors=[ValidationIssue("missing_file", f"{result_path} does not exist")])

    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return C5ValidationReport(
            ok=False,
            errors=[ValidationIssue("invalid_json", f"{result_path} is not valid JSON: {exc}")])

    if not isinstance(payload, dict):
        return C5ValidationReport(
            ok=False,
            errors=[ValidationIssue("invalid_shape", "top-level JSON must be an object")])

    errors: list[ValidationIssue] = []
    if payload.get("condition_group") != expected_condition_group:
        errors.append(ValidationIssue(
            "wrong_condition_group",
            f"condition_group is {payload.get('condition_group')!r}, "
            f"expected {expected_condition_group!r}"))

    candidate_errors, warnings, c0_flags = _validate_ranked_candidates_and_evidence(
        payload, repo_root=repo_root, candidate_actions_path=candidate_actions_path,
        c0_ranked_candidates_path=c0_ranked_candidates_path, run_dir=run_dir,
        require_generated_code=False, extra_independent_analysis_evidence=False,
    )
    errors.extend(candidate_errors)
    from geo_strategist.experiments.decision_reporting_contract import (
        validate_reporting_payload,
    )

    known_ids = _load_known_candidate_ids(
        candidate_actions_path or (repo_root / DEFAULT_CANDIDATE_ACTIONS_PATH))
    try:
        validate_reporting_payload(
            payload,
            condition_group=expected_condition_group,
            candidate_ids=known_ids,
            evidence_roots=[run_dir, repo_root],
        )
    except (TypeError, ValueError) as exc:
        errors.append(ValidationIssue("reporting_contract", str(exc)))

    ok = not errors and not c0_flags
    return C5ValidationReport(ok=ok, errors=errors, warnings=warnings, c0_substitution_flags=c0_flags)
