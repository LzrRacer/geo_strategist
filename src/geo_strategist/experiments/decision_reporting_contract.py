"""Canonical observable decision-analysis reporting contract for C1-C8.

The contract standardizes what conditions report while leaving their execution
capabilities unchanged.  C1-C4 are single-pass reasoning-only conditions;
C5-C8 may execute native coding-agent work, but do not receive project Skills.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


REPORTING_CONTRACT_VERSION = "decision_analysis_reporting_v2"
ExecutionStatus = Literal["executed", "reasoned_only", "not_performed"]
CandidateOutcomeStatus = Literal["selected", "retained", "excluded", "deferred", "replacement"]
DecisionStatus = Literal["proceed", "conditional", "defer", "replace", "reject"]

VANILLA_CONDITIONS = frozenset({"C1", "C2", "C3", "C4"})
NATIVE_AGENT_CONDITIONS = frozenset({"C5", "C6", "C7", "C8"})
SKILLS_CONDITIONS = frozenset({"C9", "C10", "C11", "C12"})

# Words that a ``reasoned_only`` entry must not use, because they assert the
# model actually executed or empirically validated something. Restricted to
# unambiguous execution/validation verbs and artifacts: broader
# data-descriptive words (measured, calculated, computed, confirmed, observed)
# are deliberately excluded, because a single-pass condition legitimately
# reasons over pre-computed scores in its data context and naturally says e.g.
# "the demand computed in the provided data" or "the evidence confirms" without
# claiming it ran anything. Genuine false-execution claims still trip on the
# execution verbs below, so the C5-C8 fairness safeguard is preserved.
_EMPIRICAL_WORDING = re.compile(
    r"\b(?:executed|ran|tested|validated|benchmark(?:ed)?|simulation|"
    r"script output|test result)\b",
    re.IGNORECASE,
)

# Negation markers. An empirical word inside a negated clause is an honest
# denial of execution ("No travel-time test was executed", "no financial model
# was run"), which is exactly the correct way to phrase a reasoned_only test
# result -- so it must not be flagged as a false execution claim.
_NEGATION = re.compile(
    r"\b(?:no|not|n't|without|never|cannot|can't|couldn't|unable|neither|nor|"
    r"none|lack(?:s|ed|ing)?|absent|unavailable)\b",
    re.IGNORECASE,
)


def _asserts_empirical_execution(text: str) -> bool:
    """True only if some clause affirmatively asserts empirical execution.

    A clause that contains an execution word but is negated (or denies that any
    test/run happened) is an honest reasoned_only statement, not a false claim,
    and does not count. Clauses are split on sentence punctuation so a negation
    in one clause cannot rescue an affirmative claim in another.
    """

    for clause in re.split(r"[.;:\n]", text):
        if _EMPIRICAL_WORDING.search(clause) and not _NEGATION.search(clause):
            return True
    return False


class CandidateOutcome(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    rank: int = Field(ge=1)
    outcome: CandidateOutcomeStatus
    summary: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class DecisionAlternativeReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    alternative_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    execution_status: ExecutionStatus
    candidate_outcomes: list[CandidateOutcome]
    selection_reason: str
    rejection_reason: str
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_evidence(self) -> "DecisionAlternativeReport":
        if self.execution_status == "executed" and not (
            self.evidence_refs or any(row.evidence_refs for row in self.candidate_outcomes)
        ):
            raise ValueError("executed alternative requires at least one auditable evidence_ref")
        if self.execution_status == "reasoned_only":
            text = ". ".join([self.objective, self.selection_reason, self.rejection_reason]
                             + [row.summary for row in self.candidate_outcomes])
            if _asserts_empirical_execution(text):
                raise ValueError(
                    "reasoned_only alternative uses wording that implies empirical execution")
        if self.execution_status == "not_performed" and not self.rejection_reason.strip():
            raise ValueError("not_performed alternative requires a rejection_reason explanation")
        if not self.candidate_outcomes and self.execution_status != "not_performed":
            raise ValueError("candidate_outcomes must report outcomes or explicit exclusions")
        return self


class ValidationTestReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    test_id: str = Field(min_length=1)
    objective: str = Field(min_length=1)
    status: ExecutionStatus
    result: str = Field(min_length=1)
    decision_effect: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_evidence(self) -> "ValidationTestReport":
        if self.status == "executed" and not self.evidence_refs:
            raise ValueError("executed validation test requires at least one auditable evidence_ref")
        if self.status == "reasoned_only" and _asserts_empirical_execution(self.result):
            raise ValueError(
                "reasoned_only validation test result uses wording that implies empirical execution")
        if self.status == "not_performed" and not self.result.strip():
            raise ValueError("not_performed validation test requires an explanation in result")
        return self


class ExcludedCandidate(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    reason: str = Field(min_length=1)
    source_alternative_ids: list[str]


class SynthesisReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    rule: str = Field(min_length=1)
    source_alternative_ids: list[str]
    tradeoffs: list[str]
    excluded_candidates: list[ExcludedCandidate]


class CriticalIssueReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str | None
    issue: str = Field(min_length=1)
    decision_effect: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class ReversalConditionReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str | None
    condition: str = Field(min_length=1)
    decision_change: str = Field(min_length=1)
    evidence_refs: list[str] = Field(default_factory=list)


class FinalDecisionReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    decision_status: DecisionStatus
    status_reason: str = Field(min_length=1)
    blocking_conditions: list[str]
    required_next_steps: list[str]

    @model_validator(mode="after")
    def require_next_steps(self) -> "FinalDecisionReport":
        if not self.required_next_steps:
            raise ValueError("required_next_steps must contain prioritized candidate-specific actions")
        return self


class DecisionAnalysisReportingV2(BaseModel):
    """Required reporting surface embedded in every C1-C8 response."""

    model_config = ConfigDict(extra="allow")

    reporting_contract_version: Literal["decision_analysis_reporting_v2"]
    decision_alternatives: list[DecisionAlternativeReport]
    validation_tests: list[ValidationTestReport]
    synthesis: SynthesisReport
    critical_issues: list[CriticalIssueReport]
    reversal_conditions: list[ReversalConditionReport]
    final_decisions: list[FinalDecisionReport]

    @model_validator(mode="after")
    def validate_references(self) -> "DecisionAnalysisReportingV2":
        alternative_ids = [row.alternative_id for row in self.decision_alternatives]
        if len(alternative_ids) != len(set(alternative_ids)):
            raise ValueError("decision_alternatives contain duplicate alternative_id values")
        test_ids = [row.test_id for row in self.validation_tests]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("validation_tests contain duplicate test_id values")
        declared = set(alternative_ids)
        references = list(self.synthesis.source_alternative_ids)
        for row in self.synthesis.excluded_candidates:
            references.extend(row.source_alternative_ids)
        dangling = sorted(set(references) - declared)
        if dangling:
            raise ValueError(f"source_alternative_ids reference undeclared alternatives: {dangling}")
        final_ids = [row.candidate_id for row in self.final_decisions]
        if len(final_ids) != len(set(final_ids)):
            raise ValueError("final_decisions contain duplicate candidate_id values")
        return self


def validate_reporting_payload(
    payload: Any,
    *,
    condition_group: str,
    candidate_ids: set[str] | None = None,
    require_ranked_match: bool = True,
    evidence_roots: list[Path] | None = None,
) -> DecisionAnalysisReportingV2:
    """Validate a C1-C8 payload with precise Pydantic JSON-path errors."""

    if not isinstance(payload, dict):
        raise ValueError("$: expected a JSON object")
    try:
        report = DecisionAnalysisReportingV2.model_validate(payload)
    except ValidationError as exc:
        details = []
        for error in exc.errors(include_url=False):
            path = "$" + "".join(
                f"[{part}]" if isinstance(part, int) else f".{part}" for part in error["loc"])
            details.append(f"{path}: {error['msg']}")
        raise ValueError("; ".join(details)) from exc

    # Required reporting categories must not be silently omitted. An empty
    # array reads to the judge as "no analysis was needed"; the honest way to
    # report "nothing to say here" is an explicit entry, not omission.
    if not report.decision_alternatives:
        raise ValueError(
            "$.decision_alternatives: must not be empty; the final slate must "
            "derive from at least one declared alternative")
    if not report.validation_tests:
        raise ValueError(
            "$.validation_tests: must not be empty; report an honest "
            "`not_performed` test with an explanation rather than omitting the category")
    if not report.reversal_conditions:
        raise ValueError(
            "$.reversal_conditions: must not be empty; report a slate-level entry "
            "(candidate_id null) stating that no defensible reversal condition could "
            "be derived rather than omitting the category")

    ranked = payload.get("ranked_candidates")
    if require_ranked_match:
        if not isinstance(ranked, list) or not ranked:
            raise ValueError("$.ranked_candidates: expected a non-empty array")
        ranked_ids = [str(row.get("candidate_id") or "") for row in ranked
                      if isinstance(row, dict)]
        final_ids = [row.candidate_id for row in report.final_decisions]
        if len(ranked_ids) != len(ranked) or set(ranked_ids) != set(final_ids):
            raise ValueError(
                "$.final_decisions: must contain exactly one entry for every ranked candidate")

    nested_candidates: list[tuple[str, str]] = []
    for ai, alternative in enumerate(report.decision_alternatives):
        nested_candidates.extend(
            (f"$.decision_alternatives[{ai}].candidate_outcomes[{oi}].candidate_id",
             outcome.candidate_id)
            for oi, outcome in enumerate(alternative.candidate_outcomes))
    nested_candidates.extend(
        (f"$.synthesis.excluded_candidates[{i}].candidate_id", row.candidate_id)
        for i, row in enumerate(report.synthesis.excluded_candidates))
    nested_candidates.extend(
        (f"$.critical_issues[{i}].candidate_id", row.candidate_id)
        for i, row in enumerate(report.critical_issues) if row.candidate_id is not None)
    nested_candidates.extend(
        (f"$.reversal_conditions[{i}].candidate_id", row.candidate_id)
        for i, row in enumerate(report.reversal_conditions) if row.candidate_id is not None)
    nested_candidates.extend(
        (f"$.final_decisions[{i}].candidate_id", row.candidate_id)
        for i, row in enumerate(report.final_decisions))
    if candidate_ids is not None:
        for path, candidate_id in nested_candidates:
            if candidate_id not in candidate_ids:
                raise ValueError(f"{path}: unknown candidate_id {candidate_id!r}")

    if condition_group in VANILLA_CONDITIONS:
        considered = [row for row in report.decision_alternatives
                      if row.execution_status == "reasoned_only"]
        if len(considered) < 2:
            raise ValueError(
                "$.decision_alternatives: C1-C4 require at least two considered alternatives")
        for index, alternative in enumerate(report.decision_alternatives):
            if alternative.execution_status == "executed":
                raise ValueError(
                    f"$.decision_alternatives[{index}].execution_status: "
                    "C1-C4 cannot report executed analysis")
        for index, test in enumerate(report.validation_tests):
            if test.status == "executed":
                raise ValueError(
                    f"$.validation_tests[{index}].status: "
                    "C1-C4 cannot report executed empirical validation")
        if payload.get("skill_trace") not in (None, []):
            raise ValueError("$.skill_trace: C1-C4 must leave the execution trace empty")
        call_summary = payload.get("model_call_summary")
        if isinstance(call_summary, dict) and isinstance(call_summary.get("total_requests"), int):
            if call_summary["total_requests"] > 1:
                raise ValueError("$.model_call_summary.total_requests: C1-C4 permit one request")
    if evidence_roots is not None:
        roots = [Path(root).resolve() for root in evidence_roots]

        def resolves(reference: str) -> bool:
            path_text = reference.split("#", 1)[0].strip()
            if not path_text:
                return False
            path = Path(path_text)
            return path.exists() if path.is_absolute() else any((root / path).exists() for root in roots)

        for index, alternative in enumerate(report.decision_alternatives):
            if alternative.execution_status != "executed":
                continue
            refs = alternative.evidence_refs + [
                ref for outcome in alternative.candidate_outcomes for ref in outcome.evidence_refs]
            if not any(resolves(ref) for ref in refs):
                raise ValueError(
                    f"$.decision_alternatives[{index}].evidence_refs: "
                    "executed claim has no resolvable evidence reference")
        for index, test in enumerate(report.validation_tests):
            if test.status == "executed" and not any(resolves(ref) for ref in test.evidence_refs):
                raise ValueError(
                    f"$.validation_tests[{index}].evidence_refs: "
                    "executed claim has no resolvable evidence reference")
    return report


def reporting_schema_example(condition_group: str) -> dict[str, Any]:
    """Return the shared prompt schema with neutral structural placeholders."""

    status = "reasoned_only"
    alternative = {
        "alternative_id": "alternative-1", "objective": "materially distinct strategy or rule",
        "execution_status": status,
        "candidate_outcomes": [{"candidate_id": "exact candidate id", "rank": 1,
                                 "outcome": "retained", "summary": "concise result",
                                 "evidence_refs": []}],
        "selection_reason": "why it contributed, or empty", "rejection_reason": "why rejected, or empty",
        "evidence_refs": [],
    }
    alternatives = [alternative]
    if condition_group in VANILLA_CONDITIONS:
        alternatives.append({**alternative, "alternative_id": "alternative-2",
                             "objective": "second materially distinct strategy or rule"})
    return {
        "reporting_contract_version": REPORTING_CONTRACT_VERSION,
        "decision_alternatives": alternatives,
        "validation_tests": [{"test_id": "unique-stable-id", "objective": "uncertainty addressed",
                              "status": status, "result": "result or reason not performed",
                              "decision_effect": "effect on ranking, confidence, status, or due diligence",
                              "evidence_refs": []}],
        "synthesis": {"rule": "explicit reproducible combination rule", "source_alternative_ids": [],
                      "tradeoffs": ["specific tradeoff"],
                      "excluded_candidates": [{"candidate_id": "exact candidate id",
                                               "reason": "exclusion reason",
                                               "source_alternative_ids": []}]},
        "critical_issues": [{"candidate_id": None, "issue": "specific issue",
                             "decision_effect": "decision effect", "evidence_refs": []}],
        "reversal_conditions": [{"candidate_id": None, "condition": "threshold, scenario, or missing fact",
                                 "decision_change": "resulting recommendation change", "evidence_refs": []}],
        "final_decisions": [{"candidate_id": "exact final candidate id",
                             "decision_status": "conditional",
                             "status_reason": "status rationale", "blocking_conditions": [],
                             "required_next_steps": ["priority 1 action"]}],
    }


def reporting_prompt_fragment(condition_group: str) -> str:
    """Canonical prompt fragment shared by direct and manual C1-C12 paths."""

    vanilla = condition_group in VANILLA_CONDITIONS
    skills = condition_group in SKILLS_CONDITIONS
    if vanilla:
        mode = (
            "This is a single-pass vanilla response. Do not use tools, execute code, run branch search, "
            "perform iterative review, or make additional model requests. Report considered alternatives "
            "and unexecuted tests as `reasoned_only` or `not_performed`; never use `executed`."
        )
    elif skills:
        mode = (
            "You are running the project's Skills-unified branch-search contract. You may inspect files, "
            "write and execute code, run branch search, design and run robustness tests, review, and revise "
            "using the Skills operators. Report the alternatives, tests, and analyses you actually executed "
            "as `executed` with auditable evidence; report considered-but-unexecuted ones as `reasoned_only` "
            "or `not_performed`. This reporting contract is in addition to (not a replacement for) the "
            "`skill_trace` lifecycle you already record."
        )
    else:
        mode = (
            "You may inspect files, write and execute code, debug, validate, review, revise, and explore "
            "portfolios using native coding-agent capabilities. Do not use the project Skills contract."
        )
    requirements = (
        "Consider at least two materially distinct alternatives in this one response. " if vanilla else
        "List every materially distinct alternative actually explored, including negative or failed work. "
    )
    return f"""## Observable decision-analysis reporting contract

{mode}

{requirements}For each alternative, report candidate-level outcomes. Label every alternative and
validation test exactly `executed`, `reasoned_only`, or `not_performed`.

- `executed`: work was actually performed and has an auditable artifact, command/result,
  data-source reference, generated-code path, or structured result. Every executed item
  must include an evidence reference; unsupported execution claims are invalid.
- `reasoned_only`: an alternative or expected test outcome was considered without tools,
  code, or empirical validation. Do not describe it as tested, measured, calculated,
  validated, or confirmed.
- `not_performed`: the work was not considered or could not be performed. Explain why;
  do not silently omit a reporting category.

Provide an explicit reproducible synthesis rule, source alternatives, tradeoffs, competitive
exclusions, critical issues and their decision effects, candidate-specific reversal conditions
(or an explicit slate-level statement that none is defensible from available data), and exactly
one final decision per ranked candidate with status reason, blocking conditions, and prioritized
next steps. Preserve negative, failed, inconclusive, and contradictory findings. Candidate IDs in
every nested field must come from the supplied universe. Do not reveal private chain-of-thought;
report only concise analysis summaries, actions, artifacts, results, comparisons, and decisions.
Candidate outcomes use `selected`, `retained`, `excluded`, `deferred`, or `replacement`.
Final decision statuses use `proceed`, `conditional`, `defer`, `replace`, or `reject`.

Required JSON fields (merge these with `condition_group`, `ranked_candidates`, qualitative
discussion, provenance, `review_comments`, `skill_trace`, and `model_call_summary`):

```json
{json.dumps(reporting_schema_example(condition_group), ensure_ascii=False, indent=2)}
```
"""
