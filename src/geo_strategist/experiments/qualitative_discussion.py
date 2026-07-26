"""Candidate-level qualitative site discussion, built only from validated facts.

One shared builder serves every condition (C0-C14 and manual-harness
ingestion): prose is rendered deterministically from fields that already
carry provenance — evaluation-model components, raw municipal facts (census
population projections, MLIT land prices, Yahoo facility counts), evidence
grades, evidence gaps, reviewer output, and condition-level narrative. It
never asserts anything the data does not contain: missing values are stated
as unconfirmed/requiring verification, facility names appear only when
source-traceable, and no cost, travel-time, or regulatory figure is invented.

Each entry is also attached to the proposal record
(``proposal["qualitative_discussion"]``) so the comparison judge scores the
discussion from structured data rather than markdown parsing.
"""

from __future__ import annotations

from typing import Any

from geo_strategist.experiments.location_costing import location_cost_model

_DIMENSIONS: tuple[str, ...] = (
    "regional", "population", "demand_supply", "access",
    "cost_financial", "preferred_action", "review_comments",
)

_ACTION_LABELS = {
    "build": "build a new facility",
    "reorganize": "reorganize existing facilities",
    "consolidate": "consolidate existing facilities",
    "expand": "expand an existing facility",
}


def _fmt_int(value: Any) -> str | None:
    if isinstance(value, (int, float)):
        return f"{int(round(value)):,}"
    return None


def _fmt_pct(value: Any, digits: int = 1) -> str | None:
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}%"
    return None


def _level(value: Any, *, high: float = 0.66, low: float = 0.33) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    return "high" if value >= high else "moderate" if value >= low else "low"


def _regional_paragraph(proposal: dict[str, Any], facts: dict[str, Any]) -> str:
    components = proposal.get("score_components") or {}
    action = proposal.get("action_type")
    muni = proposal.get("municipality")
    parts: list[str] = []
    supply = components.get("supply_shortage")
    if action == "build" and isinstance(supply, (int, float)):
        if supply >= 0.5:
            parts.append(
                f"{muni} shows a comparatively thin existing hospital supply for its "
                "demand profile, which is why it enters the candidate list as a "
                "potential new-facility area.")
        else:
            parts.append(
                f"{muni} already carries meaningful hospital supply; the build case "
                "rests on demand rather than an outright supply gap.")
    elif action in ("reorganize", "consolidate") and isinstance(supply, (int, float)):
        parts.append(
            f"{muni} has a comparatively dense existing facility base "
            f"({_fmt_int(facts.get('hospital_count')) or 'count unconfirmed'} hospital(s) "
            "in the Yahoo Local Search records), which is what makes it a "
            f"{action} candidate rather than a greenfield site.")
    growth = facts.get("population_pct_change_2020_2050")
    if isinstance(growth, (int, float)):
        if growth > 0.02:
            parts.append("The census projection classifies it as a growing urban area through 2050.")
        elif growth < -0.15:
            parts.append("The census projection classifies it as a strongly shrinking area through 2050.")
        else:
            parts.append("The census projection shows a broadly stable population through 2050.")
    else:
        parts.append("Long-term population trend for this municipality is unconfirmed in the current data.")
    return " ".join(parts) or f"Regional characteristics of {muni} beyond the scored components are unconfirmed."


def _population_paragraph(facts: dict[str, Any], components: dict[str, Any]) -> str:
    total = _fmt_int(facts.get("population_total_2025"))
    seniors = _fmt_int(facts.get("population_65_plus_2025"))
    share_now = _fmt_pct(facts.get("share_65_plus_2025_pct"))
    share_2050 = _fmt_pct(facts.get("share_65_plus_2050_pct"))
    change = facts.get("population_pct_change_2020_2050")
    parts: list[str] = []
    if total:
        sentence = f"The census projection puts the 2025 population at {total}"
        if seniors and share_now:
            sentence += f", of whom {seniors} ({share_now}) are aged 65+"
        sentence += "."
        parts.append(sentence)
        if share_2050:
            parts.append(f"By 2050 the 65+ share is projected to reach {share_2050}.")
        if isinstance(change, (int, float)):
            direction = "grow" if change > 0 else "decline"
            parts.append(
                f"Total population is projected to {direction} by "
                f"{abs(change) * 100:.1f}% between 2020 and 2050"
                + (", a decline risk that any investment case must absorb." if change < 0 else "."))
    else:
        parts.append("Raw population figures for this municipality are unconfirmed and require additional verification.")
    aging = _level(components.get("aging"))
    if aging:
        parts.append(f"Relative to the study area, elderly-demand pressure scores {aging}.")
    return " ".join(parts)


def _demand_supply_paragraph(proposal: dict[str, Any], facts: dict[str, Any]) -> str:
    components = proposal.get("score_components") or {}
    parts: list[str] = []
    demand = _level(components.get("demand"))
    if demand:
        parts.append(f"Healthcare demand pressure is {demand} for the study area.")
    hospitals = facts.get("hospital_count")
    density = facts.get("supply_density_per_100k")
    if isinstance(hospitals, (int, float)):
        sentence = f"Yahoo Local Search records show {int(hospitals)} hospital(s) in the municipality"
        if isinstance(density, (int, float)):
            sentence += f" ({density:.1f} per 100k residents)"
        parts.append(sentence + ".")
    else:
        parts.append("Nearby facility data for this municipality is incomplete; the supply picture needs field verification.")
    name = proposal.get("target_facility_name")
    if name and proposal.get("source_evidence_refs"):
        parts.append(
            f"A source-traceable facility record identifies {name} as the working "
            f"{proposal.get('action_type')} target; its clinical functions and "
            "suitability are not asserted beyond that record.")
    elif proposal.get("action_type") in ("reorganize", "consolidate"):
        parts.append("No source-traceable target facility is attached; the concrete target must come from a registry search.")
    return " ".join(parts)


def _access_paragraph(components: dict[str, Any]) -> str:
    supply = _level(components.get("supply_shortage"))
    lead = (
        f"Emergency-access relevance is inferred from the supply-gap component ({supply})."
        if supply else
        "Emergency-access relevance could not be scored from the available data."
    )
    return (
        lead + " Travel-time data is not available in the current evidence base, so "
        "geographic proximity is only a proxy; actual ambulance and transit times "
        "must be measured before any siting decision."
    )


def _cost_paragraph(facts: dict[str, Any], cost_model: dict[str, Any],
                    components: dict[str, Any],
                    sourced_amounts: list[str]) -> str:
    parts: list[str] = []
    land_median = facts.get("land_price_median_jpy_per_sqm")
    if isinstance(land_median, (int, float)):
        amount = f"¥{_fmt_int(land_median)}"
        sourced_amounts.append(amount)
        parts.append(
            f"MLIT land-price records ({facts.get('land_price_sample_count')} samples, "
            f"{facts.get('land_price_year')}) put the municipal median at "
            f"{amount}/m².")
    else:
        parts.append("Municipal land-price data is not available; land cost exposure is unquantified.")
    scenario = location_cost_model(facts, cost_model)
    factor = scenario.get("location_cost_factor")
    construction = scenario.get("estimated_construction_cost_per_bed_jpy_mm")
    invest = scenario.get("estimated_initial_investment_per_bed_jpy_mm")
    payback = scenario.get("estimated_payback_years")
    if isinstance(invest, (int, float)):
        amount = f"¥{invest:,.0f}"
        sourced_amounts.append(amount)
        if isinstance(construction, (int, float)):
            sourced_amounts.append(f"¥{construction:,.0f}")
        parts.append(
            "Using the proportional land-cost scenario assumption, the local "
            f"cost factor is {factor:.2f} versus the prefecture workbook median; "
            + (f"scaled construction cost is about ¥{construction:,.0f}M per bed and "
               if isinstance(construction, (int, float)) else "")
            + f"scaled initial investment is about {amount}M per bed"
            + (f" and a median payback around {payback:.0f} years" if isinstance(payback, (int, float)) else "")
            + ". These are scenario_assumption values from MLIT land medians and "
            "workbook model estimates, not a candidate costing.")
    elif cost_model:
        parts.append(
            "The prefecture workbook gives archetype cost medians, but a "
            "location-specific proportional estimate cannot be computed without "
            "both a municipal MLIT land median and a workbook land-price median.")
    financial = _level(components.get("financial"))
    if financial:
        parts.append(f"Financial plausibility scores {financial} on the prefecture-level model estimate.")
    parts.append(
        "Site-specific acquisition cost, construction/renovation cost, zoning and "
        "building constraints are unverified and require specialist cost due diligence.")
    return " ".join(parts)


def _action_paragraph(proposal: dict[str, Any]) -> str:
    action = str(proposal.get("action_type"))
    components = proposal.get("score_components") or {}
    supporting = [name for name in ("demand", "aging", "supply_shortage", "financial", "land")
                  if isinstance(components.get(name), (int, float)) and components[name] >= 0.5]
    return (
        f"The recommended action is `{action}` ({_ACTION_LABELS.get(action, action)}), "
        "carried over from the validated candidate action for this municipality"
        + (f"; the strongest supporting components are {', '.join(supporting)}."
           if supporting else "; component support is mixed and the action should be "
           "re-examined during due diligence.")
    )


_CRITIQUE_COMPONENTS = ("land", "supply_shortage", "financial")

_COMPONENT_REASON_FIELDS = {
    "land": "land_score_unavailable_reason",
    "supply_shortage": "healthcare_supply_score_unavailable_reason",
    "financial": "cash_flow_score_unavailable_reason",
}


def _missing_components(proposal: dict[str, Any]) -> list[str]:
    components = proposal.get("score_components") or {}
    return [name for name in _CRITIQUE_COMPONENTS
            if components.get(name) is None]


def _own_data_quality_critique(proposal: dict[str, Any], data: Any) -> list[str]:
    """This candidate's own null-component findings, with source reasons."""

    missing = _missing_components(proposal)
    if not missing:
        return []
    score_row = (getattr(data, "scores_by_key", None) or {}).get(
        (proposal.get("prefecture"), proposal.get("municipality")), {})
    findings = []
    for name in missing:
        reason = score_row.get(_COMPONENT_REASON_FIELDS.get(name, ""), None)
        findings.append(
            f"the {name.replace('_', ' ')} component is null for this municipality"
            + (f" ({reason})" if reason else ""))
    return [
        "Data-quality finding for this candidate: " + "; ".join(findings)
        + f". Its rank {proposal.get('rank')} therefore rests on the remaining "
        "components and should be re-examined once the missing data is obtained."
    ]


def _slate_anomaly_critique(proposal: dict[str, Any],
                            slate: list[dict[str, Any]]) -> list[str]:
    """Cross-candidate check: name slate peers ranked on incomplete data."""

    peers = [p for p in slate
             if p.get("candidate_id") != proposal.get("candidate_id")
             and _missing_components(p)]
    if not peers or _missing_components(proposal):
        return []
    described = ", ".join(
        f"{p.get('candidate_id')} (rank {p.get('rank')}, "
        f"null {'/'.join(_missing_components(p))})"
        for p in peers[:4])
    return [
        f"Slate-level consistency check: account for null score components "
        f"among slate peers — {described} — and clarify how they achieved "
        "their ranks without complete data before treating the slate ordering "
        "as final."
    ]


def district_data_quality_note(data: Any) -> str:
    """District-wide integrity note: candidate-universe entries whose
    municipality has no validated land-price record, so readers can question
    their universe ranks. Computed once per slate from the score layer and
    rendered exactly once at report level — never inside a candidate's own
    review section, which must stay candidate-specific."""

    scores = getattr(data, "scores_by_key", None) or {}
    no_land = {key for key, row in scores.items()
               if row.get("land_score_available") is False}
    if not no_land:
        return ""
    affected = [c for c in (getattr(data, "candidates", None) or [])
                if (c.get("prefecture"), c.get("municipality")) in no_land]
    if not affected:
        return ""
    affected.sort(key=lambda c: c.get("score_rank_overall") or 10 ** 6)
    municipalities = sorted({str(c.get("municipality")) for c in affected})
    named = ", ".join(
        f"{c.get('candidate_id')} (universe rank {c.get('score_rank_overall')})"
        for c in affected[:2])
    return (
        f"District data-integrity note: account for null land scores in "
        f"{len(no_land)} study-area municipalities (candidate-universe entries "
        f"affected: {', '.join(municipalities[:4])}) and clarify how the "
        f"related candidates ({named}) achieved their universe ranks without "
        "complete land data before comparing them against this slate.")


def _reviewer_verdict(proposal: dict[str, Any]) -> list[str]:
    """The persona-review verdict specific to this candidate."""

    parts: list[str] = []
    scores = proposal.get("reviewer_scores") or {}
    numeric = {name: value for name, value in scores.items()
               if isinstance(value, (int, float))}
    if numeric:
        worst = sorted(numeric.items(), key=lambda item: item[1])[:2]
        parts.append(
            "Persona reviewers scored this candidate lowest on "
            + " and ".join(f"{name.replace('_', ' ')} ({value:g})"
                           for name, value in worst)
            + ", so those angles deserve the earliest scrutiny.")
    review_summary = proposal.get("review_summary")
    if review_summary and review_summary != "not_available":
        parts.append(f"Review status: {review_summary}.")
    revision = proposal.get("revision_summary")
    if revision and revision != "not_available":
        parts.append(f"Revision applied: {str(revision)[:240]}")
    return parts


def _candidate_review_summary(
    proposal: dict[str, Any],
    packet: dict[str, Any] | None,
    *,
    review_threads: list[dict[str, Any]] | None = None,
    candidate_deliberation_state: str = "not_run",
) -> str:
    """Summary of this candidate's structured deliberation packet.

    Only major/blocking findings and their author responses are summarized
    here (the full findings/response tables render separately in the
    report); invalidated findings are never surfaced. Returns a
    explicit status note when no packet was generated for this candidate."""

    if not packet:
        threads = review_threads or []
        errors = [str(thread.get("error") or "") for thread in threads
                  if thread.get("error")]
        if errors:
            joined = " ".join(errors).lower()
            if "429" in joined or "rate_limited" in joined or "rate limited" in joined:
                return (
                    "Candidate-level review was attempted, but reviewer LLM "
                    "calls were rate-limited (for example http_429); no "
                    "validated deliberation packet was available for this "
                    "candidate in this run.")
            return (
                "Candidate-level review was attempted, but reviewer LLM calls "
                "failed before a validated deliberation packet was available "
                f"for this candidate ({len(errors)} reviewer error(s)).")
        if threads:
            return (
                "Candidate-level review was attempted for this candidate, but "
                "the reviewer threads produced no validated findings packet.")
        if candidate_deliberation_state == "skipped_deterministic":
            return (
                "Candidate-level reviewer packets were not generated because "
                "this is the deterministic baseline condition.")
        return (
            "Candidate-level reviewer packets were not generated for this "
            "condition because candidate deliberation was not run or was "
            "disabled for this run.")
    findings = packet.get("reviewer_findings") or []
    responses_by_id = {
        str(r.get("finding_id")): r for r in (packet.get("author_responses") or [])
    }
    major_or_blocking = [f for f in findings if f.get("severity") in ("blocking", "major")]
    parts: list[str] = []
    if not major_or_blocking:
        parts.append("Candidate-level review recorded no major or blocking issues.")
    else:
        for finding in major_or_blocking[:4]:
            response = responses_by_id.get(str(finding.get("finding_id"))) or {}
            note = (
                f"[{finding.get('severity')}] {finding.get('issue')} — author response: "
                f"{response.get('response_status', 'unresolved')}")
            if response.get("why_still_proceed"):
                note += f" ({response['why_still_proceed']})"
            parts.append(note)
    position = packet.get("final_candidate_position") or "retain"
    reason = packet.get("final_reason") or ""
    parts.append(f"Final deliberation position: {position}." + (f" {reason}" if reason else ""))
    return " ".join(parts)


def _review_paragraph(proposal: dict[str, Any],
                      slate: list[dict[str, Any]],
                      data: Any,
                      narrative_sections: dict[str, str] | None,
                      *,
                      review_packet: dict[str, Any] | None = None,
                      review_threads: list[dict[str, Any]] | None = None,
                      candidate_deliberation_state: str = "not_run",
                      include_candidate_review: bool = True) -> str:
    parts: list[str] = []
    parts.extend(_reviewer_verdict(proposal))
    parts.extend(_own_data_quality_critique(proposal, data))
    parts.extend(_slate_anomaly_critique(proposal, slate))
    rationale = proposal.get("llm_rationale")
    if rationale:
        parts.append(f"Model rationale (model_estimate): {str(rationale)[:400]}")
    followups = [item for item in (proposal.get("required_due_diligence") or [])
                 if "follow-up" in item.lower()][:3]
    for item in followups:
        parts.append(item)
    judge_note = (narrative_sections or {}).get("judge_rationale")
    if judge_note:
        parts.append(f"Judge note: {str(judge_note)[:300]}")
    gaps = proposal.get("evidence_gaps") or []
    if gaps:
        parts.append("Open evidence gaps: " + ", ".join(str(g) for g in gaps[:4]) + ".")
    # The shared post-hoc candidate-review/author-response augmentation (and
    # its "review was not run/disabled" boilerplate) is a separate,
    # optional layer on top of the base single-pass evidence above — Vanilla
    # LLM conditions never receive it, so include_candidate_review=False
    # omits this sentence entirely rather than stating it was skipped.
    if include_candidate_review:
        parts.append(_candidate_review_summary(
            proposal, review_packet,
            review_threads=review_threads,
            candidate_deliberation_state=candidate_deliberation_state))
    return " ".join(parts) or "No reviewer findings were recorded for this candidate."


def build_qualitative_site_discussions(
    proposals: list[dict[str, Any]],
    data: Any,
    *,
    narrative_sections: dict[str, str] | None = None,
    review_packet_by_candidate_id: dict[str, Any] | None = None,
    review_threads_by_candidate_id: dict[str, list[dict[str, Any]]] | None = None,
    candidate_deliberation_state: str = "not_run",
    include_candidate_review: bool = True,
) -> list[dict[str, Any]]:
    """One structured discussion entry per proposal, from validated facts only.

    ``review_packet_by_candidate_id`` supplies each candidate's own
    ``CandidateReviewPacket`` (as a dict); study-area-wide notes are never
    read here — those render once at report level via
    ``district_data_quality_note``. ``include_candidate_review=False``
    (Vanilla LLM conditions) omits the shared post-hoc candidate-review
    augmentation from the review-comments paragraph entirely — no packet
    content and no "review was not run" boilerplate — while every other
    candidate-specific paragraph (model rationale, evidence gaps, persona
    critique) is built exactly as for any other condition."""

    review_packet_by_candidate_id = review_packet_by_candidate_id or {}
    review_threads_by_candidate_id = review_threads_by_candidate_id or {}
    entries: list[dict[str, Any]] = []
    for proposal in proposals:
        key = (proposal.get("prefecture"), proposal.get("municipality"))
        facts = (getattr(data, "municipal_facts_by_key", None) or {}).get(key, {})
        cost_model = (getattr(data, "cost_model_by_prefecture", None) or {}).get(
            proposal.get("prefecture"), {})
        components = proposal.get("score_components") or {}
        sourced_amounts: list[str] = []
        review_packet = review_packet_by_candidate_id.get(str(proposal.get("candidate_id")))
        review_threads = review_threads_by_candidate_id.get(
            str(proposal.get("candidate_id")), [])
        entry = {
            "candidate_id": proposal.get("candidate_id"),
            "rank": proposal.get("rank"),
            "heading": (
                f"{proposal.get('rank')}. {proposal.get('candidate_id')} — "
                f"{proposal.get('municipality')} ({proposal.get('prefecture')}) / "
                f"{proposal.get('action_type')}"),
            "paragraphs": {
                "regional": _regional_paragraph(proposal, facts),
                "population": _population_paragraph(facts, components),
                "demand_supply": _demand_supply_paragraph(proposal, facts),
                "access": _access_paragraph(components),
                "cost_financial": _cost_paragraph(facts, cost_model, components, sourced_amounts),
                "preferred_action": _action_paragraph(proposal),
                "review_comments": _review_paragraph(
                    proposal, proposals, data, narrative_sections,
                    review_packet=review_packet,
                    review_threads=review_threads,
                    candidate_deliberation_state=candidate_deliberation_state,
                    include_candidate_review=include_candidate_review),
            },
            "facts_used": facts,
            "sourced_amounts": sourced_amounts,
            "review_packet": review_packet,
        }
        entries.append(entry)
        proposal["qualitative_discussion"] = {
            "heading": entry["heading"], "paragraphs": entry["paragraphs"],
            "sourced_amounts": sourced_amounts,
        }
    return entries


def unsourced_currency_flags(text: str, sourced_amounts: list[str]) -> list[str]:
    """Currency-claim flags excluding amounts that carry a data source.

    The generic fabrication guard flags every currency amount; the
    qualitative discussion legitimately embeds *sourced* figures (MLIT land
    medians, workbook model estimates), so those registered amounts are
    exempted here. Exemption matches the full rendered token (symbol,
    separators stripped, unit-suffix variants the builder emits) — never bare
    digits, so a sourced ¥350 (million/bed) does not exempt a fabricated
    350億円 claim.
    """

    from geo_strategist.experiments.live_common import narrative_fabrication_flags

    def _norm(value: str) -> str:
        return value.replace(",", "").replace(" ", "")

    sourced_tokens: set[str] = set()
    for amount in sourced_amounts:
        token = _norm(amount)
        if not token:
            continue
        # The builder renders registered amounts as "<amount>/m²" (land
        # medians) and "<amount>M per bed" (workbook estimates).
        sourced_tokens.update({token, f"{token}M", f"{token}/m²"})
    flags = []
    for flag in narrative_fabrication_flags(text, None):
        amount = _norm(flag.split(":", 1)[-1])
        if amount in sourced_tokens:
            continue
        flags.append(flag)
    return flags


def _candidate_deliberation_subsections(packet: dict[str, Any]) -> list[str]:
    """The structured findings/responses/position subsections for a candidate
    that has a real review packet. The ambiguous one-line "Main review
    comments" field is fallback-only and is never rendered alongside these."""

    from geo_strategist.reporting import markdown_table

    findings = packet.get("reviewer_findings") or []
    responses_by_id = {
        str(r.get("finding_id")): r for r in (packet.get("author_responses") or [])
    }
    lines = ["#### Candidate-specific reviewer findings", ""]
    if findings:
        lines.append(markdown_table(
            ["reviewer", "severity", "issue", "evidence used", "recommendation"],
            [[f.get("reviewer_id"), f.get("severity"),
              str(f.get("issue") or "")[:200],
              ", ".join(f.get("evidence_refs") or [])[:160] or "none",
              f.get("recommendation")] for f in findings],
        ))
    else:
        lines.append("No candidate-specific reviewer findings survived provenance checks.")
    lines.extend(["", "#### Report-author responses", ""])
    if findings:
        lines.append(markdown_table(
            ["finding", "response status", "response", "why still proceed",
             "mitigation", "residual risk"],
            [[str(f.get("issue") or "")[:80],
              (responses_by_id.get(str(f.get("finding_id"))) or {}).get("response_status", "unresolved"),
              str((responses_by_id.get(str(f.get("finding_id"))) or {}).get("response") or "")[:200],
              str((responses_by_id.get(str(f.get("finding_id"))) or {}).get("why_still_proceed") or "")[:160],
              str((responses_by_id.get(str(f.get("finding_id"))) or {}).get("mitigation") or "")[:160],
              str((responses_by_id.get(str(f.get("finding_id"))) or {}).get("residual_risk") or "")[:160]]
             for f in findings],
        ))
    else:
        lines.append("No author responses recorded.")
    position = packet.get("final_candidate_position") or "retain"
    reason = packet.get("final_reason") or ""
    lines.extend([
        "", "#### Final deliberation position", "",
        f"**{position}**" + (f" — {reason}" if reason else ""), "",
    ])
    if packet.get("judge_flags"):
        lines.extend([
            "Provenance/consistency judge flags (informational, not rendered as findings): "
            + "; ".join(str(flag) for flag in packet["judge_flags"][:5]), "",
        ])
    return lines


def render_qualitative_site_discussion_section(entries: list[dict[str, Any]]) -> str:
    lines = ["## Qualitative Site Discussion / 候補地ごとの定性的検討", ""]
    if not entries:
        lines.append("No candidates were available for qualitative discussion.")
        return "\n".join(lines) + "\n"
    for entry in entries:
        paragraphs = entry["paragraphs"]
        lines.extend([
            f"### {entry['heading']}",
            "",
            paragraphs["regional"] + " " + paragraphs["population"],
            "",
            paragraphs["demand_supply"] + " " + paragraphs["access"],
            "",
            paragraphs["cost_financial"],
            "",
            paragraphs["preferred_action"],
            "",
        ])
        packet = entry.get("review_packet")
        if packet:
            lines.extend(_candidate_deliberation_subsections(packet))
        else:
            lines.extend([
                "**Main review comments:** " + paragraphs["review_comments"],
                "",
            ])
    return "\n".join(lines)


def discussion_dimension_coverage(entry: dict[str, Any]) -> dict[str, bool]:
    """Which of the seven required dimensions carry substantive text."""

    paragraphs = entry.get("paragraphs") or {}
    return {name: bool(str(paragraphs.get(name) or "").strip()) and
            len(str(paragraphs.get(name) or "")) >= 40 for name in _DIMENSIONS}
