#!/usr/bin/env python
"""Audit committed prototype files for evidence-handling correctness.

This script inspects git-tracked files (excluding secrets and generated paths)
to find:

1. Wording that incorrectly prohibits concrete details (actual hospital names,
   municipalities, coordinates, or figures) merely because they are specific.
2. Concrete details that lack evidence-status labels.
3. Final-recommendation, exact-site-selection, or investment-decision language
   not properly blocked or graded.
4. Suspicious numeric operational phrases that appear unsupported.

This repo has two tracks with different final-recommendation policies:

- The C0-C13/E13 workflow-control track keeps final recommendations, E10/E11
  readiness, and exact-site selection hard-blocked (`forbidden`/`blocked`
  language).
- The S1-S7/E14 site-selection/investment decision-support track allows a
  graded recommendation (`site_specific_recommendation`,
  `municipality_level_recommendation_site_tbd`) once every populated
  concrete claim clears an explicit evidence grade
  (`verified_source`..`unverified_candidate`/`rejected_or_blocked`; see
  `docs/agent/site_selection_pipeline.md`). This is decision-support
  language, not unsupported/unblocked language, so it is recognized as safe
  when it appears alongside evidence-grade/decision-support markers (see
  `_DECISION_SUPPORT_CONTEXT_RE` below) rather than being flagged as
  unblocked forbidden language.

The script never opens .env, never prints secret-like strings, never calls live
APIs or LLMs, and never prints raw LLM or API responses.

Usage:
    .venv/bin/python scripts/audit_prototype_safety.py [root]
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

console = Console()

# ── Paths to skip entirely ────────────────────────────────────────────────────

SKIP_PATHS = {
    ".env",
    ".venv",
    ".cache",
    ".data",
    ".runs",
    ".scratch",
    ".junk",
}

SKIP_PATH_PREFIXES = (
    "references/local",
)

REPOSITORY_SCAN_FIXTURE_PATHS = {
    Path("tests/test_audit_prototype_safety.py"),
}

# ── File extensions to inspect ────────────────────────────────────────────────

TEXT_SUFFIXES = {
    ".md", ".py", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".ini",
}

# ── Wording that incorrectly prohibits concrete details ──────────────────────
#
# These patterns appear in text where the author is saying "do not use actual
# hospital names / municipalities / concrete figures" when the correct policy
# is "use concrete details when sourced, and label evidence status."
#
# The patterns use word-boundary matching to avoid false positives.

_OVER_RESTRICTIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "prohibits_actual_hospital_names",
        re.compile(
            r"(?:remove|do not include|avoid|omit|exclude|not include)\s+"
            r"(?:actual|real|specific|concrete)?\s*hospital\s+name",
            re.IGNORECASE,
        ),
    ),
    (
        "prohibits_municipalities",
        re.compile(
            r"(?:remove|do not include|avoid|omit|exclude)\s+"
            r"(?:actual|real|specific|concrete)?\s*municipalit",
            re.IGNORECASE,
        ),
    ),
    (
        "replace_real_with_synthetic",
        re.compile(
            r"replace\s+(?:real|actual|specific)\s+(?:details?|names?|values?|figures?|data)\s+"
            r"with\s+(?:synthetic|placeholder|dummy|fake|mock)",
            re.IGNORECASE,
        ),
    ),
    (
        "anonymize_hospitals",
        re.compile(
            r"anonymize\s+(?:all\s+)?hospitals?",
            re.IGNORECASE,
        ),
    ),
    (
        "prototype_must_not_contain_real_names",
        re.compile(
            r"prototype\s+must\s+not\s+contain\s+(?:real|actual|specific)",
            re.IGNORECASE,
        ),
    ),
    (
        "do_not_use_actual_data",
        re.compile(
            r"do\s+not\s+use\s+actual\s+(?:hospital|municipality|coordinate|population|land|address)\s+data",
            re.IGNORECASE,
        ),
    ),
    (
        "only_synthetic_examples",
        re.compile(
            r"only\s+(?:synthetic|toy|placeholder|dummy|fake)\s+examples?\s+(?:are\s+)?(?:allowed|permitted|acceptable)",
            re.IGNORECASE,
        ),
    ),
    (
        "avoid_specific_details",
        re.compile(
            r"(?:avoid|omit|remove|exclude)\s+(?:all\s+)?specific\s+(?:names?|details?|figures?|values?|addresses?|coordinates?)",
            re.IGNORECASE,
        ),
    ),
    (
        "remove_concrete_figures",
        re.compile(
            r"remove\s+(?:all\s+)?concrete\s+(?:figures?|values?|numbers?|details?)",
            re.IGNORECASE,
        ),
    ),
]

# ── Final-recommendation / exact-site language that is NOT blocked ─────────────
#
# We look for phrases suggesting final recommendations are allowed, but exclude:
# - lines within regex pattern strings or "in text" membership checks (detection code)
# - lines with blocking/denial context nearby

_FINAL_REC_RE = re.compile(
    r"\b(?:issue|produce|make|generate|select|confirm|approve)\s+(?:a\s+)?(?:final\s+recommendation|"
    r"definitive\s+recommendation|investment\s+decision|exact\s+site\s+selection)\b",
    re.IGNORECASE,
)

_BLOCKED_CONTEXT_RE = re.compile(
    r"\b(?:forbidden|blocked|must\s+not|shall\s+not|cannot|not\s+implemented|"
    r"NOT|BLOCKED|FORBIDDEN|prohibited|illegal|prevent|deny|denying|gate|"
    r"remain\s+blocked|not\s+produce|does\s+not\s+produce|will\s+not)\b",
    re.IGNORECASE,
)

# S1-S7/E14 decision-support markers: final-recommendation-shaped language
# next to one of these is an intentionally evidence-graded, human-reviewed
# decision-support claim, not unblocked/unsupported forbidden language.
_DECISION_SUPPORT_CONTEXT_RE = re.compile(
    r"\b(?:evidence[_\s]grade|decision[_\s]support|decision.support\s+disclaimer|"
    r"human\s+(?:expert\s+)?due\s+diligence|not\s+a\s+certified|"
    r"site_specific_recommendation|municipality_level_recommendation_site_tbd|"
    r"not_recommended_insufficient_evidence|unverified_candidate|"
    r"rejected_or_blocked|verified_source|scenario_assumption|model_estimate)\b",
    re.IGNORECASE,
)

# Patterns indicating the line is detection/testing code, not a usage
_DETECTION_CODE_RE = re.compile(
    r"""(?:r['"]{1,3}|in\s+text|re\.compile|\.search|\.match|pytest\.fail|assert|"""
    r"""if\s+["']|#\s*check|#\s*detect|#\s*test|flagged\s+by|prohibited\s+phrase)""",
    re.IGNORECASE,
)

# ── Evidence-status label patterns (used to verify that concrete claims are labeled) ─

_EVIDENCE_STATUS_LABELS = re.compile(
    r"\b(?:verified|unverified|illustrative|synthetic|blocked|"
    r"scenario_assumption|source_document_verified|live_api_verified|"
    r"manual_workbook_input|derived_from_project_input|"
    r"verified_project_input|blocked_by_missing_evidence|"
    r"evidence_status|evidence.status|evidence.gap|"
    r"supported_by_selected_evidence|not\s+verified|"
    r"workbook[\s_]fact|workbook[\s_]input|"
    r"scenario\s+assumption|documented\s+assumption|"
    r"experimental|not\s+final|not\s+a\s+recommendation)\b",
    re.IGNORECASE,
)

# Concrete-detail patterns: names, figures, coordinates, etc.
_CONCRETE_HOSPITAL_NAME_RE = re.compile(
    r"(?:病院|Hospital|医療センター|Medical\s+Center|クリニック|Clinic)\s*[「」『』「」【】]?[^\s「」『』「」【】]{2,20}",
    re.IGNORECASE,
)
_CONCRETE_COORDINATE_RE = re.compile(
    r"(?:latitude|longitude|lat|lon|lng|緯度|経度)\s*[=:]\s*[-]?\d{1,3}\.\d+",
    re.IGNORECASE,
)
_CONCRETE_POPULATION_RE = re.compile(
    r"(?:population|人口)\s*[=:]\s*[\d,]{4,}",
    re.IGNORECASE,
)
_CONCRETE_CASH_FLOW_RE = re.compile(
    r"(?:NPV|IRR|payback|revenue|CAPEX|OPEX|cash.?flow)\s*[=:]\s*[\d,¥$€]+",
    re.IGNORECASE,
)
_CONCRETE_BED_COUNT_RE = re.compile(
    r"(?:beds?|病床)\s*[=:]\s*\d{2,}",
    re.IGNORECASE,
)

# Secret-like patterns — used for detection only; values are never printed
_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|app[_-]?id|secret|token|password|passwd|ESTAT_APP_ID)\s*[=:]\s*\S{6,}",
)


@dataclass
class AuditFinding:
    path: str
    line: int
    category: str
    summary: str
    severity: str  # "error", "warning", "info"
    excerpt: str = ""  # short excerpt, never secrets


@dataclass
class AuditReport:
    findings: list[AuditFinding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0
    concrete_details_found: int = 0
    over_restrictive_phrases: int = 0
    unblocked_final_rec_phrases: int = 0
    evidence_labeled_details: int = 0


def _get_tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        p = Path(line)
        paths.append(p)
    return paths


def _should_skip(rel: Path) -> bool:
    parts = rel.parts
    if not parts:
        return True
    if str(rel) == ".env":
        return True
    # Skip if any part matches a blocked dir
    for part in parts:
        if part in SKIP_PATHS:
            return True
    # Skip references/local prefix
    for prefix in SKIP_PATH_PREFIXES:
        if str(rel).startswith(prefix):
            return True
    return False


def _safe_excerpt(line: str, max_len: int = 120) -> str:
    """Return a short excerpt that never contains secret-like substrings."""
    # Remove any secret-looking tokens
    cleaned = _SECRET_RE.sub("[SECRET_REDACTED]", line.strip())
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "…"
    return cleaned


def _audit_file(root: Path, rel: Path, report: AuditReport) -> None:
    abs_path = root / rel
    if abs_path.suffix.lower() not in TEXT_SUFFIXES:
        report.files_skipped += 1
        return

    try:
        text = abs_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        report.files_skipped += 1
        return

    report.files_scanned += 1
    path_str = str(rel)

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line_lower = raw_line.lower()

        # 1. Check for over-restrictive wording
        for category, pattern in _OVER_RESTRICTIVE_PATTERNS:
            if pattern.search(raw_line):
                report.over_restrictive_phrases += 1
                report.findings.append(AuditFinding(
                    path=path_str,
                    line=lineno,
                    category=category,
                    summary="Wording may incorrectly prohibit concrete details.",
                    severity="warning",
                    excerpt=_safe_excerpt(raw_line),
                ))

        # 2. Check for unblocked final-recommendation language
        if _FINAL_REC_RE.search(raw_line):
            # Skip detection/testing code: regex patterns, membership checks, asserts
            if not _DETECTION_CODE_RE.search(raw_line):
                surrounding = "\n".join(
                    text.splitlines()[max(0, lineno - 3): lineno + 2]
                )
                if not _BLOCKED_CONTEXT_RE.search(surrounding) and not _DECISION_SUPPORT_CONTEXT_RE.search(surrounding):
                    report.unblocked_final_rec_phrases += 1
                    report.findings.append(AuditFinding(
                        path=path_str,
                        line=lineno,
                        category="unblocked_final_recommendation_language",
                        summary="Final-recommendation language without blocking context.",
                        severity="warning",
                        excerpt=_safe_excerpt(raw_line),
                    ))

        # 3. Note concrete details and check for evidence-status labels.
        #
        # Coordinate, population, cash-flow, and bed-count patterns are reasonably
        # specific. Hospital-name detection is limited to Markdown/YAML prose (not
        # Python, where "Hospital" commonly appears in variable names and comments).
        is_prose_file = abs_path.suffix.lower() in (".md", ".yaml", ".yml", ".txt")
        has_concrete = (
            _CONCRETE_COORDINATE_RE.search(raw_line)
            or _CONCRETE_POPULATION_RE.search(raw_line)
            or _CONCRETE_CASH_FLOW_RE.search(raw_line)
            or _CONCRETE_BED_COUNT_RE.search(raw_line)
            or (is_prose_file and _CONCRETE_HOSPITAL_NAME_RE.search(raw_line))
        )
        if has_concrete:
            report.concrete_details_found += 1
            # Check if the surrounding context (±2 lines) carries evidence-status labels
            start = max(0, lineno - 3)
            end = min(len(text.splitlines()), lineno + 2)
            context_block = "\n".join(text.splitlines()[start:end])
            if _EVIDENCE_STATUS_LABELS.search(context_block):
                report.evidence_labeled_details += 1
            else:
                # Flag unlabeled concrete details in committed reports only.
                # Planning docs under docs/context/ use operational language
                # in a methodological/schema context that does not require labels.
                if "docs/reports/" in path_str or "reports/" in path_str:
                    report.findings.append(AuditFinding(
                        path=path_str,
                        line=lineno,
                        category="concrete_detail_without_evidence_label",
                        summary="Concrete detail found without nearby evidence-status label.",
                        severity="info",
                        excerpt=_safe_excerpt(raw_line),
                    ))


def run_audit(root: Path) -> AuditReport:
    report = AuditReport()
    tracked = _get_tracked_files(root)
    for rel in tracked:
        if rel in REPOSITORY_SCAN_FIXTURE_PATHS:
            report.files_skipped += 1
            continue
        if _should_skip(rel):
            report.files_skipped += 1
            continue
        _audit_file(root, rel, report)
    return report


def _print_report(report: AuditReport) -> int:
    console.print("\n[bold]Prototype Safety Audit[/bold]")
    console.print(f"  Files scanned  : {report.files_scanned}")
    console.print(f"  Files skipped  : {report.files_skipped}")
    console.print(f"  Concrete details found : {report.concrete_details_found}")
    console.print(f"  Evidence-labeled       : {report.evidence_labeled_details}")
    console.print(f"  Over-restrictive phrases : {report.over_restrictive_phrases}")
    console.print(f"  Unblocked final-rec phrases : {report.unblocked_final_rec_phrases}")

    errors = [f for f in report.findings if f.severity == "error"]
    warnings = [f for f in report.findings if f.severity == "warning"]
    infos = [f for f in report.findings if f.severity == "info"]

    if errors:
        console.print(f"\n[red]ERRORS ({len(errors)}):[/red]")
        for f in errors:
            console.print(f"  [{f.path}:{f.line}] {f.category}: {f.summary}")
            if f.excerpt:
                console.print(f"    {f.excerpt}")

    if warnings:
        console.print(f"\n[yellow]WARNINGS ({len(warnings)}):[/yellow]")
        for f in warnings:
            console.print(f"  [{f.path}:{f.line}] {f.category}: {f.summary}")
            if f.excerpt:
                console.print(f"    {f.excerpt}")

    if infos:
        console.print(f"\n[blue]INFO ({len(infos)}):[/blue]")
        for f in infos[:20]:  # cap to avoid flooding output
            console.print(f"  {f.path}:{f.line}  {f.category}")
            if f.excerpt:
                console.print(f"    {f.excerpt}")
        if len(infos) > 20:
            console.print(f"  … and {len(infos) - 20} more info findings.")

    if errors:
        console.print("\n[red]Audit FAILED — errors found.[/red]")
        return 1

    if report.over_restrictive_phrases:
        console.print("\n[yellow]Audit WARNING — over-restrictive wording found.[/yellow]")
        return 0  # warnings do not fail the audit

    console.print("\n[green]Audit passed.[/green]")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Repository root to audit (default: current directory).",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = run_audit(root)
    return _print_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
