"""Sanitizes a condition's final Markdown report into the text shown to the
LLM checklist judge for report-visible decision-analysis quality.

The judge must score only what a reader could observe in the anonymized
final report — never condition metadata, provider/model/harness identity,
execution/exclusion status, model-call or generated-code internals, run
artifact/figure paths, or shared boilerplate that is identical across every
condition and therefore cannot discriminate between them.

Parsing is heading-based (Markdown ``##`` boundaries), not fragile global
string replacement: an H2 section is either kept in full (with all of its
nested ``###``/``####`` content and body text) or dropped in full by an
explicit heading-text match. This is deliberate — a section must never be
dropped merely because its heading contains a word like "review" or
"branch"; those are exactly the sections (candidate-level review comments,
branch-by-branch objective results) the new rubric is designed to evaluate.
"""

from __future__ import annotations

import re

# H2 ("## ") section headings excluded from the judge view because they are
# shared, non-discriminating boilerplate (near-identical prose/definitions
# across every condition) or identity/operational metadata rather than
# report-visible, condition-specific evidence. Matched by exact heading
# text, not a substring/prefix, so an evidence-bearing heading is never
# dropped by accidental overlap.
EXCLUDED_H2_HEADINGS: frozenset[str] = frozenset({
    "Evaluation criteria and selection method",
    "Selection funnel (nationwide → final slate)",
    "Method (model-reported)",
    "Execution notes",
    "Study-area data quality note",
    "Generated-code execution",
    "Model-call summary",
    "Figures",
    "Run artifacts",
    "Limitations / Required Due Diligence",
})

# Front-matter bullet lines (between the H1 title and the first H2 heading,
# see live_report.py:write_condition_report) that are pure condition
# identity/operational metadata — never report-visible evidence.
_IDENTITY_BULLET_PREFIXES: tuple[str, ...] = (
    "- Condition:",
    "- Provider/model/harness:",
    "- Execution mode:",
    "- Exclusion reason:",
    "- Live steps/calls used:",
)

_H1_RE = re.compile(r"^#\s+(.*)$")
_H2_RE = re.compile(r"^##\s+(.*)$")

_ANONYMOUS_TITLE_SUFFIX = ": Hospital Location / Reorganization Proposal"


def sanitize_report_for_judge(report_text: str, method_alias: str) -> str:
    """Return ``report_text`` with the title replaced by ``method_alias``,
    identity-bearing front-matter bullets removed, and every H2 section in
    ``EXCLUDED_H2_HEADINGS`` dropped in full. Every other H2 section
    (including its nested sub-headings and body) is kept verbatim — this
    function never rewrites or truncates evidence-bearing report text, it
    only removes whole non-discriminating/identity-bearing blocks.

    Returns an empty string for empty/whitespace-only input; the caller
    (``condition_comparison_judge.py``) treats that as "no report
    available" and must not fall back to structured condition-record data.
    """

    lines = (report_text or "").splitlines()
    if not any(line.strip() for line in lines):
        return ""

    out: list[str] = []

    index = 0
    if lines and _H1_RE.match(lines[0]):
        out.append(f"# {method_alias}{_ANONYMOUS_TITLE_SUFFIX}")
        index = 1
    while index < len(lines) and not _H2_RE.match(lines[index]):
        line = lines[index]
        if not any(line.strip().startswith(prefix) for prefix in _IDENTITY_BULLET_PREFIXES):
            out.append(line)
        index += 1

    current_block: list[str] = []
    include_current = True

    def flush() -> None:
        if include_current and current_block:
            out.extend(current_block)

    while index < len(lines):
        line = lines[index]
        match = _H2_RE.match(line)
        if match:
            flush()
            heading = match.group(1).strip()
            include_current = heading not in EXCLUDED_H2_HEADINGS
            current_block = [line]
        else:
            current_block.append(line)
        index += 1
    flush()

    return "\n".join(out).strip() + "\n"
