"""Figures for the candidate-level deliberation pipeline.

Shares the palette and axis styling of ``reporting.figures`` so review
figures read as part of the same visual system as the rest of a condition
report. Every function returns ``None`` (never raises) when there is
nothing to plot, so a figure-generation failure never stops the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geo_strategist.reporting.figures import (
    CATEGORICAL,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
)

_SEVERITY_ORDER = ("blocking", "major", "moderate", "minor")
_SEVERITY_COLORS = {
    "blocking": "#e34948",
    "major": "#eda100",
    "moderate": "#2a78d6",
    "minor": "#1baf7a",
}
_RESPONSE_ORDER = ("accepted", "partially_accepted", "rejected")
_RESPONSE_COLORS = {
    "accepted": "#1baf7a",
    "partially_accepted": "#eda100",
    "rejected": "#e34948",
}


def _candidate_label(packet: Any) -> str:
    return str(getattr(packet, "candidate_id", "?"))


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def review_severity_figure(
    packets: list[Any],
    output_path: Path,
    *,
    title: str,
) -> Path | None:
    """Stacked horizontal bar: blocking/major/moderate/minor findings per candidate."""

    if not packets:
        return None
    labels = [_candidate_label(p) for p in packets]
    counts = {
        severity: [
            sum(1 for f in p.reviewer_findings if f.severity == severity)
            for p in packets
        ]
        for severity in _SEVERITY_ORDER
    }
    if not any(any(values) for values in counts.values()):
        return None

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(labels) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    left = [0.0] * len(labels)
    for severity in _SEVERITY_ORDER:
        values = counts[severity]
        ax.barh(labels, values, left=left, height=0.6,
               color=_SEVERITY_COLORS[severity], edgecolor=SURFACE, linewidth=1.0,
               label=severity, zorder=3)
        left = [a + b for a, b in zip(left, values)]
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel("finding count", fontsize=9, color=TEXT_SECONDARY)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                       ncol=4, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def reviewer_coverage_figure(
    packets: list[Any],
    output_path: Path,
    *,
    title: str,
) -> Path | None:
    """Reviewer x candidate heatmap of finding counts."""

    if not packets:
        return None
    candidates = [_candidate_label(p) for p in packets]
    reviewers = sorted({
        f.reviewer_id for p in packets for f in p.reviewer_findings
    })
    if not reviewers:
        return None
    matrix = [
        [sum(1 for f in p.reviewer_findings if f.reviewer_id == reviewer) for p in packets]
        for reviewer in reviewers
    ]

    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.9 * len(candidates) + 2.0), max(3.5, 0.45 * len(reviewers) + 1.5)),
        dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    vmax = max((max(row) for row in matrix), default=0) or 1
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(candidates)))
    ax.set_xticklabels(candidates, rotation=25, ha="right", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(reviewers)))
    ax.set_yticklabels([r.replace("_", " ") for r in reviewers], fontsize=8, color=TEXT_SECONDARY)
    for i in range(len(reviewers)):
        for j in range(len(candidates)):
            value = matrix[i][j]
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8,
                   color=TEXT_PRIMARY if value <= vmax / 2 else "white")
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def author_response_status_figure(
    packets: list[Any],
    output_path: Path,
    *,
    title: str,
) -> Path | None:
    """Stacked bar: accepted / partially_accepted / rejected counts per candidate."""

    if not packets:
        return None
    labels = [_candidate_label(p) for p in packets]
    counts = {
        status: [
            sum(1 for r in p.author_responses if r.response_status == status)
            for p in packets
        ]
        for status in _RESPONSE_ORDER
    }
    if not any(any(values) for values in counts.values()):
        return None

    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(labels)), 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    positions = range(len(labels))
    bottom = [0] * len(labels)
    for status in _RESPONSE_ORDER:
        values = counts[status]
        ax.bar(positions, values, bottom=bottom, color=_RESPONSE_COLORS[status],
               edgecolor=SURFACE, linewidth=1.0, label=status, zorder=3)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("response count", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path


def residual_risk_figure(
    packets: list[Any],
    output_path: Path,
    *,
    title: str,
) -> Path | None:
    """Unresolved major/blocking findings (no accepted mitigation) per candidate."""

    if not packets:
        return None
    labels = []
    values = []
    for packet in packets:
        responses_by_id = {r.finding_id: r for r in packet.author_responses}
        unresolved = 0
        for finding in packet.reviewer_findings:
            if finding.severity not in ("blocking", "major"):
                continue
            response = responses_by_id.get(finding.finding_id)
            if response is None or response.response_status == "rejected":
                unresolved += 1
        labels.append(_candidate_label(packet))
        values.append(unresolved)
    if not any(values):
        return None

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(labels) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    bars = ax.barh(labels, values, height=0.55, color=CATEGORICAL[5], zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.02 + 0.02,
               bar.get_y() + bar.get_height() / 2, str(value),
               va="center", ha="left", fontsize=9, color=TEXT_PRIMARY)
    ax.set_xlim(0, max(values) * 1.2 + 1)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel("unresolved major/blocking findings", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output_path
