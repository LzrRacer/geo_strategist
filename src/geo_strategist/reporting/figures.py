"""Static report figures (matplotlib) for proposal and comparison reports.

Follows the project's data-viz method: magnitude comparisons use a single
sequential hue; component identity uses the fixed categorical hue order (never
cycled); text stays in ink colors; grid and axes are recessive; stacked
segments carry a surface-colored gap. Figures are written as PNG next to the
Markdown report that embeds them. The tables in the report body are the
accessible fallback for every figure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_BLUE = "#2a78d6"
# Fixed categorical order (validated reference palette slots 1-7).
CATEGORICAL = ("#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948", "#e87ba4")

COMPONENT_ORDER = (
    "demand",
    "aging",
    "supply_shortage",
    "financial",
    "land",
    "demographic_risk",
    "evidence_completeness",
)

# Municipality names in the study area are Japanese; use a font that can
# render them if available, otherwise matplotlib falls back with warnings only.
matplotlib.rcParams["font.family"] = [
    "Noto Sans CJK JP", "IPAPGothic", "IPAGothic", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def figure_data_path(path: Path) -> Path:
    """Canonical JSON sidecar path for a generated figure PNG."""

    return path.parent / "data" / f"{path.stem}.json"


def write_figure_data(path: Path, payload: dict[str, Any]) -> Path:
    """Persist the data used to build a figure, next to report figures."""

    sidecar = figure_data_path(path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return sidecar


def _style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.xaxis.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def candidate_scores_figure(proposals: list[dict[str, Any]], path: Path, *, title: str) -> Path | None:
    """Horizontal bar of top-k composite scores (one measure, one hue)."""

    rows = [p for p in proposals if p.get("composite_score") is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda p: p["composite_score"])
    labels = [f"{p.get('municipality')} ({p.get('action_type')})" for p in rows]
    values = [float(p["composite_score"]) for p in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(rows) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    bars = ax.barh(labels, values, height=0.55, color=SERIES_BLUE, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center", ha="left", fontsize=9, color=TEXT_PRIMARY,
        )
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel("composite score", fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def component_breakdown_figure(
    proposals: list[dict[str, Any]],
    weights: dict[str, float],
    path: Path,
    *,
    title: str,
) -> Path | None:
    """Stacked horizontal bar of weighted component contributions per candidate."""

    rows = [p for p in proposals if p.get("score_components")]
    if not rows:
        return None
    rows = sorted(rows, key=lambda p: p.get("composite_score") or 0.0)
    labels = [f"{p.get('municipality')} ({p.get('action_type')})" for p in rows]

    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(rows) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    left = [0.0] * len(rows)
    for index, component in enumerate(COMPONENT_ORDER):
        weight = float(weights.get(component, 0.0))
        contributions = []
        for proposal in rows:
            value = (proposal.get("score_components") or {}).get(component)
            contributions.append(weight * float(value) if isinstance(value, (int, float)) else 0.0)
        ax.barh(
            labels, contributions, left=left, height=0.6,
            color=CATEGORICAL[index % len(CATEGORICAL)],
            edgecolor=SURFACE, linewidth=1.2,
            label=component.replace("_", " "), zorder=3,
        )
        left = [a + b for a, b in zip(left, contributions)]
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel("weighted contribution", fontsize=9, color=TEXT_SECONDARY)
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4,
        frameon=False, fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def condition_comparison_figure(
    scores: list[tuple[str, float]],
    path: Path,
    *,
    title: str,
    xlabel: str,
) -> Path | None:
    """Horizontal bar comparing one score across conditions (one hue)."""

    if not scores:
        return None
    ordered = sorted(scores, key=lambda item: item[1])
    labels = [name for name, _value in ordered]
    values = [float(value) for _name, value in ordered]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(ordered) + 1.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)
    bars = ax.barh(labels, values, height=0.55, color=SERIES_BLUE, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_width() + (max(values) or 1) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center", ha="left", fontsize=9, color=TEXT_PRIMARY,
        )
    ax.set_xlim(0, (max(values) or 1) * 1.12)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def grouped_bar_figure(
    categories: list[str],
    series: dict[str, list[float | None]],
    path: Path,
    *,
    title: str,
    ylabel: str,
) -> Path | None:
    """Vertical grouped bars; series identity uses the fixed categorical order."""

    if not categories or not series:
        return None
    fig, ax = plt.subplots(figsize=(max(7.2, 1.2 * len(categories)), 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    count = len(series)
    width = 0.8 / count
    for index, (label, values) in enumerate(series.items()):
        positions = [i + (index - (count - 1) / 2) * width for i in range(len(categories))]
        heights = [float(v) if isinstance(v, (int, float)) else 0.0 for v in values]
        ax.bar(positions, heights, width=width * 0.95,
               color=CATEGORICAL[index % len(CATEGORICAL)], label=label, zorder=3)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22),
                       ncol=min(4, count), frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def success_failure_figure(
    labels: list[str],
    successes: list[int],
    failures: list[int],
    path: Path,
    *,
    title: str,
) -> Path | None:
    """Stacked success/failure counts (e.g. generated-code execution)."""

    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(max(6.0, 1.0 * len(labels)), 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color="#e4e3df", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    positions = range(len(labels))
    ax.bar(positions, successes, color="#1baf7a", label="succeeded",
           edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.bar(positions, failures, bottom=successes, color="#e34948", label="failed",
           edgecolor=SURFACE, linewidth=1.0, zorder=3)
    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("count", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left", pad=10)
    legend = ax.legend(loc="upper right", frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def radar_figure(
    dimensions: list[str],
    series: dict[str, list[float]],
    path: Path,
    *,
    title: str,
    max_value: float = 5.0,
) -> Path | None:
    """Radar chart comparing conditions across score dimensions."""

    import math

    if not dimensions or not series:
        return None
    angles = [2 * math.pi * i / len(dimensions) for i in range(len(dimensions))]
    angles_closed = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(8.4, 8.4), dpi=150,
                           subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for index, (label, values) in enumerate(series.items()):
        closed = list(values) + list(values)[:1]
        color = CATEGORICAL[index % len(CATEGORICAL)]
        ax.plot(angles_closed, closed, color=color, linewidth=1.4, label=label)
        ax.fill(angles_closed, closed, color=color, alpha=0.06)
    ax.set_xticks(angles)
    ax.set_xticklabels([d.replace("_", " ") for d in dimensions], fontsize=7,
                       color=TEXT_SECONDARY)
    ax.set_ylim(0, max_value)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, pad=24)
    legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.04),
                       ncol=3, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def call_distribution_figure(
    model_counts: dict[str, int],
    path: Path,
    *,
    title: str,
) -> Path | None:
    """Horizontal bar of request counts per model (one measure, one hue)."""

    if not model_counts:
        return None
    return condition_comparison_figure(
        [(model, float(count)) for model, count in sorted(model_counts.items())],
        path, title=title, xlabel="requests",
    )


# Sequential blue ramp (light -> dark) for magnitude-on-map encoding;
# steps 100/200/350/450/550/700 of the validated reference ramp.
_SEQUENTIAL_BLUE = ("#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
_CANDIDATE_ACCENT = "#eda100"  # categorical slot 3; identity overlay, not magnitude


def metric_map_figure(
    metric_points: dict[str, list[dict]],
    candidate_points: list[dict],
    path: Path,
    *,
    title: str,
    metric_label: str,
    value_format: str = "{:,.0f}",
) -> Path | None:
    """Per-prefecture panels with municipality anchors colored by one metric.

    ``metric_points`` maps prefecture -> [{"lon", "lat", "value",
    "municipality"}]; anchors are municipality centroids derived from that
    municipality's Yahoo facility geocodes (schematic, not cartographic
    boundaries). Color encodes the metric on a single sequential hue shared
    across panels; municipalities without a metric value are hollow gray
    (missing, never imputed). Candidate sites are overlaid as accent-ringed
    markers with their rank so readers can see each pick against the metric
    surface.
    """

    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    prefectures = [p for p, rows in metric_points.items() if rows]
    values = [row["value"] for rows in metric_points.values() for row in rows
              if isinstance(row.get("value"), (int, float))]
    if not prefectures or not values:
        return None
    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_BLUE)
    norm = Normalize(vmin=min(values), vmax=max(values))

    fig, axes = plt.subplots(1, len(prefectures),
                             figsize=(4.4 * len(prefectures), 5.2), dpi=150)
    if len(prefectures) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    for ax, prefecture in zip(axes, prefectures):
        rows = metric_points[prefecture]
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color("#e4e3df")
        missing = [r for r in rows if not isinstance(r.get("value"), (int, float))]
        present = [r for r in rows if isinstance(r.get("value"), (int, float))]
        if missing:
            ax.scatter([r["lon"] for r in missing], [r["lat"] for r in missing],
                       s=18, facecolors="none", edgecolors="#c9c7c2",
                       linewidths=0.8, zorder=2, label="no data")
        if present:
            ax.scatter([r["lon"] for r in present], [r["lat"] for r in present],
                       s=42, c=[r["value"] for r in present], cmap=cmap, norm=norm,
                       edgecolors=SURFACE, linewidths=0.6, zorder=3)
        marks = sorted((c for c in candidate_points if c["prefecture"] == prefecture),
                       key=lambda c: (c["lat"], c["lon"]))
        for index, candidate in enumerate(marks):
            ax.scatter([candidate["lon"]], [candidate["lat"]], s=170, zorder=4,
                       facecolors="none", edgecolors=_CANDIDATE_ACCENT,
                       linewidths=2.0, marker="o")
            # Alternate offsets so labels of adjacent wards do not collide.
            side = -1 if index % 2 else 1
            ax.annotate(
                f"#{candidate['rank']} {candidate['municipality']}",
                (candidate["lon"], candidate["lat"]),
                textcoords="offset points",
                xytext=(7 * side, 7 + 11 * (index // 2)),
                ha="left" if side > 0 else "right",
                fontsize=8, color=TEXT_PRIMARY, zorder=5)
        ax.margins(0.14)
        ax.set_title(prefecture, fontsize=10, color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
        ax.set_xlabel("longitude", fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylabel("latitude", fontsize=8, color=TEXT_SECONDARY)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    colorbar = fig.colorbar(mappable, ax=axes, orientation="horizontal",
                            fraction=0.05, pad=0.14, aspect=45)
    colorbar.set_label(metric_label, fontsize=9, color=TEXT_SECONDARY)
    colorbar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    colorbar.outline.set_edgecolor("#e4e3df")
    fig.suptitle(title, fontsize=12, color=TEXT_PRIMARY, y=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def candidate_map_figure(
    background_points: dict[str, list[tuple[float, float]]],
    candidate_points: list[dict],
    path: Path,
    *,
    title: str,
) -> Path | None:
    """Schematic per-prefecture location panels.

    Background dots are geocoded facility records (Yahoo Local Search);
    candidate markers are municipality centroids derived as the mean of that
    municipality's facility geocodes. Panels are schematic scatter plots in
    lon/lat, not cartographic base maps.
    """

    prefectures = [p for p in background_points if background_points[p]]
    if not prefectures or not candidate_points:
        return None
    fig, axes = plt.subplots(1, len(prefectures),
                             figsize=(4.4 * len(prefectures), 4.6), dpi=150)
    if len(prefectures) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    for ax, prefecture in zip(axes, prefectures):
        points = background_points[prefecture]
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color("#e4e3df")
        ax.scatter([lon for lon, _lat in points], [lat for _lon, lat in points],
                   s=4, color="#c9c7c2", zorder=2, label="facilities (Yahoo geocodes)")
        marks = [c for c in candidate_points if c["prefecture"] == prefecture]
        for candidate in marks:
            ax.scatter([candidate["lon"]], [candidate["lat"]], s=90, zorder=4,
                       color=SERIES_BLUE, edgecolor=SURFACE, linewidth=1.2, marker="o")
            ax.annotate(
                f"#{candidate['rank']} {candidate['municipality']}",
                (candidate["lon"], candidate["lat"]),
                textcoords="offset points", xytext=(6, 6),
                fontsize=8, color=TEXT_PRIMARY, zorder=5)
        ax.set_title(prefecture, fontsize=10, color=TEXT_PRIMARY)
        ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)
        ax.set_xlabel("longitude", fontsize=8, color=TEXT_SECONDARY)
        ax.set_ylabel("latitude", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle(title, fontsize=12, color=TEXT_PRIMARY, y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
