"""
RQ3 Analysis: RAGAS Metric Consistency
=======================================
Reads results/rq3_raw.csv produced by rq3_experiment.py and outputs:
  - results/rq3_stats.csv        — per-sample per-metric descriptive statistics
  - results/rq3_icc.csv          — ICC(2,1) per metric with interpretation
  - results/rq3_fig1a_score_distributions_primary.png   (faithfulness, context precision)
  - results/rq3_fig1b_score_distributions_secondary.png (answer relevance, context recall)
  - results/rq3_fig2_faithfulness_per_sample.png
  - results/rq3_fig3_heatmap.png
  - results/rq3_fig4_run_stability.png

Plot style matches analysis/rq4/rq4_experiment.py's ACADEMIC_STYLE so both
RQ chapters use the same fonts, colors, and axis conventions.

Usage
-----
    python rq3_analysis.py

    # Override input/output paths via env vars:
    RAW_CSV=results/rq3_raw.csv \
    OUT_DIR=results \
    python rq3_analysis.py
"""

import csv
import math
import os
import sys
from pathlib import Path

import matplotlib
from dotenv import load_dotenv

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Local helpers — helper/icc.py must exist relative to this script
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "helpers"))
from icc import ReliabilityAnalysis  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Paths are resolved relative to this script file, not the working directory.
_HERE = Path(__file__).resolve().parent
# Run standalone from a local venv, not through docker-compose's env_file, so
# load .env explicitly before reading any of the overrides below.
load_dotenv(_HERE.parent.parent / ".env")
RAW_CSV = Path(
    os.getenv("RAW_CSV", str(_HERE / "../../results/rq3/data/rq3_raw.csv"))
).resolve()
STATS_DIR = Path(
    os.getenv("STATS_DIR", str(_HERE / "../../results/rq3/csvs"))
).resolve()
IMGS_DIR = Path(
    os.getenv("IMGS_DIR", str(_HERE / "../../results/rq3/images"))
).resolve()
STATS_DIR.mkdir(parents=True, exist_ok=True)
IMGS_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
METRIC_LABEL = {
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer Relevance",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}

# The two metrics with genuine, sample dependent spread, shown together in
# fig1a since they carry the actual reliability discussion in the results
# section.
PRIMARY_METRICS = ["faithfulness", "context_precision"]
# Near ceiling (context recall) or tightly bound (answer relevance), shown
# in fig1b for completeness rather than for their own discussion.
SECONDARY_METRICS = ["answer_relevance", "context_recall"]

# Colorblind safe qualitative palette (Okabe and Ito, 2008), yellow excluded
# on request. Each metric also gets a distinct marker shape below, so the
# encoding still works in grayscale or under any color vision deficiency,
# not only standard vision.
PALETTE_CB = {
    "faithfulness": "#0072B2",  # blue
    "answer_relevance": "#D55E00",  # vermillion
    "context_precision": "#009E73",  # bluish green
    "context_recall": "#CC79A7",  # reddish purple
}
MARKERS = {
    "faithfulness": "o",
    "answer_relevance": "s",
    "context_precision": "^",
    "context_recall": "D",
}

# Same style block as analysis/rq4/rq4_experiment.py, kept identical so RQ3
# and RQ4 figures read as one consistent set in the report.
ACADEMIC_STYLE = {
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_raw(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "run_id": int(row["run_id"]),
                    "sample_id": int(row["sample_id"]),
                    "faithfulness": float(row["faithfulness"])
                    if row["faithfulness"]
                    else None,
                    "answer_relevance": float(row["answer_relevance"])
                    if row["answer_relevance"]
                    else None,
                    "context_precision": float(row["context_precision"])
                    if row["context_precision"]
                    else None,
                    "context_recall": float(row["context_recall"])
                    if row["context_recall"]
                    else None,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# Statistics helpers (stdlib only)
# ---------------------------------------------------------------------------
def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _cv(vals: list[float]) -> float | None:
    m = _mean(vals)
    s = _std(vals)
    if m is None or s is None or m == 0:
        return None
    return s / m


def build_matrix(rows: list[dict], metric: str) -> dict[int, dict[int, float]]:
    """Return {sample_id: {run_id: score}}."""
    mat: dict[int, dict[int, float]] = {}
    for r in rows:
        if r[metric] is not None:
            mat.setdefault(r["sample_id"], {})[r["run_id"]] = r[metric]
    return mat


# ---------------------------------------------------------------------------
# ICC(2,1) — delegates to ReliabilityAnalysis in helper/icc.py
# ---------------------------------------------------------------------------
def compute_icc(rows: list[dict], metric: str) -> dict:
    mat = build_matrix(rows, metric)
    sids = sorted(mat.keys())
    runs = sorted({r["run_id"] for r in rows})
    data = []
    for sid in sids:
        row_vals = [mat[sid].get(rid) for rid in runs]
        if any(v is None for v in row_vals):
            continue
        data.append([float(v) for v in row_vals])

    arr = np.array(data)  # shape (n_samples, k_runs)
    n, k = arr.shape

    # Perfect agreement — all scores identical, ICC is trivially 1
    icc = 1.0 if arr.std() == 0 else float(ReliabilityAnalysis.compute_icc(arr))

    if icc < 0.5:
        interpretation = "poor"
    elif icc < 0.75:
        interpretation = "moderate"
    elif icc < 0.9:
        interpretation = "good"
    else:
        interpretation = "excellent"

    return {
        "metric": metric,
        "icc": round(icc, 6),
        "interpretation": interpretation,
        "n_samples": n,
        "n_runs": k,
    }


# ---------------------------------------------------------------------------
# Per-sample descriptive statistics
# ---------------------------------------------------------------------------
def compute_sample_stats(rows: list[dict]) -> list[dict]:
    stats = []
    for metric in METRICS:
        mat = build_matrix(rows, metric)
        for sid in sorted(mat.keys()):
            vals = list(mat[sid].values())
            stats.append(
                {
                    "metric": metric,
                    "sample_id": sid,
                    "mean": round(_mean(vals), 6) if _mean(vals) is not None else None,
                    "std": round(_std(vals), 6) if _std(vals) is not None else None,
                    "cv": round(_cv(vals), 6) if _cv(vals) is not None else None,
                    "min": round(min(vals), 6),
                    "max": round(max(vals), 6),
                    "n_runs": len(vals),
                }
            )
    return stats


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def write_stats_csv(stats: list[dict]):
    path = STATS_DIR / "rq3_stats.csv"
    fields = ["metric", "sample_id", "mean", "std", "cv", "min", "max", "n_runs"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(stats)
    print(f"Saved: {path}")


def write_icc_csv(icc_results: list[dict]):
    path = STATS_DIR / "rq3_icc.csv"
    fields = ["metric", "icc", "interpretation", "n_samples", "n_runs"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(icc_results)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _boxplot_legend_elements(
    color: str = "#888888", include_outlier: bool = True
) -> list:
    """Proxy artists explaining box plot semantics for a general audience.

    Kept in one neutral color rather than matching each panel's metric
    color, since a single shared legend sits above panels of different
    colors in fig1a and fig1b. include_outlier is False when the figure
    being built has no flier points at all, so the legend never lists a
    symbol that is not actually present anywhere on the chart. The label
    is deliberately just "Outlier", the 1.5x IQR convention and the fact
    that outliers are still counted in every statistic belongs in the
    report text next to the figure, not on the chart itself.
    """
    elements = [
        Patch(
            facecolor=color,
            alpha=0.6,
            edgecolor="#555555",
            label="Interquartile range (25th to 75th percentile)",
        ),
        Line2D([0], [0], color="black", linewidth=1.5, label="Median"),
        Line2D([0], [0], color="#555555", linewidth=1, label="Whiskers (1.5x IQR)"),
    ]
    if include_outlier:
        elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=color,
                markeredgecolor="#555555",
                markersize=5,
                alpha=0.6,
                label="Outlier",
            )
        )
    return elements


def _plot_score_distribution_pair(
    rows: list[dict], n_runs: int, metrics: list[str], path: Path, subtitle: str
):
    """Shared implementation behind fig1a and fig1b.

    Two metrics per figure rather than four, so each panel gets roughly
    twice the width of the original single 1x4 layout, plus a shared
    legend explaining what the box, whiskers, and outlier points mean.
    """
    with plt.rc_context(ACADEMIC_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        has_outliers = False
        for ax, m in zip(axes, metrics):
            color = PALETTE_CB[m]
            vals_per_run = [
                [r[m] for r in rows if r["run_id"] == rid and r[m] is not None]
                for rid in range(1, n_runs + 1)
            ]
            bp = ax.boxplot(
                vals_per_run,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                whiskerprops=dict(color="#555555"),
                capprops=dict(color="#555555"),
                # A fixed circle marker, the same for every metric, so the
                # single shared legend entry below always matches what is
                # actually drawn. MARKERS[m] is still used for fig4's line
                # series, but outliers here are not tied to a metric shape.
                flierprops=dict(
                    marker="o",
                    markersize=4,
                    alpha=0.5,
                    markerfacecolor=color,
                    markeredgecolor="#555555",
                ),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            if any(len(flier.get_xdata()) > 0 for flier in bp["fliers"]):
                has_outliers = True
            ax.set_title(METRIC_LABEL[m], fontweight="bold")
            ax.set_xlabel("Run")
            ax.set_ylabel("Score")
            ax.set_ylim(-0.05, 1.05)
            step = max(1, n_runs // 5)
            ticks = list(range(1, n_runs + 1, step))
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticks)
            ax.grid(True, axis="y", alpha=0.5)
            ax.grid(False, axis="x")

        n_samples = len({r["sample_id"] for r in rows})
        fig.suptitle(
            f"Score Distribution per Run, {subtitle} "
            f"(amnesty_qa, N={n_samples} samples x {n_runs} runs)",
            fontweight="bold",
            y=1.1,
        )
        legend_elements = _boxplot_legend_elements(include_outlier=has_outliers)
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=len(legend_elements),
            bbox_to_anchor=(0.5, 1.0),
            fontsize=9,
        )
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved: {path}")


def fig1a_primary_metrics(rows: list[dict], n_runs: int):
    """Faithfulness and context precision, the two metrics that carry the
    actual reliability discussion."""
    _plot_score_distribution_pair(
        rows,
        n_runs,
        PRIMARY_METRICS,
        IMGS_DIR / "rq3_fig1a_score_distributions_primary.png",
        "Primary Metrics",
    )


def fig1b_secondary_metrics(rows: list[dict], n_runs: int):
    """Answer relevance and context recall, shown for completeness. Both
    sit close to a ceiling or floor and move far less than the primary
    pair."""
    _plot_score_distribution_pair(
        rows,
        n_runs,
        SECONDARY_METRICS,
        IMGS_DIR / "rq3_fig1b_score_distributions_secondary.png",
        "Secondary Metrics",
    )


def fig2_faithfulness_per_sample(rows: list[dict]):
    """Bar chart of per sample faithfulness mean and standard deviation.

    The 0.5 threshold line from the original version is gone, it implied a
    pass or fail cutoff this experiment never established. Every bar is
    labeled with its exact mean value (three decimals, matching the report
    text convention), positioned above the error bar cap rather than the
    bar top, so the label never overlaps the whisker. Without this, a bar
    at or near zero height (sample 20 in the reference dataset, mean
    0.000, std 0.000) reads as missing data rather than a real, perfectly
    consistent score.

    Standard deviation is symmetric and has no concept of faithfulness's
    own 0 to 1 bound. A sample with a skewed distribution, many runs
    pinned at 1.0 with a scattered lower tail, can have mean + 1 SD exceed
    1.0 (sample 17 in the reference dataset: mean 0.898, SD 0.177, mean +
    SD = 1.075), which is not an error but does look like one when drawn.
    The whiskers here are clipped to [0, 1] for display; the SD value
    itself, printed nowhere on this chart but available in rq3_stats.csv,
    is left uncapped.
    """
    with plt.rc_context(ACADEMIC_STYLE):
        mat = build_matrix(rows, "faithfulness")
        sids = sorted(mat.keys())
        means, stds = [], []
        for sid in sids:
            vals = list(mat[sid].values())
            means.append(_mean(vals) or 0.0)
            stds.append(_std(vals) or 0.0)

        # Asymmetric, clipped whiskers: drawn extent never implies a score
        # outside [0, 1], even though the underlying SD is computed and
        # reported (in rq3_stats.csv) without that clipping.
        lower_err = [min(s, m) for m, s in zip(means, stds)]
        upper_err = [min(s, 1.0 - m) for m, s in zip(means, stds)]

        color = PALETTE_CB["faithfulness"]
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(sids))
        ax.bar(
            x,
            means,
            yerr=[lower_err, upper_err],
            capsize=4,
            color=color,
            alpha=0.75,
            ecolor="#333333",
            linewidth=0.8,
        )

        for xi, mean_val, upper in zip(x, means, upper_err):
            ax.annotate(
                f"{mean_val:.3f}",
                xy=(xi, mean_val + upper),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#333333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"S{s}" for s in sids])
        ax.set_xlabel("Sample")
        ax.set_ylabel("Faithfulness score")
        ax.set_ylim(0, 1.15)
        ax.grid(True, axis="y", alpha=0.5)
        ax.grid(False, axis="x")

        # Title and legend both placed with fig-level coordinates (not
        # ax.set_title, which is axes-relative and does not compose
        # predictably with fig.legend's figure-relative anchor), so their
        # vertical order is set directly rather than fought over.
        fig.suptitle(
            "Per Sample Faithfulness, Mean \u00b1 SD across 20 Runs",
            fontweight="bold",
            y=1.08,
        )
        # Legend placed above the axes rather than inside a corner: with 20
        # bars, some reach close enough to 1.0 that an in-plot legend (any
        # corner) ends up sitting on top of a bar's own value label.
        legend_elements = [
            Patch(facecolor=color, alpha=0.75, label="Mean faithfulness"),
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.2,
                label="1 SD across 20 runs (clipped to [0, 1])",
            ),
        ]
        fig.legend(
            handles=legend_elements,
            loc="upper center",
            ncol=2,
            bbox_to_anchor=(0.5, 0.99),
        )

        plt.tight_layout()
        path = IMGS_DIR / "rq3_fig2_faithfulness_per_sample.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved: {path}")


def fig3_heatmap(rows: list[dict]):
    """Heatmap of mean score per sample per metric.

    Content and layout unchanged, only the shared academic style is applied
    for font and spine consistency with the other figures. Note for later:
    RdYlGn is not colorblind safe, worth a diverging colorblind safe map
    (e.g. PRGn or BrBG) if this ever leaves the appendix for the main text.
    """
    with plt.rc_context(ACADEMIC_STYLE):
        n_samples = max(r["sample_id"] for r in rows)
        heatmap = np.zeros((n_samples, len(METRICS)))
        for mi, m in enumerate(METRICS):
            mat = build_matrix(rows, m)
            for si, sid in enumerate(sorted(mat.keys())):
                vals = list(mat[sid].values())
                heatmap[si, mi] = _mean(vals) or 0.0

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(
            [METRIC_LABEL[m] for m in METRICS], fontsize=9, rotation=15, ha="right"
        )
        ax.set_yticks(range(n_samples))
        ax.set_yticklabels([f"Sample {i + 1}" for i in range(n_samples)], fontsize=8)
        for si in range(n_samples):
            for mi in range(len(METRICS)):
                val = heatmap[si, mi]
                ax.text(
                    mi,
                    si,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black" if 0.25 < val < 0.85 else "white",
                )
        plt.colorbar(im, ax=ax, label="Mean Score")
        ax.set_title("Mean Score per Sample per Metric", fontweight="bold")
        plt.tight_layout()
        path = IMGS_DIR / "rq3_fig3_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved: {path}")


def fig4_run_stability(rows: list[dict], n_runs: int):
    """Small multiples of run level mean score, one panel per metric.

    The original single shared 0 to 1 axis made context_recall and
    context_precision look almost perfectly flat next to faithfulness,
    whose real range is roughly ten times wider. Each panel here gets its
    own y-limits, scaled to that metric's actual data range with padding,
    the same principle already used for bar heights in
    analysis/rq4/rq4_experiment.py's plot_error. Each metric also gets a
    distinct marker shape in addition to its color, and yellow is excluded
    from the palette, so the figure still reads correctly for colorblind
    viewers or in grayscale print.
    """
    with plt.rc_context(ACADEMIC_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        run_ids = list(range(1, n_runs + 1))

        for ax, m in zip(axes.flat, METRICS):
            color = PALETTE_CB[m]
            run_means = [
                _mean([r[m] for r in rows if r["run_id"] == rid and r[m] is not None])
                or 0.0
                for rid in run_ids
            ]
            ax.plot(
                run_ids,
                run_means,
                marker=MARKERS[m],
                markersize=5,
                color=color,
                linewidth=1.5,
                markerfacecolor=color,
                markeredgecolor="#333333",
                markeredgewidth=0.5,
            )
            y_min, y_max = min(run_means), max(run_means)
            y_pad = max((y_max - y_min) * 0.2, 0.01)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlabel("Run")
            ax.set_ylabel("Mean score")
            ax.set_title(METRIC_LABEL[m], fontweight="bold")
            ax.set_xticks(run_ids[::2])
            ax.set_xticklabels(run_ids[::2])
            ax.grid(True, axis="y", alpha=0.5)
            ax.grid(False, axis="x")

        fig.suptitle(
            "Run Level Mean Score Stability, Independent Scale per Metric",
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        path = IMGS_DIR / "rq3_fig4_run_stability.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
def print_summary(rows: list[dict], icc_results: list[dict], stats: list[dict]):
    n_runs = max(r["run_id"] for r in rows)
    n_samples = max(r["sample_id"] for r in rows)
    print(f"\n{'=' * 60}")
    print("RQ3 Analysis Summary")
    print(f"  Runs: {n_runs}   Samples: {n_samples}   Total rows: {len(rows)}")
    print(f"{'=' * 60}")

    print("\nICC(2,1) per metric (Koo & Li 2016):")
    for res in icc_results:
        print(f"  {res['metric']:20s}  ICC={res['icc']:.6f}  ({res['interpretation']})")

    print("\nCV summary per metric (across samples):")
    for m in METRICS:
        cvs = [s["cv"] for s in stats if s["metric"] == m and s["cv"] is not None]
        if cvs:
            print(
                f"  {m:20s}  mean_CV={_mean(cvs):.4f}  "
                f"max_CV={max(cvs):.4f}  "
                f"n_CV>0.1={sum(1 for c in cvs if c > 0.1)}/{len(cvs)}"
            )
        else:
            print(f"  {m:20s}  CV=0 (all scores identical across runs)")

    print("\nCross-run variation check:")
    for m in METRICS:
        mat = build_matrix(rows, m)
        varying = sum(
            1 for sid in mat if max(mat[sid].values()) - min(mat[sid].values()) > 1e-9
        )
        print(f"  {m:20s}  samples with any variation = {varying}/{n_samples}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {RAW_CSV} ...")
    rows = load_raw(RAW_CSV)
    n_runs = max(r["run_id"] for r in rows)

    stats = compute_sample_stats(rows)
    icc_results = [compute_icc(rows, m) for m in METRICS]

    write_stats_csv(stats)
    write_icc_csv(icc_results)

    fig1a_primary_metrics(rows, n_runs)
    fig1b_secondary_metrics(rows, n_runs)
    fig2_faithfulness_per_sample(rows)
    fig3_heatmap(rows)
    fig4_run_stability(rows, n_runs)

    print_summary(rows, icc_results, stats)


if __name__ == "__main__":
    main()
