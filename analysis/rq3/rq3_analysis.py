"""
RQ3 Analysis: RAGAS Metric Consistency
=======================================
Reads results/rq3_raw.csv produced by rq3_experiment.py and outputs:
  - results/rq3_stats.csv        — per-sample per-metric descriptive statistics
  - results/rq3_icc.csv          — ICC(2,1) per metric with interpretation
  - results/rq3_fig1_score_distributions.png
  - results/rq3_fig2_faithfulness_per_sample.png
  - results/rq3_fig3_heatmap.png
  - results/rq3_fig4_run_stability.png

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
PALETTE = ["#2C7BB6", "#D7191C", "#1A9641", "#FDAE61"]


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
def fig1_score_distributions(rows: list[dict], n_runs: int):
    """Boxplot of score distribution per metric per run."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    for ax, m, col in zip(axes, METRICS, PALETTE):
        vals_per_run = [
            [r[m] for r in rows if r["run_id"] == rid and r[m] is not None]
            for rid in range(1, n_runs + 1)
        ]
        bp = ax.boxplot(
            vals_per_run,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(color="#555"),
            capprops=dict(color="#555"),
            flierprops=dict(marker="o", markersize=3, alpha=0.4, color=col),
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        ax.set_title(METRIC_LABEL[m], fontsize=11, fontweight="bold")
        ax.set_xlabel("Run", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        step = max(1, n_runs // 5)
        ticks = list(range(1, n_runs + 1, step))
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=7)

    fig.suptitle(
        f"Score Distribution per Run (amnesty_qa, N=20 samples × {n_runs} runs)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    plt.tight_layout()
    path = IMGS_DIR / "rq3_fig1_score_distributions.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig2_faithfulness_per_sample(rows: list[dict]):
    """Bar chart of per-sample faithfulness mean ± std."""
    mat = build_matrix(rows, "faithfulness")
    sids = sorted(mat.keys())
    means = []
    stds = []
    for sid in sids:
        vals = list(mat[sid].values())
        means.append(_mean(vals) or 0.0)
        stds.append(_std(vals) or 0.0)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(sids))
    ax.bar(
        x,
        means,
        yerr=stds,
        capsize=4,
        color=PALETTE[0],
        alpha=0.7,
        ecolor="#333",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in sids], fontsize=8)
    ax.set_xlabel("Sample", fontsize=10)
    ax.set_ylabel("Faithfulness Score", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, label="0.5 threshold")
    ax.set_title(
        "Per-Sample Faithfulness: Mean ± Std across Runs",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = IMGS_DIR / "rq3_fig2_faithfulness_per_sample.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig3_heatmap(rows: list[dict]):
    """Heatmap of mean score per sample per metric."""
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
    ax.set_title("Mean Score per Sample per Metric", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = IMGS_DIR / "rq3_fig3_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig4_run_stability(rows: list[dict], n_runs: int):
    """Line plot of run-level mean score per metric across all runs."""
    fig, ax = plt.subplots(figsize=(10, 4))
    run_ids = list(range(1, n_runs + 1))
    for m, col in zip(METRICS, PALETTE):
        run_means = [
            _mean([r[m] for r in rows if r["run_id"] == rid and r[m] is not None])
            or 0.0
            for rid in run_ids
        ]
        ax.plot(
            run_ids,
            run_means,
            marker="o",
            markersize=4,
            label=METRIC_LABEL[m],
            color=col,
            linewidth=1.5,
        )
    ax.set_xlabel("Run", fontsize=10)
    ax.set_ylabel("Mean Score", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(run_ids)
    ax.set_xticklabels(run_ids, fontsize=8)
    ax.set_title(
        "Run-Level Mean Score Stability across Runs", fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
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

    fig1_score_distributions(rows, n_runs)
    fig2_faithfulness_per_sample(rows)
    fig3_heatmap(rows)
    fig4_run_stability(rows, n_runs)

    print_summary(rows, icc_results, stats)


if __name__ == "__main__":
    main()
