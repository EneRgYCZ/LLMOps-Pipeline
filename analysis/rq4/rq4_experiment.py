"""
RQ4 VRAM validation: experiment and analysis in a single script.

Purpose
-------
Empirically validates the VRAM estimation formula (Equation eq:estimation in
sections/methodology.tex) by sweeping Ollama's context length (num_ctx),
measuring actual GPU memory usage, and comparing it against the formula's
prediction.

    M_total(N) = (P * b_w) + (0.55 + 0.08 * P) + (N * 2 * L * (d / g) * b_kv) * 1e-9

Design
------
Weights (P, b_w), layer count (L), hidden dimension (d) and the GQA grouping
factor (g) are fixed for a given model and quantisation, so the only free
term is the KV cache, which scales with context length N. Sweeping N traces
out the predicted line and lets measured VRAM be checked against both the
intercept (weights + framework overhead) and the slope (KV cache growth).

Layout
------
This script lives at analysis/rq4/rq4_experiment.py and writes to
results/rq4/{data,csvs,images}, resolved relative to this file so the
project can be checked out anywhere and still reproduce.

    results/rq4/data    raw per run measurements (rq4_vram_raw.csv)
    results/rq4/csvs    aggregated summary table (rq4_summary.csv)
    results/rq4/images  fit plot and residual plot

Measurement backend
--------------------
VRAMReader is an abstract interface with two implementations: NvidiaSmiReader
(default, used in this experiment) and a PrometheusReader stub. Nvidia smi is
used by default because it needs no extra infrastructure and gives an on
demand reading rather than one delayed by a scrape interval. The Prometheus
path is left in place so switching backends later is a one line change
(--backend prometheus) plus filling in the query in PrometheusReader.

Usage
-----
    python rq4_experiment.py                      # sweep + measure + analyse, one shot
    python rq4_experiment.py --repeats 3           # fewer repeats per N
    python rq4_experiment.py --model <ollama_tag>  # overrides OLLAMA_MODEL env var
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths (all relative to this file)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent  # analysis/rq4
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # repo root
RESULTS_DIR = PROJECT_ROOT / "results" / "rq4"
DATA_DIR = RESULTS_DIR / "data"
CSV_DIR = RESULTS_DIR / "csvs"
IMAGES_DIR = RESULTS_DIR / "images"

RAW_CSV_PATH = DATA_DIR / "rq4_vram_raw.csv"
SUMMARY_CSV_PATH = CSV_DIR / "rq4_summary.csv"
FIT_PLOT_PATH = IMAGES_DIR / "rq4_predicted_vs_measured.png"
RESIDUAL_PLOT_PATH = IMAGES_DIR / "rq4_residuals.png"

for directory in (DATA_DIR, CSV_DIR, IMAGES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Formula constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Architectural and quantisation parameters for the estimation formula.

    Defaults correspond to Ministral 8B Instruct under Q4_K_M, as used
    throughout the thesis (see Table tab:quantisation-formats and the Week 1
    log entry for the architectural values). b_kv assumes Ollama's default
    fp16 K/V cache; adjust here if a quantised KV cache is used instead.
    """

    P: float = 8.0e9
    b_w: float = 0.56
    L: int = 34
    d: int = 4096
    g: int = 4
    b_kv: float = 2.0

    def predicted_vram_gb(self, context_length: int) -> float:
        weights_gb = self.P * self.b_w * 1e-9
        overhead_gb = 0.55 + 0.08 * (self.P * 1e-9)
        kv_bytes = context_length * 2 * self.L * (self.d / self.g) * self.b_kv
        kv_gb = kv_bytes * 1e-9
        return weights_gb + overhead_gb + kv_gb


# ---------------------------------------------------------------------------
# VRAM reader interface
# ---------------------------------------------------------------------------


class VRAMReader(ABC):
    """Common interface for GPU memory measurement backends.

    Any backend returns memory used, in MiB, for a single GPU index. This is
    what lets NvidiaSmiReader and a future PrometheusReader be swapped
    without changing the experiment loop.
    """

    @abstractmethod
    def read_used_mib(self, gpu_index: int) -> float:
        raise NotImplementedError


class NvidiaSmiReader(VRAMReader):
    """Reads instantaneous GPU memory usage via nvidia smi. Default backend."""

    def read_used_mib(self, gpu_index: int) -> float:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())


class PrometheusReader(VRAMReader):
    """Reads GPU memory usage from Prometheus, backed by the DCGM exporter.

    Not implemented yet. Kept here so switching the measurement backend
    later only requires filling in the query below and passing
    --backend prometheus; no other part of the script needs to change. The
    query should be an instant query against DCGM_FI_DEV_FB_USED filtered by
    GPU index, for example:

        DCGM_FI_DEV_FB_USED{gpu="<gpu_index>"}

    against http://<prometheus_host>:9090/api/v1/query. A short polling loop
    across the request window, rather than a single instant query, would
    also let this backend report the prefill memory peak, which nvidia smi's
    single post load reading misses.
    """

    def __init__(self, prometheus_url: str = "http://localhost:9090") -> None:
        self.prometheus_url = prometheus_url

    def read_used_mib(self, gpu_index: int) -> float:
        raise NotImplementedError(
            "PrometheusReader is a placeholder. Implement the DCGM query "
            "above before using --backend prometheus."
        )


def get_reader(
    backend: str, prometheus_url: str = "http://localhost:9090"
) -> VRAMReader:
    if backend == "nvidia_smi":
        return NvidiaSmiReader()
    if backend == "prometheus":
        return PrometheusReader(prometheus_url=prometheus_url)
    raise ValueError(f"Unknown VRAM backend: {backend}")


# ---------------------------------------------------------------------------
# Ollama control helpers
# ---------------------------------------------------------------------------


@dataclass
class OllamaConfig:
    host: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    container: str = field(
        default_factory=lambda: os.environ.get("OLLAMA_CONTAINER", "ollama")
    )
    model: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M"
        )
    )


def stop_model(config: OllamaConfig) -> None:
    """Unloads the model and clears Ollama's prompt cache.

    Same fix used in the RQ3 experiment: without this, a request can reuse
    a cached KV state from a previous context length, biasing the memory
    reading toward whichever context length loaded first.
    """
    subprocess.run(
        ["docker", "exec", config.container, "ollama", "stop", config.model],
        capture_output=True,
        text=True,
    )
    time.sleep(2)


def load_model_with_context(
    config: OllamaConfig, context_length: int, prompt: str
) -> None:
    """Forces a model load at a given context length via one generate call."""
    response = requests.post(
        f"{config.host}/api/generate",
        json={
            "model": config.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": context_length},
        },
        timeout=300,
    )
    response.raise_for_status()


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

RAW_CSV_FIELDS = [
    "timestamp_utc",
    "context_length",
    "repeat_index",
    "measured_vram_mib",
    "measured_vram_gb",
    "baseline_vram_mib",
    "delta_vram_gb",
    "predicted_vram_gb",
    "gpu_index",
]


def append_row(row: dict) -> None:
    write_header = not RAW_CSV_PATH.exists()
    with open(RAW_CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_experiment(
    context_lengths: list[int],
    repeats: int,
    gpu_index: int,
    ollama_config: OllamaConfig,
    reader: VRAMReader,
    model_spec: ModelSpec,
    prompt: str,
) -> None:
    print(f"Baseline read on GPU {gpu_index} before any load")
    baseline_mib = reader.read_used_mib(gpu_index)
    print(f"Baseline VRAM used: {baseline_mib:.1f} MiB")

    total_runs = len(context_lengths) * repeats
    run_count = 0

    for context_length in context_lengths:
        predicted_gb = model_spec.predicted_vram_gb(context_length)
        for repeat_index in range(repeats):
            run_count += 1
            print(
                f"[{run_count}/{total_runs}] N={context_length} repeat={repeat_index + 1}/{repeats}"
            )

            stop_model(ollama_config)
            load_model_with_context(ollama_config, context_length, prompt)
            time.sleep(1)

            measured_mib = reader.read_used_mib(gpu_index)
            measured_gb = measured_mib / 1024.0
            delta_gb = (measured_mib - baseline_mib) / 1024.0

            append_row(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "context_length": context_length,
                    "repeat_index": repeat_index,
                    "measured_vram_mib": measured_mib,
                    "measured_vram_gb": round(measured_gb, 4),
                    "baseline_vram_mib": baseline_mib,
                    "delta_vram_gb": round(delta_gb, 4),
                    "predicted_vram_gb": round(predicted_gb, 4),
                    "gpu_index": gpu_index,
                }
            )

    print(f"Raw results written to {RAW_CSV_PATH}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_raw_data() -> pd.DataFrame:
    if not RAW_CSV_PATH.exists():
        raise FileNotFoundError(f"No raw data at {RAW_CSV_PATH}.")
    return pd.read_csv(RAW_CSV_PATH)


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates repeats per context length and computes error metrics.

    measured_vram_gb is used directly (not the baseline delta) since the
    formula predicts total device memory for the model process, and on a
    dedicated GPU the two should coincide; delta_vram_gb remains in the raw
    CSV in case a shared GPU needs that correction instead.
    """
    grouped = df.groupby("context_length")["measured_vram_gb"].agg(
        ["mean", "std", "count"]
    )
    grouped = grouped.rename(
        columns={
            "mean": "measured_mean_gb",
            "std": "measured_std_gb",
            "count": "n_repeats",
        }
    )
    grouped["measured_std_gb"] = grouped["measured_std_gb"].fillna(0.0)

    predicted = df.groupby("context_length")["predicted_vram_gb"].first()
    grouped["predicted_gb"] = predicted

    grouped["error_gb"] = grouped["measured_mean_gb"] - grouped["predicted_gb"]
    grouped["abs_error_gb"] = grouped["error_gb"].abs()
    grouped["pct_error"] = 100.0 * grouped["error_gb"] / grouped["predicted_gb"]

    return grouped.reset_index().sort_values("context_length")


def compute_overall_metrics(summary: pd.DataFrame) -> dict:
    mae = summary["abs_error_gb"].mean()
    mape = summary["pct_error"].abs().mean()
    ss_res = (summary["error_gb"] ** 2).sum()
    ss_tot = (
        (summary["measured_mean_gb"] - summary["measured_mean_gb"].mean()) ** 2
    ).sum()
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"mae_gb": mae, "mape_pct": mape, "r_squared": r_squared}


def save_summary(summary: pd.DataFrame, metrics: dict) -> None:
    summary_out = summary.copy()
    for key, value in metrics.items():
        summary_out[key] = value
    summary_out.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"Summary written to {SUMMARY_CSV_PATH}")
    print(
        f"Overall fit: MAE={metrics['mae_gb']:.3f} GB, "
        f"MAPE={metrics['mape_pct']:.2f}%, R^2={metrics['r_squared']:.4f}"
    )


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

COLOR_EXPECTED = "#1f4e79"
COLOR_MEASURED = "#c0392b"
COLOR_ERROR_POS = "#c0392b"
COLOR_ERROR_NEG = "#2e7d5b"


def plot_fit(summary: pd.DataFrame, model_spec: ModelSpec, model_name: str) -> None:
    """Expected value curve vs. measured points, both connected by lines.

    The expected curve is drawn with markers at the exact sampled context
    lengths (not just a smooth line) so it lines up visually against the
    measured series at each N, which is what makes the comparison readable
    at a glance rather than requiring the reader to cross reference values.
    """
    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))

        context_lengths = summary["context_length"]
        expected = [model_spec.predicted_vram_gb(int(n)) for n in context_lengths]

        ax.plot(
            context_lengths,
            expected,
            marker="s",
            markersize=6,
            linewidth=2,
            color=COLOR_EXPECTED,
            label="Expected value (Eq. eq:estimation)",
        )
        ax.errorbar(
            context_lengths,
            summary["measured_mean_gb"],
            yerr=summary["measured_std_gb"],
            marker="o",
            markersize=6,
            linewidth=2,
            color=COLOR_MEASURED,
            ecolor=COLOR_MEASURED,
            capsize=3,
            elinewidth=1,
            label=model_name,
        )

        ax.set_xscale("log")
        ax.set_xticks(context_lengths)
        ax.set_xticklabels(context_lengths.astype(int).astype(str))
        ax.set_xlabel("Context length N (tokens, log scale)")
        ax.set_ylabel("VRAM usage (GB)")
        ax.set_title("Expected vs. measured VRAM usage across context length")
        ax.legend(loc="upper left")
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        fig.tight_layout()
        fig.savefig(FIT_PLOT_PATH, dpi=200)
        plt.close(fig)
    print(f"Fit plot written to {FIT_PLOT_PATH}")


def plot_residuals(summary: pd.DataFrame, model_name: str) -> None:
    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(8, 4))

        colors = [
            COLOR_ERROR_POS if v >= 0 else COLOR_ERROR_NEG for v in summary["pct_error"]
        ]
        ax.bar(
            summary["context_length"].astype(str),
            summary["pct_error"],
            color=colors,
            alpha=0.85,
        )
        for i, pct in enumerate(summary["pct_error"]):
            ax.annotate(
                f"{pct:+.1f}%",
                xy=(i, pct),
                xytext=(0, 4 if pct >= 0 else -12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#333333",
            )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xlabel("Context length N (tokens)")
        ax.set_ylabel("Error (%)")
        ax.set_title(f"{model_name}: prediction error by context length")
        ax.grid(True, axis="y", alpha=0.6)

        fig.tight_layout()
        fig.savefig(RESIDUAL_PLOT_PATH, dpi=200)
        plt.close(fig)
    print(f"Residual plot written to {RESIDUAL_PLOT_PATH}")


def run_analysis(model_spec: ModelSpec, model_name: str) -> None:
    df = load_raw_data()
    summary = build_summary(df)
    metrics = compute_overall_metrics(summary)
    save_summary(summary, metrics)
    plot_fit(summary, model_spec, model_name)
    plot_residuals(summary, model_name)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFAULT_PROMPT = (
    "Summarise, in a few sentences, the main considerations involved in "
    "deploying a large language model in production."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RQ4 VRAM validation: sweep, measure, analyse, one shot"
    )
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--gpu-index", type=int, default=int(os.environ.get("GPU_INDEX", 0))
    )
    parser.add_argument(
        "--backend",
        choices=["nvidia_smi", "prometheus"],
        default=os.environ.get("VRAM_BACKEND", "nvidia_smi"),
    )
    parser.add_argument(
        "--prometheus-url",
        default=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
    )
    parser.add_argument("--model", default=None, help="Overrides OLLAMA_MODEL env var")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = ModelSpec()

    ollama_config = OllamaConfig()
    if args.model:
        ollama_config.model = args.model

    reader = get_reader(args.backend, prometheus_url=args.prometheus_url)

    run_experiment(
        context_lengths=args.context_lengths,
        repeats=args.repeats,
        gpu_index=args.gpu_index,
        ollama_config=ollama_config,
        reader=reader,
        model_spec=model_spec,
        prompt=args.prompt,
    )
    run_analysis(model_spec, ollama_config.model)


if __name__ == "__main__":
    main()
