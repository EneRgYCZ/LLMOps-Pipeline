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
import numpy as np
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
        default_factory=lambda: os.environ.get("OLLAMA_MODEL", "ministral-8b-instruct")
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
        raise FileNotFoundError(
            f"No raw data at {RAW_CSV_PATH}. Run the experiment first with "
            "'python rq4_experiment.py run'."
        )
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


def plot_predicted_vs_measured(
    df: pd.DataFrame, summary: pd.DataFrame, model_spec: ModelSpec
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # smooth predicted curve across the observed range, not just the sampled points
    n_min, n_max = df["context_length"].min(), df["context_length"].max()
    n_smooth = np.linspace(n_min, n_max, 200)
    predicted_smooth = [model_spec.predicted_vram_gb(int(n)) for n in n_smooth]
    ax.plot(
        n_smooth,
        predicted_smooth,
        label="Predicted (Eq. eq:estimation)",
        color="#1f77b4",
        linewidth=2,
    )

    ax.errorbar(
        summary["context_length"],
        summary["measured_mean_gb"],
        yerr=summary["measured_std_gb"],
        fmt="o",
        color="#d62728",
        capsize=4,
        label="Measured (mean ± std across repeats)",
    )

    ax.set_xlabel("Context length N (tokens)")
    ax.set_ylabel("VRAM usage (GB)")
    ax.set_title("Predicted vs. measured VRAM usage across context length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIT_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Fit plot written to {FIT_PLOT_PATH}")


def plot_residuals(summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in summary["error_gb"]]
    ax.bar(summary["context_length"].astype(str), summary["error_gb"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Context length N (tokens)")
    ax.set_ylabel("Measured minus predicted (GB)")
    ax.set_title("Prediction residuals by context length")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESIDUAL_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Residual plot written to {RESIDUAL_PLOT_PATH}")


def run_analysis(model_spec: ModelSpec) -> None:
    df = load_raw_data()
    summary = build_summary(df)
    metrics = compute_overall_metrics(summary)
    save_summary(summary, metrics)
    plot_predicted_vs_measured(df, summary, model_spec)
    plot_residuals(summary)


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
        description="RQ4 VRAM validation: experiment and analysis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="Run the experiment, then analyse the results"
    )
    run_parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=DEFAULT_CONTEXT_LENGTHS
    )
    run_parser.add_argument("--repeats", type=int, default=5)
    run_parser.add_argument(
        "--gpu-index", type=int, default=int(os.environ.get("GPU_INDEX", 0))
    )
    run_parser.add_argument(
        "--backend",
        choices=["nvidia_smi", "prometheus"],
        default=os.environ.get("VRAM_BACKEND", "nvidia_smi"),
    )
    run_parser.add_argument(
        "--prometheus-url",
        default=os.environ.get("PROMETHEUS_URL", "http://localhost:9090"),
    )
    run_parser.add_argument(
        "--model", default=None, help="Overrides OLLAMA_MODEL env var"
    )
    run_parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    run_parser.add_argument(
        "--skip-analysis", action="store_true", help="Collect data only"
    )

    subparsers.add_parser(
        "analyze", help="Analyse an existing raw CSV without running the experiment"
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_spec = ModelSpec()

    if args.command == "run":
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

        if not args.skip_analysis:
            run_analysis(model_spec)

    elif args.command == "analyze":
        run_analysis(model_spec)


if __name__ == "__main__":
    main()
