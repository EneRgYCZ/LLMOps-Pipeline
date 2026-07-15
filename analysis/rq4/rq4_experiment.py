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
import re
import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Model registry
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


# All five 8B-class models with their architectural parameters.
# b_kv=2.0 (fp16 KV cache, Ollama default) and b_w=0.56 (Q4_K_M) for all.
MODEL_REGISTRY: dict[str, ModelSpec] = {
    "ministral-3:8b-instruct-2512-q4_K_M": ModelSpec(
        P=8.0e9, b_w=0.56, L=34, d=4096, g=4, b_kv=2.0
    ),
    "llama3.1:8b": ModelSpec(P=8.03e9, b_w=0.56, L=32, d=4096, g=4, b_kv=2.0),
    "qwen3:8b": ModelSpec(P=8.2e9, b_w=0.56, L=36, d=4096, g=4, b_kv=2.0),
    "qwen2.5:7b": ModelSpec(P=7.61e9, b_w=0.56, L=28, d=3584, g=7, b_kv=2.0),
    "mistral-7b-instruct-q4_K_M": ModelSpec(
        P=7.24e9, b_w=0.56, L=32, d=4096, g=4, b_kv=2.0
    ),
}


def sanitize_tag(tag: str) -> str:
    """Sanitise an Ollama tag for use as a directory name.

    Replaces ':' with '_' since colons are not safe on all filesystems.
    """
    return tag.replace(":", "_")


# The default model tag for backwards compatibility
DEFAULT_MODEL_TAG = "ministral-3:8b-instruct-2512-q4_K_M"

# All five model tags for the "all" option
ALL_MODEL_TAGS = list(MODEL_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Wikipedia-based prompt generation
# ---------------------------------------------------------------------------

# Cache for Wikipedia text
_WIKI_TEXT_CACHE: str | None = None

COMPLETION_PROMPT = "Continue the sentence:"


def _count_tokens_simple(text: str) -> int:
    """Simple token counter that approximates LLM tokenization.
    
    Splits on whitespace and common punctuation. For English text, this is
    typically within 10-20% of actual LLM token counts, which is sufficient
    for generating prompts of approximately the target length.
    """
    return len(re.findall(r"\w+|[^\w\s]", text))


def _get_wikipedia_text() -> str:
    """Fetch and cache Wikipedia text for prompt generation.
    
    Tries multiple sources in order:
    1. Hugging Face datasets (wikimedia/wikipedia) - tries English first, then simple
    2. Wikipedia API (multiple large articles concatenated)
    3. Hardcoded fallback text (sufficient for ~10k tokens before repetition)
    
    The text is cached after first fetch to avoid repeated downloads.
    Note: For very long context lengths (>text length), the text is repeated.
    """
    global _WIKI_TEXT_CACHE
    if _WIKI_TEXT_CACHE is not None:
        return _WIKI_TEXT_CACHE

    # Try Hugging Face datasets - English Wikipedia first
    try:
        from datasets import load_dataset
        # Stream instead of downloading the full split - we only need one article
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        _WIKI_TEXT_CACHE = next(iter(dataset))["text"]
        return _WIKI_TEXT_CACHE
    except ImportError:
        pass  # datasets library not installed
    except Exception:
        # Try Simple English Wikipedia as fallback
        try:
            dataset = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
            _WIKI_TEXT_CACHE = next(iter(dataset))["text"]
            return _WIKI_TEXT_CACHE
        except Exception:
            pass  # datasets loading failed

    # Fallback: Wikipedia API - fetch multiple articles
    try:
        articles = ["Artificial_intelligence", "Machine_learning", "Deep_learning", 
                    "Natural_language_processing", "Neural_network"]
        all_texts = []
        for title in articles:
            response = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "format": "json",
                    "prop": "extracts",
                    "explaintext": True,
                    "titles": title,
                },
                timeout=30,
            )
            data = response.json()
            pages = data.get("query", {}).get("pages", {})
            if pages:
                page = next(iter(pages.values()))
                extract = page.get("extract", "")
                if extract:
                    all_texts.append(extract)
        if all_texts:
            _WIKI_TEXT_CACHE = " ".join(all_texts)
            return _WIKI_TEXT_CACHE
    except Exception:
        pass

    # Final fallback: hardcoded text (~2000+ tokens when concatenated)
    # This is long enough that repetition only kicks in for very large N
    _WIKI_TEXT_CACHE = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to the natural intelligence displayed by humans and other animals. "
        "Leading AI textbooks define the field as the study of intelligent agents: "
        "any system that perceives its environment and takes actions that maximize "
        "its chance of successfully achieving its goals. Colloquially, the term "
        "artificial intelligence is often used to describe machines that mimic "
        "cognitive functions that humans associate with the human mind, such as "
        "learning and problem solving. The scope of AI is disputed: as machines "
        "become increasingly capable, tasks considered to require intelligence are "
        "often removed from the definition of AI, a phenomenon known as the AI effect. "
        "For instance, optical character recognition is frequently excluded from "
        "things considered to be AI, having become a routine technology. AI was founded "
        "as an academic discipline in 1956, and in the years since has experienced "
        "several waves of optimism, followed by disappointment and the loss of "
        "funding, known as AI winter, followed by new approaches, success and "
        "renewed funding. For most of its history, AI research has been divided into "
        "subfields that often fail to communicate with each other. These sub-fields "
        "are based on technical considerations, such as particular goals, the use "
        "of particular tools, or deep philosophical differences. Subsymbolic AI "
        "includes evolutionary computation and swarm intelligence algorithms. "
        "Deep learning models such as recurrent neural networks and large language "
        "models have been particularly successful in solving problems that require "
        "understanding of natural language, speech recognition, and computer vision. "
        "However, these models are often criticized for their lack of explainability "
        "and potential for bias. The discipline was founded on the claim that "
        "a central property of humans, intelligence, can be so precisely described "
        "that a machine can be made to simulate it. This raises philosophical "
        "arguments about the mind and the ethics of creating artificial beings "
        "endowed with human-like intelligence, issues which have been explored "
        "by myth, fiction and philosophy since antiquity. "
        "Machine learning is the study of computer algorithms that improve automatically "
        "through experience and by the use of data. It is seen as a subset of artificial "
        "intelligence. Machine learning algorithms build a mathematical model based on "
        "sample data, known as training data, in order to make predictions or decisions "
        "without being explicitly programmed to perform the task. Machine learning is "
        "closely related to computational statistics, which focuses on making predictions "
        "using computers. The study of mathematical optimization delivers methods, theory "
        "and application domains to the field of machine learning. Data mining is a "
        "related field of study, focusing on exploratory data analysis through "
        "unsupervised learning. In its application across business problems, machine "
        "learning is also referred to as predictive analytics. The goal of machine "
        "learning is to understand the structure of data and fit that data into models "
        "that can be understood and utilized by humans. Machine learning is the science "
        "of getting computers to learn and act like humans do, and improve their learning "
        "over time in autonomous fashion, by feeding them data and information in the "
        "form of observations and real-world interactions. The primary objective is to "
        "allow the computers learn automatically without human assistance or intervention "
        "and adjust actions accordingly. Neural networks are a set of algorithms, modeled "
        "loosely after the human brain, that are designed to recognize patterns. They "
        "interpret sensory data through machine perception, labeling or clustering raw "
        "input. The patterns they recognize are numerical, contained in vectors, into "
        "which all real-world data, be it images, sound, text or time series, must be "
        "translated. Neural networks are particularly useful for solving problems that "
        "are difficult to solve with traditional rule-based programming. They have "
        "achieved remarkable success in recent years, particularly in the fields of "
        "computer vision, natural language processing, and game playing. The most "
        "common type of neural network used today is the feedforward neural network, "
        "also known as a multilayer perceptron, which consists of multiple layers of "
        "nodes, or neurons, with connections between them. Each connection has a "
        "weight that determines the strength of the signal passed from one neuron "
        "to another. During training, these weights are adjusted to minimize the "
        "difference between the predicted output and the actual output. "
    )
    return _WIKI_TEXT_CACHE


def generate_prompt(context_length: int) -> str:
    """Generate a prompt of approximately context_length tokens.
    
    For context_length <= text_length: takes a prefix from Wikipedia text and 
    appends 'Continue the sentence:' so the total is approximately context_length.
    
    For context_length > text_length: uses context_length tokens from Wikipedia 
    text (repeating if needed) WITHOUT the completion prompt, since adding it
    after repeated text wouldn't be semantically meaningful.
    
    Uses a simple tokenizer that approximates LLM tokenization. The actual
    token count may vary slightly from the target.
    """
    wiki_text = _get_wikipedia_text()
    text_token_count = _count_tokens_simple(wiki_text)
    completion_token_count = _count_tokens_simple(COMPLETION_PROMPT)

    if context_length <= completion_token_count:
        # Context too short for completion prompt, truncate it
        return COMPLETION_PROMPT[:context_length]

    # Calculate how many tokens we need from Wikipedia text
    target_prefix_tokens = context_length - completion_token_count

    # If the text is long enough to accommodate prefix + completion
    if target_prefix_tokens <= text_token_count:
        # Build prefix + completion prompt
        all_words = wiki_text.split()
        prefix = ""
        for word in all_words:
            test_prefix = prefix + (" " + word if prefix else word)
            new_token_count = _count_tokens_simple(test_prefix)
            if new_token_count > target_prefix_tokens:
                break
            prefix = test_prefix
        return prefix + (" " if prefix else "") + COMPLETION_PROMPT
    else:
        # For very long contexts, just return N tokens from Wikipedia text
        # (repeating if needed) - no completion prompt since it wouldn't make sense
        return _generate_n_tokens(wiki_text, context_length)


def _generate_n_tokens(text: str, n: int) -> str:
    """Generate exactly n tokens from text, repeating if needed."""
    all_words = text.split()
    tokens = []
    while _count_tokens_simple(" ".join(tokens)) < n:
        tokens.extend(all_words)
    # Trim to exactly n tokens (approximately)
    result = ""
    for word in tokens:
        test = result + (" " + word if result else word)
        if _count_tokens_simple(test) > n:
            break
        result = test
    return result


# ---------------------------------------------------------------------------
# Paths (all relative to this file)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent  # analysis/rq4
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # repo root

# Run standalone from a local venv, not through docker-compose's env_file, so
# load .env explicitly before any of the os.environ.get(...) calls below.
load_dotenv(PROJECT_ROOT / ".env")

RESULTS_DIR = PROJECT_ROOT / "results" / "rq4"
DATA_DIR = RESULTS_DIR / "data"
CSV_DIR = RESULTS_DIR / "csvs"
IMAGES_DIR = RESULTS_DIR / "images"

# Cross-model comparison output
CROSS_MODEL_CSV_PATH = CSV_DIR / "rq4_cross_model_summary.csv"

for directory in (DATA_DIR, CSV_DIR, IMAGES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


def get_model_paths(model_tag: str) -> tuple[Path, Path, Path, Path, Path]:
    """Get the per-model output paths for data, summary CSV, LaTeX, and plots."""
    sanitized = sanitize_tag(model_tag)
    data_path = DATA_DIR / sanitized / "rq4_vram_raw.csv"
    summary_csv_path = CSV_DIR / sanitized / "rq4_summary.csv"
    summary_tex_path = CSV_DIR / sanitized / "rq4_summary_table.tex"
    fit_plot_path = IMAGES_DIR / sanitized / "rq4_predicted_vs_measured.png"
    error_plot_path = IMAGES_DIR / sanitized / "rq4_errors.png"
    return data_path, summary_csv_path, summary_tex_path, fit_plot_path, error_plot_path


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


def is_model_pulled(model_tag: str, container: str = "ollama") -> bool:
    """Check if a model is already pulled in the Ollama container."""
    try:
        result = subprocess.run(
            ["docker", "exec", container, "ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Check if the exact tag appears in the output
        return model_tag in result.stdout
    except subprocess.CalledProcessError:
        return False


def pull_model(model_tag: str, container: str = "ollama") -> None:
    """Pull a model if not already present."""
    if is_model_pulled(model_tag, container):
        print(f"Model {model_tag} already pulled, skipping.")
    else:
        print(f"Pulling {model_tag}...")
        subprocess.run(
            ["docker", "exec", container, "ollama", "pull", model_tag],
            check=True,
        )
        print(f"Successfully pulled {model_tag}")


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


def append_row(row: dict, data_path: Path) -> None:
    write_header = not data_path.exists()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(data_path, "a", newline="") as f:
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
    prompt: str | None,
    data_path: Path,
) -> None:
    if data_path.exists():
        data_path.unlink()
        print(f"Removed existing {data_path} to start a fresh run.")

    # Pre-fetch Wikipedia text once before the loop
    if prompt is None:
        wiki_text = _get_wikipedia_text()
        print(f"Using Wikipedia-based prompts (first {len(wiki_text.split())} words)")

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
                f"[{run_count}/{total_runs}] N={context_length} "
                f"repeat={repeat_index + 1}/{repeats} "
                f"(predicted {predicted_gb:.2f} GB)"
            )

            stop_model(ollama_config)
            
            # Generate prompt for this context length
            if prompt is None:
                # Use Wikipedia-based prompt
                actual_prompt = generate_prompt(context_length)
            else:
                # Use the provided fixed prompt (for backward compatibility)
                actual_prompt = prompt
            
            load_model_with_context(ollama_config, context_length, actual_prompt)
            time.sleep(1)

            measured_mib = reader.read_used_mib(gpu_index)
            measured_gb = measured_mib / 1024.0
            delta_gb = (measured_mib - baseline_mib) / 1024.0

            print(f"  measured {measured_gb:.2f} GB")

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
                },
                data_path,
            )

    print(f"Raw results written to {data_path}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def load_raw_data(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"No raw data at {data_path}.")
    return pd.read_csv(data_path)


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


def save_summary(summary: pd.DataFrame, metrics: dict, summary_csv_path: Path) -> None:
    summary_out = summary.copy()
    for key, value in metrics.items():
        summary_out[key] = value
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_out.to_csv(summary_csv_path, index=False)
    print(f"Summary written to {summary_csv_path}")
    print(
        f"Overall fit: MAE={metrics['mae_gb']:.3f} GB, "
        f"MAPE={metrics['mape_pct']:.2f}%, R^2={metrics['r_squared']:.4f}"
    )


# ---------------------------------------------------------------------------
# Regime split analysis
# ---------------------------------------------------------------------------

# The thesis text describes the large-N regime as accurate at "under 2%"
# (observed 1.74% at N=114688), so 2% is the threshold used to separate the
# overestimation regime from the accurate regime, rather than an arbitrary
# round number.
REGIME_THRESHOLD_PCT = 2.0


def _regime_metrics(subset: pd.DataFrame) -> dict:
    if subset.empty:
        return {"mae_gb": float("nan"), "mape_pct": float("nan"), "n_points": 0}
    return {
        "mae_gb": subset["abs_error_gb"].mean(),
        "mape_pct": subset["pct_error"].abs().mean(),
        "n_points": len(subset),
    }


def _fit_log_trend(subset: pd.DataFrame) -> tuple[float, float]:
    """Least-squares slope/intercept of pct_error against log2(N).

    A plain linear fit, not a changepoint model: one is fit on the
    overestimation regime alone (where error is shrinking as N grows) and
    one on the full range, so the thesis can quote how much steeper the
    error trend is close to the breakpoint than over the whole sweep.
    """
    if len(subset) < 2:
        return float("nan"), float("nan")
    log_n = np.log2(subset["context_length"].astype(float))
    slope, intercept = np.polyfit(log_n, subset["pct_error"], 1)
    return float(slope), float(intercept)


def compute_regime_split(
    summary: pd.DataFrame, threshold_pct: float = REGIME_THRESHOLD_PCT
) -> dict:
    """Splits the sweep into an overestimation regime and an accurate regime.

    The thesis claim is that |pct_error| shrinks monotonically as N grows,
    crossing from "clearly overestimating" to "accurate" (below
    threshold_pct) somewhere in the sweep. Rather than a general changepoint
    search (out of scope here), this finds the last N still above
    threshold_pct and the first N at or below it, then linearly interpolates
    the crossing point in log2(N) space to get a single breakpoint value. If
    every point already lies on one side of the threshold (e.g. a
    perfect-fit sanity check), the breakpoint falls back to the nearest edge
    of the sweep so the two regimes are still well defined.
    """
    s = summary.sort_values("context_length").reset_index(drop=True)
    abs_err = s["pct_error"].abs()
    above = abs_err > threshold_pct

    above_idx = s.index[above]
    if len(above_idx) == 0:
        breakpoint_n = float(s["context_length"].min())
    else:
        last_above = above_idx.max()
        below_after = s.index[(s.index > last_above) & (~above)]
        if len(below_after) == 0:
            breakpoint_n = float(s["context_length"].max())
        else:
            first_below = below_after.min()
            n_above = float(s.loc[last_above, "context_length"])
            n_below = float(s.loc[first_below, "context_length"])
            err_above = float(abs_err.loc[last_above])
            err_below = float(abs_err.loc[first_below])
            log_above, log_below = np.log2(n_above), np.log2(n_below)
            if err_above == err_below:
                frac = 0.5
            else:
                frac = (err_above - threshold_pct) / (err_above - err_below)
            frac = min(max(frac, 0.0), 1.0)
            breakpoint_n = float(2 ** (log_above + frac * (log_below - log_above)))

    regime_overestimate = s[s["context_length"] <= breakpoint_n]
    regime_accurate = s[s["context_length"] > breakpoint_n]

    return {
        "threshold_pct": threshold_pct,
        "breakpoint_n": breakpoint_n,
        "overestimate": _regime_metrics(regime_overestimate),
        "accurate": _regime_metrics(regime_accurate),
        "trend_overestimate": _fit_log_trend(regime_overestimate),
        "trend_full": _fit_log_trend(s),
    }


def print_regime_split(regime: dict) -> None:
    o, a = regime["overestimate"], regime["accurate"]
    slope_o, _ = regime["trend_overestimate"]
    slope_f, _ = regime["trend_full"]
    print(
        f"Regime breakpoint: N ~= {regime['breakpoint_n']:,.0f} tokens "
        f"(|pct_error| crosses {regime['threshold_pct']:.1f}%)"
    )
    print(
        f"  Overestimation regime (N <= breakpoint, n={o['n_points']}): "
        f"MAE={o['mae_gb']:.3f} GB, MAPE={o['mape_pct']:.2f}%"
    )
    print(
        f"  Accurate regime (N > breakpoint, n={a['n_points']}): "
        f"MAE={a['mae_gb']:.3f} GB, MAPE={a['mape_pct']:.2f}%"
    )
    print(
        f"  log2(N) trend in pct_error: overestimation-regime slope="
        f"{slope_o:.3f} pp/doubling, full-range slope={slope_f:.3f} pp/doubling"
    )


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------

_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
}


def _escape_latex(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(ch, ch) for ch in text)


def write_latex_table(
    summary: pd.DataFrame,
    model_name: str,
    path: Path,
) -> None:
    """Writes the summary as a booktabs-style .tex fragment for \\input{}."""
    s = summary.sort_values("context_length")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{rrrr}",
        r"\toprule",
        r"$N$ (tokens) & Predicted (GB) & Measured (GB) & Error (\)) \\",
        r"\midrule",
    ]
    for _, row in s.iterrows():
        n = f"{int(row['context_length']):,}"
        predicted = f"{row['predicted_gb']:.2f}"
        measured = f"{row['measured_mean_gb']:.2f} \\pm {row['measured_std_gb']:.2f}"
        error = f"{row['pct_error']:+.2f}"
        lines.append(f"{n} & {predicted} & {measured} & {error} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    caption = _escape_latex(
        f"Predicted vs. measured VRAM usage across context length for {model_name}."
    )
    lines.append(rf"\caption{{{caption}}}")
    lines.append(r"\label{tab:rq4-vram-summary}")
    lines.append(r"\end{table}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX summary table written to {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


def plot_fit(
    summary: pd.DataFrame,
    model_spec: ModelSpec,
    model_name: str,
    fit_plot_path: Path,
) -> None:
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
        ax.set_xticklabels(
            [
                f"{int(n) // 1024}k" if n >= 1024 else str(int(n))
                for n in context_lengths
            ],
            rotation=0,
            ha="center",
        )
        ax.set_xlabel("Context length N (tokens, log scale)")
        ax.set_ylabel("VRAM usage (GB)")
        ax.set_title("Expected vs. measured VRAM usage across context length")
        ax.legend(loc="upper left", frameon=False)
        ax.grid(True, axis="y", alpha=0.6)
        ax.grid(False, axis="x")

        fig.tight_layout()
        fit_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fit_plot_path, dpi=200)
        plt.close(fig)
    print(f"Fit plot written to {fit_plot_path}")


def plot_error(
    summary: pd.DataFrame,
    model_name: str,
    error_plot_path: Path,
) -> None:
    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=(9, 4))

        colors = [
            COLOR_ERROR_POS if v >= 0 else COLOR_ERROR_NEG for v in summary["pct_error"]
        ]
        labels = [
            f"{int(n) // 1024}k" if int(n) >= 1024 else str(int(n))
            for n in summary["context_length"]
        ]
        ax.bar(labels, summary["pct_error"], color=colors, alpha=0.85)

        # Auto-scaled ylim is fit to bar heights only, not the offset annotation
        # text above/below each bar. A small-magnitude bar near the axis floor
        # (e.g. a lone negative bar) then gets its label pushed past the axis
        # line into the tick labels. Padding both ends of the range, with a
        # floor so a near-zero min/max still reserves real room, keeps every
        # annotation clear of the plot border regardless of the data mix.
        y_min = summary["pct_error"].min()
        y_max = summary["pct_error"].max()
        y_pad = max((y_max - y_min) * 0.15, 2.0)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

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
        error_plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(error_plot_path, dpi=200)
        plt.close(fig)
    print(f"Error plot written to {error_plot_path}")


def run_analysis(
    model_spec: ModelSpec,
    model_name: str,
    data_path: Path,
    summary_csv_path: Path,
    summary_tex_path: Path,
    fit_plot_path: Path,
    error_plot_path: Path,
) -> dict:
    """Run analysis for a single model and return the metrics dict for cross-model comparison."""
    df = load_raw_data(data_path)
    summary = build_summary(df)
    metrics = compute_overall_metrics(summary)
    save_summary(summary, metrics, summary_csv_path)
    regime = compute_regime_split(summary)
    print_regime_split(regime)
    write_latex_table(summary, model_name, summary_tex_path)
    plot_fit(summary, model_spec, model_name, fit_plot_path)
    plot_error(summary, model_name, error_plot_path)

    # Return metrics for cross-model comparison
    return {
        "model_tag": model_name,
        "mae_gb": metrics["mae_gb"],
        "mape_pct": metrics["mape_pct"],
        "r_squared": metrics["r_squared"],
        "breakpoint_n": regime["breakpoint_n"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Doubling sequence from 512 to 65536, then 114688 (112*1024) pushing toward
# the hardware ceiling of the NVIDIA L4 (23 GiB VRAM).
DEFAULT_CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 114688]


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
    parser.add_argument(
        "--model-tag",
        default=DEFAULT_MODEL_TAG,
        choices=["all"] + list(MODEL_REGISTRY.keys()),
        help="Model tag from registry or 'all' to run all five models",
    )
    return parser.parse_args()


def migrate_existing_results():
    """Migrate existing Ministral results to the new per-model directory layout."""
    old_data_path = DATA_DIR / "rq4_vram_raw.csv"
    old_summary_path = CSV_DIR / "rq4_summary.csv"
    old_tex_path = CSV_DIR / "rq4_summary_table.tex"
    old_fit_plot = IMAGES_DIR / "rq4_predicted_vs_measured.png"
    old_error_plot = IMAGES_DIR / "rq4_errors.png"

    # Only migrate if old files exist and new structure doesn't
    sanitized = sanitize_tag(DEFAULT_MODEL_TAG)
    new_data_dir = DATA_DIR / sanitized
    new_csv_dir = CSV_DIR / sanitized
    new_image_dir = IMAGES_DIR / sanitized

    # Check if migration is needed
    if (
        old_data_path.exists()
        or old_summary_path.exists()
        or old_fit_plot.exists()
        or old_error_plot.exists()
    ):
        if not (
            new_data_dir.exists() or new_csv_dir.exists() or new_image_dir.exists()
        ):
            print("Migrating existing Ministral results to new layout...")

            # Create directories
            new_data_dir.mkdir(parents=True, exist_ok=True)
            new_csv_dir.mkdir(parents=True, exist_ok=True)
            new_image_dir.mkdir(parents=True, exist_ok=True)

            # Move files
            if old_data_path.exists():
                new_data_path = new_data_dir / "rq4_vram_raw.csv"
                old_data_path.rename(new_data_path)
                print(f"  Moved {old_data_path} -> {new_data_path}")

            if old_summary_path.exists():
                new_summary_path = new_csv_dir / "rq4_summary.csv"
                old_summary_path.rename(new_summary_path)
                print(f"  Moved {old_summary_path} -> {new_summary_path}")

            if old_tex_path.exists():
                new_tex_path = new_csv_dir / "rq4_summary_table.tex"
                old_tex_path.rename(new_tex_path)
                print(f"  Moved {old_tex_path} -> {new_tex_path}")

            if old_fit_plot.exists():
                new_fit_path = new_image_dir / "rq4_predicted_vs_measured.png"
                old_fit_plot.rename(new_fit_path)
                print(f"  Moved {old_fit_plot} -> {new_fit_path}")

            if old_error_plot.exists():
                new_error_path = new_image_dir / "rq4_errors.png"
                old_error_plot.rename(new_error_path)
                print(f"  Moved {old_error_plot} -> {new_error_path}")

            print("Migration complete.")


def main() -> None:
    args = parse_args()

    # Migrate existing results first
    migrate_existing_results()

    # Get the model tag(s) to run
    model_tags = [args.model_tag] if args.model_tag != "all" else ALL_MODEL_TAGS

    # For "all" mode, check and pull models first
    if args.model_tag == "all":
        for tag in model_tags:
            pull_model(tag, "ollama")
    else:
        # For single model, check if it's in the registry
        if args.model_tag not in MODEL_REGISTRY:
            raise ValueError(f"Model tag {args.model_tag} not found in MODEL_REGISTRY")

    reader = get_reader(args.backend, prometheus_url=args.prometheus_url)

    # Collect metrics for cross-model comparison
    all_metrics = []

    for model_tag in model_tags:
        print(f"\n{'=' * 60}")
        print(f"Running experiment for model: {model_tag}")
        print(f"{'=' * 60}")

        model_spec = MODEL_REGISTRY[model_tag]

        # Create OllamaConfig for this model
        ollama_config = OllamaConfig()
        ollama_config.model = model_tag

        # Get paths for this model
        (
            data_path,
            summary_csv_path,
            summary_tex_path,
            fit_plot_path,
            error_plot_path,
        ) = get_model_paths(model_tag)

        # Run experiment (using Wikipedia-based prompts by default)
        run_experiment(
            context_lengths=args.context_lengths,
            repeats=args.repeats,
            gpu_index=args.gpu_index,
            ollama_config=ollama_config,
            reader=reader,
            model_spec=model_spec,
            prompt=None,  # Use Wikipedia-based prompts
            data_path=data_path,
        )

        # Run analysis
        metrics = run_analysis(
            model_spec=model_spec,
            model_name=model_tag,
            data_path=data_path,
            summary_csv_path=summary_csv_path,
            summary_tex_path=summary_tex_path,
            fit_plot_path=fit_plot_path,
            error_plot_path=error_plot_path,
        )
        all_metrics.append(metrics)

        print(f"Completed analysis for {model_tag}")

    # Generate cross-model summary if all five models were run
    if args.model_tag == "all":
        # Check if all five models have data
        missing_models = []
        for tag in ALL_MODEL_TAGS:
            sanitized = sanitize_tag(tag)
            data_path = DATA_DIR / sanitized / "rq4_vram_raw.csv"
            if not data_path.exists():
                missing_models.append(tag)

        if missing_models:
            print(
                f"\nCannot generate cross-model summary: missing data for {missing_models}"
            )
        else:
            # Generate the cross-model summary
            csv_path = CSV_DIR / "rq4_cross_model_summary.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "model_tag",
                        "mae_gb",
                        "mape_pct",
                        "r_squared",
                        "breakpoint_n",
                    ],
                )
                writer.writeheader()
                for metrics in all_metrics:
                    writer.writerow(metrics)

            print(f"\nCross-model summary written to {csv_path}")


if __name__ == "__main__":
    main()

