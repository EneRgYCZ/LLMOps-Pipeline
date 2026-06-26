"""
RQ3 Experiment: RAGAS Metric Consistency Under a Self-Judging Model Configuration
====================================================================================
Dataset : explodinggradients/amnesty_qa (english_v3) — the canonical dataset
          used in the original RAGAS paper (Es et al., EACL 2024).
Judge   : Ministral 8B Instruct Q4_K_M via local Ollama (same model as inference).
Runs    : N_RUNS passes over all 20 samples (default 20).
Output  : results/rq3_raw.csv          — one row per (run, sample)
          results/rq3_run_summary.csv  — per-run mean + std per metric
          results/rq3_experiment.log   — full execution log

Usage
-----
    # Make sure Ollama is running and the model is pulled:
    #   ollama pull ministral-3:8b-instruct-2512-q4_K_M
    python rq3_experiment.py

    # Override defaults via env vars:
    OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODEL=ministral-3:8b-instruct-2512-q4_K_M \
    N_RUNS_RQ3_TEST=20 \
    python rq3_experiment.py

Dependencies (all already in requirements.txt + datasets):
    ragas==0.4.3
    langchain-ollama==0.2.3
    datasets
"""

import asyncio
import csv
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
N_RUNS = int(os.getenv("N_RUNS_RQ3_TEST", "20"))

RESULTS_DIR = Path("results")
RAW_CSV = RESULTS_DIR / "rq3_raw.csv"
SUMMARY_CSV = RESULTS_DIR / "rq3_run_summary.csv"
LOG_FILE = RESULTS_DIR / "rq3_experiment.log"

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger("rq3")


# ---------------------------------------------------------------------------
# Statistics helpers (stdlib only — no scipy/pandas dependency)
# ---------------------------------------------------------------------------
def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def _std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _summary_stats(rows: list[dict], metric: str) -> dict:
    """Return mean, std, min, max for a metric over a list of result rows."""
    vals = [r[metric] for r in rows if r.get(metric) is not None]
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(_mean(vals), 4),
        "std": round(_std(vals), 4) if _std(vals) is not None else None,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }


# ---------------------------------------------------------------------------
# RAGAS + Ollama wiring (0.4.x API — no deprecated wrappers)
# ---------------------------------------------------------------------------
def build_ragas_components():
    """
    Return (llm, embeddings) using the 0.4.x llm_factory API.
    Ollama exposes an OpenAI-compatible endpoint at /v1.
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key="ollama",
        base_url=f"{OLLAMA_HOST}/v1",
    )
    # Raise max_tokens to avoid truncating verbose structured outputs
    # (faithfulness verification can produce many statement-level lines).
    llm = llm_factory(OLLAMA_MODEL, provider="openai", client=client, max_tokens=4096)

    # HuggingFaceEmbeddings runs locally on CPU — no Ollama embedding endpoint needed.
    # all-MiniLM-L6-v2 is already used by the pipeline's RAG module.
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    return llm, embeddings


def build_metrics(llm, embeddings):
    """Instantiate the four core RAGAS metrics with the local judge."""
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevance": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_amnesty_qa():
    """
    Load explodinggradients/amnesty_qa english_v3.
    Column names in v3: user_input, response, retrieved_contexts, reference.
    """
    from datasets import load_dataset

    log.info("Loading amnesty_qa (english_v3) from HuggingFace ...")
    ds = load_dataset("explodinggradients/amnesty_qa", "english_v3")
    samples = []
    for row in ds["eval"]:
        samples.append(
            {
                "question": row["user_input"],
                "answer": row["response"],
                "contexts": row["retrieved_contexts"],
                "ground_truth": row["reference"],
            }
        )
    log.info("Loaded %d samples.", len(samples))
    return samples


# ---------------------------------------------------------------------------
# Single-sample scoring
# ---------------------------------------------------------------------------
async def score_sample(sample: dict, metrics: dict) -> dict:
    """
    Score one sample with all four metrics.
    Returns dict {metric_name: float | None}.
    """
    import inspect

    available_args = {
        "user_input": sample["question"],
        "response": sample["answer"],
        "retrieved_contexts": sample["contexts"],
        "reference": sample["ground_truth"],
    }

    scores = {}
    for name, metric in metrics.items():
        try:
            sig = inspect.signature(metric.ascore)
            kwargs = {k: v for k, v in available_args.items() if k in sig.parameters}
            result = await metric.ascore(**kwargs)
            score = result.value if hasattr(result, "value") else result
            scores[name] = float(score) if score is not None else None
        except Exception as exc:
            log.warning("  metric=%s  error=%s", name, exc)
            scores[name] = None
    return scores


# ---------------------------------------------------------------------------
# One full run over all samples
# ---------------------------------------------------------------------------
async def run_pass(run_id: int, samples: list, metrics: dict) -> list[dict]:
    """Score all samples in one pass. Returns rows ready for CSV writing."""
    rows = []
    for idx, sample in enumerate(samples):
        sample_id = idx + 1
        log.info(
            "  run=%d  sample=%d/%d  question=%.60s ...",
            run_id,
            sample_id,
            len(samples),
            sample["question"],
        )

        t0 = time.monotonic()
        scores = await score_sample(sample, metrics)
        elapsed = time.monotonic() - t0

        row = {
            "run_id": run_id,
            "sample_id": sample_id,
            "question": sample["question"],
            "elapsed_seconds": round(elapsed, 2),
            **{m: scores.get(m) for m in METRICS},
        }
        rows.append(row)

        score_str = "  ".join(
            f"{m}={scores.get(m):.3f}" if scores.get(m) is not None else f"{m}=None"
            for m in METRICS
        )
        log.info("    %s  (%.1fs)", score_str, elapsed)

    return rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
RAW_FIELDNAMES = ["run_id", "sample_id", "question", "elapsed_seconds"] + METRICS

# Summary includes mean + std + min + max per metric per run
SUMMARY_FIELDNAMES = ["run_id", "n_samples"] + [
    f"{stat}_{m}" for m in METRICS for stat in ("mean", "std", "min", "max")
]


def write_raw_header():
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RAW_FIELDNAMES).writeheader()


def append_raw_rows(rows: list[dict]):
    with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES, extrasaction="ignore")
        writer.writerows(rows)


def write_summary(all_rows: list[dict]):
    """Write per-run descriptive statistics (mean, std, min, max) to the summary CSV."""
    from collections import defaultdict

    run_buckets: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
        run_buckets[row["run_id"]].append(row)

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for run_id in sorted(run_buckets):
            bucket = run_buckets[run_id]
            summary_row = {"run_id": run_id, "n_samples": len(bucket)}
            for m in METRICS:
                stats = _summary_stats(bucket, m)
                summary_row[f"mean_{m}"] = stats["mean"]
                summary_row[f"std_{m}"] = stats["std"]
                summary_row[f"min_{m}"] = stats["min"]
                summary_row[f"max_{m}"] = stats["max"]
            writer.writerow(summary_row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    log.info("=" * 70)
    log.info("RQ3 Experiment — RAGAS metric consistency")
    log.info("Started at %s", datetime.now().isoformat())
    log.info(
        "OLLAMA_HOST=%s  OLLAMA_MODEL=%s  EMBED_MODEL=%s  N_RUNS=%d",
        OLLAMA_HOST,
        OLLAMA_MODEL,
        EMBED_MODEL,
        N_RUNS,
    )
    log.info("=" * 70)

    import httpx

    try:
        r = httpx.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        r.raise_for_status()
        log.info("Ollama reachable at %s", OLLAMA_HOST)
    except Exception as exc:
        log.error("Cannot reach Ollama at %s: %s", OLLAMA_HOST, exc)
        sys.exit(1)

    samples = load_amnesty_qa()

    log.info(
        "Building RAGAS judge (model=%s, embeddings=all-MiniLM-L6-v2 local) ...",
        OLLAMA_MODEL,
    )
    llm, embeddings = build_ragas_components()
    metrics = build_metrics(llm, embeddings)

    write_raw_header()
    all_rows = []

    for run_id in range(1, N_RUNS + 1):
        log.info("-" * 60)
        log.info("Starting run %d / %d", run_id, N_RUNS)
        log.info("-" * 60)

        t_run = time.monotonic()
        rows = await run_pass(run_id, samples, metrics)
        run_elapsed = time.monotonic() - t_run

        append_raw_rows(rows)
        all_rows.extend(rows)

        # Log mean ± std for each metric after each run
        for m in METRICS:
            stats = _summary_stats(rows, m)
            if stats["mean"] is not None:
                log.info(
                    "  run=%d  %s: mean=%.4f  std=%.4f  min=%.4f  max=%.4f  (n=%d)",
                    run_id,
                    m,
                    stats["mean"],
                    stats["std"] or 0.0,
                    stats["min"],
                    stats["max"],
                    stats["n"],
                )
        log.info("  run=%d  total_time=%.1fs", run_id, run_elapsed)

    write_summary(all_rows)

    log.info("=" * 70)
    log.info("Experiment complete.")
    log.info("Raw results   : %s", RAW_CSV.resolve())
    log.info("Run summary   : %s", SUMMARY_CSV.resolve())
    log.info("Log file      : %s", LOG_FILE.resolve())
    log.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
