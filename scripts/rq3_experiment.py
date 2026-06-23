"""
RQ3 Experiment: RAGAS Metric Consistency Under a Self-Judging Model Configuration
====================================================================================
Dataset : explodinggradients/amnesty_qa (english_v3) — the canonical dataset
          used in the original RAGAS paper (Es et al., EACL 2024).
Judge   : Ministral 8B Instruct Q4_K_M via local Ollama (same model as inference).
Runs    : N_RUNS passes over all 20 samples (default 3).
Output  : results/rq3_raw.csv          — one row per (run, sample)
          results/rq3_run_summary.csv  — per-run averages
          results/rq3_experiment.log   — full execution log

Usage
-----
    # Make sure Ollama is running and the model is pulled:
    #   ollama pull ministral-3:8b-instruct-2512-q4_K_M
    #   ollama pull nomic-embed-text
    python rq3_experiment.py

    # Override defaults via env vars:
    OLLAMA_HOST=http://localhost:11434 \
    OLLAMA_MODEL=ministral-3:8b-instruct-2512-q4_K_M \
    EMBED_MODEL=nomic-embed-text \
    N_RUNS=3 \
    python rq3_experiment.py

Dependencies (all already in requirements.txt + datasets):
    ragas==0.4.3
    langchain-ollama==0.2.3
    datasets
"""

import asyncio
import csv
import logging
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
N_RUNS = int(os.getenv("N_RUNS", "3"))

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
# RAGAS + Ollama wiring (0.4.x API — no deprecated wrappers)
# ---------------------------------------------------------------------------
def build_ragas_components():
    """
    Return (llm, embeddings) using the 0.4.x llm_factory / embedding_factory API.
    Ollama exposes an OpenAI-compatible endpoint at /v1, so we use provider='openai'.
    """
    from openai import AsyncOpenAI
    from ragas.embeddings import HuggingFaceEmbeddings
    from ragas.llms import llm_factory

    client = AsyncOpenAI(
        api_key="ollama",  # Ollama does not require a real key
        base_url=f"{OLLAMA_HOST}/v1",
    )
    # Default max_tokens=1024 truncates this model's verbose structured
    # output (e.g. faithfulness statement verification); raise the cap.
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
    Returns a list of dicts with normalised keys used throughout this script.
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

    # ragas 0.4.x collections metrics each take a different subset of these
    # named arguments via .ascore(**kwargs); filter per metric's signature.
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
    """
    Score all samples in one pass.
    Returns list of result dicts ready for CSV writing.
    """
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


def write_raw_header():
    with open(RAW_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=RAW_FIELDNAMES).writeheader()


def append_raw_rows(rows: list[dict]):
    with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDNAMES, extrasaction="ignore")
        writer.writerows(rows)


def write_summary(all_rows: list[dict]):
    """Write per-run averages to the summary CSV."""
    from collections import defaultdict

    run_buckets: dict[int, list[dict]] = defaultdict(list)
    for row in all_rows:
        run_buckets[row["run_id"]].append(row)

    summary_fields = ["run_id", "n_samples"] + [f"mean_{m}" for m in METRICS]

    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for run_id in sorted(run_buckets):
            bucket = run_buckets[run_id]
            summary = {"run_id": run_id, "n_samples": len(bucket)}
            for m in METRICS:
                vals = [r[m] for r in bucket if r[m] is not None]
                summary[f"mean_{m}"] = round(sum(vals) / len(vals), 4) if vals else None
            writer.writerow(summary)


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

    # Validate Ollama is reachable before doing anything else
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

        for m in METRICS:
            vals = [r[m] for r in rows if r[m] is not None]
            avg = sum(vals) / len(vals) if vals else None
            log.info(
                "  run=%d  mean_%s=%.4f  (n=%d)",
                run_id,
                m,
                avg if avg else 0.0,
                len(vals),
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
