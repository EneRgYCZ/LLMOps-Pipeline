import asyncio
import csv
import logging
import math
import os
import subprocess
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
N_RUNS = int(os.getenv("N_RUNS_RQ3_TEST", "5"))

# Restart the Ollama model process between runs to clear llama.cpp's prompt
# cache. Default ON — set to "0" to disable (e.g. for quick debugging runs
# where cache effects do not matter).
RESTART_BETWEEN_RUNS = os.getenv("RESTART_BETWEEN_RUNS", "1") == "1"

# Seconds to wait after `ollama stop` before the next run starts, giving the
# server time to fully unload the model and release VRAM before reload.
RESTART_SETTLE_SECONDS = float(os.getenv("RESTART_SETTLE_SECONDS", "3"))

# Paths resolved relative to this file (not the working directory) so the
# project can be checked out anywhere and still reproduce, matching the
# results/rq4/{data,csvs,images} layout used by analysis/rq4/rq4_experiment.py.
SCRIPT_DIR = Path(__file__).resolve().parent  # analysis/rq3
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # repo root
RESULTS_DIR = PROJECT_ROOT / "results" / "rq3"
DATA_DIR = RESULTS_DIR / "data"

RAW_CSV = DATA_DIR / "rq3_raw.csv"
SUMMARY_CSV = DATA_DIR / "rq3_run_summary.csv"
LOG_FILE = DATA_DIR / "rq3_experiment.log"

METRICS = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]

# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------
DATA_DIR.mkdir(parents=True, exist_ok=True)

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
# Ollama model restart (cache invalidation)
# ---------------------------------------------------------------------------
# Name of the Docker container running Ollama. If set, the restart command
# runs as `docker exec <container> ollama stop <model>` instead of a bare
# host-level `ollama stop`, since Ollama typically runs containerized.
# Leave empty/unset to fall back to a direct host-level call (e.g. when
# Ollama runs natively, not in Docker).
OLLAMA_CONTAINER = os.getenv("OLLAMA_CONTAINER", "ollama")


def restart_ollama_model():
    """
    Stop the Ollama model process to flush llama.cpp's in-memory prompt
    cache, then briefly wait. The model will be reloaded automatically by
    Ollama on the next request, at the cost of a cold-start delay on the
    first sample of the following run.

    Runs inside the Docker container (`docker exec <container> ollama stop
    <model>`) when OLLAMA_CONTAINER is set, since Ollama in this pipeline
    runs containerized and the host has no `ollama` CLI on PATH. Set
    OLLAMA_CONTAINER="" to fall back to a direct host-level `ollama stop`.
    """
    log.info("Restarting Ollama model %s to clear prompt cache ...", OLLAMA_MODEL)

    if OLLAMA_CONTAINER:
        cmd = ["docker", "exec", OLLAMA_CONTAINER, "ollama", "stop", OLLAMA_MODEL]
    else:
        cmd = ["ollama", "stop", OLLAMA_MODEL]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            log.warning(
                "  '%s' exited with code %d: %s",
                " ".join(cmd),
                result.returncode,
                result.stderr.strip(),
            )
        else:
            log.info("  Model stopped successfully.")
    except FileNotFoundError:
        log.error(
            "  Command not found (%s) — cannot restart model. "
            "Cache invalidation will not occur between runs. "
            "Check OLLAMA_CONTAINER / docker availability.",
            cmd[0],
        )
    except subprocess.TimeoutExpired:
        log.warning("  '%s' timed out after 30s, continuing anyway.", " ".join(cmd))
    except Exception as exc:
        log.warning("  Unexpected error while stopping model: %s", exc)

    time.sleep(RESTART_SETTLE_SECONDS)


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
    # temperature=0.1 follows the recommendation in the temperature-vs-LLM-judge
    # literature (peak self-consistency at t=0.1).
    #
    # CRITICAL: top_p must be set explicitly. ragas's llm_factory defaults
    # top_p to match the temperature value (top_p=0.1 here) unless overridden,
    # which collapses nucleus sampling to near-greedy decoding regardless of
    # temperature -- temperature only perturbs the candidate pool that top_p
    # selects, and a pool of size ~1 leaves nothing for temperature to act on.
    # Confirmed empirically: with top_p left at its temperature-matched
    # default, repeated raw completions and RAGAS scores were bit-identical
    # across 20 runs; setting top_p=0.95 restored genuine run-to-run variation.
    llm = llm_factory(
        OLLAMA_MODEL,
        provider="openai",
        client=client,
        max_tokens=4096,
        temperature=0.1,
        top_p=0.95,
    )

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
        "OLLAMA_HOST=%s  OLLAMA_MODEL=%s  EMBED_MODEL=%s  N_RUNS=%d  "
        "RESTART_BETWEEN_RUNS=%s",
        OLLAMA_HOST,
        OLLAMA_MODEL,
        EMBED_MODEL,
        N_RUNS,
        RESTART_BETWEEN_RUNS,
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

    samples = load_amnesty_qa()[:5]

    log.info(
        "Building RAGAS judge (model=%s, embeddings=all-MiniLM-L6-v2 local) ...",
        OLLAMA_MODEL,
    )
    llm, embeddings = build_ragas_components()
    metrics = build_metrics(llm, embeddings)

    write_raw_header()
    all_rows = []

    for run_id in range(1, N_RUNS + 1):
        # Clear the prompt cache before every run except the very first,
        # which starts from a cold model anyway.
        if RESTART_BETWEEN_RUNS and run_id > 1:
            restart_ollama_model()

        log.info("-" * 60)
        log.info("Starting run %d / %d", run_id, N_RUNS)
        log.info("-" * 60)

        t_run = time.monotonic()
        rows = await run_pass(run_id, samples, metrics)
        run_elapsed = time.monotonic() - t_run

        append_raw_rows(rows)
        all_rows.extend(rows)

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
