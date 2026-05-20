"""Evaluation module for the LLMOps pipeline.

Computes RAG quality metrics on inference requests using RAGAS (LLM-as-judge).
Runs in a background asyncio worker; inference is never blocked by evaluation.

Dependencies: ragas==0.2.13, langchain-community>=0.3.18.
RAGAS imports ragas.executor at module-load time which calls nest_asyncio.apply().
nest_asyncio cannot patch uvloop. uvicorn must run with --loop asyncio (not uvloop).

Configuration (all via env vars):
    EVAL_ENABLED            bool  default False — no eval worker started when False
    EVAL_JUDGE_HOST         str   default OLLAMA_HOST — LLM endpoint for RAGAS judge
    EVAL_JUDGE_MODEL        str   default OLLAMA_MODEL — model used for judging
    EVAL_EMBEDDING_MODEL    str   default nomic-embed-text — Ollama embedding model
                                  (must support /api/embeddings; separate from judge LLM)
    EVAL_QUEUE_MAX_SIZE     int   default 100 — max pending evals before drops
    EVAL_SAMPLE_RATE        float default 1.0 — fraction of requests to evaluate
    EVAL_DB_PATH            str   default /data/evaluations.db — SQLite path
    EVAL_TIMEOUT_SECONDS    float default 60.0 — per-metric timeout before abort
    EVAL_FAITHFULNESS_HALLUCINATION_THRESHOLD float default 0.5
    EVAL_REFERENCES_PATH    str   optional — JSON file with ground-truth references

Metrics computed per request:
    faithfulness, answer_relevance, context_precision — when retrieved_chunks non-empty
    context_recall — only when reference_answer is provided (else None)
    is_hallucination — derived: faithfulness < threshold (else None)

SQLite per-replica isolation:
    Each pod/container writes to its own EVAL_DB_PATH. In the k8s StatefulSet
    deployment each replica has an independent PVC, so databases are isolated.
    Cross-replica aggregation is performed at analysis time, not at the DB level.

Streaming limitation:
    Evaluation is limited to non-streaming requests. Streaming requests are
    skipped with a debug log. Known limitation; future work.
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass

from opentelemetry import trace
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

EVAL_REQUESTS_TOTAL = Counter(
    "eval_requests_total",
    "Evaluations completed (success or failure)",
    ["deployment_environment"],
)
EVAL_DROPPED_TOTAL = Counter(
    "eval_dropped_total",
    "Tasks dropped due to full queue",
)
EVAL_ERRORS_TOTAL = Counter(
    "eval_errors_total",
    "Evaluations that failed (timeout, judge error, etc.)",
    ["error_type"],
)
EVAL_QUEUE_DEPTH = Gauge(
    "eval_queue_depth",
    "Current queue size",
)
EVAL_DURATION = Histogram(
    "eval_duration_seconds",
    "Time to compute all metrics for one task",
    buckets=[1, 2.5, 5, 10, 20, 30, 60, 120],
)
EVAL_FAITHFULNESS = Histogram(
    "eval_faithfulness",
    "RAGAS faithfulness score distribution",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
EVAL_ANSWER_RELEVANCE = Histogram(
    "eval_answer_relevance",
    "RAGAS answer relevance distribution",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
EVAL_CONTEXT_PRECISION = Histogram(
    "eval_context_precision",
    "RAGAS context precision distribution",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
EVAL_CONTEXT_RECALL = Histogram(
    "eval_context_recall",
    "RAGAS context recall distribution (only when reference provided)",
    buckets=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
)
EVAL_HALLUCINATIONS_TOTAL = Counter(
    "eval_hallucinations_total",
    "Requests flagged as hallucinations",
    ["deployment_environment"],
)

_tracer = trace.get_tracer("evaluation")

_CREATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evaluations (
    request_id TEXT PRIMARY KEY,
    timestamp REAL NOT NULL,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    chunks_count INTEGER NOT NULL,
    reference_answer TEXT,
    latency_seconds REAL NOT NULL,
    eval_duration_seconds REAL NOT NULL,
    faithfulness REAL,
    answer_relevance REAL,
    context_precision REAL,
    context_recall REAL,
    is_hallucination INTEGER,
    error TEXT,
    deployment_environment TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evaluations_timestamp ON evaluations(timestamp);
CREATE INDEX IF NOT EXISTS idx_evaluations_environment ON evaluations(deployment_environment);
"""


class EvaluationConfig:
    """Loaded from environment variables."""

    def __init__(self) -> None:
        raw_enabled = os.getenv("EVAL_ENABLED", "false").strip().lower()
        self.eval_enabled: bool = raw_enabled in ("true", "1", "yes")
        self.eval_judge_host: str = os.getenv(
            "EVAL_JUDGE_HOST",
            os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        )
        self.eval_judge_model: str = os.getenv(
            "EVAL_JUDGE_MODEL",
            os.getenv("OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M"),
        )
        self.eval_embedding_model: str = os.getenv(
            "EVAL_EMBEDDING_MODEL", "nomic-embed-text"
        )
        self.eval_queue_max_size: int = int(os.getenv("EVAL_QUEUE_MAX_SIZE", "100"))
        self.eval_sample_rate: float = float(os.getenv("EVAL_SAMPLE_RATE", "1.0"))
        self.eval_db_path: str = os.getenv("EVAL_DB_PATH", "/data/evaluations.db")
        self.eval_timeout_seconds: float = float(os.getenv("EVAL_TIMEOUT_SECONDS", "60.0"))
        self.eval_faithfulness_hallucination_threshold: float = float(
            os.getenv("EVAL_FAITHFULNESS_HALLUCINATION_THRESHOLD", "0.5")
        )


@dataclass
class EvaluationTask:
    """Unit of evaluation work enqueued by the wrapper."""

    request_id: str
    timestamp: float
    query: str
    response: str
    retrieved_chunks: list[dict]
    reference_answer: str | None
    latency_seconds: float
    deployment_environment: str


@dataclass
class EvaluationResult:
    """Output of one evaluation."""

    request_id: str
    timestamp: float
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None
    is_hallucination: bool | None
    latency_seconds: float
    eval_duration_seconds: float
    error: str | None


def load_references(path: str) -> dict[str, str]:
    """Load reference answers from a JSON file for use at wrapper startup.

    Format: [{"query": "...", "reference": "..."}, ...]
    Matching is exact string match on query (no fuzzy matching — out of scope).
    Returns empty dict if path is missing or file is unreadable.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        refs = {item["query"]: item["reference"] for item in data}
        logger.info("Loaded %d reference answers from %s", len(refs), path)
        return refs
    except FileNotFoundError:
        logger.warning("References file not found: %s", path)
        return {}
    except Exception as exc:
        logger.warning("Failed to load references from %s: %s", path, exc)
        return {}


class EvaluationService:
    """Owns the queue, the background worker task, and metric definitions.

    Lifecycle: call start() from FastAPI startup, stop() from FastAPI shutdown.
    The worker is resilient — any exception in _evaluate() is caught, logged,
    and converted to an EvaluationResult with error set and None metrics.
    """

    def __init__(self, config: EvaluationConfig, deployment_environment: str = "dev") -> None:
        self._config = config
        self._deployment_env = deployment_environment
        self._queue: asyncio.Queue[EvaluationTask] = asyncio.Queue(
            maxsize=config.eval_queue_max_size
        )
        self._worker_task: asyncio.Task | None = None
        self._db: sqlite3.Connection | None = None
        self._llm_wrapper = None
        self._emb_wrapper = None
        self._init_ragas()
        self._init_db()

    def _init_ragas(self) -> None:
        try:
            from langchain_community.llms import Ollama as LangchainOllama
            from langchain_community.embeddings import OllamaEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper

            langchain_llm = LangchainOllama(
                base_url=self._config.eval_judge_host,
                model=self._config.eval_judge_model,
            )
            langchain_emb = OllamaEmbeddings(
                base_url=self._config.eval_judge_host,
                model=self._config.eval_embedding_model,
            )
            self._llm_wrapper = LangchainLLMWrapper(langchain_llm)
            self._emb_wrapper = LangchainEmbeddingsWrapper(langchain_emb)
            logger.info(
                "RAGAS judge initialised: host=%s model=%s",
                self._config.eval_judge_host,
                self._config.eval_judge_model,
            )
        except Exception as exc:
            logger.error("Failed to initialise RAGAS judge wrappers: %s", exc)

    def _init_db(self) -> None:
        try:
            db_dir = os.path.dirname(self._config.eval_db_path)
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
            self._db = sqlite3.connect(self._config.eval_db_path, check_same_thread=False)
            self._db.executescript(_CREATE_SCHEMA_SQL)
            self._db.commit()
            logger.info("SQLite evaluation DB ready at %s", self._config.eval_db_path)
        except Exception as exc:
            logger.warning(
                "Failed to init SQLite DB at %s: %s — persistence disabled",
                self._config.eval_db_path,
                exc,
            )
            self._db = None

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._worker_loop(), name="eval-worker")
        logger.info("Evaluation worker started")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        try:
            await asyncio.wait_for(self._queue.join(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(
                "Evaluation queue drain timed out; %d tasks abandoned",
                self._queue.qsize(),
            )
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        if self._db:
            self._db.close()
        logger.info("Evaluation worker stopped")

    def enqueue(self, task: EvaluationTask) -> bool:
        """Non-blocking enqueue. Returns False and increments drop counter if full."""
        try:
            self._queue.put_nowait(task)
            EVAL_QUEUE_DEPTH.set(self._queue.qsize())
            return True
        except asyncio.QueueFull:
            EVAL_DROPPED_TOTAL.inc()
            logger.debug("Eval queue full; dropping task %s", task.request_id)
            return False

    async def _worker_loop(self) -> None:
        while True:
            task = await self._queue.get()
            EVAL_QUEUE_DEPTH.set(self._queue.qsize())
            try:
                result = await self._evaluate(task)
                self._record_prometheus(result)
                self._persist(task, result)
            except Exception as exc:
                logger.exception(
                    "Unexpected worker error for request %s: %s", task.request_id, exc
                )
                EVAL_ERRORS_TOTAL.labels(error_type="other").inc()
            finally:
                self._queue.task_done()

    async def _evaluate(self, task: EvaluationTask) -> EvaluationResult:
        start = time.monotonic()
        faithfulness = None
        answer_relevance = None
        context_precision = None
        context_recall = None
        error = None

        with _tracer.start_as_current_span("evaluation.compute") as span:
            span.set_attribute("eval.request_id", task.request_id)
            span.set_attribute("eval.has_reference", task.reference_answer is not None)
            span.set_attribute("eval.chunks_count", len(task.retrieved_chunks))

            metric_errors: list[str] = []

            if self._llm_wrapper is None:
                error = "judge_init_failed"
                EVAL_ERRORS_TOTAL.labels(error_type="judge_error").inc()
            else:
                try:
                    from ragas.metrics import (
                        Faithfulness,
                        AnswerRelevancy,
                        ContextPrecision,
                        ContextRecall,
                    )
                    from ragas.dataset_schema import SingleTurnSample

                    sample = SingleTurnSample(
                        user_input=task.query,
                        response=task.response,
                        retrieved_contexts=[c["text"] for c in task.retrieved_chunks],
                        reference=task.reference_answer,
                    )

                    from ragas.metrics import LLMContextPrecisionWithoutReference

                    faithfulness = await self._score_metric(
                        Faithfulness(llm=self._llm_wrapper),
                        sample,
                        "faithfulness",
                        metric_errors,
                    )
                    answer_relevance = await self._score_metric(
                        AnswerRelevancy(
                            llm=self._llm_wrapper,
                            embeddings=self._emb_wrapper,
                        ),
                        sample,
                        "answer_relevance",
                        metric_errors,
                    )
                    # ContextPrecision requires reference; use without-reference variant
                    # when no ground truth is available.
                    if task.reference_answer is not None:
                        context_precision = await self._score_metric(
                            ContextPrecision(llm=self._llm_wrapper),
                            sample,
                            "context_precision",
                            metric_errors,
                        )
                        context_recall = await self._score_metric(
                            ContextRecall(llm=self._llm_wrapper),
                            sample,
                            "context_recall",
                            metric_errors,
                        )
                    else:
                        context_precision = await self._score_metric(
                            LLMContextPrecisionWithoutReference(llm=self._llm_wrapper),
                            sample,
                            "context_precision",
                            metric_errors,
                        )

                    if metric_errors:
                        error = "; ".join(metric_errors)

                except Exception as exc:
                    error = f"judge_error: {exc}"
                    logger.warning("RAGAS evaluation failed for %s: %s", task.request_id, exc)
                    EVAL_ERRORS_TOTAL.labels(error_type="judge_error").inc()

            is_hallucination: bool | None = None
            if faithfulness is not None:
                is_hallucination = (
                    faithfulness
                    < self._config.eval_faithfulness_hallucination_threshold
                )

            if faithfulness is not None:
                span.set_attribute("eval.faithfulness", faithfulness)
            if answer_relevance is not None:
                span.set_attribute("eval.answer_relevance", answer_relevance)
            if context_precision is not None:
                span.set_attribute("eval.context_precision", context_precision)
            if context_recall is not None:
                span.set_attribute("eval.context_recall", context_recall)
            if is_hallucination is not None:
                span.set_attribute("eval.is_hallucination", is_hallucination)

        return EvaluationResult(
            request_id=task.request_id,
            timestamp=task.timestamp,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            is_hallucination=is_hallucination,
            latency_seconds=task.latency_seconds,
            eval_duration_seconds=time.monotonic() - start,
            error=error,
        )

    async def _score_metric(
        self, metric, sample, metric_name: str, errors: list[str]
    ) -> float | None:
        """Evaluate one RAGAS metric with timeout. Returns None on any failure.

        Appends a short error description to `errors` on failure so the caller
        can build a summary error string for EvaluationResult.
        """
        try:
            score = await asyncio.wait_for(
                metric.single_turn_ascore(sample),
                timeout=self._config.eval_timeout_seconds,
            )
            return float(score)
        except asyncio.TimeoutError:
            logger.warning("Metric %s timed out for request", metric_name)
            EVAL_ERRORS_TOTAL.labels(error_type="timeout").inc()
            errors.append(f"{metric_name}:timeout")
            return None
        except Exception as exc:
            logger.warning("Metric %s failed: %s", metric_name, exc)
            EVAL_ERRORS_TOTAL.labels(error_type="judge_error").inc()
            errors.append(f"{metric_name}:{exc}")
            return None

    def _record_prometheus(self, result: EvaluationResult) -> None:
        EVAL_REQUESTS_TOTAL.labels(deployment_environment=self._deployment_env).inc()
        EVAL_DURATION.observe(result.eval_duration_seconds)
        if result.faithfulness is not None:
            EVAL_FAITHFULNESS.observe(result.faithfulness)
        if result.answer_relevance is not None:
            EVAL_ANSWER_RELEVANCE.observe(result.answer_relevance)
        if result.context_precision is not None:
            EVAL_CONTEXT_PRECISION.observe(result.context_precision)
        if result.context_recall is not None:
            EVAL_CONTEXT_RECALL.observe(result.context_recall)
        if result.is_hallucination:
            EVAL_HALLUCINATIONS_TOTAL.labels(deployment_environment=self._deployment_env).inc()

    def _persist(self, task: EvaluationTask, result: EvaluationResult) -> None:
        if self._db is None:
            return
        try:
            self._db.execute(
                """
                INSERT OR REPLACE INTO evaluations (
                    request_id, timestamp, query, response, chunks_count,
                    reference_answer, latency_seconds, eval_duration_seconds,
                    faithfulness, answer_relevance, context_precision, context_recall,
                    is_hallucination, error, deployment_environment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.request_id,
                    result.timestamp,
                    task.query,
                    task.response,
                    len(task.retrieved_chunks),
                    task.reference_answer,
                    result.latency_seconds,
                    result.eval_duration_seconds,
                    result.faithfulness,
                    result.answer_relevance,
                    result.context_precision,
                    result.context_recall,
                    int(result.is_hallucination) if result.is_hallucination is not None else None,
                    result.error,
                    self._deployment_env,
                ),
            )
            self._db.commit()
        except Exception as exc:
            logger.warning("Failed to persist evaluation %s: %s", result.request_id, exc)
