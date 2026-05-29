import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import openlit
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

from settings import SettingsFactory, Settings
from telemetry import OpenTelemetryHelper

ACTIVE_REQUESTS = Gauge("llm_active_requests", "Current in-flight LLM inference requests")
REQUEST_TOTAL = Counter("llm_requests_total", "Total LLM inference requests", ["endpoint"])
REQUEST_ERRORS = Counter("llm_request_errors_total", "Total failed LLM requests", ["endpoint"])
REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "End-to-end LLM request latency",
    ["endpoint"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300],
)

_settings: Settings | None = None
_rag_service = None
_eval_service = None
_references: dict[str, str] = {}


def _get_ollama_host() -> str:
    return _settings.ollama_host if _settings else os.getenv("OLLAMA_HOST", "http://localhost:11434")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _settings, _rag_service, _eval_service, _references
    _settings = SettingsFactory.from_env()
    OpenTelemetryHelper.configure(_settings)
    OpenTelemetryHelper.init_openlit()

    if _settings.rag_enabled:
        import asyncio
        from rag import RAGConfig, RAGService

        rag_config = RAGConfig()
        # RAGService.__init__ loads a SentenceTransformer model synchronously.
        # Run in a thread so the event loop stays responsive during startup (health probe).
        _rag_service = await asyncio.to_thread(RAGService, rag_config)

    if _settings.eval_enabled:
        from evaluation import EvaluationConfig, EvaluationService, load_references

        eval_config = EvaluationConfig()
        _eval_service = EvaluationService(
            eval_config,
            deployment_environment=_settings.otel_deployment_environment,
        )
        await _eval_service.start()

        if _settings.eval_references_path:
            _references = load_references(_settings.eval_references_path)

    # Pre-initialize labeled series so Prometheus emits them even before first request.
    for ep in ("api/chat", "api/generate"):
        REQUEST_TOTAL.labels(endpoint=ep)
        REQUEST_ERRORS.labels(endpoint=ep)
        REQUEST_DURATION.labels(endpoint=ep)

    yield

    if _eval_service is not None:
        await _eval_service.stop()


app = FastAPI(title="LLMOps Wrapper", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict:
    async with httpx.AsyncClient(timeout=5) as client:
        r = await client.get(f"{_get_ollama_host()}/api/tags")
        r.raise_for_status()
    return {"status": "ok", "ollama": "reachable"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/tags")
async def tags() -> Response:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{_get_ollama_host()}/api/tags")
    return Response(content=r.content, media_type="application/json", status_code=r.status_code)


@app.post("/api/chat", response_model=None)
async def chat(request: Request) -> Response | StreamingResponse:
    return await _proxy_request(request, "/api/chat")


@app.post("/api/generate", response_model=None)
async def generate(request: Request) -> Response | StreamingResponse:
    return await _proxy_request(request, "/api/generate")


def _extract_last_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and "text" in p
                )
    return ""


def _extract_response_text(body: bytes, path: str) -> str:
    try:
        data = json.loads(body)
        if path == "/api/chat":
            return data.get("message", {}).get("content", "")
        if path == "/api/generate":
            return data.get("response", "")
    except Exception:
        pass
    return ""


async def _proxy_request(request: Request, path: str) -> Response | StreamingResponse:
    body = await request.json()

    # Capture original query before RAG augmentation for evaluation
    original_query = ""
    rag_chunks: list[dict] = []

    if _rag_service is not None:
        if path == "/api/chat":
            user_msg = _extract_last_user_message(body.get("messages", []))
            if user_msg:
                original_query = user_msg
                rag_chunks = _rag_service.retrieve(user_msg)
                body["messages"] = _rag_service.augment_messages(body["messages"], rag_chunks)
        elif path == "/api/generate":
            prompt = body.get("prompt", "")
            if prompt:
                original_query = prompt
                rag_chunks = _rag_service.retrieve(prompt)
                body["prompt"] = _rag_service.augment_prompt(prompt, rag_chunks)
    else:
        if path == "/api/chat":
            original_query = _extract_last_user_message(body.get("messages", []))
        elif path == "/api/generate":
            original_query = body.get("prompt", "")

    is_stream = body.get("stream", True)
    endpoint = path.lstrip("/")

    REQUEST_TOTAL.labels(endpoint=endpoint).inc()
    ACTIVE_REQUESTS.inc()
    start = time.monotonic()

    async def _stream_gen():
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                async with client.stream("POST", f"{_get_ollama_host()}{path}", json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
        except Exception:
            REQUEST_ERRORS.labels(endpoint=endpoint).inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()
            REQUEST_DURATION.labels(endpoint=endpoint).observe(time.monotonic() - start)

    if is_stream:
        if _eval_service is not None:
            import logging
            logging.getLogger(__name__).debug(
                "Skipping eval for streaming request (known limitation)"
            )
        return StreamingResponse(_stream_gen(), media_type="application/x-ndjson")

    response_bytes: list[bytes] = []
    try:
        async for chunk in _stream_gen():
            response_bytes.append(chunk)
    except Exception:
        raise

    response_body = b"".join(response_bytes)
    latency = time.monotonic() - start

    if _eval_service is not None and original_query and path in ("/api/chat", "/api/generate"):
        if random.random() <= _settings.eval_sample_rate:
            from evaluation import EvaluationTask

            response_text = _extract_response_text(response_body, path)
            reference = _references.get(original_query) if _references else None
            task = EvaluationTask(
                request_id=str(uuid.uuid4()),
                timestamp=time.time(),
                query=original_query,
                response=response_text,
                retrieved_chunks=rag_chunks,
                reference_answer=reference,
                latency_seconds=latency,
                deployment_environment=_settings.otel_deployment_environment,
            )
            _eval_service.enqueue(task)

    return Response(content=response_body, media_type="application/json")
