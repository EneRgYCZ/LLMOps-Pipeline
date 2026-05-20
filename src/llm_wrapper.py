import os
import time

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


def _get_ollama_host() -> str:
    return _settings.ollama_host if _settings else os.getenv("OLLAMA_HOST", "http://localhost:11434")


app = FastAPI(title="LLMOps Wrapper", version="1.0.0")


@app.on_event("startup")
async def startup() -> None:
    global _settings, _rag_service
    _settings = SettingsFactory.from_env()
    OpenTelemetryHelper.configure(_settings)
    OpenTelemetryHelper.init_openlit()

    if _settings.rag_enabled:
        from rag import RAGConfig, RAGService

        rag_config = RAGConfig()
        _rag_service = RAGService(rag_config)


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


async def _proxy_request(request: Request, path: str) -> Response | StreamingResponse:
    body = await request.json()

    if _rag_service is not None:
        if path == "/api/chat":
            user_msg = _extract_last_user_message(body.get("messages", []))
            if user_msg:
                chunks = _rag_service.retrieve(user_msg)
                body["messages"] = _rag_service.augment_messages(body["messages"], chunks)
        elif path == "/api/generate":
            prompt = body.get("prompt", "")
            if prompt:
                chunks = _rag_service.retrieve(prompt)
                body["prompt"] = _rag_service.augment_prompt(prompt, chunks)

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
        return StreamingResponse(_stream_gen(), media_type="application/x-ndjson")

    chunks: list[bytes] = []
    try:
        async for chunk in _stream_gen():
            chunks.append(chunk)
    except Exception:
        raise
    return Response(content=b"".join(chunks), media_type="application/json")
