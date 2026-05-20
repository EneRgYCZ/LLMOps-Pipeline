"""Integration tests for FastAPI endpoints in llm_wrapper.py with mocked backends."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import llm_wrapper
from prometheus_client import REGISTRY

OLLAMA_BASE = "http://localhost:11434"


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    with respx.mock:
        respx.get(f"{OLLAMA_BASE}/api/tags").mock(
            return_value=httpx.Response(200, json={"models": []})
        )
        response = await async_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "ollama": "reachable"}


# ---------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_endpoint(async_client):
    response = await async_client.get("/metrics")

    assert response.status_code == 200
    text = response.text
    assert "llm_active_requests" in text
    assert "llm_requests_total" in text
    assert "llm_request_errors_total" in text
    assert "llm_request_duration_seconds" in text


# ---------------------------------------------------------------------------
# /api/chat — no RAG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_proxy_without_rag(async_client, monkeypatch):
    monkeypatch.setattr(llm_wrapper, "_rag_service", None)

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "hi"}})
        )
        response = await async_client.post("/api/chat", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })

    assert response.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    # No system message injected
    assert not any(m["role"] == "system" for m in sent["messages"])


# ---------------------------------------------------------------------------
# /api/chat — with RAG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_proxy_with_rag(async_client, monkeypatch):
    augmented = [
        {"role": "system", "content": "Use the following context...\n\n[1] relevant chunk"},
        {"role": "user", "content": "hello"},
    ]
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [{"text": "relevant chunk", "metadata": {}, "similarity": 0.9}]
    mock_rag.augment_messages.return_value = augmented
    monkeypatch.setattr(llm_wrapper, "_rag_service", mock_rag)

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
        )
        response = await async_client.post("/api/chat", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })

    assert response.status_code == 200
    mock_rag.retrieve.assert_called_once_with("hello")
    mock_rag.augment_messages.assert_called_once()
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"] == augmented


# ---------------------------------------------------------------------------
# /api/generate — with RAG
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_proxy_with_rag(async_client, monkeypatch):
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = [{"text": "context chunk", "metadata": {}, "similarity": 0.9}]
    mock_rag.augment_prompt.return_value = "Context:\n[1] context chunk\n\nQuestion: explain X"
    monkeypatch.setattr(llm_wrapper, "_rag_service", mock_rag)

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE}/api/generate").mock(
            return_value=httpx.Response(200, json={"response": "answer"})
        )
        response = await async_client.post("/api/generate", json={
            "model": "test",
            "prompt": "explain X",
            "stream": False,
        })

    assert response.status_code == 200
    mock_rag.retrieve.assert_called_once_with("explain X")
    mock_rag.augment_prompt.assert_called_once()
    sent = json.loads(route.calls[0].request.content)
    assert "context chunk" in sent["prompt"]


# ---------------------------------------------------------------------------
# /api/chat — RAG enabled but Chroma down (graceful degradation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_proxy_rag_chroma_down(async_client, monkeypatch):
    mock_rag = MagicMock()
    mock_rag.retrieve.return_value = []  # Chroma unreachable → empty chunks
    mock_rag.augment_messages.return_value = [{"role": "user", "content": "hello"}]
    monkeypatch.setattr(llm_wrapper, "_rag_service", mock_rag)

    with respx.mock:
        route = respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}})
        )
        response = await async_client.post("/api/chat", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        })

    assert response.status_code == 200
    sent = json.loads(route.calls[0].request.content)
    assert not any(m.get("role") == "system" for m in sent["messages"])


# ---------------------------------------------------------------------------
# Error counter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_error_increments_counter(async_client, monkeypatch):
    monkeypatch.setattr(llm_wrapper, "_rag_service", None)

    before = (
        REGISTRY.get_sample_value("llm_request_errors_total", {"endpoint": "api/chat"}) or 0
    )

    with respx.mock:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with pytest.raises(Exception):
            await async_client.post("/api/chat", json={
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            })

    after = (
        REGISTRY.get_sample_value("llm_request_errors_total", {"endpoint": "api/chat"}) or 0
    )
    assert after == before + 1


# ---------------------------------------------------------------------------
# Streaming response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_response(async_client, monkeypatch):
    monkeypatch.setattr(llm_wrapper, "_rag_service", None)

    with respx.mock:
        respx.post(f"{OLLAMA_BASE}/api/chat").mock(
            return_value=httpx.Response(200, content=b'{"done":true}')
        )
        response = await async_client.post("/api/chat", json={
            "model": "test",
            "messages": [{"role": "user", "content": "hello"}],
            # stream defaults to True via body.get("stream", True)
        })

    assert response.status_code == 200
    assert "application/x-ndjson" in response.headers.get("content-type", "")
