"""Shared pytest fixtures for the LLMOps pipeline test suite."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_SRC = Path(__file__).parent.parent / "src"
_SCRIPTS = Path(__file__).parent.parent / "scripts"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# Must be set before importing llm_wrapper (module-level openlit import)
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
os.environ.setdefault("RAG_ENABLED", "false")
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")

# Import once at module level — prevents Prometheus "Duplicated timeseries" errors
import llm_wrapper  # noqa: E402
from llm_wrapper import app  # noqa: E402


# ---------------------------------------------------------------------------
# External-service mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_openlit(monkeypatch):
    """Patch openlit.init to be a no-op."""
    monkeypatch.setattr("openlit.init", lambda **kwargs: None)


@pytest.fixture
def mock_sentence_transformer(monkeypatch):
    """Patch SentenceTransformer so it never downloads a model.
    encode() returns a fixed numpy array of shape (384,)."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *args, **kwargs: mock_model,
    )
    return mock_model


@pytest.fixture
def mock_chroma_collection():
    """A mock ChromaDB collection with three documents at varying distances."""
    collection = MagicMock()
    collection.query.return_value = {
        "documents": [["doc one", "doc two", "doc three"]],
        "metadatas": [[{"source": "a.txt"}, {"source": "b.txt"}, {"source": "c.txt"}]],
        "distances": [[0.1, 0.3, 0.9]],
    }
    return collection


@pytest.fixture
def mock_chroma_client(monkeypatch, mock_chroma_collection):
    """Patch chromadb.HttpClient to return a mock client."""
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_chroma_collection
    monkeypatch.setattr("chromadb.HttpClient", lambda **kwargs: mock_client)
    return mock_client


@pytest.fixture
def mock_chroma_unreachable(monkeypatch):
    """Patch chromadb.HttpClient to raise ConnectionError on instantiation."""

    def _raise(**kwargs):
        raise ConnectionError("ChromaDB unreachable")

    monkeypatch.setattr("chromadb.HttpClient", _raise)


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rag_env(monkeypatch):
    """Set RAG-related environment variables for testing."""
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("RAG_TOP_K", "5")
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.3")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    monkeypatch.setenv("CHROMA_HOST", "localhost")
    monkeypatch.setenv("CHROMA_PORT", "8000")
    monkeypatch.setenv("CHROMA_COLLECTION", "documents")


@pytest.fixture
def rag_env_disabled(monkeypatch):
    """Set RAG_ENABLED=false."""
    monkeypatch.setenv("RAG_ENABLED", "false")


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings():
    """A minimal Settings object for injection into llm_wrapper globals."""
    from settings import Settings

    return Settings(
        otlp_base="http://localhost:4318",
        metrics_endpoint="http://localhost:4318/v1/metrics",
        traces_endpoint="http://localhost:4318/v1/traces",
        ollama_host="http://localhost:11434",
        chat_host="http://localhost:8000",
        ollama_model="test-model",
        otel_service_name="test-service",
        otel_deployment_environment="test",
        rag_enabled=False,
        chroma_host="localhost",
        chroma_port=8000,
        chroma_collection="documents",
        rag_top_k=5,
        rag_embedding_model="all-MiniLM-L6-v2",
        rag_min_similarity=0.3,
        eval_enabled=False,
        eval_judge_host="http://localhost:11434",
        eval_judge_model="test-model",
        eval_queue_max_size=100,
        eval_sample_rate=1.0,
        eval_db_path="/tmp/test_evaluations.db",
        eval_timeout_seconds=60.0,
        eval_faithfulness_hallucination_threshold=0.5,
        eval_references_path=None,
    )


@pytest.fixture
async def async_client(mock_openlit, test_settings, monkeypatch):
    """httpx.AsyncClient bound to the FastAPI app via ASGITransport.

    Bypasses startup event — sets _settings and _rag_service directly.
    """
    import httpx

    monkeypatch.setattr(llm_wrapper, "_settings", test_settings)
    monkeypatch.setattr(llm_wrapper, "_rag_service", None)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
