"""Unit tests for RAGConfig, RAGService, augment_messages, and augment_prompt."""

import os

import pytest
from prometheus_client import REGISTRY


# ---------------------------------------------------------------------------
# RAGConfig
# ---------------------------------------------------------------------------


def test_rag_config_defaults(monkeypatch):
    for var in [
        "RAG_ENABLED", "CHROMA_HOST", "CHROMA_PORT", "CHROMA_COLLECTION",
        "RAG_TOP_K", "RAG_EMBEDDING_MODEL", "RAG_MIN_SIMILARITY",
    ]:
        monkeypatch.delenv(var, raising=False)

    from rag import RAGConfig

    cfg = RAGConfig()
    assert cfg.rag_enabled is False
    assert cfg.chroma_host == "localhost"
    assert cfg.chroma_port == 8000
    assert cfg.collection == "documents"
    assert cfg.top_k == 5
    assert cfg.embedding_model == "all-MiniLM-L6-v2"
    assert cfg.min_similarity == pytest.approx(0.3)


def test_rag_config_from_env(monkeypatch):
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("CHROMA_HOST", "chroma-server")
    monkeypatch.setenv("CHROMA_PORT", "9000")
    monkeypatch.setenv("CHROMA_COLLECTION", "my-docs")
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom-model")
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.6")

    from rag import RAGConfig

    cfg = RAGConfig()
    assert cfg.rag_enabled is True
    assert cfg.chroma_host == "chroma-server"
    assert cfg.chroma_port == 9000
    assert cfg.collection == "my-docs"
    assert cfg.top_k == 10
    assert cfg.embedding_model == "custom-model"
    assert cfg.min_similarity == pytest.approx(0.6)


@pytest.mark.parametrize("value,expected", [
    ("true", True),
    ("True", True),
    ("TRUE", True),
    ("1", True),
    ("yes", True),
    ("YES", True),
    ("false", False),
    ("0", False),
    ("no", False),
    ("", False),
])
def test_rag_config_boolean_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("RAG_ENABLED", value)

    from rag import RAGConfig

    cfg = RAGConfig()
    assert cfg.rag_enabled is expected


# ---------------------------------------------------------------------------
# RAGService.retrieve
# ---------------------------------------------------------------------------


def test_retrieve_returns_chunks(mock_sentence_transformer, mock_chroma_client, rag_env):
    """dist 0.1 → sim 0.9 (pass), dist 0.3 → sim 0.7 (pass), dist 0.9 → sim 0.1 (fail)."""
    from rag import RAGConfig, RAGService

    cfg = RAGConfig()
    svc = RAGService(cfg)
    chunks = svc.retrieve("test query")

    assert len(chunks) == 2
    assert chunks[0]["text"] == "doc one"
    assert chunks[1]["text"] == "doc two"
    assert chunks[0]["similarity"] == pytest.approx(0.9)
    assert chunks[1]["similarity"] == pytest.approx(0.7)


def test_retrieve_chroma_unreachable(mock_sentence_transformer, mock_chroma_unreachable, rag_env):
    from rag import RAGConfig, RAGService, RAG_RETRIEVAL_ERRORS

    before = RAG_RETRIEVAL_ERRORS._value.get()
    cfg = RAGConfig()
    svc = RAGService(cfg)
    result = svc.retrieve("query")

    assert result == []
    assert RAG_RETRIEVAL_ERRORS._value.get() > before


def test_retrieve_reconnects_after_failure(monkeypatch, mock_sentence_transformer, rag_env):
    """First call with Chroma down → []. Re-patch Chroma to succeed → chunks returned."""
    import chromadb
    from unittest.mock import MagicMock
    from rag import RAGConfig, RAGService

    monkeypatch.setattr("chromadb.HttpClient", lambda **kwargs: (_ for _ in ()).throw(ConnectionError("down")))

    cfg = RAGConfig()
    svc = RAGService(cfg)
    assert svc.retrieve("query") == []

    # Now make Chroma reachable
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["reconnected doc"]],
        "metadatas": [[{"source": "x.txt"}]],
        "distances": [[0.1]],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    monkeypatch.setattr("chromadb.HttpClient", lambda **kwargs: mock_client)

    result = svc.retrieve("query")
    assert len(result) == 1
    assert result[0]["text"] == "reconnected doc"


# ---------------------------------------------------------------------------
# augment_messages
# ---------------------------------------------------------------------------


def test_augment_messages_with_chunks(mock_sentence_transformer, mock_chroma_client, rag_env):
    from rag import RAGConfig, RAGService

    cfg = RAGConfig()
    svc = RAGService(cfg)
    messages = [{"role": "user", "content": "hello"}]
    chunks = [
        {"text": "chunk alpha", "metadata": {}, "similarity": 0.9},
        {"text": "chunk beta", "metadata": {}, "similarity": 0.8},
    ]

    result = svc.augment_messages(messages, chunks)

    assert result[0]["role"] == "system"
    assert "chunk alpha" in result[0]["content"]
    assert "chunk beta" in result[0]["content"]
    assert result[-1]["role"] == "user"


def test_augment_messages_existing_system(mock_sentence_transformer, mock_chroma_client, rag_env):
    from rag import RAGConfig, RAGService

    cfg = RAGConfig()
    svc = RAGService(cfg)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]
    chunks = [{"text": "context chunk", "metadata": {}, "similarity": 0.9}]

    result = svc.augment_messages(messages, chunks)

    assert result[0]["role"] == "system"
    assert "context chunk" in result[0]["content"]
    assert "You are a helpful assistant." in result[0]["content"]
    # No duplicate system message
    assert sum(1 for m in result if m["role"] == "system") == 1


def test_augment_messages_no_chunks(mock_sentence_transformer, mock_chroma_client, rag_env):
    from rag import RAGConfig, RAGService

    cfg = RAGConfig()
    svc = RAGService(cfg)
    messages = [{"role": "user", "content": "hello"}]

    result = svc.augment_messages(messages, [])

    assert result is messages


# ---------------------------------------------------------------------------
# augment_prompt
# ---------------------------------------------------------------------------


def test_augment_prompt_with_chunks(mock_sentence_transformer, mock_chroma_client, rag_env):
    from rag import RAGConfig, RAGService

    cfg = RAGConfig()
    svc = RAGService(cfg)
    chunks = [{"text": "important context", "metadata": {}, "similarity": 0.9}]

    result = svc.augment_prompt("What is X?", chunks)

    assert "important context" in result
    assert "What is X?" in result
    assert result.index("important context") < result.index("What is X?")


def test_augment_prompt_no_chunks(mock_sentence_transformer, mock_chroma_client, rag_env):
    from rag import RAGConfig, RAGService

    cfg = RAGConfig()
    svc = RAGService(cfg)

    result = svc.augment_prompt("What is X?", [])

    assert result == "What is X?"


# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------


def test_retrieve_emits_prometheus_metrics(mock_sentence_transformer, mock_chroma_client, rag_env):
    from rag import RAGConfig, RAGService, RAG_RETRIEVAL_DURATION, RAG_CHUNKS_RETRIEVED

    before_duration = RAG_RETRIEVAL_DURATION._sum.get()
    before_chunks = RAG_CHUNKS_RETRIEVED._sum.get()

    cfg = RAGConfig()
    svc = RAGService(cfg)
    svc.retrieve("test query")

    assert RAG_RETRIEVAL_DURATION._sum.get() > before_duration
    assert RAG_CHUNKS_RETRIEVED._sum.get() >= before_chunks
