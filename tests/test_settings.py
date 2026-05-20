"""Tests for SettingsFactory with environment variable manipulation."""

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_rag_vars(monkeypatch):
    for var in [
        "RAG_ENABLED", "CHROMA_HOST", "CHROMA_PORT", "CHROMA_COLLECTION",
        "RAG_TOP_K", "RAG_EMBEDDING_MODEL", "RAG_MIN_SIMILARITY",
    ]:
        monkeypatch.delenv(var, raising=False)


def _no_dotenv(monkeypatch):
    """Prevent load_env_file() from loading .env so test env vars are authoritative."""
    monkeypatch.setattr("settings.EnvConfigHelper.load_env_file", lambda: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_settings_defaults(monkeypatch):
    _no_dotenv(monkeypatch)
    _clear_rag_vars(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.delenv("OTEL_DEPLOYMENT_ENVIRONMENT", raising=False)

    from settings import SettingsFactory

    s = SettingsFactory.from_env()

    assert s.otlp_base == "http://localhost:4318"
    assert s.ollama_host == "http://localhost:11434"
    assert s.rag_enabled is False
    assert s.rag_top_k == 5
    assert s.rag_min_similarity == pytest.approx(0.3)
    assert s.chroma_host == "localhost"
    assert s.chroma_port == 8000
    assert s.chroma_collection == "documents"


def test_settings_rag_fields(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("RAG_ENABLED", "true")
    monkeypatch.setenv("CHROMA_HOST", "chroma-svc")
    monkeypatch.setenv("CHROMA_PORT", "9100")
    monkeypatch.setenv("CHROMA_COLLECTION", "my-col")
    monkeypatch.setenv("RAG_TOP_K", "7")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "custom-embed")
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.5")

    from settings import SettingsFactory

    s = SettingsFactory.from_env()

    assert s.rag_enabled is True
    assert s.chroma_host == "chroma-svc"
    assert s.chroma_port == 9100
    assert s.chroma_collection == "my-col"
    assert s.rag_top_k == 7
    assert s.rag_embedding_model == "custom-embed"
    assert s.rag_min_similarity == pytest.approx(0.5)


def test_settings_rag_disabled_by_default(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("RAG_ENABLED", raising=False)

    from settings import SettingsFactory

    s = SettingsFactory.from_env()
    assert s.rag_enabled is False


def test_settings_otlp_validation(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "not-a-url")

    from settings import SettingsFactory

    with pytest.raises(ValueError, match="not a valid http/https URL"):
        SettingsFactory.from_env()


def test_settings_derived_endpoints(monkeypatch):
    _no_dotenv(monkeypatch)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)

    from settings import SettingsFactory

    s = SettingsFactory.from_env()

    assert s.metrics_endpoint == "http://otel-collector:4318/v1/metrics"
    assert s.traces_endpoint == "http://otel-collector:4318/v1/traces"
