import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    otlp_base: str
    metrics_endpoint: str
    traces_endpoint: str
    ollama_host: str
    ollama_model: str
    otel_service_name: str
    otel_deployment_environment: str
    rag_enabled: bool
    chroma_host: str
    chroma_port: int
    chroma_collection: str
    rag_top_k: int
    rag_embedding_model: str
    rag_min_similarity: float
    eval_enabled: bool
    eval_judge_host: str
    eval_judge_model: str
    eval_queue_max_size: int
    eval_sample_rate: float
    eval_db_path: str
    eval_timeout_seconds: float
    eval_faithfulness_hallucination_threshold: float
    eval_references_path: str | None


class EnvConfigHelper:
    @staticmethod
    def load_env_file() -> None:
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=repo_root / ".env", override=False)

    @staticmethod
    def read(name: str, default: str | None = None) -> str:
        value = os.getenv(name, "").strip()
        if value:
            return value
        if default is not None:
            return default
        raise ValueError(f"Required environment variable '{name}' is not set")

    @staticmethod
    def _normalize_otlp_base(raw: str) -> str:
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"OTEL_EXPORTER_OTLP_ENDPOINT '{raw}' is not a valid http/https URL"
            )
        return raw.rstrip("/")


class SettingsFactory:
    @staticmethod
    def from_env() -> Settings:
        EnvConfigHelper.load_env_file()

        otlp_base = EnvConfigHelper._normalize_otlp_base(
            EnvConfigHelper.read("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        )

        # If the per-signal env vars are absent or identical to the base URL,
        # derive the standard OTLP paths automatically.  This lets users set
        # only OTEL_EXPORTER_OTLP_ENDPOINT and get working defaults.
        def _signal_endpoint(env_var: str, suffix: str) -> str:
            val = os.getenv(env_var, "").strip().rstrip("/")
            return val if val and val != otlp_base else f"{otlp_base}{suffix}"

        raw_rag_enabled = EnvConfigHelper.read("RAG_ENABLED", "false").lower()
        raw_eval_enabled = EnvConfigHelper.read("EVAL_ENABLED", "false").lower()

        ollama_host = EnvConfigHelper.read("OLLAMA_HOST", "http://localhost:11434")
        ollama_model = EnvConfigHelper.read(
            "OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M"
        )

        eval_refs_raw = os.getenv("EVAL_REFERENCES_PATH", "").strip()

        return Settings(
            otlp_base=otlp_base,
            metrics_endpoint=_signal_endpoint(
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "/v1/metrics"
            ),
            traces_endpoint=_signal_endpoint(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "/v1/traces"
            ),
            ollama_host=ollama_host,
            ollama_model=ollama_model,
            otel_service_name=EnvConfigHelper.read("OTEL_SERVICE_NAME", "llmops-chat"),
            otel_deployment_environment=EnvConfigHelper.read(
                "OTEL_DEPLOYMENT_ENVIRONMENT", "dev"
            ),
            rag_enabled=raw_rag_enabled in ("true", "1", "yes"),
            chroma_host=EnvConfigHelper.read("CHROMA_HOST", "localhost"),
            chroma_port=int(EnvConfigHelper.read("CHROMA_PORT", "8000")),
            chroma_collection=EnvConfigHelper.read("CHROMA_COLLECTION", "documents"),
            rag_top_k=int(EnvConfigHelper.read("RAG_TOP_K", "5")),
            rag_embedding_model=EnvConfigHelper.read(
                "RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
            ),
            rag_min_similarity=float(EnvConfigHelper.read("RAG_MIN_SIMILARITY", "0.3")),
            eval_enabled=raw_eval_enabled in ("true", "1", "yes"),
            eval_judge_host=EnvConfigHelper.read("EVAL_JUDGE_HOST", ollama_host),
            eval_judge_model=EnvConfigHelper.read("EVAL_JUDGE_MODEL", ollama_model),
            eval_queue_max_size=int(EnvConfigHelper.read("EVAL_QUEUE_MAX_SIZE", "100")),
            eval_sample_rate=float(EnvConfigHelper.read("EVAL_SAMPLE_RATE", "1.0")),
            eval_db_path=EnvConfigHelper.read("EVAL_DB_PATH", "/data/evaluations.db"),
            eval_timeout_seconds=float(
                EnvConfigHelper.read("EVAL_TIMEOUT_SECONDS", "60.0")
            ),
            eval_faithfulness_hallucination_threshold=float(
                EnvConfigHelper.read(
                    "EVAL_FAITHFULNESS_HALLUCINATION_THRESHOLD", "0.5"
                )
            ),
            eval_references_path=eval_refs_raw if eval_refs_raw else None,
        )
