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

        return Settings(
            otlp_base=otlp_base,
            metrics_endpoint=_signal_endpoint(
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "/v1/metrics"
            ),
            traces_endpoint=_signal_endpoint(
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "/v1/traces"
            ),
            ollama_host=EnvConfigHelper.read("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=EnvConfigHelper.read(
                "OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M"
            ),
            otel_service_name=EnvConfigHelper.read("OTEL_SERVICE_NAME", "llmops-chat"),
            otel_deployment_environment=EnvConfigHelper.read(
                "OTEL_DEPLOYMENT_ENVIRONMENT", "dev"
            ),
        )
