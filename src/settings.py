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


class EnvConfigHelper:
    @staticmethod
    def load_env_file() -> None:
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(dotenv_path=repo_root / ".env", override=False)

    @staticmethod
    def read(name: str, default: str | None = None) -> str:
        value = os.getenv(name)
        if value is None:
            return "" if default is None else default
        value = value.strip()
        if not value and default is not None:
            return default
        return value

    @staticmethod
    def normalize_otlp_base(raw_endpoint: str) -> str:
        parsed = urlparse(raw_endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "http://localhost:4318"
        return raw_endpoint.rstrip("/")


class SettingsFactory:
    @staticmethod
    def from_env() -> Settings:
        EnvConfigHelper.load_env_file()

        otlp_base = EnvConfigHelper.normalize_otlp_base(
            EnvConfigHelper.read("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
        )
        metrics_endpoint = EnvConfigHelper.read("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", f"{otlp_base}/v1/metrics")
        traces_endpoint = EnvConfigHelper.read("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{otlp_base}/v1/traces")

        if metrics_endpoint.rstrip("/") == otlp_base:
            metrics_endpoint = f"{otlp_base}/v1/metrics"

        if traces_endpoint.rstrip("/") == otlp_base:
            traces_endpoint = f"{otlp_base}/v1/traces"

        return Settings(
            otlp_base=otlp_base,
            metrics_endpoint=metrics_endpoint,
            traces_endpoint=traces_endpoint,
            ollama_host=EnvConfigHelper.read("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=EnvConfigHelper.read("OLLAMA_MODEL", "ministral-3:8b-instruct-2512-q4_K_M"),
        )
