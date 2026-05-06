import os

import openlit

from settings import Settings


class OpenTelemetryHelper:
    _DEFAULTS = {
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_SERVICE_NAME": "llmops-chat",
        "OTEL_DEPLOYMENT_ENVIRONMENT": "dev",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "none",
    }

    @classmethod
    def configure(cls, settings: Settings) -> None:
        os.environ.update(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": settings.otlp_base,
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": settings.metrics_endpoint,
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": settings.traces_endpoint,
            }
        )
        for key, value in cls._DEFAULTS.items():
            os.environ.setdefault(key, value)

    @staticmethod
    def init_openlit() -> None:
        openlit.init(detailed_tracing=False)
