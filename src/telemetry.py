import os

import openlit

from settings import Settings


class OpenTelemetryHelper:
    _FIXED_DEFAULTS = {
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "none",
    }

    @classmethod
    def configure(cls, settings: Settings) -> None:
        # Set signal endpoints from Settings (already derived/validated from env).
        os.environ.update(
            {
                "OTEL_EXPORTER_OTLP_ENDPOINT": settings.otlp_base,
                "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": settings.metrics_endpoint,
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": settings.traces_endpoint,
                "OTEL_SERVICE_NAME": settings.otel_service_name,
                "OTEL_DEPLOYMENT_ENVIRONMENT": settings.otel_deployment_environment,
            }
        )
        # Apply protocol/exporter defaults only if not already in the environment
        # so that callers can override them without touching Settings.
        for key, value in cls._FIXED_DEFAULTS.items():
            os.environ.setdefault(key, value)

    @staticmethod
    def init_openlit() -> None:
        openlit.init()
