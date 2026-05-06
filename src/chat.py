from chat_app import ChatApplication
from settings import SettingsFactory
from telemetry import OpenTelemetryHelper


def main() -> None:
    settings = SettingsFactory.from_env()
    OpenTelemetryHelper.configure(settings)
    OpenTelemetryHelper.init_openlit()
    ChatApplication(settings).run()


if __name__ == "__main__":
    main()