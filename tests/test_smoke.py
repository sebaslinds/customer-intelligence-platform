from config.settings import get_settings


def test_settings_loads() -> None:
    settings = get_settings()

    assert settings.app_name == "customer-intelligence-platform"
    assert settings.api_port == 8000
