import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import api.security as security
from api.security import api_guard, clear_rate_limit_state
from config.settings import get_settings


def build_test_client() -> TestClient:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(api_guard)])
    def protected() -> dict[str, str]:
        return {"status": "ok"}

    return TestClient(app)


def reset_settings_cache() -> None:
    get_settings.cache_clear()
    clear_rate_limit_state()


@pytest.fixture(autouse=True)
def clean_security_state():
    reset_settings_cache()
    yield
    reset_settings_cache()


def test_api_guard_allows_local_requests_when_auth_is_disabled(monkeypatch) -> None:
    monkeypatch.delenv("API_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    reset_settings_cache()

    client = build_test_client()

    response = client.get("/protected")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_api_guard_requires_valid_api_key_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("API_AUTH_ENABLED", "true")
    monkeypatch.setenv("API_KEY", "test-secret")
    reset_settings_cache()

    client = build_test_client()

    assert client.get("/protected").status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/protected", headers={"X-API-Key": "test-secret"}).status_code == 200


def test_api_guard_rate_limits_by_client_and_path(monkeypatch) -> None:
    monkeypatch.delenv("API_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "2")
    reset_settings_cache()

    client = build_test_client()

    assert client.get("/protected").status_code == 200
    assert client.get("/protected").status_code == 200
    assert client.get("/protected").status_code == 429


def test_api_guard_uses_redis_rate_limit_when_configured(monkeypatch) -> None:
    class FakeRedisClient:
        def __init__(self) -> None:
            self.results = [1, 1, 0]
            self.calls: list[tuple[str, int, int]] = []

        def eval(
            self,
            script: str,
            number_of_keys: int,
            key: str,
            window_milliseconds: int,
            limit: int,
            member: str,
        ) -> int:
            self.calls.append((key, window_milliseconds, limit))
            return self.results.pop(0)

    redis_client = FakeRedisClient()
    monkeypatch.delenv("API_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(security, "_get_redis_client", lambda redis_url: redis_client)
    reset_settings_cache()

    client = build_test_client()

    assert client.get("/protected").status_code == 200
    assert client.get("/protected").status_code == 200
    assert client.get("/protected").status_code == 429
    assert len(redis_client.calls) == 3
    assert redis_client.calls[0][1:] == (60_000, 2)


def test_api_guard_falls_back_to_memory_when_redis_fails(monkeypatch) -> None:
    class FailingRedisClient:
        def eval(
            self,
            script: str,
            number_of_keys: int,
            key: str,
            window_milliseconds: int,
            limit: int,
            member: str,
        ) -> int:
            raise security.RedisError("connection unavailable")

    monkeypatch.delenv("API_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setenv("API_RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(security, "_get_redis_client", lambda redis_url: FailingRedisClient())
    reset_settings_cache()

    client = build_test_client()

    assert client.get("/protected").status_code == 200
    assert client.get("/protected").status_code == 429
