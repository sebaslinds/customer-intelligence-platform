from collections import defaultdict, deque
from time import monotonic

from fastapi import Header, HTTPException, Request, status

from config.settings import get_settings

RATE_LIMIT_WINDOW_SECONDS = 60

_request_windows: dict[str, deque[float]] = defaultdict(deque)


def clear_rate_limit_state() -> None:
    _request_windows.clear()


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    settings = get_settings()
    auth_is_required = settings.api_auth_enabled or bool(settings.api_key)

    if not auth_is_required:
        return

    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is enabled, but API_KEY is not configured.",
        )

    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


def enforce_rate_limit(request: Request) -> None:
    settings = get_settings()
    limit = settings.api_rate_limit_per_minute

    if limit <= 0:
        return

    client_host = request.client.host if request.client else "unknown"
    key = f"{client_host}:{request.url.path}"
    now = monotonic()
    window = _request_windows[key]

    while window and now - window[0] >= RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )

    window.append(now)


def api_guard(request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    verify_api_key(x_api_key)
    enforce_rate_limit(request)
