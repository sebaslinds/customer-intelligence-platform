from collections import defaultdict, deque
from hashlib import sha256
from logging import getLogger
from time import monotonic
from typing import Any
from uuid import uuid4

from fastapi import Header, HTTPException, Request, status

from config.settings import get_settings

try:
    from redis import Redis
    from redis.exceptions import RedisError
except ImportError:
    Redis = None
    RedisError = Exception

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_WINDOW_MILLISECONDS = RATE_LIMIT_WINDOW_SECONDS * 1000

logger = getLogger(__name__)

_request_windows: dict[str, deque[float]] = defaultdict(deque)
_redis_client: Any | None = None
_redis_client_url: str | None = None
_redis_warning_logged = False

_REDIS_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local window_ms = tonumber(ARGV[1])
local limit = tonumber(ARGV[2])
local member = ARGV[3]
local current_time = redis.call("TIME")
local now_ms = current_time[1] * 1000 + math.floor(current_time[2] / 1000)

redis.call("ZREMRANGEBYSCORE", key, 0, now_ms - window_ms)

local request_count = tonumber(redis.call("ZCARD", key))
if request_count >= limit then
    redis.call("PEXPIRE", key, window_ms)
    return 0
end

redis.call("ZADD", key, now_ms, member)
redis.call("PEXPIRE", key, window_ms)
return 1
"""


def clear_rate_limit_state() -> None:
    global _redis_client, _redis_client_url, _redis_warning_logged

    _request_windows.clear()
    if _redis_client is not None:
        try:
            _redis_client.close()
        except (AttributeError, RedisError):
            pass
    _redis_client = None
    _redis_client_url = None
    _redis_warning_logged = False


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

    if settings.redis_url and _enforce_redis_rate_limit(request, settings.redis_url, settings.redis_rate_limit_prefix, limit):
        return

    _enforce_local_rate_limit(request, limit)


def _enforce_local_rate_limit(request: Request, limit: int) -> None:
    client_host = request.client.host if request.client else "unknown"
    key = f"{client_host}:{request.url.path}"
    now = monotonic()
    window = _request_windows[key]

    while window and now - window[0] >= RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    if len(window) >= limit:
        _raise_rate_limit_exceeded()

    window.append(now)


def _enforce_redis_rate_limit(request: Request, redis_url: str, key_prefix: str, limit: int) -> bool:
    redis_client = _get_redis_client(redis_url)
    if redis_client is None:
        return False

    raw_key = f"{request.client.host if request.client else 'unknown'}:{request.url.path}"
    key_digest = sha256(raw_key.encode("utf-8")).hexdigest()
    rate_limit_key = f"{key_prefix}:{key_digest}"
    member = uuid4().hex

    try:
        result = redis_client.eval(
            _REDIS_RATE_LIMIT_SCRIPT,
            1,
            rate_limit_key,
            RATE_LIMIT_WINDOW_MILLISECONDS,
            limit,
            member,
        )
    except RedisError as error:
        _warn_redis_fallback(error)
        return False

    if int(result) == 0:
        _raise_rate_limit_exceeded()

    return True


def _get_redis_client(redis_url: str) -> Any | None:
    global _redis_client, _redis_client_url

    if Redis is None:
        _warn_redis_fallback()
        return None

    if _redis_client is None or _redis_client_url != redis_url:
        if _redis_client is not None:
            try:
                _redis_client.close()
            except (AttributeError, RedisError):
                pass
        _redis_client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        _redis_client_url = redis_url

    return _redis_client


def _warn_redis_fallback(error: Exception | None = None) -> None:
    global _redis_warning_logged

    if _redis_warning_logged:
        return

    if error:
        logger.warning("Redis rate limiting unavailable; falling back to in-memory limits: %s", error)
    else:
        logger.warning("Redis URL configured but redis package is unavailable; falling back to in-memory limits.")
    _redis_warning_logged = True


def _raise_rate_limit_exceeded() -> None:
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail="Rate limit exceeded. Try again later.",
    )


def api_guard(request: Request, x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    verify_api_key(x_api_key)
    enforce_rate_limit(request)
