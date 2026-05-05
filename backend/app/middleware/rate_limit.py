"""Per-endpoint rate limiting backed by Valkey."""

from __future__ import annotations

from litestar.exceptions import ClientException

from app.task_manager import get_valkey


async def check_rate_limit(identifier: str, max_requests: int, window_seconds: int) -> None:
    """Increment counter for identifier and raise 429 if exceeded.

    Uses Valkey INCR + EXPIRE for a fixed window.
    """
    valkey = get_valkey()
    key = f"rate_limit:{identifier}"
    pipe = valkey.pipeline()
    pipe.incr(key)
    pipe.ttl(key)
    count, ttl = await pipe.execute()

    if count == 1 or ttl < 0:
        await valkey.expire(key, window_seconds)

    if count > max_requests:
        raise ClientException(
            status_code=429, detail="Rate limit exceeded. Please try again later."
        )
