"""In-process token-bucket rate limiting.

Good for a single API instance; with several instances behind a load balancer, put the same
limits in the edge (Cloud Armor / Vercel firewall) or swap the store for Redis.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock

from fastapi import HTTPException, Request, status


class TokenBucket:
    def __init__(self, capacity: int, per_seconds: float, clock: Callable[[], float] = time.monotonic):
        self.capacity, self.rate, self.clock = capacity, capacity / per_seconds, clock
        self._state: dict[str, tuple[float, float]] = {}
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = self.clock()
        with self._lock:
            tokens, last = self._state.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            if tokens < 1:
                self._state[key] = (tokens, now)
                return False
            self._state[key] = (tokens - 1, now)
            if len(self._state) > 100_000:  # bound memory under abuse
                self._state.clear()
            return True


def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


_BUCKETS: list[TokenBucket] = []


def reset_all() -> None:
    """Forget all counters (tests, or after a config change)."""
    for b in _BUCKETS:
        with b._lock:
            b._state.clear()


def limiter(name: str, capacity: int, per_seconds: float):
    bucket = TokenBucket(capacity, per_seconds)
    _BUCKETS.append(bucket)

    def dep(request: Request) -> None:
        user_cookie = request.cookies.get("ic_session", "")[:16]
        if not bucket.allow(f"{name}:{client_ip(request)}:{user_cookie}"):
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "too many requests, slow down",
                headers={"Retry-After": str(int(per_seconds / capacity) + 1)},
            )

    dep.bucket = bucket  # type: ignore[attr-defined]
    return dep
