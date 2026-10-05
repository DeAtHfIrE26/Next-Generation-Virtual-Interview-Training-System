"""In-process token-bucket rate limiting.

Good for a single API instance; with several instances behind a load balancer, put the same
limits in the edge (Cloud Armor / Vercel firewall) or swap the store for Redis.
"""

from __future__ import annotations

import os
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
    """Client address for rate limiting.

    ``X-Forwarded-For`` is attacker-controlled except for the entries appended by our own
    proxies, so it is used only when ``TRUSTED_PROXY_HOPS`` says how many trusted proxies sit in
    front of the API (for Vercel -> Cloud Run: 2). Otherwise the socket peer is used.
    """
    hops = int(os.getenv("TRUSTED_PROXY_HOPS", "0") or 0)
    fwd = request.headers.get("x-forwarded-for", "")
    if hops > 0 and fwd:
        parts = [p.strip() for p in fwd.split(",") if p.strip()]
        if parts:
            return parts[-hops] if len(parts) >= hops else parts[0]
    return request.client.host if request.client else "unknown"


_BUCKETS: list[TokenBucket] = []


def reset_all() -> None:
    """Forget all counters (tests, or after a config change)."""
    for b in _BUCKETS:
        with b._lock:
            b._state.clear()


def limiter(name: str, capacity: int, per_seconds: float, *, per_session: bool = True):
    """``per_session=False`` keys on IP only (unauthenticated routes, where a client could
    otherwise mint fresh buckets by sending random cookies). ``RATE_LIMIT_MULTIPLIER`` scales
    every limit (for load tests and staging)."""
    capacity = max(1, round(capacity * float(os.getenv("RATE_LIMIT_MULTIPLIER", "1") or 1)))
    bucket = TokenBucket(capacity, per_seconds)
    _BUCKETS.append(bucket)

    def dep(request: Request) -> None:
        who = request.cookies.get("ic_session", "")[:16] if per_session else ""
        if not bucket.allow(f"{name}:{client_ip(request)}:{who}"):
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "too many requests, slow down",
                headers={"Retry-After": str(int(per_seconds / capacity) + 1)},
            )

    dep.bucket = bucket  # type: ignore[attr-defined]
    return dep
