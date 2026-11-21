"""In-memory request limits that protect a public demo's free API quota.

Two limits apply to the endpoints that call the model provider:

* RATE_LIMIT_PER_HOUR: units per client IP in any rolling hour (default 30).
* DAILY_LIMIT: units for all clients together in any rolling 24 hours
  (default 500), so many visitors cannot exhaust a free daily quota.

Ingesting a page or asking a question costs 1 unit; an evaluation costs
1 unit per case. Set a limit to 0 to turn it off. Limits are kept in memory,
so they reset when the server restarts, which is fine for a small demo.
"""

import math
import os
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

HOUR = 3600
DAY = 24 * HOUR


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def client_ip(request: Request) -> str:
    """Best-effort client address.

    Behind a proxy such as Render's, the proxy appends the real client
    address to X-Forwarded-For, so the right-most entry is the one the proxy
    saw; entries further left can be set by the client and are ignored.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded.strip():
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


class RequestLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._per_ip: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
        self._all: deque[tuple[float, int]] = deque()

    @staticmethod
    def _prune(events: deque[tuple[float, int]], now: float, window: int) -> int:
        while events and events[0][0] <= now - window:
            events.popleft()
        return sum(cost for _, cost in events)

    @staticmethod
    def _retry_after(events: deque[tuple[float, int]], now: float, window: int) -> int:
        return max(1, math.ceil(events[0][0] + window - now)) if events else 1

    def check(self, request: Request, cost: int = 1) -> None:
        """Record a request or raise HTTP 429 if it would exceed a limit."""
        per_hour = _env_int("RATE_LIMIT_PER_HOUR", 30)
        per_day = _env_int("DAILY_LIMIT", 500)
        ip = client_ip(request)
        now = self._clock()

        with self._lock:
            ip_events = self._per_ip[ip]
            if per_hour and self._prune(ip_events, now, HOUR) + cost > per_hour:
                retry = self._retry_after(ip_events, now, HOUR)
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "You have reached this demo's hourly limit. "
                        f"Please try again in about {math.ceil(retry / 60)} minute(s)."
                    ),
                    headers={"Retry-After": str(retry)},
                )
            if per_day and self._prune(self._all, now, DAY) + cost > per_day:
                retry = self._retry_after(self._all, now, DAY)
                raise HTTPException(
                    status_code=429,
                    detail="This demo has reached its daily usage limit. Please try again tomorrow.",
                    headers={"Retry-After": str(retry)},
                )
            ip_events.append((now, cost))
            self._all.append((now, cost))


limiter = RequestLimiter()
