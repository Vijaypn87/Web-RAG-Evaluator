"""Shared test setup."""

import pytest

from backend import limits


@pytest.fixture(autouse=True)
def no_request_limits(monkeypatch):
    """Turn the public-demo request limits off unless a test enables them."""
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "0")
    monkeypatch.setenv("DAILY_LIMIT", "0")
    monkeypatch.setattr(limits, "limiter", limits.RequestLimiter())
    from backend import main

    monkeypatch.setattr(main, "limiter", limits.limiter)
