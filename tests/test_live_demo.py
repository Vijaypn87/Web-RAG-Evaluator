"""Tests for the hosted-demo pieces: Gemini provider, request limits, /api/info."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import RateLimitError

from backend import limits, main, rag
from backend.rag import KnowledgeBaseError, MissingOpenAIAPIKeyError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "PAGE_FETCHER", "GEMINI_API_KEY", "GEMINI_BASE_URL",
        "GEMINI_CHAT_MODEL", "GEMINI_EMBEDDING_MODEL", "GEMINI_EVALUATOR_MODEL",
        "CONTEXT_DEV_API_KEY", "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


# Gemini provider
def test_gemini_requires_its_own_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(MissingOpenAIAPIKeyError, match="GEMINI_API_KEY"):
        rag._openai_client()


def test_gemini_client_uses_google_endpoint(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = rag._openai_client()
    assert "generativelanguage.googleapis.com" in str(client.base_url)


def test_gemini_model_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    settings = rag.model_settings()
    assert settings["chat_model"] == "gemini-flash-lite-latest"
    assert settings["embedding_model"] == "gemini-embedding-001"
    assert settings["evaluator_model"] == "gemini-flash-lite-latest"
    monkeypatch.setenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
    assert rag.model_settings()["evaluator_model"] == "gemini-3.5-flash-lite"


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.content, Exception):
            raise self.content
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _fake_client(monkeypatch, content):
    completions = FakeCompletions(content)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(rag, "_openai_client", lambda: client)
    return completions


def test_gemini_answer_uses_chat_completions_without_seed(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    completions = _fake_client(monkeypatch, "Paris [S1].")
    assert rag._generate_answer("x") == "Paris [S1]."
    call = completions.calls[0]
    assert call["model"] == "gemini-flash-lite-latest"
    assert call["temperature"] == 0
    assert "seed" not in call


def test_gemini_evaluation_uses_relaxed_schema_and_strips_fences(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    metric = {"score": 0.8, "explanation": "Good."}
    payload = json.dumps({n: metric for n in ("retrieval_relevance", "groundedness", "answer_correctness")})
    completions = _fake_client(monkeypatch, f"```json\n{payload}\n```")
    result = rag._generate_evaluation("data")
    assert result.answer_correctness.score == 0.8
    schema_text = json.dumps(completions.calls[0]["response_format"])
    assert "additionalProperties" not in schema_text
    assert "minimum" not in schema_text


def test_gemini_quota_error_is_friendly(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    error = RateLimitError("quota", response=httpx.Response(429, request=request), body=None)
    _fake_client(monkeypatch, error)
    with pytest.raises(KnowledgeBaseError, match="free usage limit"):
        rag._generate_answer("x")


# /api/info
def test_info_endpoint_reports_active_setup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("PAGE_FETCHER", "local")
    data = TestClient(main.app).get("/api/info").json()
    assert data["llm_provider"] == "gemini"
    assert data["page_fetcher"] == "local"
    assert "api_key" not in json.dumps(data).lower()


# Request limits
class FakeClock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now


def _request(ip="1.2.3.4", forwarded=None):
    headers = [(b"x-forwarded-for", forwarded.encode())] if forwarded else []
    scope = {"type": "http", "headers": headers, "client": (ip, 1234)}
    from starlette.requests import Request

    return Request(scope)


def test_client_ip_uses_rightmost_forwarded_entry():
    assert limits.client_ip(_request(forwarded="6.6.6.6, 9.9.9.9")) == "9.9.9.9"
    assert limits.client_ip(_request(ip="5.5.5.5")) == "5.5.5.5"


def test_per_ip_hourly_limit_and_reset(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "3")
    monkeypatch.setenv("DAILY_LIMIT", "0")
    clock = FakeClock()
    limiter = limits.RequestLimiter(clock=clock)

    for _ in range(3):
        limiter.check(_request())
    with pytest.raises(Exception) as excinfo:
        limiter.check(_request())
    assert excinfo.value.status_code == 429
    assert int(excinfo.value.headers["Retry-After"]) > 0

    limiter.check(_request(ip="8.8.8.8"))  # another visitor is unaffected
    clock.now += limits.HOUR + 1
    limiter.check(_request())  # the window has rolled over


def test_evaluation_cost_counts_cases(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "5")
    monkeypatch.setenv("DAILY_LIMIT", "0")
    limiter = limits.RequestLimiter(clock=FakeClock())
    limiter.check(_request(), cost=4)
    with pytest.raises(Exception) as excinfo:
        limiter.check(_request(), cost=2)
    assert excinfo.value.status_code == 429


def test_daily_limit_applies_across_visitors(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "0")
    monkeypatch.setenv("DAILY_LIMIT", "2")
    limiter = limits.RequestLimiter(clock=FakeClock())
    limiter.check(_request(ip="1.1.1.1"))
    limiter.check(_request(ip="2.2.2.2"))
    with pytest.raises(Exception) as excinfo:
        limiter.check(_request(ip="3.3.3.3"))
    assert "daily" in excinfo.value.detail


def test_api_returns_429_when_limited(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "1")
    monkeypatch.setattr(main, "answer_question", lambda q: {"answer": "a", "citations": [], "retrieved_chunks": []})
    client = TestClient(main.app)
    assert client.post("/api/ask", json={"question": "q"}).status_code == 200
    limited = client.post("/api/ask", json={"question": "q"})
    assert limited.status_code == 429
    assert "hourly limit" in limited.json()["detail"]
