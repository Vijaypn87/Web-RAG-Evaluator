"""Tests for the free mode: Ollama provider and the local page fetcher.

No network calls: model clients and HTTP pools are replaced with fakes, and
DNS lookups are patched.
"""

import json
import socket
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from backend import rag
from backend.rag import KnowledgeBaseError, RAGInputError, WebsiteIngestionError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in (
        "LLM_PROVIDER", "PAGE_FETCHER", "OPENAI_API_KEY", "CONTEXT_DEV_API_KEY",
        "OLLAMA_BASE_URL", "OLLAMA_CHAT_MODEL", "OLLAMA_EMBEDDING_MODEL",
        "OLLAMA_EVALUATOR_MODEL", "OPENAI_CHAT_MODEL", "OPENAI_EMBEDDING_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


# Provider selection
def test_provider_defaults_to_openai():
    assert rag.llm_provider() == "openai"


def test_invalid_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RAGInputError):
        rag.llm_provider()


def test_ollama_needs_no_api_key_and_uses_local_url(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Ollama")
    client = rag._openai_client()
    assert "localhost:11434" in str(client.base_url)


def test_openai_still_requires_a_key():
    with pytest.raises(rag.MissingOpenAIAPIKeyError):
        rag._openai_client()


def test_ollama_model_defaults_and_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    settings = rag.model_settings()
    assert settings["chat_model"] == "llama3.2"
    assert settings["evaluator_model"] == "llama3.2"
    assert settings["embedding_model"] == "nomic-embed-text"
    monkeypatch.setenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    assert rag.model_settings()["evaluator_model"] == "qwen2.5:7b"


# Page fetcher selection
def test_page_fetcher_is_local_without_context_dev_key():
    assert rag._page_fetcher() == "local"


def test_page_fetcher_uses_context_dev_when_keyed(monkeypatch):
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "x")
    assert rag._page_fetcher() == "contextdev"
    monkeypatch.setenv("PAGE_FETCHER", "local")
    assert rag._page_fetcher() == "local"


def test_invalid_page_fetcher(monkeypatch):
    monkeypatch.setenv("PAGE_FETCHER", "selenium")
    with pytest.raises(RAGInputError):
        rag._page_fetcher()


# Ollama model calls
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


def test_ollama_answer_uses_chat_completions_deterministically(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    completions = _fake_client(monkeypatch, "  Paris [S1].  ")
    assert rag._generate_answer("Source passages: ...") == "Paris [S1]."
    call = completions.calls[0]
    assert call["model"] == "llama3.2"
    assert call["temperature"] == 0
    assert call["messages"][0] == {"role": "system", "content": rag.ANSWER_INSTRUCTIONS}


def test_ollama_empty_answer_is_an_error(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _fake_client(monkeypatch, "")
    with pytest.raises(KnowledgeBaseError, match="Ollama returned an empty answer"):
        rag._generate_answer("x")


def _evaluation_json(explanation="Fine."):
    metric = {"score": 0.5, "explanation": explanation}
    return json.dumps({
        "retrieval_relevance": metric,
        "groundedness": metric,
        "answer_correctness": metric,
    })


def test_ollama_evaluation_parses_json_and_sends_schema(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    completions = _fake_client(monkeypatch, _evaluation_json())
    result = rag._generate_evaluation("data")
    assert result.groundedness.score == 0.5
    schema = completions.calls[0]["response_format"]["json_schema"]["schema"]
    assert "$defs" not in json.dumps(schema)
    assert set(schema["required"]) == {"retrieval_relevance", "groundedness", "answer_correctness"}


def test_long_explanations_are_trimmed_not_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _fake_client(monkeypatch, _evaluation_json("word " * 100))
    result = rag._generate_evaluation("data")
    assert len(result.retrieval_relevance.explanation) <= 180


@pytest.mark.parametrize("content", ["not json", json.dumps({"groundedness": 1})])
def test_bad_evaluator_output_is_a_clear_error(monkeypatch, content):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    _fake_client(monkeypatch, content)
    with pytest.raises(KnowledgeBaseError, match="expected format"):
        rag._generate_evaluation("data")


def test_ollama_not_running_gives_helpful_message(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    error = APIConnectionError(request=httpx.Request("POST", "http://localhost:11434/v1"))
    _fake_client(monkeypatch, error)
    with pytest.raises(KnowledgeBaseError, match="Make sure the Ollama app is running"):
        rag._generate_answer("x")


# Local page fetcher
class FakeResponse:
    def __init__(self, status=200, headers=None, body=b""):
        self.status = status
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self.body = body

    def stream(self, size):
        for start in range(0, len(self.body), size):
            yield self.body[start : start + size]

    def release_conn(self):
        pass


class FakePool:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def request(self, method, url, redirect, preload_content):
        assert redirect is False
        self.urls.append(url)
        return self.responses[url]


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every host to a public address unless it is listed as private."""
    private = {"internal.example": "10.0.0.7"}

    def fake_getaddrinfo(host, port, type=0):
        ip = private.get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


PAGE_HTML = b"""<html><head><title>Test Page</title></head><body>
<nav>Menu Home About</nav>
<article><h1>Test Page</h1>
<p>The answer to the question is forty-two, according to this article.<sup><a href="#cite_note-1">[1]</a></sup></p>
<p>A second paragraph adds more detail so the extractor keeps the article body.</p>
</article><footer>Copyright</footer></body></html>"""


def test_local_fetch_extracts_main_content(public_dns):
    pool = FakePool({"https://site.example/a": FakeResponse(body=PAGE_HTML)})
    page = rag._fetch_with_trafilatura("https://site.example/a", pool=pool)
    assert "forty-two" in page["markdown"]
    assert "<sup>" not in page["markdown"]
    assert page["metadata"]["title"] == "Test Page"
    assert page["url"] == "https://site.example/a"


def test_local_fetch_follows_public_redirects(public_dns):
    pool = FakePool({
        "https://site.example/old": FakeResponse(301, {"location": "/new"}),
        "https://site.example/new": FakeResponse(body=PAGE_HTML),
    })
    final_url, html = rag._download_html("https://site.example/old", pool=pool)
    assert final_url == "https://site.example/new"
    assert "forty-two" in html


@pytest.mark.parametrize(
    "location",
    ["http://127.0.0.1/admin", "http://localhost:8000/", "http://internal.example/"],
)
def test_local_fetch_blocks_redirects_to_private_hosts(public_dns, location):
    pool = FakePool({"https://site.example/r": FakeResponse(302, {"location": location})})
    with pytest.raises(RAGInputError):
        rag._download_html("https://site.example/r", pool=pool)
    assert pool.urls == ["https://site.example/r"]


def test_local_fetch_blocks_hosts_resolving_to_private_ips(public_dns):
    with pytest.raises(RAGInputError):
        rag._download_html("http://internal.example/", pool=FakePool({}))


def test_local_fetch_redirect_loop(public_dns):
    pool = FakePool({"https://site.example/loop": FakeResponse(302, {"location": "/loop"})})
    with pytest.raises(WebsiteIngestionError, match="too many times"):
        rag._download_html("https://site.example/loop", pool=pool)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (FakeResponse(404), "HTTP 404"),
        (FakeResponse(200, {"content-type": "application/pdf"}, b"%PDF"), "not an HTML"),
        (FakeResponse(200, body=b"x" * (rag.LOCAL_FETCH_MAX_BYTES + 1)), "too large"),
    ],
)
def test_local_fetch_rejects_bad_responses(public_dns, response, message):
    pool = FakePool({"https://site.example/p": response})
    with pytest.raises(WebsiteIngestionError, match=message):
        rag._download_html("https://site.example/p", pool=pool)


def test_clean_extracted_markdown_removes_footnotes():
    text = "A fact.[\\[1\\]](https://w.example#cite_note-1)<sup>[\\[2\\]](https://w.example#n2)</sup> More \\[3\\] text."
    assert rag._clean_extracted_markdown(text) == "A fact. More  text."
