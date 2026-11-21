"""API tests for /api/evaluate with the RAG pipeline stubbed out."""

import pytest
from fastapi.testclient import TestClient

from backend import main
from backend.main import MAX_EVALUATION_CASES, app
from backend.rag import KnowledgeBaseError, MissingOpenAIAPIKeyError

client = TestClient(app)


def _case(n: int) -> dict:
    return {"question": f"Question {n}?", "expected_answer": f"Answer {n}."}


def _score(question: str, expected_answer: str) -> dict:
    return {
        "question": question,
        "expected_answer": expected_answer,
        "generated_answer": "An answer [S1].",
        "retrieved_chunks": [],
        "retrieval_relevance": 1.0,
        "groundedness": 1.0,
        "answer_correctness": 1.0,
        "citation_validity": 1.0,
        "overall_score": 1.0,
        "passed": True,
        "explanations": {},
    }


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_one_failed_case_does_not_discard_the_others(monkeypatch):
    def fake_evaluate(question, expected_answer):
        if question == "Question 2?":
            raise KnowledgeBaseError("OpenAI could not evaluate this case.")
        return _score(question, expected_answer)

    monkeypatch.setattr(main, "evaluate_case", fake_evaluate)
    response = client.post("/api/evaluate", json=[_case(1), _case(2), _case(3)])

    assert response.status_code == 200
    rows = response.json()
    assert [row["passed"] for row in rows] == [True, False, True]
    assert rows[1]["error"] == "OpenAI could not evaluate this case."
    assert rows[1]["overall_score"] == 0
    assert rows[0]["error"] is None


def test_all_cases_failing_fails_the_request(monkeypatch):
    def fake_evaluate(question, expected_answer):
        raise KnowledgeBaseError("No knowledge base exists. Ingest a website first.")

    monkeypatch.setattr(main, "evaluate_case", fake_evaluate)
    response = client.post("/api/evaluate", json=[_case(1), _case(2)])

    assert response.status_code == 502
    assert "Ingest a website first" in response.json()["detail"]


def test_missing_api_key_fails_the_whole_batch(monkeypatch):
    def fake_evaluate(question, expected_answer):
        raise MissingOpenAIAPIKeyError("OPENAI_API_KEY is not configured on the backend.")

    monkeypatch.setattr(main, "evaluate_case", fake_evaluate)
    response = client.post("/api/evaluate", json=[_case(1)])

    assert response.status_code == 503


@pytest.mark.parametrize("count", [0, MAX_EVALUATION_CASES + 1])
def test_case_count_limits(monkeypatch, count):
    monkeypatch.setattr(main, "evaluate_case", _score)
    response = client.post("/api/evaluate", json=[_case(n) for n in range(count)])
    assert response.status_code == 422


def test_max_cases_allowed(monkeypatch):
    monkeypatch.setattr(main, "evaluate_case", _score)
    response = client.post(
        "/api/evaluate",
        json=[_case(n) for n in range(MAX_EVALUATION_CASES)],
    )
    assert response.status_code == 200
    assert len(response.json()) == MAX_EVALUATION_CASES
