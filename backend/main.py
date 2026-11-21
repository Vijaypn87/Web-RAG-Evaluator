"""FastAPI entry point for the Web RAG Evaluator."""

import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from backend.limits import limiter
from backend.rag import (
    KnowledgeBaseError,
    MissingContextDevAPIKeyError,
    MissingOpenAIAPIKeyError,
    RAGInputError,
    WebsiteIngestionError,
    answer_question,
    create_knowledge_base,
    evaluate_case,
    fetch_website_markdown,
    model_settings,
)

load_dotenv()

app = FastAPI(
    title="Web RAG Evaluator API",
    description="Small API for a tutorial-sized website RAG system.",
    version="0.1.0",
)

# Each case costs one embedding, one answer, and one evaluator call, and the
# request runs them one after another, so keep a batch small.
MAX_EVALUATION_CASES = 10

frontend_origin = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Return a lightweight readiness response."""
    return {"status": "ok"}


@app.get("/api/info", tags=["system"])
def info() -> dict[str, str]:
    """Which page fetcher and models are active, so the UI can label them."""
    try:
        return model_settings()
    except RAGInputError as exc:
        raise _http_error(exc) from exc


class IngestRequest(BaseModel):
    """Payload for ingesting one public webpage."""

    model_config = ConfigDict(extra="forbid")

    url: HttpUrl


class IngestResponse(BaseModel):
    """Small ingestion summary returned to the frontend."""

    url: str
    markdown_preview: str
    character_count: int
    chunk_count: int
    title: str | None = None


class AskRequest(BaseModel):
    """Payload for asking one grounded question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)


class CitationResponse(BaseModel):
    """One validated source citation used in the answer."""

    label: str
    source_url: str
    chunk_number: int


class RetrievedChunkResponse(BaseModel):
    """One retrieved passage shown alongside an answer."""

    label: str
    text: str
    source_url: str
    chunk_number: int
    distance: float


class AskResponse(BaseModel):
    """Grounded answer and its retrieved evidence."""

    answer: str
    citations: list[CitationResponse]
    retrieved_chunks: list[RetrievedChunkResponse]


class EvaluationCaseRequest(BaseModel):
    """One question and the answer the RAG system should produce."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)
    expected_answer: str = Field(min_length=1, max_length=4_000)


class EvaluationCaseResponse(BaseModel):
    """Scores for one completed RAG evaluation case."""

    question: str
    expected_answer: str
    generated_answer: str
    retrieval_relevance: float = Field(ge=0, le=1)
    groundedness: float = Field(ge=0, le=1)
    answer_correctness: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    passed: bool
    # The judge's one-sentence reason for each LLM-scored metric.
    explanations: dict[str, str] = Field(default_factory=dict)
    error: str | None = None


RAG_ERRORS = (RAGInputError, WebsiteIngestionError, KnowledgeBaseError)
# Errors that would make every case fail the same way, so the whole request
# should fail instead of returning a table of identical errors.
BATCH_ERRORS = (RAGInputError, MissingContextDevAPIKeyError, MissingOpenAIAPIKeyError)


def _http_error(error: Exception) -> HTTPException:
    if isinstance(error, RAGInputError):
        return HTTPException(status_code=422, detail=str(error))
    if isinstance(error, (MissingContextDevAPIKeyError, MissingOpenAIAPIKeyError)):
        return HTTPException(status_code=503, detail=str(error))
    return HTTPException(status_code=502, detail=str(error))


@app.post("/api/ingest", response_model=IngestResponse, tags=["knowledge-base"])
def ingest_website(payload: IngestRequest, request: Request) -> IngestResponse:
    """Ingest one webpage and keep it as the app's current knowledge source."""
    limiter.check(request)
    try:
        website = fetch_website_markdown(str(payload.url))
        markdown = website["markdown"]
        chunks = create_knowledge_base(markdown, website["url"])
    except RAG_ERRORS as exc:
        raise _http_error(exc) from exc

    metadata = website["metadata"]

    preview_length = 500
    markdown_preview = markdown[:preview_length]
    if len(markdown) > preview_length:
        markdown_preview += "…"

    return IngestResponse(
        url=website["url"],
        markdown_preview=markdown_preview,
        character_count=len(markdown),
        chunk_count=len(chunks),
        title=metadata.get("title"),
    )


@app.post("/api/ask", response_model=AskResponse, tags=["question-answering"])
def ask_question(payload: AskRequest, request: Request) -> AskResponse:
    """Answer one question using only the current website knowledge base."""
    limiter.check(request)
    try:
        result = answer_question(payload.question)
    except RAG_ERRORS as exc:
        raise _http_error(exc) from exc

    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        retrieved_chunks=result["retrieved_chunks"],
    )


@app.post(
    "/api/evaluate",
    response_model=list[EvaluationCaseResponse],
    tags=["evaluations"],
)
def evaluate_cases(
    cases: Annotated[
        list[EvaluationCaseRequest],
        Body(min_length=1, max_length=MAX_EVALUATION_CASES),
    ],
    request: Request,
) -> list[EvaluationCaseResponse]:
    """Run the normal RAG pipeline and evaluator once for every case.

    A failure in one case is reported on that case's row instead of discarding
    the results of the others. If every case fails, the request fails.
    """
    limiter.check(request, cost=len(cases))
    results: list[EvaluationCaseResponse] = []
    first_error: Exception | None = None

    for case in cases:
        try:
            result = evaluate_case(case.question, case.expected_answer)
        except BATCH_ERRORS as exc:
            raise _http_error(exc) from exc
        except RAG_ERRORS as exc:
            first_error = first_error or exc
            results.append(
                EvaluationCaseResponse(
                    question=case.question,
                    expected_answer=case.expected_answer,
                    generated_answer="",
                    retrieval_relevance=0,
                    groundedness=0,
                    answer_correctness=0,
                    citation_validity=0,
                    overall_score=0,
                    passed=False,
                    error=str(exc),
                )
            )
            continue

        results.append(EvaluationCaseResponse.model_validate(result))

    if first_error is not None and all(result.error for result in results):
        raise _http_error(first_error) from first_error

    return results
