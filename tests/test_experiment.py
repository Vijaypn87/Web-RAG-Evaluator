"""Tests for the experiment metrics and an offline end-to-end run.

The end-to-end tests use a real Chroma store in a temp folder but replace
OpenAI embeddings with a deterministic bag-of-words vector, and Context.dev
with a pre-filled page cache, so they make no network calls.
"""

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from backend import rag
from backend.rag import RAGInputError, chunk_markdown
from experiments import run_experiment
from experiments.metrics import (
    evidence_rank,
    hit_at_k,
    mean_and_ci95,
    normalize_text,
    reciprocal_rank,
    summarize,
    summary_markdown,
)

SAMPLE_DATASET = Path(__file__).resolve().parent.parent / "experiments" / "datasets" / "sample_rag_wikipedia.json"


# Metrics
def test_normalize_text_strips_markdown_links_and_emphasis():
    text = "The **[Eiffel Tower](https://x.test/e)** is  *tall*."
    assert normalize_text(text) == "the eiffel tower is tall ."


def test_evidence_rank_finds_first_matching_chunk():
    chunks = ["nothing here", "prompts and stuffing", "it is called [prompt](u) stuffing"]
    # Rank 3: the link is stripped before matching; chunk 2 is a near miss.
    assert evidence_rank(chunks, ["Prompt Stuffing"]) == 3
    assert evidence_rank(chunks, ["missing phrase"]) is None
    assert evidence_rank(chunks, []) is None


def test_hit_and_reciprocal_rank():
    assert hit_at_k(2, 4) == 1.0
    assert hit_at_k(5, 4) == 0.0
    assert hit_at_k(None, 4) == 0.0
    assert reciprocal_rank(2, 4) == 0.5
    assert reciprocal_rank(5, 4) == 0.0


def test_mean_and_ci95():
    assert mean_and_ci95([]) == (None, None, 0)
    assert mean_and_ci95([0.5]) == (0.5, None, 1)
    mean, ci, n = mean_and_ci95([0.0, 1.0, 0.0, 1.0])
    assert (mean, n) == (0.5, 4)
    assert ci == pytest.approx(1.96 * math.sqrt((1 / 3) / 4))


def test_summarize_groups_by_configuration_and_skips_errors():
    rows = [
        {"chunk_size": 400, "overlap": 50, "top_k": 2, "chunk_count": 10, "hit_at_k": 1.0, "passed": True, "overall_score": 0.9},
        {"chunk_size": 400, "overlap": 50, "top_k": 2, "chunk_count": 10, "hit_at_k": 0.0, "passed": False, "overall_score": 0.5},
        {"chunk_size": 400, "overlap": 50, "top_k": 2, "chunk_count": 10, "error": "boom"},
        {"chunk_size": 800, "overlap": 100, "top_k": 2, "chunk_count": 5, "hit_at_k": 1.0, "passed": True, "overall_score": 1.0},
    ]
    summary = summarize(rows)
    assert [(s["chunk_size"], s["top_k"]) for s in summary] == [(400, 2), (800, 2)]
    first = summary[0]
    assert first["cases"] == 3 and first["errors"] == 1
    assert first["hit_at_k"] == 0.5
    assert first["overall_score"] == 0.7
    assert first["pass_rate"] == 0.5
    table = summary_markdown(summary, judged=True)
    assert "| 400 | 50 | 2 | 10 |" in table


# Chunking parameters
@pytest.mark.parametrize("size", [300, 800, 1500])
def test_chunk_size_parameter_controls_chunk_length(size):
    paragraph = "Filler sentence for chunking tests. " * 30
    markdown = "\n\n".join(f"## Part {n}\n\n{paragraph}" for n in range(8))
    chunks = chunk_markdown(markdown, chunk_size=size, overlap=size // 8)
    assert all(len(c["text"]) <= size + 10 for c in chunks)


def test_smaller_chunks_make_more_chunks():
    markdown = "\n\n".join(f"Paragraph {n}. " + "word " * 60 for n in range(30))
    small = chunk_markdown(markdown, chunk_size=400, overlap=50)
    large = chunk_markdown(markdown, chunk_size=1200, overlap=150)
    assert len(small) > len(large)


def test_zero_overlap_is_allowed():
    markdown = "\n\n".join("Sentence number %d here." % n * 10 for n in range(10))
    assert chunk_markdown(markdown, chunk_size=300, overlap=0)


@pytest.mark.parametrize(("size", "overlap"), [(50, 0), (400, 200), (400, -1)])
def test_invalid_chunk_parameters(size, overlap):
    with pytest.raises(RAGInputError):
        chunk_markdown("text", chunk_size=size, overlap=overlap)


# Dataset loading and CLI
def test_sample_dataset_loads():
    data = run_experiment.load_dataset(SAMPLE_DATASET)
    cases = data["sources"][0]["cases"]
    assert len(cases) == 10
    assert all(case["evidence"] for case in cases)


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "d.json"
    case = {"id": "x", "question": "q", "expected_answer": "a"}
    path.write_text(json.dumps({"sources": [{"url": "https://a.test", "cases": [case, case]}]}))
    with pytest.raises(ValueError, match="Duplicate"):
        run_experiment.load_dataset(path)


def test_dry_run_needs_no_keys(capsys):
    assert run_experiment.main(["--dataset", str(SAMPLE_DATASET), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Chat answer calls: 90" in out  # 10 cases x 3 sizes x 3 k


# Offline end-to-end run
PAGE = """# Retrieval

## Origins
Retrieval-augmented generation was first proposed in 2020 by researchers.

## Storage
The embeddings are stored in a vector database for fast lookup.

## Chunking
Chunk size involves a trade-off between focused retrieval and contextual completeness.

## Hybrid
Hybrid search adds a traditional text search to the vector search results.
""" + "\n\n".join(f"## Filler {n}\n\n" + "Unrelated filler words about gardening. " * 15 for n in range(6))


def _fake_embed(texts, model):
    """Deterministic bag-of-words embedding so similar text retrieves similar text."""
    vectors = []
    for text in texts:
        vector = [0.0] * 256
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        vectors.append([v / norm for v in vector])
    return vectors


@pytest.fixture
def offline(tmp_path, monkeypatch):
    url = "https://example.test/rag"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = hashlib.sha1(url.encode()).hexdigest()[:16]
    (cache_dir / f"{key}.json").write_text(json.dumps({"url": url, "markdown": PAGE}))

    dataset = {
        "name": "offline",
        "sources": [
            {
                "url": url,
                "cases": [
                    {"id": "a", "question": "When was retrieval-augmented generation first proposed?",
                     "expected_answer": "2020", "evidence": ["first proposed in 2020"]},
                    {"id": "b", "question": "Where are the embeddings stored?",
                     "expected_answer": "A vector database", "evidence": ["stored in a vector database"]},
                ],
            }
        ],
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset))
    monkeypatch.setattr(rag, "_embed_texts", _fake_embed)
    monkeypatch.delenv("CHROMA_PATH", raising=False)
    return {"tmp": tmp_path, "dataset": dataset_path, "cache": cache_dir}


def _args(offline, *extra):
    return [
        "--dataset", str(offline["dataset"]),
        "--cache-dir", str(offline["cache"]),
        "--chroma-path", str(offline["tmp"] / "chroma"),
        "--out", str(offline["tmp"] / "out"),
        "--chunk-sizes", "200,600",
        "--top-k", "1,3",
        *extra,
    ]


def test_retrieval_only_run_end_to_end(offline):
    assert run_experiment.main(_args(offline, "--retrieval-only")) == 0

    out = offline["tmp"] / "out"
    summary = list(csv_rows(out / "summary.csv"))
    assert {(r["chunk_size"], r["top_k"]) for r in summary} == {
        ("200", "1"), ("200", "3"), ("600", "1"), ("600", "3"),
    }
    assert all(r["errors"] == "0" for r in summary)
    # Bag-of-words retrieval finds these easy questions at k=3.
    assert all(float(r["hit_at_k"]) == 1.0 for r in summary if r["top_k"] == "3")
    assert (out / "summary.md").read_text(encoding="utf-8").count("|") > 10
    assert json.loads((out / "config.json").read_text())["retrieval_only"] is True


def test_full_run_records_judge_scores_and_errors(offline, monkeypatch):
    calls = []

    def fake_evaluate(question, expected_answer, *, top_k, collection_name):
        calls.append((top_k, collection_name))
        if question.startswith("Where") and top_k == 1:
            raise rag.KnowledgeBaseError("judge timed out")
        retrieved = rag.retrieve_chunks(question, top_k=top_k, collection_name=collection_name)
        return {
            "retrieved_chunks": retrieved,
            "generated_answer": "answer [S1]",
            "retrieval_relevance": 1.0,
            "groundedness": 1.0,
            "answer_correctness": 0.5,
            "citation_validity": 1.0,
            "overall_score": 0.875,
            "passed": True,
        }

    monkeypatch.setattr(rag, "evaluate_case", fake_evaluate)
    assert run_experiment.main(_args(offline, "--repeats", "2")) == 0

    # 2 cases x 2 sizes x 2 k x 2 repeats
    assert len(calls) == 16
    assert {name for _, name in calls} == {"exp_s0_c200_o25", "exp_s0_c600_o75"}
    rows = list(csv_rows(offline["tmp"] / "out" / "cases.csv"))
    assert sum(1 for r in rows if r.get("error")) == 4
    summary = {(r["chunk_size"], r["top_k"]): r for r in csv_rows(offline["tmp"] / "out" / "summary.csv")}
    assert summary[("200", "1")]["errors"] == "2"
    assert float(summary[("200", "3")]["overall_score"]) == 0.875


def test_experiment_does_not_touch_app_collection(offline):
    run_experiment.main(_args(offline, "--retrieval-only"))
    import chromadb

    client = chromadb.PersistentClient(path=str(offline["tmp"] / "chroma"))
    names = {c.name for c in client.list_collections()}
    assert rag.COLLECTION_NAME not in names


def csv_rows(path):
    import csv

    with path.open(encoding="utf-8") as handle:
        yield from csv.DictReader(handle)
