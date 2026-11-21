"""Unit tests for the RAG helpers. None of these call OpenAI or Context.dev."""

import pytest

from backend import rag
from backend.rag import (
    INSUFFICIENT_INFORMATION_ANSWER,
    TARGET_CHUNK_SIZE,
    RAGInputError,
    _citation_labels,
    _citation_validity,
    _expand_citation_groups,
    _validate_public_url,
    _validated_citation_labels,
    chunk_markdown,
)


def _chunks(count: int) -> list[dict]:
    return [
        {
            "label": f"[S{n}]",
            "text": f"passage {n}",
            "source_url": "https://example.com",
            "chunk_number": n,
            "distance": 0.1,
        }
        for n in range(1, count + 1)
    ]


def _result(answer: str, raw_labels: list[str], supplied: int = 4) -> dict:
    return {
        "answer": answer,
        "citations": [],
        "retrieved_chunks": _chunks(supplied),
        "source_url": "https://example.com",
        "raw_citation_labels": raw_labels,
    }


# Citation group expansion
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Fact [S1, S2].", "Fact [S1][S2]."),
        ("Fact [S1;S3].", "Fact [S1][S3]."),
        ("Fact [ S1 , S2 , S4 ].", "Fact [S1][S2][S4]."),
        ("Fact [S1].", "Fact [S1]."),
        ("Fact [Source 1].", "Fact [Source 1]."),
    ],
)
def test_expand_citation_groups(text, expected):
    assert _expand_citation_groups(text) == expected


def test_grouped_citations_are_kept_after_validation():
    answer = _expand_citation_groups("Paris is the capital [S1, S2].")
    cleaned, labels = _validated_citation_labels(answer, {"[S1]", "[S2]"})
    assert cleaned == "Paris is the capital [S1][S2]."
    assert labels == ["[S1]", "[S2]"]


def test_invented_labels_are_removed_from_displayed_answer():
    cleaned, labels = _validated_citation_labels(
        "A [S1]. B [S9]. C [S1].", {"[S1]", "[S2]"}
    )
    assert cleaned == "A [S1]. B. C [S1]."
    assert labels == ["[S1]"]


# Citation validity (previously always 1.0)
def test_citation_validity_penalises_invented_labels():
    raw = _citation_labels("A [S1]. B [S9].")
    assert _citation_validity(_result("A [S1]. B.", raw)) == 0.5


def test_citation_validity_all_valid():
    raw = _citation_labels("A [S1]. B [S2].")
    assert _citation_validity(_result("A [S1]. B [S2].", raw)) == 1.0


def test_citation_validity_all_invented():
    raw = _citation_labels("A [S7]. B [S8].")
    assert _citation_validity(_result("A. B.", raw)) == 0.0


def test_citation_validity_uncited_answer_scores_zero():
    assert _citation_validity(_result("The capital is Paris.", [])) == 0.0


def test_citation_validity_insufficient_answer_without_citations():
    result = _result(INSUFFICIENT_INFORMATION_ANSWER, [])
    assert _citation_validity(result) == 1.0


def test_citation_validity_insufficient_answer_with_citations():
    result = _result(INSUFFICIENT_INFORMATION_ANSWER, ["[S1]"])
    assert _citation_validity(result) == 0.0


# URL validation
@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000",
        "http://api.localhost/",
        "http://127.0.0.1/",
        "http://10.0.0.5/",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/",
        "ftp://example.com/",
        "https://user:pass@example.com/",
    ],
)
def test_validate_public_url_rejects(url):
    with pytest.raises(RAGInputError):
        _validate_public_url(url)


def test_validate_public_url_accepts_public():
    assert _validate_public_url("https://example.com/a") == "https://example.com/a"


# Chunking
def test_chunk_markdown_empty():
    assert chunk_markdown("   \n ") == []


def test_chunk_markdown_keeps_heading_with_paragraph():
    chunks = chunk_markdown("# Title\n\nFirst paragraph.")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "# Title\n\nFirst paragraph."
    assert chunks[0]["chunk_id"] == "chunk-0001"


def test_chunk_markdown_splits_long_text_near_target_size():
    paragraph = "This is one sentence of filler text. " * 20
    markdown = "\n\n".join(f"## Section {n}\n\n{paragraph}" for n in range(6))
    chunks = chunk_markdown(markdown)

    assert len(chunks) > 1
    assert [c["chunk_number"] for c in chunks] == list(range(1, len(chunks) + 1))
    # Overlap and paragraph joins can add a few characters beyond the target.
    assert all(len(c["text"]) <= TARGET_CHUNK_SIZE + 10 for c in chunks)


def test_chunk_markdown_covers_all_content():
    markdown = "\n\n".join(f"Paragraph number {n} with unique-{n}." for n in range(80))
    joined = "\n".join(c["text"] for c in chunk_markdown(markdown))
    for n in range(80):
        assert f"unique-{n}." in joined


def test_split_long_block_without_spaces():
    pieces = rag._split_long_block("x" * 2_000, 700)
    assert [len(p) for p in pieces] == [700, 700, 600]
