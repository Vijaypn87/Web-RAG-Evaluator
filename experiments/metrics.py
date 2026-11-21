"""Pure helpers for scoring and summarising retrieval experiments.

Nothing here calls an API, so it is easy to unit-test.
"""

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

# Metrics produced by the LLM evaluator (plus citation validity) per case.
JUDGE_METRICS = (
    "retrieval_relevance",
    "groundedness",
    "answer_correctness",
    "citation_validity",
    "overall_score",
)
# Columns that identify one configuration in the grid.
CONFIG_KEYS = ("chunk_size", "overlap", "top_k")


def normalize_text(text: str) -> str:
    """Lower-case text and strip Markdown so evidence matching is robust.

    Context.dev returns Markdown, so a sentence on the page may appear as
    "the [Eiffel Tower](https://...) is *tall*". Evidence strings are written
    as plain text, so links, emphasis and extra whitespace are removed first.
    """
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [label](url) -> label
    text = re.sub(r"[*_`#>|]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def evidence_rank(chunk_texts: Sequence[str], evidence: Sequence[str]) -> int | None:
    """Return the 1-based rank of the first chunk containing any evidence string.

    Returns None when no retrieved chunk contains the evidence.
    """
    needles = [normalize_text(item) for item in evidence if item.strip()]
    if not needles:
        return None
    for rank, text in enumerate(chunk_texts, start=1):
        haystack = normalize_text(text)
        if any(needle in haystack for needle in needles):
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    """1.0 if the evidence was found within the first k chunks, else 0.0."""
    return float(rank is not None and rank <= k)


def reciprocal_rank(rank: int | None, k: int) -> float:
    """1/rank if the evidence was found within the first k chunks, else 0.0."""
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def mean_and_ci95(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    """Return (mean, half-width of a normal-approximation 95% CI, n).

    With small n this interval is only a rough guide; report n alongside it.
    """
    numbers = [float(value) for value in values if value is not None]
    n = len(numbers)
    if n == 0:
        return None, None, 0
    mean = sum(numbers) / n
    if n == 1:
        return mean, None, 1
    variance = sum((value - mean) ** 2 for value in numbers) / (n - 1)
    return mean, 1.96 * math.sqrt(variance / n), n


def summarize(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-case rows into one summary row per configuration.

    Rows with an "error" are counted but excluded from metric means.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in CONFIG_KEYS)].append(row)

    summary: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = groups[key]
        ok = [row for row in group if not row.get("error")]
        entry: dict[str, Any] = dict(zip(CONFIG_KEYS, key))
        entry["cases"] = len(group)
        entry["errors"] = len(group) - len(ok)
        chunk_counts = {row["chunk_count"] for row in group if row.get("chunk_count")}
        entry["chunk_count"] = (
            round(sum(chunk_counts) / len(chunk_counts), 1) if chunk_counts else None
        )

        for metric in JUDGE_METRICS + ("hit_at_k", "reciprocal_rank", "latency_s"):
            mean, ci, n = mean_and_ci95(
                row[metric] for row in ok if row.get(metric) is not None
            )
            entry[metric] = None if mean is None else round(mean, 3)
            entry[f"{metric}_ci95"] = None if ci is None else round(ci, 3)
            entry[f"{metric}_n"] = n

        passed = [row["passed"] for row in ok if row.get("passed") is not None]
        entry["pass_rate"] = round(sum(passed) / len(passed), 3) if passed else None
        summary.append(entry)
    return summary


def _cell(entry: dict[str, Any], metric: str) -> str:
    value = entry.get(metric)
    if value is None:
        return "–"
    ci = entry.get(f"{metric}_ci95")
    return f"{value:.2f} ± {ci:.2f}" if ci is not None else f"{value:.2f}"


def summary_markdown(summary: Sequence[dict[str, Any]], *, judged: bool) -> str:
    """Render the summary as a Markdown table for the README or a report."""
    columns = [
        ("Chunk size", None),
        ("Overlap", None),
        ("top-k", None),
        ("Chunks", None),
        ("Context (chars)", None),
    ]
    columns += [("Hit@k", "hit_at_k"), ("MRR", "reciprocal_rank")]
    if judged:
        columns += [
            ("Retrieval", "retrieval_relevance"),
            ("Grounded", "groundedness"),
            ("Correct", "answer_correctness"),
            ("Citations", "citation_validity"),
            ("Overall", "overall_score"),
            ("Pass rate", None),
        ]
    columns += [("Latency (s)", "latency_s"), ("Errors", None)]

    header = "| " + " | ".join(name for name, _ in columns) + " |"
    divider = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, divider]
    for entry in summary:
        cells = [
            str(entry["chunk_size"]),
            str(entry["overlap"]),
            str(entry["top_k"]),
            "–" if entry["chunk_count"] is None else f"{entry['chunk_count']:g}",
            f"~{int(entry['chunk_size']) * int(entry['top_k']):,}",
            _cell(entry, "hit_at_k"),
            _cell(entry, "reciprocal_rank"),
        ]
        if judged:
            cells += [
                _cell(entry, "retrieval_relevance"),
                _cell(entry, "groundedness"),
                _cell(entry, "answer_correctness"),
                _cell(entry, "citation_validity"),
                _cell(entry, "overall_score"),
                "–" if entry["pass_rate"] is None else f"{entry['pass_rate']:.0%}",
            ]
        cells += [_cell(entry, "latency_s"), str(entry["errors"])]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "Values are means over cases (and repeats) ± a 95% confidence half-width. "
        "Hit@k and MRR use the dataset's evidence strings and need no LLM. "
        "Context is roughly chunk size × top-k: larger chunks contain more text, "
        "so compare configurations with similar context sizes for a fair test."
    )
    return "\n".join(lines)
