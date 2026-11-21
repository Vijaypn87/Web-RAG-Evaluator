"""Grid experiment over chunk size and top-k for the Web RAG Evaluator.

Run from the repository root, for example:

    python -m experiments.run_experiment --dataset experiments/datasets/sample_rag_wikipedia.json --dry-run
    python -m experiments.run_experiment --dataset experiments/datasets/sample_rag_wikipedia.json --retrieval-only
    python -m experiments.run_experiment --dataset experiments/datasets/my_dataset.json

Each configuration gets its own Chroma collection in a separate store
(experiments/.chroma by default), so the web app's knowledge base is untouched.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from experiments.metrics import (
    CONFIG_KEYS,
    JUDGE_METRICS,
    evidence_rank,
    hit_at_k,
    reciprocal_rank,
    summarize,
    summary_markdown,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent


# Dataset loading
def load_dataset(path: Path) -> dict[str, Any]:
    """Load and sanity-check a dataset file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Dataset needs a non-empty 'sources' list.")

    seen_ids: set[str] = set()
    for source_index, source in enumerate(sources):
        if not source.get("url"):
            raise ValueError(f"Source {source_index} has no 'url'.")
        cases = source.get("cases")
        if not isinstance(cases, list) or not cases:
            raise ValueError(f"Source {source['url']} has no 'cases'.")
        for case_index, case in enumerate(cases):
            case.setdefault("id", f"s{source_index}-q{case_index + 1}")
            if case["id"] in seen_ids:
                raise ValueError(f"Duplicate case id: {case['id']}")
            seen_ids.add(case["id"])
            for field in ("question", "expected_answer"):
                if not str(case.get(field, "")).strip():
                    raise ValueError(f"Case {case['id']} is missing '{field}'.")
            evidence = case.get("evidence", [])
            case["evidence"] = [evidence] if isinstance(evidence, str) else list(evidence)
    data.setdefault("name", path.stem)
    return data


def _int_list(text: str) -> list[int]:
    values = sorted({int(part) for part in text.split(",") if part.strip()})
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one integer.")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chunk-sizes", type=_int_list, default=[400, 800, 1200])
    parser.add_argument("--top-k", type=_int_list, default=[2, 4, 8])
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.125,
        help="Overlap as a fraction of chunk size (0.125 of 800 = the app's 100).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run every judged case this many times to average out LLM noise.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Only measure Hit@k and MRR from evidence strings (embedding cost only).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and API call counts, then exit.")
    parser.add_argument("--out", type=Path, default=None, help="Results folder.")
    parser.add_argument("--chroma-path", type=Path, default=EXPERIMENTS_DIR / ".chroma")
    parser.add_argument("--cache-dir", type=Path, default=EXPERIMENTS_DIR / ".cache")
    args = parser.parse_args(argv)

    if not 0 <= args.overlap_ratio < 0.5:
        parser.error("--overlap-ratio must be in [0, 0.5).")
    if args.repeats < 1:
        parser.error("--repeats must be at least 1.")
    if min(args.top_k) < 1:
        parser.error("--top-k values must be at least 1.")
    if min(args.chunk_sizes) < 100:
        parser.error("--chunk-sizes values must be at least 100.")
    return args


def overlap_for(chunk_size: int, ratio: float) -> int:
    return round(chunk_size * ratio)


def plan_summary(dataset: dict[str, Any], args: argparse.Namespace) -> str:
    """Describe the grid and roughly how many API calls it will make."""
    sources = len(dataset["sources"])
    cases = sum(len(source["cases"]) for source in dataset["sources"])
    sizes, ks = len(args.chunk_sizes), len(args.top_k)
    lines = [
        f"Dataset: {dataset['name']} ({sources} source(s), {cases} case(s))",
        f"Chunk sizes: {args.chunk_sizes}  overlap ratio: {args.overlap_ratio}",
        f"top-k: {args.top_k}",
        f"Mode: {'retrieval only' if args.retrieval_only else 'full (answer + judge)'}",
        "",
        f"Context.dev page fetches: up to {sources} (cached after the first run)",
        f"Embedding batches for chunks: ~{sources * sizes}",
    ]
    if args.retrieval_only:
        lines.append(f"Question embeddings: {cases * sizes}")
        lines.append("Chat/judge calls: 0")
    else:
        runs = cases * sizes * ks * args.repeats
        lines.append(f"Question embeddings: {runs}")
        lines.append(f"Chat answer calls: {runs}")
        lines.append(f"Judge calls: {runs}")
    return "\n".join(lines)


# Page fetching with a local cache
def fetch_markdown_cached(url: str, cache_dir: Path) -> dict[str, Any]:
    """Fetch a page once with Context.dev and reuse it on later runs.

    Caching keeps every configuration on exactly the same page text, even if
    the live page changes between runs.
    """
    from backend.rag import fetch_website_markdown

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()[:16]}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    website = fetch_website_markdown(url)
    payload = {
        "requested_url": url,
        "url": website["url"],
        "markdown": website["markdown"],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=EXPERIMENTS_DIR.parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _error_row(base: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {**base, "error": f"{type(error).__name__}: {error}"}


# The experiment
def run(args: argparse.Namespace, dataset: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Run the grid and return (per-case rows, per-configuration summary)."""
    # Imported here so --dry-run works without API keys or Chroma installed.
    from backend import rag

    # backend.rag reads CHROMA_PATH on every call, so this keeps experiment
    # collections out of the web app's store.
    os.environ["CHROMA_PATH"] = str(args.chroma_path)

    rows: list[dict[str, Any]] = []
    max_k = max(args.top_k)

    for source_index, source in enumerate(dataset["sources"]):
        page = fetch_markdown_cached(source["url"], args.cache_dir)
        print(f"\nSource {source_index}: {page['url']} ({len(page['markdown']):,} chars)")

        for chunk_size in args.chunk_sizes:
            overlap = overlap_for(chunk_size, args.overlap_ratio)
            collection = f"exp_s{source_index}_c{chunk_size}_o{overlap}"
            chunks = rag.create_knowledge_base(
                page["markdown"],
                page["url"],
                collection_name=collection,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            print(f"  chunk_size={chunk_size} overlap={overlap}: {len(chunks)} chunks")

            for case in source["cases"]:
                base = {
                    "source_url": page["url"],
                    "case_id": case["id"],
                    "question": case["question"],
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "chunk_count": len(chunks),
                }

                if args.retrieval_only:
                    # Retrieval is deterministic, so retrieve once at the
                    # largest k and score every smaller k from that ranking.
                    started = time.perf_counter()
                    try:
                        retrieved = rag.retrieve_chunks(
                            case["question"], top_k=max_k, collection_name=collection
                        )
                    except Exception as exc:  # keep the run going; record it
                        for k in args.top_k:
                            rows.append(_error_row({**base, "top_k": k, "repeat": 1}, exc))
                        continue
                    latency = time.perf_counter() - started
                    rank = evidence_rank([c["text"] for c in retrieved], case["evidence"])
                    for k in args.top_k:
                        rows.append(
                            {
                                **base,
                                "top_k": k,
                                "repeat": 1,
                                "evidence_rank": rank,
                                "hit_at_k": hit_at_k(rank, k) if case["evidence"] else None,
                                "reciprocal_rank": reciprocal_rank(rank, k) if case["evidence"] else None,
                                "latency_s": round(latency, 3),
                            }
                        )
                    continue

                for k in args.top_k:
                    for repeat in range(1, args.repeats + 1):
                        row_base = {**base, "top_k": k, "repeat": repeat}
                        started = time.perf_counter()
                        try:
                            result = rag.evaluate_case(
                                case["question"],
                                case["expected_answer"],
                                top_k=k,
                                collection_name=collection,
                            )
                        except Exception as exc:  # keep the run going; record it
                            rows.append(_error_row(row_base, exc))
                            print(f"    {case['id']} k={k} r={repeat}: ERROR {exc}")
                            continue
                        latency = time.perf_counter() - started
                        rank = evidence_rank(
                            [c["text"] for c in result["retrieved_chunks"]],
                            case["evidence"],
                        )
                        rows.append(
                            {
                                **row_base,
                                **{metric: result[metric] for metric in JUDGE_METRICS},
                                "passed": result["passed"],
                                "evidence_rank": rank,
                                "hit_at_k": hit_at_k(rank, k) if case["evidence"] else None,
                                "reciprocal_rank": reciprocal_rank(rank, k) if case["evidence"] else None,
                                "latency_s": round(latency, 3),
                                "generated_answer": result["generated_answer"],
                            }
                        )
                        print(
                            f"    {case['id']} k={k} r={repeat}: "
                            f"overall={result['overall_score']:.2f} rank={rank}"
                        )

    return rows, summarize(rows)


def write_results(
    out_dir: Path,
    rows: list[dict],
    summary: list[dict],
    args: argparse.Namespace,
    dataset: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, records: list[dict]) -> None:
        fields: list[str] = []
        for record in records:
            fields.extend(key for key in record if key not in fields)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    write_csv(out_dir / "cases.csv", rows)
    write_csv(out_dir / "summary.csv", summary)

    table = summary_markdown(summary, judged=not args.retrieval_only)
    (out_dir / "summary.md").write_text(
        f"# {dataset['name']}\n\n{plan_summary(dataset, args)}\n\n{table}\n",
        encoding="utf-8",
    )
    config = {
        "dataset": str(args.dataset),
        "dataset_name": dataset["name"],
        "chunk_sizes": args.chunk_sizes,
        "top_k": args.top_k,
        "overlap_ratio": args.overlap_ratio,
        "repeats": args.repeats,
        "retrieval_only": args.retrieval_only,
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
        "evaluator_model": os.getenv("OPENAI_EVALUATOR_MODEL")
        or os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini"),
        "git_commit": _git_commit(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "config_keys": list(CONFIG_KEYS),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = parse_args(argv)
    dataset = load_dataset(args.dataset)

    print(plan_summary(dataset, args))
    if args.dry_run:
        return 0

    rows, summary = run(args, dataset)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    mode = "retrieval" if args.retrieval_only else "full"
    out_dir = args.out or EXPERIMENTS_DIR / "results" / f"{dataset['name']}-{mode}-{stamp}"
    write_results(out_dir, rows, summary, args, dataset)

    print()
    print(summary_markdown(summary, judged=not args.retrieval_only))
    print(f"\nResults written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
