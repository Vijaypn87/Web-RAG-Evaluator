# Retrieval experiment: chunk size × top-k

This folder runs the RAG pipeline over a grid of **chunk sizes** and **top-k**
values on a fixed question set, and writes tables you can put in a report.

**Research question:** how do chunk size and the number of retrieved chunks
affect retrieval quality and answer quality, and where is the trade-off?

## What gets measured

| Metric | Needs an LLM? | Meaning |
|---|---|---|
| **Hit@k** | No | Did any of the top-k chunks contain the case's `evidence` phrase? |
| **MRR** | No | 1 / rank of the first chunk containing the evidence (0 if not in top-k). |
| Retrieval relevance | Yes (judge) | Judge's view of how relevant the retrieved chunks are. |
| Groundedness | Yes (judge) | Is the answer supported by the retrieved chunks? |
| Answer correctness | Yes (judge) | Does the answer match the expected answer? |
| Citation validity | No | Share of written `[S#]` citations that point to a supplied chunk. |
| Overall / pass rate | — | Mean of the four scores; pass at ≥ 0.70. |
| Latency | No | Seconds per case (retrieval + answer + judge). |

Hit@k and MRR are deterministic and cheap, so they are the most reliable
numbers. The judge scores come from an LLM and vary between runs; use
`--repeats` to average that out and always report the ± interval.

## 1. Write a dataset

Copy [`datasets/TEMPLATE.json`](datasets/TEMPLATE.json). For each case write:

* `question`: something the page answers.
* `expected_answer`: the answer in your own words.
* `evidence`: one or more **short exact phrases copied from the page** that
  contain the answer. These power Hit@k and MRR. Keep them short (3–8 words)
  so a chunk boundary is unlikely to split them.

Aim for **30–50 cases**, ideally across **2–4 different pages** (for example a
Wikipedia article, a product documentation page and a news article), and mix
easy fact lookups with questions that need two parts of the page.
[`datasets/sample_rag_wikipedia.json`](datasets/sample_rag_wikipedia.json)
has 10 checked cases you can use to make sure everything runs.

## 2. Run it

From the repository root, with `.env` filled in:

```powershell
# See the grid and how many API calls it will make. No keys needed.
.\.venv\Scripts\python.exe -m experiments.run_experiment --dataset experiments/datasets/sample_rag_wikipedia.json --dry-run

# Cheap first pass: Hit@k and MRR only (embedding calls only).
.\.venv\Scripts\python.exe -m experiments.run_experiment --dataset experiments/datasets/sample_rag_wikipedia.json --retrieval-only

# Full run: answers + judge, each case repeated 3 times.
.\.venv\Scripts\python.exe -m experiments.run_experiment --dataset experiments/datasets/my_dataset.json --repeats 3
```

Useful options: `--chunk-sizes 300,600,900,1200`, `--top-k 1,2,4,8`,
`--overlap-ratio 0` (to test whether overlap matters), `--out <folder>`.

The page is fetched once and cached in `experiments/.cache/`, so every
configuration (and every later run) uses exactly the same text. Experiment
collections live in `experiments/.chroma/` and never touch the web app's
knowledge base.

## 3. Outputs

Each run writes a folder under `experiments/results/`:

* `summary.md`: the results table, ready to paste into a report.
* `summary.csv`: one row per configuration, with 95% intervals and n.
* `cases.csv`: one row per case, k, size and repeat, including the
  generated answer and any error. Use it to find failure examples.
* `config.json`: models, settings and git commit, so the run is reproducible.

Commit the results folders you report on.

## 4. Writing it up (2–4 pages)

1. **Question and setup**: the research question, pages used, number of
   cases, models, grid, repeats.
2. **Results**: the summary table, plus one chart (for example Hit@k against
   top-k, one line per chunk size).
3. **Analysis**: where is the best trade-off? Does a bigger k help answer
   correctness, or does extra context hurt groundedness? Do small chunks
   miss context that larger chunks keep?
4. **Failure examples**: two or three cases from `cases.csv` where retrieval
   found the evidence but the answer was still wrong (or the reverse), and why.
5. **Limitations**: small dataset, one judge model judging itself, evidence
   matching is exact text, results vary between runs.

The limitations section matters: it shows you understand what the numbers
can and cannot tell you.
