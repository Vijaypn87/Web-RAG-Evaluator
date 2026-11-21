# rag_eval_45

Dataset: rag_eval_45 (3 source(s), 45 case(s))
Chunk sizes: [400, 800, 1200]  overlap ratio: 0.125
top-k: [2, 4, 8]
Mode: full (answer + judge)

Page fetches: up to 3 (cached after the first run)
Embedding batches for chunks: ~9
Question embeddings: 405
Chat answer calls: 405
Judge calls: 405

| Chunk size | Overlap | top-k | Chunks | Context (chars) | Hit@k | MRR | Retrieval | Grounded | Correct | Citations | Overall | Pass rate | Latency (s) | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 | 50 | 2 | 97 | ~800 | 0.69 ± 0.15 | 0.65 ± 0.14 | 0.73 ± 0.05 | 0.80 ± 0.07 | 0.73 ± 0.08 | 0.53 ± 0.15 | 0.70 ± 0.05 | 40% | 10.40 ± 0.89 | 0 |
| 400 | 50 | 4 | 97 | ~1,600 | 0.87 ± 0.11 | 0.71 ± 0.12 | 0.70 ± 0.08 | 0.80 ± 0.08 | 0.73 ± 0.08 | 0.76 ± 0.13 | 0.75 ± 0.07 | 64% | 10.05 ± 0.15 | 0 |
| 400 | 50 | 8 | 97 | ~3,200 | 0.97 ± 0.05 | 0.73 ± 0.11 | 0.68 ± 0.08 | 0.78 ± 0.09 | 0.75 ± 0.08 | 0.71 ± 0.13 | 0.73 ± 0.06 | 60% | 10.47 ± 0.19 | 0 |
| 800 | 100 | 2 | 48.7 | ~1,600 | 0.82 ± 0.12 | 0.74 ± 0.12 | 0.72 ± 0.07 | 0.81 ± 0.08 | 0.74 ± 0.09 | 0.69 ± 0.14 | 0.74 ± 0.06 | 56% | 10.33 ± 0.16 | 0 |
| 800 | 100 | 4 | 48.7 | ~3,200 | 0.92 ± 0.09 | 0.78 ± 0.11 | 0.71 ± 0.08 | 0.79 ± 0.09 | 0.71 ± 0.09 | 0.71 ± 0.13 | 0.73 ± 0.07 | 56% | 10.47 ± 0.21 | 0 |
| 800 | 100 | 8 | 48.7 | ~6,400 | 0.97 ± 0.05 | 0.79 ± 0.10 | 0.71 ± 0.08 | 0.78 ± 0.09 | 0.71 ± 0.09 | 0.64 ± 0.14 | 0.71 ± 0.07 | 53% | 11.44 ± 0.23 | 0 |
| 1200 | 150 | 2 | 31.3 | ~2,400 | 0.90 ± 0.10 | 0.80 ± 0.11 | 0.68 ± 0.08 | 0.80 ± 0.08 | 0.74 ± 0.09 | 0.78 ± 0.12 | 0.75 ± 0.06 | 67% | 10.53 ± 0.18 | 0 |
| 1200 | 150 | 4 | 31.3 | ~4,800 | 1.00 ± 0.00 | 0.82 ± 0.09 | 0.71 ± 0.08 | 0.78 ± 0.09 | 0.75 ± 0.09 | 0.64 ± 0.14 | 0.72 ± 0.07 | 58% | 11.20 ± 0.30 | 0 |
| 1200 | 150 | 8 | 31.3 | ~9,600 | 1.00 ± 0.00 | 0.82 ± 0.09 | 0.75 ± 0.07 | 0.80 ± 0.08 | 0.76 ± 0.09 | 0.56 ± 0.15 | 0.72 ± 0.06 | 51% | 12.26 ± 0.30 | 0 |

Values are means over cases (and repeats) ± a 95% confidence half-width. Hit@k and MRR use the dataset's evidence strings and need no LLM. Context is roughly chunk size × top-k: larger chunks contain more text, so compare configurations with similar context sizes for a fair test.
