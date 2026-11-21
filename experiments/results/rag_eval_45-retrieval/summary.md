# rag_eval_45

Dataset: rag_eval_45 (3 source(s), 45 case(s))
Chunk sizes: [400, 800, 1200]  overlap ratio: 0.125
top-k: [2, 4, 8]
Mode: retrieval only

Page fetches: up to 3 (cached after the first run)
Embedding batches for chunks: ~9
Question embeddings: 135
Chat/judge calls: 0

| Chunk size | Overlap | top-k | Chunks | Context (chars) | Hit@k | MRR | Latency (s) | Errors |
|---|---|---|---|---|---|---|---|---|
| 400 | 50 | 2 | 97 | ~800 | 0.69 ± 0.15 | 0.65 ± 0.14 | 2.20 ± 0.14 | 0 |
| 400 | 50 | 4 | 97 | ~1,600 | 0.87 ± 0.11 | 0.71 ± 0.12 | 2.20 ± 0.14 | 0 |
| 400 | 50 | 8 | 97 | ~3,200 | 0.97 ± 0.05 | 0.73 ± 0.11 | 2.20 ± 0.14 | 0 |
| 800 | 100 | 2 | 48.7 | ~1,600 | 0.82 ± 0.12 | 0.74 ± 0.12 | 2.09 ± 0.01 | 0 |
| 800 | 100 | 4 | 48.7 | ~3,200 | 0.92 ± 0.09 | 0.78 ± 0.11 | 2.09 ± 0.01 | 0 |
| 800 | 100 | 8 | 48.7 | ~6,400 | 0.97 ± 0.05 | 0.79 ± 0.10 | 2.09 ± 0.01 | 0 |
| 1200 | 150 | 2 | 31.3 | ~2,400 | 0.90 ± 0.10 | 0.80 ± 0.11 | 2.10 ± 0.01 | 0 |
| 1200 | 150 | 4 | 31.3 | ~4,800 | 1.00 ± 0.00 | 0.82 ± 0.09 | 2.10 ± 0.01 | 0 |
| 1200 | 150 | 8 | 31.3 | ~9,600 | 1.00 ± 0.00 | 0.82 ± 0.09 | 2.10 ± 0.01 | 0 |

Values are means over cases (and repeats) ± a 95% confidence half-width. Hit@k and MRR use the dataset's evidence strings and need no LLM. Context is roughly chunk size × top-k: larger chunks contain more text, so compare configurations with similar context sizes for a fair test.
