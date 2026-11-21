# sample_rag_wikipedia

Dataset: sample_rag_wikipedia (1 source(s), 10 case(s))
Chunk sizes: [400, 800, 1200]  overlap ratio: 0.125
top-k: [2, 4, 8]
Mode: retrieval only

Context.dev page fetches: up to 1 (cached after the first run)
Embedding batches for chunks: ~3
Question embeddings: 30
Chat/judge calls: 0

| Chunk size | Overlap | top-k | Chunks | Hit@k | MRR | Latency (s) | Errors |
|---|---|---|---|---|---|---|---|
| 400 | 50 | 2 | 93 | 0.60 ± 0.32 | 0.55 ± 0.31 | 3.00 ± 0.70 | 0 |
| 400 | 50 | 4 | 93 | 0.80 ± 0.26 | 0.62 ± 0.27 | 3.00 ± 0.70 | 0 |
| 400 | 50 | 8 | 93 | 0.90 ± 0.20 | 0.63 ± 0.25 | 3.00 ± 0.70 | 0 |
| 800 | 100 | 2 | 47 | 0.80 ± 0.26 | 0.80 ± 0.26 | 2.11 ± 0.01 | 0 |
| 800 | 100 | 4 | 47 | 0.80 ± 0.26 | 0.80 ± 0.26 | 2.11 ± 0.01 | 0 |
| 800 | 100 | 8 | 47 | 1.00 ± 0.00 | 0.83 ± 0.22 | 2.11 ± 0.01 | 0 |
| 1200 | 150 | 2 | 31 | 1.00 ± 0.00 | 0.95 ± 0.10 | 2.11 ± 0.01 | 0 |
| 1200 | 150 | 4 | 31 | 1.00 ± 0.00 | 0.95 ± 0.10 | 2.11 ± 0.01 | 0 |
| 1200 | 150 | 8 | 31 | 1.00 ± 0.00 | 0.95 ± 0.10 | 2.11 ± 0.01 | 0 |

Values are means over cases (and repeats) ± a 95% confidence half-width. Hit@k and MRR use the dataset's evidence strings and need no LLM.
