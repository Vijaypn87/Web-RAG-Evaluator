# Web RAG Evaluator

Ask questions about any public webpage, get answers with checkable citations, and **measure** how good those answers are.

[![Live demo](https://img.shields.io/badge/Live_demo-Open-16a34a?style=flat-square&logo=render&logoColor=white)](https://web-rag-evaluator-frontend.onrender.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-1f2328?style=flat-square)](LICENSE)
![Tests](https://img.shields.io/badge/Tests-91_passing-16a34a?style=flat-square&logo=pytest&logoColor=white)

> The live demo runs on free hosting. If it has been idle, the first load takes up to a minute while the server wakes up.

## ◆ Tech stack

![Python](https://img.shields.io/badge/Python-1f2328?style=flat-square&logo=python&logoColor=3776AB)
![FastAPI](https://img.shields.io/badge/FastAPI-1f2328?style=flat-square&logo=fastapi&logoColor=009688)
![React](https://img.shields.io/badge/React-1f2328?style=flat-square&logo=react&logoColor=61DAFB)
![TypeScript](https://img.shields.io/badge/TypeScript-1f2328?style=flat-square&logo=typescript&logoColor=3178C6)
![Vite](https://img.shields.io/badge/Vite-1f2328?style=flat-square&logo=vite&logoColor=646CFF)
![ChromaDB](https://img.shields.io/badge/ChromaDB-1f2328?style=flat-square)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-1f2328?style=flat-square&logo=googlegemini&logoColor=8E75B2)
![Ollama](https://img.shields.io/badge/Ollama-1f2328?style=flat-square&logo=ollama&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_(optional)-1f2328?style=flat-square)
![Trafilatura](https://img.shields.io/badge/Trafilatura-1f2328?style=flat-square&logo=python&logoColor=FFD43B)
![pytest](https://img.shields.io/badge/pytest-1f2328?style=flat-square&logo=pytest&logoColor=0A9EDC)
![Render](https://img.shields.io/badge/Render-1f2328?style=flat-square&logo=render&logoColor=white)

## ◆ Features

- **Load any public webpage:** its main content is extracted, split into chunks and embedded into a vector store.
- **Grounded answers:** each answer uses only the retrieved passages and cites them as `[S1]`, `[S2]`, … Click a citation to highlight its passage.
- **Built-in evaluation:** every answer is scored on retrieval relevance, groundedness, correctness and citation validity.
- **Experiment framework:** compare chunk sizes and top-k with Hit@k, MRR and 95% confidence intervals.
- **Runs free:** Ollama on your own computer, or Google Gemini's free tier when hosted. OpenAI is optional.

## ◆ How it works

```text
Webpage ─▶ Extract main text ─▶ Chunk ─▶ Embed ─▶ ChromaDB
                                                     │
Question ─▶ Embed ─▶ Retrieve top 4 chunks ◀─────────┘
                           │
                           ▼
           LLM answer with [S#] citations ─▶ Citation check ─▶ LLM judge scores
```

## ◆ Evaluation

| Metric | Measures | Scored by |
|---|---|---|
| Retrieval relevance | Are the retrieved passages relevant to the question? | LLM judge |
| Groundedness | Is the answer supported by those passages? | LLM judge |
| Answer correctness | Does it match the expected answer? | LLM judge |
| Citation validity | Share of written citations that point to a real passage | Code |

The overall score is the average of the four (0–1), and a case **passes at ≥ 0.70**.

## ◆ Quick start (free, local)

**Requirements:** Python 3.10+, Node.js 20+, and [Ollama](https://ollama.com).

```bash
git clone https://github.com/Vijaypn87/Web-RAG-Evaluator.git
cd Web-RAG-Evaluator

python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
cp .env.example .env                                     # already set up for free mode

ollama pull llama3.2
ollama pull nomic-embed-text
```

Start the backend and the frontend in two terminals:

```bash
.venv/Scripts/python -m uvicorn backend.main:app --reload --port 8000
```

```bash
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173**.

<details>
<summary><b>Configuration</b></summary>

All settings live in `.env`. See [`.env.example`](.env.example) for the full list.

| Setting | Options | Default |
|---|---|---|
| `LLM_PROVIDER` | `ollama` · `gemini` · `openai` | `openai` (the example sets `ollama`) |
| `PAGE_FETCHER` | `local` (Trafilatura) · `contextdev` | `local` when no Context.dev key is set |
| `GEMINI_API_KEY` | Free key from [Google AI Studio](https://aistudio.google.com/apikey) | – |
| `RATE_LIMIT_PER_HOUR` / `DAILY_LIMIT` | Request limits for a public demo (0 = off) | `30` / `500` |

</details>

<details>
<summary><b>Deploy for free (Render + Gemini)</b></summary>

1. Get a free key at [Google AI Studio](https://aistudio.google.com/apikey).
2. On the Render **backend** service, set `LLM_PROVIDER=gemini`, `PAGE_FETCHER=local`, `GEMINI_API_KEY=<your key>` and `FRONTEND_ORIGIN=<frontend URL>`.
3. On the **frontend** static site, set `VITE_API_URL=<backend URL>`.
4. Check `<backend URL>/api/info`: it should report `gemini` and `local`.

Never commit API keys. They belong only in `.env` (git-ignored) or in Render's environment settings.

</details>

## ◆ Experiments

Run the pipeline over a grid of chunk sizes and top-k values on a fixed question set:

```bash
python -m experiments.run_experiment --dataset experiments/datasets/sample_rag_wikipedia.json --retrieval-only
```

Results (tables, CSVs and the exact config) are written to `experiments/results/`. See the [experiments guide](experiments/README.md) for writing datasets and reporting results.

## ◆ Project structure

```text
backend/       FastAPI app, RAG pipeline, providers, request limits
frontend/      React + TypeScript single-page app
experiments/   Grid runner, metrics, datasets
tests/         pytest suite (no network calls)
```

## ◆ Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## ◆ License

[MIT](LICENSE) © 2026 Vijay PN
