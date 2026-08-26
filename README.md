# Web RAG Evaluator

A full-stack **Retrieval-Augmented Generation (RAG)** application that ingests a public webpage, retrieves relevant content, generates grounded answers, and evaluates the quality of the RAG pipeline.

## 🚀 Live Demo

[**Try Web RAG Evaluator →**](https://web-rag-evaluator-frontend.onrender.com)

![Web RAG Evaluator](https://github.com/user-attachments/assets/77f9548e-9b65-4b2c-a2b1-aa9e39473555)

## ✨ Features

* 🌐 Ingest a public webpage and extract its main content using **Context.dev**
* ✂️ Split webpage content into retrieval-ready Markdown chunks
* 🧠 Generate embeddings using **OpenAI**
* 🔎 Retrieve the **4 most relevant chunks** using persistent **Chroma**
* 💬 Generate answers using the **OpenAI Responses API**
* 📎 Validate citations against retrieved source chunks
* 📊 Evaluate RAG performance across multiple quality dimensions
* ⚡ Full-stack interface built with **React, Vite, and TypeScript**
* 🔐 Keep API keys securely on the backend using environment variables

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* Context.dev
* OpenAI API
* Chroma

### Frontend

* React
* TypeScript
* Vite
* CSS

## 🏗️ Architecture

```text
                    Public Webpage
                          │
                          ▼
                    Context.dev
                          │
                          ▼
                    Markdown Content
                          │
                          ▼
                       Chunking
                          │
                          ▼
                  OpenAI Embeddings
                          │
                          ▼
                    Chroma Vector DB
                          │
                 ┌────────┴────────┐
                 │                 │
              Question          Retrieval
                 │                 │
                 └────────┬────────┘
                          ▼
                 OpenAI Responses API
                          │
                          ▼
                Grounded Answer
                          │
                          ▼
              Citation Validation
                          │
                          ▼
                  RAG Evaluation
```

## 📁 Project Structure

```text
web-rag-evaluator/
│
├── backend/
│   ├── main.py          # FastAPI routes and application state
│   └── rag.py           # RAG pipeline and evaluation logic
│
├── frontend/
│   └── src/
│       ├── App.tsx      # React application interface
│       └── index.css    # Application styling
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## 🔄 RAG Pipeline

The application follows a simple end-to-end RAG workflow:

1. **Ingestion** — A public webpage URL is submitted to the backend.
2. **Content extraction** — Context.dev converts the webpage into main-content Markdown.
3. **Chunking** — The Markdown is divided into smaller sections around headings and paragraph boundaries.
4. **Embedding** — OpenAI embeddings are generated for the chunks in batches.
5. **Storage** — Embeddings and chunk content are stored in a persistent local Chroma collection.
6. **Retrieval** — A user's question is embedded and the four closest chunks are retrieved.
7. **Generation** — The OpenAI Responses API generates an answer using only the supplied retrieved passages.
8. **Citation validation** — Returned citations are checked against the source labels provided to the model.

## 📊 RAG Evaluation

The project includes a built-in evaluation workflow.

Each evaluation case runs through the normal RAG pipeline and is then scored using a structured OpenAI evaluator.

The evaluation measures:

| Metric              | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| Retrieval Relevance | How relevant the retrieved passages are to the question         |
| Groundedness        | How well the generated answer is supported by retrieved context |
| Answer Correctness  | How closely the answer matches the expected answer              |
| Citation Validity   | Whether the generated citations correspond to supplied sources  |

Each metric is scored from **0 to 1**.

The final score is the average of the four metrics:

```text
Overall Score =
(Retrieval + Groundedness + Correctness + Citation Validity) / 4
```

A case is considered **passed when the overall score is ≥ 0.70**.

## 🚀 Setup

### 1. Clone the repository

```powershell
git clone <your-repository-url>
cd web-rag-evaluator
```

### 2. Create a Python virtual environment

```powershell
python -m venv .venv
```

### 3. Install backend dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create `.env` from the provided example:

```powershell
Copy-Item .env.example .env
```

Add your API keys:

```env
CONTEXT_DEV_API_KEY=your_context_dev_api_key
OPENAI_API_KEY=your_openai_api_key
```

**Never commit `.env` to GitHub.**

### 5. Start the backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000
```

Backend:

```text
http://localhost:8000
```

Health check:

```text
http://localhost:8000/health
```

### 6. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

## ⚠️ Known Limitation

This MVP currently supports **one webpage at a time**.

Ingesting another URL replaces the existing `website_knowledge` Chroma collection and the current in-memory page state.

This keeps the project intentionally small and easy to understand while leaving room for future improvements such as multi-document ingestion, persistent cloud vector storage, and more advanced retrieval strategies.

## 🔮 Future Improvements

Potential extensions include:

* Support for multiple webpages and documents
* Metadata-based filtering
* Hybrid retrieval
* Reranking retrieved chunks
* Larger automated evaluation datasets
* Evaluation history and analytics
* Cloud-hosted vector storage
* Production deployment with persistent storage

## 🔐 Security

API keys are kept on the **FastAPI backend** and are loaded from `.env`.

The frontend communicates only with the backend API and does not directly expose the Context.dev or OpenAI credentials.

## 📌 Project Goal

The goal of this project is to provide a **small, transparent, and explainable RAG system** where both the generated answer and the quality of the retrieval pipeline can be inspected and evaluated.
