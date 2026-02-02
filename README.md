# RAG Microservice

FastAPI-based RAG (Retrieval-Augmented Generation) service with `/query` and `/ingest` endpoints, configurable embeddings, vector store (FAISS or Chroma), and LLM (OpenAI or Ollama).

## Structure

```
rag-microservice/
├── app/
│   ├── main.py              # FastAPI entrypoint
│   ├── api/routes.py        # /query, /ingest
│   ├── core/config.py       # env vars, settings
│   ├── core/logger.py
│   ├── rag/
│   │   ├── embeddings.py    # embedding model
│   │   ├── retriever.py    # vector search
│   │   ├── generator.py    # LLM response
│   │   └── pipeline.py     # end-to-end RAG flow
│   ├── db/vector_store.py  # FAISS / Chroma
│   └── schemas/request.py
├── data/documents/         # PDFs, text files
├── docker/Dockerfile
├── tests/test_rag.py
├── .env.example
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Setup

1. Copy env and set your keys:

   ```bash
   cp .env.example .env
   # Edit .env: OPENAI_API_KEY or use LLM_PROVIDER=ollama
   ```

2. Install and run locally:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Or run with Docker:

   ```bash
   docker compose up --build
   ```

## API

- **GET /health** — Health check.
- **POST /api/v1/query** — RAG query. Body: `{"query": "Your question", "top_k": 5}`. Returns `{"answer": "...", "sources": [...]}`.
- **POST /api/v1/ingest** — Ingest documents. Form data: `files` (one or more PDF or text files). Returns `{"ingested": [...], "errors": [...]}`.

## Tests

From project root:

```bash
PYTHONPATH=. pytest tests/ -v
```

## Config (.env)

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` |
| `VECTOR_STORE_BACKEND` | `faiss` or `chroma` | `faiss` |
| `VECTOR_STORE_PATH` | Persistence path | `data/vector_store` |
| `LLM_PROVIDER` | `openai` or `ollama` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | OpenAI model | `gpt-4o-mini` |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | For Ollama | `http://localhost:11434` / `llama3.2` |
