# RAG Microservice

Backend (FastAPI) + frontend (Streamlit) for RAG: ingest documents, query with retrieval-augmented generation. No `.env` required for Docker; optional for overrides.

## Architecture

```
rag-microservice/
├── backend/                 # FastAPI RAG API
│   ├── app/
│   │   ├── main.py
│   │   ├── api/routes.py    # /query, /ingest
│   │   ├── core/config.py, logger.py
│   │   ├── rag/             # embeddings, retriever, generator, pipeline
│   │   ├── db/vector_store.py
│   │   └── schemas/
│   ├── tests/
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                # Streamlit UI
│   ├── app.py               # Query + Ingest tabs
│   ├── requirements.txt
│   └── Dockerfile
├── data/documents/          # PDFs, text files (mounted in backend)
├── docker-compose.yml       # backend + frontend (no .env required)
├── .env.example
└── README.md
```

## Run with Docker (recommended)

No `.env` needed; backend uses Ollama by default (run Ollama on host if you use it).

```bash
docker compose up --build
```

- **Backend API:** http://localhost:8000  
- **Frontend (Streamlit):** http://localhost:8501  

Use the Streamlit app to **Query** (ask questions) and **Ingest** (upload PDFs/text).

Optional overrides (e.g. OpenAI): create `.env` from `.env.example`, then:

```bash
docker compose --env-file .env up --build
```

## Test after build

1. **Health check (backend)**  
   ```bash
   curl http://localhost:8000/health
   ```  
   Expected: `{"status":"ok"}`

2. **Ingest a document**  
   - **Streamlit:** Open http://localhost:8501 → **Ingest** tab → upload a PDF or `.txt` → click **Ingest**.  
   - **curl:**  
     ```bash
     curl -X POST http://localhost:8000/api/v1/ingest \
       -F "files=@data/documents/sample.txt"
     ```  
   (Create `data/documents/sample.txt` with some text if needed.)

3. **Query**  
   - **Streamlit:** **Query** tab → type a question → **Get answer**.  
   - **curl:**  
     ```bash
     curl -X POST http://localhost:8000/api/v1/query \
       -H "Content-Type: application/json" \
       -d '{"query": "What is this document about?", "top_k": 3}'
     ```  
   Expected: JSON with `answer` and `sources`.

4. **API docs**  
   Open http://localhost:8000/docs for Swagger UI.

## Run locally

**Backend**

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend** (in another terminal)

```bash
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run app.py --server.port 8501
```

Open http://localhost:8501.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/v1/query` | RAG query. JSON: `{"query": "...", "top_k": 5}` → `{"answer": "...", "sources": [...]}` |
| POST | `/api/v1/ingest` | Ingest docs. Form: `files` (PDF/text) → `{"ingested": [...], "errors": [...]}` |

## Config (optional .env)

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODEL` | Sentence-transformers model | `all-MiniLM-L6-v2` |
| `VECTOR_STORE_BACKEND` | `faiss` or `chroma` | `faiss` |
| `VECTOR_STORE_PATH` | Persistence path | `data/vector_store` |
| `LLM_PROVIDER` | `openai` or `ollama` | `ollama` in Docker |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OLLAMA_BASE_URL` / `OLLAMA_MODEL` | For Ollama | `http://localhost:11434` / `llama3.2` |

## Tests

```bash
cd backend && PYTHONPATH=. pytest tests/ -v
```
