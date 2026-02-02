"""API routes for /query and /ingest."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import List

from app.rag.pipeline import RAGPipeline
from app.schemas.request import QueryRequest, QueryResponse, IngestResponse

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run RAG: retrieve relevant docs and generate answer."""
    pipeline = RAGPipeline()
    try:
        answer, sources = await pipeline.query(request.query, top_k=request.top_k)
        return QueryResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    """Ingest documents (PDF, text) into the vector store."""
    pipeline = RAGPipeline()
    ingested = []
    errors = []
    for f in files:
        try:
            content = await f.read()
            pipeline.ingest_document(content, filename=f.filename or "unknown")
            ingested.append(f.filename or "unknown")
        except Exception as e:
            errors.append({"file": f.filename, "error": str(e)})
    return IngestResponse(ingested=ingested, errors=errors)
