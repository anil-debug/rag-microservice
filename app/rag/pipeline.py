"""End-to-end RAG flow: ingest, retrieve, generate."""

import io

from app.rag.retriever import retrieve
from app.rag.generator import generate
from app.db.vector_store import get_vector_store
from app.rag.embeddings import embed
from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """RAG pipeline: retrieval + generation."""

    async def query(self, query: str, top_k: int = 5) -> tuple[str, list[str]]:
        """Retrieve context and generate answer."""
        results = retrieve(query, top_k=top_k)
        chunks = [text for text, _ in results]
        if not chunks:
            return "No relevant documents found.", []
        answer = generate(query, chunks)
        return answer, chunks

    def ingest_document(self, content: bytes, filename: str) -> None:
        """Parse document, chunk, embed, and add to vector store."""
        text = _extract_text(content, filename)
        if not text.strip():
            raise ValueError(f"Empty or unreadable document: {filename}")
        chunks = _chunk_text(text)
        vectors = embed(chunks)
        store = get_vector_store()
        store.add(chunks, vectors)
        logger.info("Ingested %d chunks from %s", len(chunks), filename)


def _extract_text(content: bytes, filename: str) -> str:
    """Extract plain text from PDF or raw bytes."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return content.decode("utf-8", errors="replace")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks
