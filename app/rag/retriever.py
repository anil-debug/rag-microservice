"""Vector search over stored documents."""

from app.db.vector_store import get_vector_store
from app.rag.embeddings import embed_single
from app.core.logger import get_logger

logger = get_logger(__name__)


def retrieve(query: str, top_k: int = 5) -> list[tuple[str, float]]:
    """Retrieve top_k most relevant chunks for the query."""
    store = get_vector_store()
    query_vector = embed_single(query)
    results = store.search(query_vector, top_k=top_k)
    return results
