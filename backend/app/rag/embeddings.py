"""Embedding model for document and query vectors."""

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load embedding model."""
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", settings.EMBEDDING_MODEL)
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Compute embeddings for a list of texts."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True).tolist()


def embed_single(text: str) -> list[float]:
    """Compute embedding for a single text."""
    return embed([text])[0]
