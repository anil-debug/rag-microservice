"""Vector store backend: FAISS or Chroma."""

from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_store = None


def get_vector_store():
    """Return singleton vector store instance."""
    global _store
    if _store is None:
        backend = (settings.VECTOR_STORE_BACKEND or "faiss").lower()
        path = Path(settings.VECTOR_STORE_PATH)
        path.mkdir(parents=True, exist_ok=True)
        if backend == "chroma":
            _store = _ChromaStore(path)
        else:
            _store = _FAISSStore(path)
    return _store


class _FAISSStore:
    """FAISS-based in-memory vector store with optional persistence."""

    def __init__(self, path: Path):
        import faiss
        self.path = path
        self.index_path = path / "index.faiss"
        self.metadata_path = path / "metadata.txt"
        self._index = None
        self._texts: list[str] = []
        self._dim = None
        self._load()

    def _load(self):
        if self.index_path.exists():
            import faiss
            self._index = faiss.read_index(str(self.index_path))
            self._dim = self._index.d
            if self.metadata_path.exists():
                self._texts = self.metadata_path.read_text(encoding="utf-8").strip().split("\n")
            logger.info("Loaded FAISS index with %d vectors", len(self._texts))
        else:
            self._index = None
            self._dim = None
            self._texts = []

    def add(self, texts: list[str], vectors: list[list[float]]):
        import faiss
        import numpy as np
        arr = np.array(vectors, dtype="float32")
        dim = arr.shape[1]
        if self._index is None:
            self._dim = dim
            self._index = faiss.IndexFlatL2(dim)
        self._index.add(arr)
        self._texts.extend(texts)
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.write_text("\n".join(self._texts), encoding="utf-8")
        logger.info("Added %d vectors; total %d", len(texts), len(self._texts))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        import faiss
        import numpy as np
        if self._index is None or not self._texts:
            return []
        q = np.array([query_vector], dtype="float32")
        distances, indices = self._index.search(q, min(top_k, len(self._texts)))
        out = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self._texts):
                out.append((self._texts[idx], float(distances[0][i])))
        return out


class _ChromaStore:
    """Chroma-based persistent vector store."""

    def __init__(self, path: Path):
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection("rag", metadata={"hnsw:space": "cosine"})
        logger.info("Chroma collection 'rag' ready at %s", path)

    def add(self, texts: list[str], vectors: list[list[float]]):
        ids = [f"id_{hash(t) % (2**32)}" for t in texts]
        self.collection.add(ids=ids, documents=texts, embeddings=vectors)
        logger.info("Added %d documents to Chroma", len(texts))

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "distances"],
        )
        docs = result["documents"][0] if result["documents"] else []
        dists = result["distances"][0] if result.get("distances") else [0.0] * len(docs)
        return list(zip(docs, dists))
