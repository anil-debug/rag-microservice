"""Environment variables and settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings from env."""

    PROJECT_NAME: str = "RAG Microservice"
    VERSION: str = "0.1.0"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store (faiss | chroma)
    VECTOR_STORE_BACKEND: str = "faiss"
    VECTOR_STORE_PATH: str = "data/vector_store"

    # LLM (openai | ollama | etc.)
    LLM_PROVIDER: str = "openai"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
