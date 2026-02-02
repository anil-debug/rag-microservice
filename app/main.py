"""FastAPI entrypoint for RAG microservice."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting RAG microservice")
    yield
    logger.info("Shutting down RAG microservice")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
)

app.include_router(router, prefix="/api/v1", tags=["rag"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
