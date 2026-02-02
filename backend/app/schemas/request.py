"""Request/response schemas for API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for /query."""

    query: str = Field(..., description="User question")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")


class QueryResponse(BaseModel):
    """Response for /query."""

    answer: str
    sources: list[str] = Field(default_factory=list, description="Retrieved chunk texts")


class IngestResponse(BaseModel):
    """Response for /ingest."""

    ingested: list[str] = Field(default_factory=list, description="Successfully ingested filenames")
    errors: list[dict] = Field(default_factory=list, description="Per-file errors if any")
