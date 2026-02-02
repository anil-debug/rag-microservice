"""Tests for RAG pipeline and API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    """Health endpoint returns ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_query_empty_store():
    """Query with no ingested docs returns no-relevant-docs style response."""
    r = client.post(
        "/api/v1/query",
        json={"query": "What is RAG?", "top_k": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)


def test_ingest_invalid_empty():
    """Ingest with empty or invalid file may error per file."""
    r = client.post(
        "/api/v1/ingest",
        files=[("files", ("empty.txt", b"", "text/plain"))],
    )
    assert r.status_code == 200
    data = r.json()
    assert "ingested" in data
    assert "errors" in data
