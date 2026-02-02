"""LLM response generation."""

import httpx

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

# Message returned when LLM is unreachable (Ollama not running / OpenAI key missing)
LLM_UNREACHABLE_MSG = (
    "LLM is not reachable. For Ollama: start it on your host (e.g. `ollama serve` and `ollama pull llama3.2`). "
    "Or set LLM_PROVIDER=openai and OPENAI_API_KEY in .env. "
    "Below is the retrieved context only (no generated answer):"
)


def _generate_openai(query: str, context: str) -> str:
    """Generate using OpenAI API. On auth/connection error, return retrieval-only fallback."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Answer based only on the provided context. If the context does not contain the answer, say so.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}",
                },
            ],
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        logger.exception("OpenAI generation failed: %s", e)
        if "api_key" in str(e).lower() or "connection" in str(e).lower() or "refused" in str(e).lower():
            return f"{LLM_UNREACHABLE_MSG}\n\n---\n\n{context}"
        raise


def generate(query: str, context_chunks: list[str]) -> str:
    """Generate answer from query and retrieved context using LLM."""
    context = "\n\n".join(context_chunks)

    if settings.LLM_PROVIDER == "openai":
        return _generate_openai(query, context)
    if settings.LLM_PROVIDER == "ollama":
        return _generate_ollama(query, context)
    raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")


def _generate_ollama(query: str, context: str) -> str:
    """Generate using Ollama API. On connection error, return retrieval-only fallback."""
    try:
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on the context:",
                "stream": False,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        logger.warning("Ollama unreachable (%s). Returning retrieval-only fallback.", e)
        return f"{LLM_UNREACHABLE_MSG}\n\n---\n\n{context}"
    except Exception as e:
        logger.exception("Ollama generation failed: %s", e)
        raise
