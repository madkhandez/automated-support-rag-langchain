"""
Part 5 — API Schemas: Pydantic models for the FastAPI layer.

Every request and response flowing through the REST API is validated and
serialised via these models.  Using Pydantic v2 for speed and clarity.
"""

from __future__ import annotations

from typing import Optional, Any

from pydantic import BaseModel, Field


# ── Request Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Payload for POST /api/v1/chat."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question (max 2 000 chars).",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session ID for conversational memory.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for rate-limiting.",
    )


# ── Response Models ──────────────────────────────────────────────────

class ChatResponse(BaseModel):
    """Payload returned by POST /api/v1/chat."""

    answer: str = Field(..., description="Generated answer text.")
    sources: list[Any] = Field(
        default_factory=list,
        description="Source documents (list of dicts with filename and excerpt).",
    )
    latency_ms: int = Field(
        ..., ge=0, description="End-to-end latency in milliseconds."
    )
    processing_time: Optional[float] = Field(
        default=None,
        description="End-to-end processing time in seconds.",
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Approximate token count of the answer (if available).",
    )


class IngestResponse(BaseModel):
    """Payload returned by POST /api/v1/ingest."""

    status: str = Field(..., description="'success' or 'error'.")
    chunk_count: int = Field(
        ..., ge=0, description="Number of chunks written to the vector store.",
    )
    filename: str = Field(
        ..., description="Original filename of the ingested document.",
    )


class HealthResponse(BaseModel):
    """Payload returned by GET /api/v1/health."""

    status: str = Field(..., description="Overall system status.")
    active_llm_provider: Optional[str] = Field(None, description="Current LLM provider in use.")
    provider_statuses: list[dict] = Field(default_factory=list, description="Status of all providers.")
    version: str = Field(..., description="API version string.")
    uptime: float = Field(..., ge=0.0, description="Uptime in seconds.")
    vector_store: str = Field(
        ..., description="'connected' or 'disconnected'."
    )
    llm: str = Field(
        ..., description="'connected' or 'disconnected'."
    )


class StatsResponse(BaseModel):
    """Payload returned by GET /api/v1/stats."""

    total_queries: int = Field(..., ge=0, description="Lifetime query count.")
    avg_latency_ms: float = Field(
        ..., ge=0.0, description="Average response latency (ms)."
    )
    daily_cost_usd: float = Field(
        ..., ge=0.0, description="Estimated cost for the current day."
    )


class ErrorResponse(BaseModel):
    """Structured error payload returned by the API."""

    error: str = Field(..., description="Short error description.")
    detail: Optional[str] = Field(
        default=None,
        description="Extended detail / traceback (when safe to expose).",
    )


# ── Standalone entrypoint ────────────────────────────────────────────
def main() -> None:
    """Quick validation of every schema."""
    print("=" * 60)
    print("Part 5 · API Schemas Demo")
    print("=" * 60)

    # ChatRequest
    req = ChatRequest(question="What is RAG?", session_id="s-123", user_id="u-1")
    print(f"\n  ChatRequest:    {req.model_dump()}")

    # ChatResponse
    resp = ChatResponse(
        answer="RAG is Retrieval-Augmented Generation.",
        sources=["doc1.pdf", "doc2.md"],
        latency_ms=342,
        token_count=48,
    )
    print(f"  ChatResponse:   {resp.model_dump()}")

    # IngestResponse
    ingest = IngestResponse(status="success", chunk_count=12, filename="doc.pdf")
    print(f"  IngestResponse: {ingest.model_dump()}")

    # HealthResponse
    health = HealthResponse(status="ok", version="1.0.0", uptime=42.5, vector_store="connected", llm="connected")
    print(f"  HealthResponse: {health.model_dump()}")

    # StatsResponse
    stats = StatsResponse(total_queries=1042, avg_latency_ms=287.5, daily_cost_usd=1.23)
    print(f"  StatsResponse:  {stats.model_dump()}")

    # ErrorResponse
    err = ErrorResponse(error="rate_limited", detail="Max 20 req/min exceeded.")
    print(f"  ErrorResponse:  {err.model_dump()}")

    print("\n✅ All schemas validated.")


if __name__ == "__main__":
    main()
