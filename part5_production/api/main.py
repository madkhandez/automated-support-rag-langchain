"""
Production FastAPI application for RAG pipeline.

Provides REST API endpoints for chat, health checks, and file ingestion
with built-in security, rate limiting, and input validation.
"""

import os
import tempfile
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from part5_production.security import (
    InputValidator,
    OutputFilter,
    RateLimiter,
    SecurityManager,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    question: str = Field(..., min_length=1, max_length=2000, description="User question")
    session_id: Optional[str] = Field(None, description="Optional session ID")


class ChatResponse(BaseModel):
    """Response body for the chat endpoint."""
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float = Field(default=0.0)


class HealthResponse(BaseModel):
    """Response body for the health endpoint."""
    status: str = "healthy"
    version: str = "1.0.0"
    uptime: float = 0.0


class IngestResponse(BaseModel):
    """Response body for the file ingest endpoint."""
    filename: str
    chunk_count: int
    status: str = "success"


# ---------------------------------------------------------------------------
# Pipeline stub (real implementation lives in core/)
# ---------------------------------------------------------------------------

class ProductionRAGPipeline:
    """Minimal stub of the RAG pipeline used by the API layer.
    
    In production, this delegates to the full pipeline in core/.
    """

    def __init__(self) -> None:
        self.is_ready = True

    def query(self, question: str) -> dict[str, Any]:
        """Process a question through the RAG pipeline."""
        return {
            "answer": f"Answer to: {question}",
            "sources": ["doc1.txt"],
            "confidence": 0.85,
        }

    def ingest_file(self, file_path: str, filename: str) -> int:
        """Ingest a file into the vector store. Returns chunk count."""
        # Simple placeholder: count ~200-char chunks
        with open(file_path, "r") as f:
            content = f.read()
        chunk_count = max(1, len(content) // 200)
        return chunk_count


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

_start_time = time.time()


def create_app(
    pipeline: Optional[ProductionRAGPipeline] = None,
    security: Optional[SecurityManager] = None,
) -> FastAPI:
    """Create and configure the FastAPI application.
    
    Args:
        pipeline: Optional RAG pipeline instance (created if not provided).
        security: Optional SecurityManager instance (created if not provided).
        
    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="Production RAG API",
        description="Production-grade Retrieval-Augmented Generation API",
        version="1.0.0",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Attach dependencies to app state
    app.state.pipeline = pipeline or ProductionRAGPipeline()
    app.state.security = security or SecurityManager(
        max_requests=20, window_seconds=60,
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/api/v1/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            version="1.0.0",
            uptime=time.time() - _start_time,
        )

    @app.post("/api/v1/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest, req: Request) -> ChatResponse:
        """Process a chat question through the RAG pipeline."""
        security_mgr: SecurityManager = req.app.state.security
        rag_pipeline: ProductionRAGPipeline = req.app.state.pipeline

        # Derive a client ID from request
        client_id = req.client.host if req.client else "unknown"

        # Security validation
        validation = security_mgr.validate_and_process(
            request.question, client_id=client_id,
        )

        if not validation.is_valid:
            if "Rate limit" in (validation.reason or ""):
                raise HTTPException(status_code=429, detail=validation.reason)
            raise HTTPException(status_code=400, detail=validation.reason)

        # Process through pipeline
        start = time.time()
        result = rag_pipeline.query(request.question)
        processing_time = time.time() - start

        # Filter output
        answer = security_mgr.filter_response(result.get("answer", ""))

        return ChatResponse(
            answer=answer,
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            processing_time=processing_time,
        )

    @app.post("/api/v1/ingest", response_model=IngestResponse)
    async def ingest(file: UploadFile = File(...), req: Request = None) -> IngestResponse:
        """Ingest a document into the RAG pipeline."""
        security_mgr: SecurityManager = req.app.state.security
        rag_pipeline: ProductionRAGPipeline = req.app.state.pipeline

        # Validate file type
        filename = file.filename or "unknown"
        file_validation = security_mgr.input_validator.validate_file_type(filename)
        if not file_validation.is_valid:
            raise HTTPException(status_code=400, detail=file_validation.reason)

        # Save uploaded file to temp location and ingest
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            chunk_count = rag_pipeline.ingest_file(tmp_path, filename)
            return IngestResponse(
                filename=filename,
                chunk_count=chunk_count,
                status="success",
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    return app


# Default app instance for uvicorn
app = create_app()


def main() -> None:
    """Run the API server."""
    import uvicorn
    print("🚀 Starting Production RAG API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
