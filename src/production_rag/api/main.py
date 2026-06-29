"""
Production FastAPI application for RAG pipeline.

Provides REST API endpoints for chat, health checks, and file ingestion
with built-in security, rate limiting, and input validation.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from production_rag.api.routes.chat import router as chat_router
from production_rag.api.routes.health import router as health_router
from production_rag.core.pipeline import ProductionRAGPipeline
from production_rag.security import SecurityManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle.

    The RAG pipeline is created here (not at import time) so that:
      • Tests can import ``app`` without a live OpenAI key.
      • The pipeline object is available on ``app.state`` for request handlers.
    """
    # ── Startup ──────────────────────────────────────────────────────
    app.state.pipeline = ProductionRAGPipeline()
    app.state.security = SecurityManager()
    yield
    # ── Shutdown (nothing to clean up for now) ────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Production RAG API",
        description="Production-grade Retrieval-Augmented Generation API",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    # Mount static UI
    import os
    static_dir = os.path.join(os.path.dirname(__file__), "..", "ui", "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def serve_landing():
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/app")
    async def serve_app():
        return FileResponse(os.path.join(static_dir, "app.html"))

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
