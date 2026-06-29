import time
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from ..models.schemas import ChatRequest, ChatResponse, IngestResponse
from ...security import SecurityManager

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Main chat endpoint for the RAG application."""
    start_time = time.time()

    # Get security manager from app state
    security: SecurityManager = req.app.state.security

    # Full security check: rate limiting + input validation
    user_id = request.user_id or "anonymous"
    val_result = security.validate_and_process(request.question, client_id=user_id)
    if not val_result.is_valid:
        # Distinguish rate-limit from validation errors
        if val_result.reason and "Rate limit" in val_result.reason:
            raise HTTPException(status_code=429, detail=val_result.reason)
        raise HTTPException(status_code=400, detail=val_result.reason)

    # 3. Process via RAG Pipeline
    pipeline = getattr(req.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")

    try:
        result = pipeline.query(request.question)

        processing_time = time.time() - start_time
        latency = int(processing_time * 1000)

        return ChatResponse(
            answer=result.get("answer", "Error generating answer."),
            sources=result.get("sources", []),
            latency_ms=latency,
            processing_time=round(processing_time, 4),
            token_count=result.get("token_count", 0),
        )
    except Exception as e:
        error_str = str(e).lower()
        if "quota" in error_str or "insufficient_quota" in error_str or "billing" in error_str or "api key" in error_str:
            raise HTTPException(
                status_code=502,
                detail=(
                    "All LLM providers failed. "
                    "Please check your API keys or quotas for the configured providers in .env."
                ),
            )
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...), req: Request = None):
    """Endpoint to upload and index documents."""
    # 1. Validate file via security manager
    security: SecurityManager = req.app.state.security
    val_result = security.input_validator.validate_file_type(file.filename)
    if not val_result.is_valid:
        raise HTTPException(status_code=400, detail=val_result.reason)

    content = await file.read()

    # 2. Save temporarily and ingest
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        pipeline = req.app.state.pipeline
        result = pipeline.ingest_documents(tmp_path, original_filename=file.filename)
        return IngestResponse(
            status="success",
            chunk_count=result.get("chunks_indexed", 0),
            filename=file.filename,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
