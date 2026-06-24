import time
from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from ..models.schemas import ChatRequest, ChatResponse, IngestResponse
from ...security.input_validator import InputSecurityLayer

router = APIRouter()
security = InputSecurityLayer()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Main chat endpoint for the RAG application."""
    start_time = time.time()
    
    # 1. Rate Limiting
    user_id = request.user_id or "anonymous"
    if not security.rate_limit_check(user_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
    # 2. Input Validation (Prompt Injection & PII)
    val_result = security.validate_question(request.question)
    if not val_result["is_valid"]:
        raise HTTPException(status_code=400, detail=val_result["reason"])
        
    # 3. Process via RAG Pipeline
    app = req.app
    if not hasattr(app.state, 'rag_pipeline'):
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
    try:
        # In a real app, we'd use the LangGraph agent here
        # For simplicity, we assume rag_pipeline.query returns dict with answer and sources
        result = app.state.rag_pipeline.query(request.question)
        
        latency = int((time.time() - start_time) * 1000)
        
        return ChatResponse(
            answer=result.get("answer", "Error generating answer."),
            sources=result.get("sources", []),
            latency_ms=latency,
            token_count=result.get("token_count", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile = File(...), req: Request = None):
    """Endpoint to upload and index documents."""
    # 1. Validate file
    content = await file.read()
    val_result = security.validate_document(content, file.filename)
    if not val_result["is_valid"]:
        raise HTTPException(status_code=400, detail=val_result["reason"])
        
    # 2. Save temporarily and ingest
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
        tmp.write(content)
        tmp_path = tmp.name
        
    try:
        app = req.app
        result = app.state.rag_pipeline.ingest_documents(tmp_path)
        return IngestResponse(
            status="success",
            chunks_indexed=result.get("chunks_indexed", 0),
            doc_id=file.filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
