from fastapi import APIRouter, Request
from ..models.schemas import HealthResponse, StatsResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Basic health check endpoint."""
    app = request.app
    
    # Check components
    vector_store_status = "connected" if hasattr(app.state, 'rag_pipeline') and app.state.rag_pipeline else "disconnected"
    llm_status = "connected" # Simplified for this demo
    
    return HealthResponse(
        status="healthy",
        vector_store=vector_store_status,
        llm=llm_status
    )

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    # In a real app, this would query the ProductionMonitor or DB
    return StatsResponse(
        total_queries=100,
        avg_latency_ms=450.5,
        daily_cost_usd=2.50
    )
