import time
from fastapi import APIRouter, Request
from ..models.schemas import HealthResponse, StatsResponse

router = APIRouter()

_start_time = time.time()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Basic health check endpoint."""
    app = request.app

    # Check components
    vector_store_status = (
        "connected"
        if hasattr(app.state, "pipeline") and app.state.pipeline
        else "disconnected"
    )
    llm_status = "connected"  # Simplified for this demo
    
    from production_rag.core.llm_factory import LLMFactory
    try:
        _, active_provider = LLMFactory.get_llm()
    except Exception:
        active_provider = "none"
        llm_status = "disconnected"
        
    provider_statuses = LLMFactory.get_available_providers()

    return HealthResponse(
        status="healthy" if llm_status == "connected" else "degraded",
        active_llm_provider=active_provider,
        provider_statuses=provider_statuses,
        version="1.0.0",
        uptime=round(time.time() - _start_time, 2),
        vector_store=vector_store_status,
        llm=llm_status,
    )

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    # In a real app, this would query the ProductionMonitor or DB
    return StatsResponse(
        total_queries=100,
        avg_latency_ms=450.5,
        daily_cost_usd=2.50,
    )
