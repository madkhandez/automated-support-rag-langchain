import os
import time
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from ..models.schemas import ErrorResponse
from fastapi.responses import JSONResponse

def setup_security_middleware(app):
    """Configure all security middleware for the FastAPI app."""
    
    # 1. CORS
    allowed_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8501").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 2. Custom Logging & Error Handling Middleware
    class SecurityLoggingMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            start_time = time.time()
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                response.headers["X-Process-Time"] = str(process_time)
                return response
            except Exception as e:
                # Catch unhandled exceptions and return structured JSON
                # Prevents raw exception traces from leaking to client
                return JSONResponse(
                    status_code=500,
                    content=ErrorResponse(
                        error="Internal Server Error",
                        detail="An unexpected error occurred. Please try again later."
                    ).model_dump()
                )
                
    app.add_middleware(SecurityLoggingMiddleware)
