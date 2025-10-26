"""
Main FastAPI application entrypoint for MCP AI Server.
Handles initialization, route registration, middleware, and global exceptions.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import pr_routes, rag_routes, session_routes
from backend.core.utils.logger import get_logger
from backend.api.routes.utils import error_response

logger = get_logger(__name__)

# ----------------------------------------------------
# App Initialization
# ----------------------------------------------------
app = FastAPI(
    title="MCP AI Server",
    version="2.0",
    description="Modular backend for PR analysis, documentation insights, and RAG-powered development support.",
)


# ----------------------------------------------------
# Middleware
# ----------------------------------------------------
# Enable CORS for frontend integration (Flask/Django/React/Vue etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------
# Exception Handlers
# ----------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler to avoid leaking raw tracebacks in production.
    """
    logger.exception(f"Unhandled error at {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal Server Error. Please try again later.",
        },
    )


# ----------------------------------------------------
# Route Registration
# ----------------------------------------------------
app.include_router(pr_routes.router, prefix="/pr", tags=["PR"])
app.include_router(rag_routes.router, prefix="/rag", tags=["RAG"])
app.include_router(session_routes.router, prefix="/session", tags=["Session"])


# ----------------------------------------------------
# Health Check Endpoint
# ----------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """
    Simple health check endpoint.
    """
    logger.info("Health check ping received.")
    return {
        "status": "ok",
        "message": "🚀 MCP AI Server is running",
        "version": app.version,
    }
