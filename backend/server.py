"""
MCP AI Server: central orchestration layer.
Handles PR analysis, RAG queries, chat sessions, and prompt management.
"""

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import pr_routes, rag_routes, session_routes
from backend.core.utils.logger import get_logger
from backend.config import API_TITLE, API_VERSION, DEBUG, HOST, PORT, LOGS_DIR
from backend.core.mcp_server import MCPServer

logger = get_logger(__name__)

# -----------------------------
# Initialize FastAPI app
# -----------------------------
app = FastAPI(title=API_TITLE, version=API_VERSION)

# -----------------------------
# CORS for frontend or external calls
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Instantiate MCPServer singleton
# -----------------------------
logger.info("Initializing MCPServer instance...")
mcp_server = MCPServer()
logger.info("✅ MCPServer initialized successfully")

# -----------------------------
# Dependency for injecting MCPServer
# -----------------------------
def get_mcp_server() -> MCPServer:
    return mcp_server

# -----------------------------
# Include API routes with dependency injection
# -----------------------------
app.include_router(pr_routes.router, prefix="/pr", tags=["PR"], dependencies=[Depends(get_mcp_server)])
app.include_router(rag_routes.router, prefix="/rag", tags=["RAG"], dependencies=[Depends(get_mcp_server)])
app.include_router(session_routes.router, prefix="/session", tags=["Session"], dependencies=[Depends(get_mcp_server)])

# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
async def root():
    return {"message": f"{API_TITLE} MCP Server is running!"}

# -----------------------------
# Startup and shutdown events
# -----------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 MCP AI Server starting up...")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logs directory ensured at {LOGS_DIR}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 MCP AI Server shutting down...")

# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    uvicorn.run("backend.server:app", host=HOST, port=PORT, reload=DEBUG)
