"""
MCP AI Server: Central orchestration layer for the MCP architecture.
Bridges PR analysis, retrieval-augmented generation (RAG),
chat session handling, and smart prompt management.
"""

import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import pr_routes, rag_routes, session_routes
from backend.config import API_TITLE, API_VERSION, DEBUG, HOST, PORT, LOGS_DIR
from backend.core.utils.logger import get_logger
from backend.core.mcp_server import get_mcp_server, MCPServer

logger = get_logger(__name__)

# ---------------------------------------------------------------------
# ⚙️ Initialize FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="MCP AI Orchestration Server integrating PR analysis, RAG, and prompt intelligence."
)

# ---------------------------------------------------------------------
# 🌍 Middleware (CORS, etc.)
# ---------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# 🧩 Include API Routers
# ---------------------------------------------------------------------
app.include_router(pr_routes.router, prefix="/pr", tags=["Pull Requests"])
app.include_router(rag_routes.router, prefix="/rag", tags=["Retrieval-Augmented Generation"])
app.include_router(session_routes.router, prefix="/session", tags=["Session Management"])

# ---------------------------------------------------------------------
# 🧠 Inject MCPServer Instance (Singleton)
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting MCP AI Server...")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    mcp = get_mcp_server()
    logger.info("✅ MCPServer initialized successfully.")
    logger.info(f"🗂 Logs directory: {LOGS_DIR.resolve()}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 MCP AI Server shutting down gracefully...")


# ---------------------------------------------------------------------
# 🔗 Root Endpoint
# ---------------------------------------------------------------------
@app.get("/")
async def root(mcp: MCPServer = Depends(get_mcp_server)):
    """
    Health check + MCP server overview.
    """
    return {
        "status": "ok",
        "message": f"{API_TITLE} MCP Server is running!",
        "version": API_VERSION,
        "available_services": ["PR Analysis", "RAG", "Session", "Prompting"],
    }


# ---------------------------------------------------------------------
# 💡 MCP Control Endpoints (optional utilities)
# ---------------------------------------------------------------------
@app.get("/mcp/info")
async def mcp_info(mcp: MCPServer = Depends(get_mcp_server)):
    """
    Get MCP system info and available subservices.
    """
    return {
        "mcp_status": "active",
        "services": list(vars(mcp).keys()),
        "description": "Unified AI orchestration layer for multi-agent MCP backend.",
    }


@app.get("/mcp/ping")
async def mcp_ping():
    """
    Simple MCP heartbeat endpoint.
    """
    return {"ping": "pong", "status": "alive"}


# ---------------------------------------------------------------------
# 🧭 CLI Entrypoint
# ---------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Launching MCP Server via Uvicorn...")
    uvicorn.run(
        "backend.server:app",
        host=HOST,
        port=PORT,
        reload=DEBUG,
        log_level="info"
    )
