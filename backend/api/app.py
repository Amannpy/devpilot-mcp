from fastapi import FastAPI
from backend.api.routes import pr_routes, rag_routes, session_routes

app = FastAPI(title="MCP AI Server", version="2.0")

# Include route modules
app.include_router(pr_routes.router, prefix="/pr", tags=["PR"])
app.include_router(rag_routes.router, prefix="/rag", tags=["RAG"])
app.include_router(session_routes.router, prefix="/session", tags=["Session"])

@app.get("/")
async def root():
    return {"message": "MCP AI Server is running"}
