from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import pr_routes, session_routes

#from backend.api.routes import rag_routes, pr_routes, session_routes

app = FastAPI(
    title="MCP AI Server",
    version="0.1.0",
    description="Modular AI backend integrating RAG, PR analysis, and intelligent prompting.",
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(rag_routes.router, prefix="/api/rag", tags=["RAG"])
app.include_router(pr_routes.router, prefix="/api/pr", tags=["Pull Requests"])
app.include_router(session_routes.router, prefix="/api/session", tags=["Session"])

@app.get("/")
async def root():
    return {"message": "Welcome to MCP AI Server 🚀"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.app:app", host="0.0.0.0", port=8000, reload=True)
