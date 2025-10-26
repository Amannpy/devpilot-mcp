"""
RAG (Retrieval-Augmented Generation) routes for the MCP AI Server.
Handles document ingestion, retrieval, and contextual QA.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from backend.core.mcp_server import MCPServer

router = APIRouter()


@router.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    file_type: str = Form("text"),
    mcp_server: MCPServer = Depends(),
):
    """
    Ingest a document (text/pdf/code) into the RAG database.
    """
    try:
        contents = await file.read()
        await mcp_server.ingest_document(contents.decode(), file_type)
        return {"status": "success", "message": f"{file.filename} ingested as {file_type}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}")


@router.post("/query")
async def query_rag(query: str, mcp_server: MCPServer = Depends()):
    """
    Query the RAG pipeline for contextual responses.
    """
    try:
        result = await mcp_server.query_rag(query)
        return {"status": "success", "response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")
