"""
RAG (Retrieval-Augmented Generation) related API routes.
Supports document ingestion, retrieval, and intelligent question answering.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from backend.services.rag_service import RAGService
from backend.core.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])

rag_service = RAGService()


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query for RAG search or answer generation")
    session_id: str | None = Field(None, description="Optional session ID for maintaining context")


class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Raw text content to ingest into the RAG system")
    namespace: str | None = Field("default", description="Optional namespace for indexing separation")


@router.post("/ingest/text")
async def ingest_text(request: IngestTextRequest):
    """
    Ingest raw text into the RAG knowledge base.
    """
    try:
        logger.info(f"🧠 Ingesting text into RAG namespace: {request.namespace}")
        result = await rag_service.ingest_text(request.text, namespace=request.namespace)
        return {"status": "success", "message": "Text ingested successfully", "result": result}

    except Exception as e:
        logger.exception(f"❌ Failed to ingest text: {e}")
        raise HTTPException(status_code=500, detail="Failed to ingest text")


@router.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...), namespace: str = Form("default")):
    """
    Ingest a file (PDF, TXT, or code) into the RAG system.
    """
    try:
        logger.info(f"📄 Ingesting file: {file.filename} into namespace '{namespace}'")
        content = await file.read()
        result = await rag_service.ingest_file(file.filename, content, namespace)
        return {"status": "success", "message": f"File '{file.filename}' ingested successfully", "result": result}

    except Exception as e:
        logger.exception(f"❌ File ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="File ingestion failed")


@router.post("/query")
async def query_rag(request: QueryRequest):
    """
    Perform a RAG search or generate an AI-enhanced answer from the indexed knowledge base.
    """
    try:
        logger.info(f"🔍 RAG query received: {request.query}")
        response = await rag_service.query(request.query, session_id=request.session_id)
        logger.info("✅ RAG query processed successfully")
        return {"status": "success", "response": response}

    except Exception as e:
        logger.exception(f"❌ RAG query failed: {e}")
        raise HTTPException(status_code=500, detail="Error while processing RAG query")
