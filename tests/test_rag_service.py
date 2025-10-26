import pytest
from unittest.mock import AsyncMock
from backend.services.rag_service import RAGService

@pytest.mark.asyncio
async def test_query_rag_basic(monkeypatch):
    service = RAGService()

    # Mock retriever + model
    service.rag_manager = AsyncMock()
    service.rag_manager.retrieve_context.return_value = ["context A", "context B"]
    service.rag_manager.generate_response.return_value = "Mocked RAG answer."

    query = "What is the purpose of this function?"
    response = await service.query(query, session_id="s123")

    assert isinstance(response, dict)
    assert "response" in response
    assert response["response"] == "Mocked RAG answer."
    assert "context" in response

@pytest.mark.asyncio
async def test_query_rag_empty_context(monkeypatch):
    service = RAGService()
    service.rag_manager = AsyncMock()
    service.rag_manager.retrieve_context.return_value = []
    service.rag_manager.generate_response.return_value = "No context found."

    result = await service.query("Explain code")
    assert result["response"] == "No context found."
