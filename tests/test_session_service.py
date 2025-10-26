import pytest
from backend.services.session_service import SessionService

@pytest.mark.asyncio
async def test_create_and_get_session():
    service = SessionService()
    user_id = "test_user"
    session_id = await service.create_session(user_id)

    assert isinstance(session_id, str)
    history = await service.get_history(session_id)
    assert history == []

@pytest.mark.asyncio
async def test_add_and_retrieve_messages():
    service = SessionService()
    session_id = await service.create_session("user_123")

    await service.add_message(session_id, "Hello!", role="user")
    await service.add_message(session_id, "Hi there!", role="assistant")

    history = await service.get_history(session_id)
    assert len(history) == 2
    assert history[0]["message"] == "Hello!"
    assert history[1]["role"] == "assistant"

@pytest.mark.asyncio
async def test_add_message_invalid_session():
    service = SessionService()
    with pytest.raises(ValueError):
        await service.add_message("nonexistent_id", "This should fail")
