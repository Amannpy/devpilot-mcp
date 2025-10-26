from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.prompt_service import build_prompt, detect_intent

router = APIRouter(prefix="/session", tags=["Session / Chat"])


class ChatRequest(BaseModel):
    prompt: str
    code: str | None = None
    context: str | None = None


@router.post("/chat")
async def chat_with_model(payload: ChatRequest):
    """
    Simulate a chat-like exchange where we intelligently build a prompt.
    (Later this will interface with the model for reasoning-based responses.)
    """
    try:
        intent = detect_intent(payload.prompt)
        smart_prompt = build_prompt(payload.prompt, payload.code, payload.context)

        return {
            "intent": intent.value,
            "built_prompt": smart_prompt,
            "message": "Prompt successfully constructed for model input."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
