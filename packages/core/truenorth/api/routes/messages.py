"""Send a message to a session."""

from __future__ import annotations
import os
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

router = APIRouter()

class SendMessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    session_id: str
    response: str
    emotion_state: str
    is_complete: bool
    is_escalated: bool
    profile: dict
    cost_usd: float

@router.post("/{session_id}/messages", response_model=MessageResponse)
async def send_message(session_id: str, req: SendMessageRequest,
                       x_api_key: str | None = Header(default=None)):
    from truenorth.core.engine import TrueNorthEngine
    from truenorth.llm.router import LLMRouter
    from truenorth.api.main import db_store
    from truenorth.api.routes.sessions import _load_goal_config, _verify_api_key

    _verify_api_key(x_api_key)

    if not db_store:
        raise HTTPException(503, "Database not available")

    from truenorth.storage.models import Session as SessionModel
    async with db_store.session_factory() as db:
        row = await db.get(SessionModel, session_id)
    if not row:
        raise HTTPException(404, "Session not found")

    config = _load_goal_config(row.goal_id)
    state = await db_store.load_state(session_id, config.model_dump())
    if not state:
        raise HTTPException(404, "Session state not found")

    if state.completed:
        raise HTTPException(400, "Session already completed")

    llm_router = LLMRouter(max_cost_usd=float(os.getenv("MAX_COST_PER_SESSION_USD", "0.05")))
    engine = TrueNorthEngine(config, llm_router)

    new_state, response_text = await engine.process_turn(state, req.message)
    await db_store.save_state(new_state)

    return MessageResponse(
        session_id=session_id,
        response=response_text,
        emotion_state=new_state.emotion_state,
        is_complete=new_state.completed,
        is_escalated=new_state.escalated,
        profile=new_state.collected_fields,
        cost_usd=new_state.cost_usd,
    )
