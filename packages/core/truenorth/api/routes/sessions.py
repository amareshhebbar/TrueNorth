"""Session CRUD endpoints."""

from __future__ import annotations
import uuid
import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

router = APIRouter()

class CreateSessionRequest(BaseModel):
    goal_id: str
    user_id: str | None = None
    resume_session_id: str | None = None

class SessionResponse(BaseModel):
    session_id: str
    goal_id: str
    welcome_message: str
    is_resumed: bool = False

@router.post("", response_model=SessionResponse)
async def create_session(req: CreateSessionRequest,
                          x_api_key: str | None = Header(default=None)):
    _verify_api_key(x_api_key)

    from truenorth.core.engine import TrueNorthEngine
    from truenorth.llm.router import LLMRouter
    from truenorth.api.main import db_store

    config = _load_goal_config(req.goal_id)
    router_llm = LLMRouter(max_cost_usd=float(os.getenv("MAX_COST_PER_SESSION_USD", "0.05")))
    engine = TrueNorthEngine(config, router_llm)

    if req.resume_session_id and db_store:
        existing = await db_store.load_state(req.resume_session_id, config.model_dump())
        if existing and not existing.completed:
            welcome = await engine.generate_resume_message(existing)
            return SessionResponse(
                session_id=existing.session_id,
                goal_id=existing.goal_id,
                welcome_message=welcome,
                is_resumed=True,
            )

    session_id = str(uuid.uuid4())
    state = engine.create_initial_state(session_id, req.user_id)
    welcome = await engine.generate_welcome_message(state)

    if db_store:
        await db_store.save_state(state)

    return SessionResponse(
        session_id=session_id,
        goal_id=req.goal_id,
        welcome_message=welcome,
        is_resumed=False,
    )

@router.get("/{session_id}")
async def get_session(session_id: str, x_api_key: str | None = Header(default=None)):
    _verify_api_key(x_api_key)

    from truenorth.core.session_manager import SessionManager
    from truenorth.api.main import db_store, redis_store

    sm = SessionManager(postgres=db_store, redis=redis_store)
    state = await sm.load(session_id)

    if not state:
        raise HTTPException(404, "Session not found")

    return {"session_id": session_id, "state": state}

def _load_goal_config(goal_id: str):
    from truenorth.core.yaml_loader import YamlLoader
    loader = YamlLoader(Path("examples/goals"))
    candidates = list(Path(".").rglob(f"{goal_id}.yaml"))
    if not candidates:
        raise HTTPException(404, f"Goal config '{goal_id}' not found")
    return loader.load(candidates[0])

def _verify_api_key(key: str | None):
    expected = os.getenv("TRUENORTH_API_KEY", "")
    if expected and key != expected:
        raise HTTPException(401, "Invalid API key")

@router.get("")
async def list_sessions(
    user_id: str | None = None,
    tenant_id: str | None = None,
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    x_api_key: str | None = Header(default=None)
):
    _verify_api_key(x_api_key)
    from truenorth.core.session_manager import SessionManager
    from truenorth.api.main import db_store, redis_store

    sm = SessionManager(postgres=db_store, redis=redis_store)
    return await sm.list_sessions(
        user_id=user_id, tenant_id=tenant_id, status=status, limit=limit, offset=offset
    )

@router.delete("/{session_id}")
async def delete_session(session_id: str, x_api_key: str | None = Header(default=None)):
    _verify_api_key(x_api_key)
    from truenorth.core.session_manager import SessionManager
    from truenorth.api.main import db_store, redis_store

    sm = SessionManager(postgres=db_store, redis=redis_store)
    deleted = await sm.delete(session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"status": "deleted", "session_id": session_id}
