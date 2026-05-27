"""WebSocket endpoint for streaming responses."""

from __future__ import annotations
import os
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pathlib import Path

router = APIRouter()


@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()

    from truenorth.api.main import db_store
    if not db_store:
        await websocket.send_json({"error": "Database not available"})
        await websocket.close()
        return

    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "")

            if not message:
                continue

            from truenorth.storage.models import Session as SessionModel
            async with db_store.session_factory() as db:
                row = await db.get(SessionModel, session_id)
            if not row:
                await websocket.send_json({"error": "Session not found"})
                break

            from truenorth.api.routes.sessions import _load_goal_config
            from truenorth.core.engine import TrueNorthEngine
            from truenorth.llm.router import LLMRouter

            config = _load_goal_config(row.goal_id)
            state = await db_store.load_state(session_id, config.model_dump())

            llm_router = LLMRouter()
            engine = TrueNorthEngine(config, llm_router)

            new_state, response = await engine.process_turn(state, message)
            await db_store.save_state(new_state)

            await websocket.send_json({
                "response": response,
                "emotion_state": new_state.emotion_state,
                "is_complete": new_state.completed,
                "is_escalated": new_state.escalated,
                "profile": new_state.collected_fields,
            })

            if new_state.completed or new_state.escalated:
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()
