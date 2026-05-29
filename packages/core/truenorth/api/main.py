"""FastAPI application factory."""

from __future__ import annotations
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from truenorth.storage.postgres import PostgresStore
from truenorth.storage.redis import RedisStore


db_store: PostgresStore | None = None
redis_store: RedisStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_store, redis_store

    db_url = os.getenv("DATABASE_URL", "")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    if db_url:
        db_store = PostgresStore(db_url)
        await db_store.create_tables()

    redis_store = RedisStore(redis_url)
    await redis_store.ping()

    print("✓ TrueNorth API ready")
    yield

    if db_store:
        await db_store.engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="TrueNorth API",
        description="Conversation-first AI agent framework",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from truenorth.api.routes import sessions, messages, analytics, health
    app.include_router(health.router, tags=["health"])
    app.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
    app.include_router(messages.router, prefix="/sessions", tags=["messages"])
    app.include_router(analytics.router, prefix="/analytics", tags=["analytics"])

    from truenorth.api import websocket
    app.include_router(websocket.router, tags=["websocket"])

    return app


app = create_app()
