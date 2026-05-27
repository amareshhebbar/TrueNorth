"""Redis for session caching and rate limiting."""

from __future__ import annotations
import json
import redis.asyncio as aioredis


class RedisStore:
    def __init__(self, redis_url: str):
        self.client = aioredis.from_url(redis_url, decode_responses=True)

    async def cache_session(self, session_id: str, data: dict, ttl_seconds: int = 3600):
        await self.client.setex(f"session:{session_id}", ttl_seconds, json.dumps(data))

    async def get_cached_session(self, session_id: str) -> dict | None:
        data = await self.client.get(f"session:{session_id}")
        return json.loads(data) if data else None

    async def invalidate_session(self, session_id: str):
        await self.client.delete(f"session:{session_id}")

    async def check_rate_limit(self, user_id: str, goal_id: str,
                                window: str, limit: int) -> tuple[bool, int]:
        """Returns (is_allowed, current_count)."""
        key = f"rl:{user_id}:{goal_id}:{window}"
        pipe = self.client.pipeline()
        pipe.incr(key)
        if window == "day":
            pipe.expire(key, 86400)
        elif window == "hour":
            pipe.expire(key, 3600)
        results = await pipe.execute()
        count = results[0]
        return count <= limit, count

    async def ping(self) -> bool:
        try:
            return await self.client.ping()
        except Exception:
            return False
