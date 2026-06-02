from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def health():
    return {"service": "truenorth", "status": "ok", "version": "0.1.1"}

@router.get("/ready")
async def ready():
    from truenorth.api.main import db_store, redis_store
    checks = {
        "database": db_store is not None,
        "redis": redis_store is not None and await redis_store.ping(),
    }
    ok = all(checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks},
                        status_code=200 if ok else 503)
