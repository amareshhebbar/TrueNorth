"""Analytics and cost observability endpoints."""

from fastapi import APIRouter, Query, Header
from truenorth.observability.cost_dashboard import CostDashboard
from truenorth.api.routes.sessions import _verify_api_key

router = APIRouter()
dashboard = CostDashboard()

@router.get("/cost")
async def get_cost_summary(
    goal: str = Query(...),
    period: int = Query(7),
    x_api_key: str | None = Header(default=None)
):
    _verify_api_key(x_api_key)
    return dashboard.goal_cost_summary(goal_id=goal, period_days=period).to_dict()

@router.get("/health")
async def get_health_report(
    goal: str = Query(...),
    window: int = Query(24),
    x_api_key: str | None = Header(default=None)
):
    _verify_api_key(x_api_key)
    return {"goal_id": goal, "window_hours": window, "completion_rate": 1.0}

@router.get("/health/{goal_id}")
async def agent_health(goal_id: str, days: int = 7,
                        x_api_key: str | None = Header(default=None)):
    _verify_api_key(x_api_key)
    from truenorth.api.main import db_store
    if not db_store:
        return {"error": "database not available"}

    from datetime import datetime, timedelta
    from sqlalchemy import select, func, and_
    from truenorth.storage.models import Session

    since = datetime.utcnow() - timedelta(days=days)

    async with db_store.session_factory() as db:
        total_q = select(func.count()).select_from(Session).where(
            and_(Session.goal_id == goal_id, Session.created_at >= since)
        )
        total = (await db.execute(total_q)).scalar()

        completed_q = select(func.count()).select_from(Session).where(
            and_(Session.goal_id == goal_id, Session.created_at >= since,
                 Session.completed == True)
        )
        completed = (await db.execute(completed_q)).scalar()

        avg_cost_q = select(func.avg(Session.cost_usd)).where(
            and_(Session.goal_id == goal_id, Session.created_at >= since)
        )
        avg_cost = (await db.execute(avg_cost_q)).scalar() or 0

    return {
        "goal_id": goal_id,
        "period_days": days,
        "total_sessions": total,
        "completed_sessions": completed,
        "completion_rate": round(completed / total, 3) if total else 0,
        "abandoned_sessions": total - completed,
        "avg_cost_per_session_usd": round(avg_cost, 4),
    }

@router.get("/cost/trend")
async def get_cost_trend(
    goal: str = Query(...),
    period: int = Query(7),
    x_api_key: str | None = Header(default=None)
):
    _verify_api_key(x_api_key)
    return dashboard.cost_trend(goal_id=goal, period_days=period)

@router.get("/cost/models")
async def get_model_comparison(
    goal: str = Query(...),
    period: int = Query(7),
    x_api_key: str | None = Header(default=None)
):
    _verify_api_key(x_api_key)
    return dashboard.model_comparison(goal_id=goal, period_days=period)
