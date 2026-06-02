"""Goals marketplace endpoints."""

from fastapi import APIRouter, HTTPException, Query
from truenorth.marketplace.goal_registry import GoalRegistry

router = APIRouter()
registry = GoalRegistry()  # Initializes with official curated goals

@router.get("")
async def list_goals(sector: str | None = Query(default=None)):
    """List all goals or filter by sector."""
    return registry.search(query="", sector=sector, limit=100)

@router.get("/{name}")
async def get_goal(name: str):
    """Get metadata for a specific goal."""
    goal = registry.info(name)
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    return goal