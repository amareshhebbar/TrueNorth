"""TrueNorth — Conversation-first AI agent framework."""

__version__ = "0.1.0"
__author__ = "Studio Ilios"

from truenorth.core.engine import TrueNorthEngine
from truenorth.core.graph_state import GraphState

# Removed FieldValue from both the import and the __all__ list
__all__ = ["TrueNorthEngine", "GraphState"]