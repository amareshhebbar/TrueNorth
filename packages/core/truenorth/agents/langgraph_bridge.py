"""
LangGraph compatibility bridge for TrueNorth.

This makes TrueNorth a first-class citizen in existing LangGraph graphs.
A team using LangGraph can add a TrueNorth node to their graph and get
full field-extraction, hallucination firewall, and structured output
without rewriting their pipeline.

Two integration patterns:

Pattern 1 — TrueNorth as a LangGraph node:
    The TrueNorthNode wraps a full TrueNorthEngine and exposes it as a
    LangGraph StateGraph node. Drop it into any existing graph.

        from truenorth.agents.langgraph_bridge import TrueNorthNode
        from langgraph.graph import StateGraph

        graph = StateGraph(MyState)
        graph.add_node("intake",   TrueNorthNode(goal_config=medical_intake_yaml))
        graph.add_node("process",  my_existing_processor)
        graph.add_edge("intake", "process")

Pattern 2 — LangGraph graph as a TrueNorth agent:
    The LangGraphAgent wraps any compiled LangGraph graph as a TrueNorth
    BaseAgent. Use it in the TrueNorth AgentOrchestrator.

        from truenorth.agents.langgraph_bridge import LangGraphAgent
        langgraph_agent = LangGraphAgent(compiled_graph=my_graph)
        orchestrator.register(langgraph_agent)

State compatibility:
    TrueNorth's GraphState ↔ LangGraph's TypedDict state are different
    shapes. The bridge uses StateAdapter to translate between them.
    Collected fields flow into the LangGraph state as a flat dict.

Zero coupling:
    langgraph is an optional dependency. All imports are lazy. If langgraph
    is not installed, the bridge raises ImportError with a helpful message
    only when actually used.

Sector-agnostic: plug TrueNorth's medical intake into a LangGraph clinical
decision graph, or TrueNorth's HR screening into a LangGraph hiring pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from truenorth.core.engine import TrueNorthEngine

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  State adapter — translates between TrueNorth GraphState and LangGraph state
# ─────────────────────────────────────────────────────────────────────────────

class StateAdapter:
    """
    Converts between TrueNorth's GraphState dict and a LangGraph state dict.

    TrueNorth GraphState keys of interest:
        collected_fields   {field_name: value}
        final_output       {format, content, fields, metadata}
        session_id         str
        current_turn       int
        completion_pct     float
        detected_language  str
        total_cost_usd     float

    LangGraph state gets these injected under a "truenorth" namespace key
    to avoid collisions with other nodes' state.
    """

    NAMESPACE = "truenorth"

    @classmethod
    def truenorth_to_langgraph(cls, tn_state: dict, existing: dict) -> dict:
        """
        Merge TrueNorth state into an existing LangGraph state dict.
        TrueNorth data goes under the "truenorth" key.
        """
        updated = dict(existing)
        updated[cls.NAMESPACE] = {
            "session_id":      tn_state.get("session_id", ""),
            "goal_id":         tn_state.get("goal_id", ""),
            "collected_fields":tn_state.get("collected_fields", {}),
            "completion_pct":  tn_state.get("completion_pct", 0.0),
            "final_output":    tn_state.get("final_output"),
            "total_cost_usd":  tn_state.get("total_cost_usd", 0.0),
            "detected_language": tn_state.get("detected_language", "en"),
            "current_turn":    tn_state.get("current_turn", 0),
            "is_complete":     tn_state.get("final_output") is not None,
        }
        for field_name, value in tn_state.get("collected_fields", {}).items():
            if field_name not in updated:  
                updated[field_name] = value
        return updated

    @classmethod
    def langgraph_to_truenorth(cls, lg_state: dict) -> dict:
        """
        Extract TrueNorth-relevant data from a LangGraph state dict.
        Used to seed TrueNorth with context from prior LangGraph nodes.
        """
        tn_section = lg_state.get(cls.NAMESPACE, {})
        return {
            "session_id":       tn_section.get("session_id", ""),
            "collected_fields": tn_section.get("collected_fields", {}),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  TrueNorthNode — TrueNorth engine as a LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

class TrueNorthNode:
    """
    Wrap a TrueNorth goal as a LangGraph StateGraph node.

    The node takes one user message per call, runs it through the engine,
    and updates the LangGraph state with collected fields and output.

    The LangGraph state must have a "messages" key (list of dicts with
    "role" and "content") — standard LangGraph convention.
    """

    def __init__(
        self,
        goal_config:  dict,
        router:       Optional[Any]   = None,
        cost_tracker: Optional[Any]   = None,
        session_id:   Optional[str]   = None,
        max_turns:    int             = 30,
    ):
        self._goal_config  = goal_config
        self._router       = router
        self._cost_tracker = cost_tracker
        self._session_id   = session_id
        self._max_turns    = max_turns
        self._engine: Optional["TrueNorthEngine"] = None

    # ------------------------------------------------------------------
    # LangGraph node interface
    # ------------------------------------------------------------------

    async def __call__(self, state: dict) -> dict:
        """
        LangGraph calls this for each node invocation.
        Takes the last user message from state["messages"] and runs it
        through TrueNorth. Returns updated state dict.
        """
        engine = await self._get_engine()
        messages = state.get("messages", [])

        user_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break
            elif hasattr(msg, "content"): 
                user_msg = msg.content
                break

        if not user_msg:
            logger.warning("langgraph_bridge: no user message found in state")
            return state

        response = await engine.process_message(user_msg)
        assistant_msg = {
            "role":    "assistant",
            "content": response.text or "",
        }
        updated_messages = list(messages) + [assistant_msg]

        # Merge TrueNorth state into LangGraph state
        tn_state  = engine.state.to_dict() if hasattr(engine.state, "to_dict") else {}
        new_state = StateAdapter.truenorth_to_langgraph(tn_state, state)
        new_state["messages"] = updated_messages
        new_state["truenorth"]["last_response"] = response.text

        return new_state

    def should_continue(self, state: dict) -> str:
        """
        Routing function for LangGraph conditional edges.
        Returns "continue" if more turns needed, "end" if complete.
        Designed to be passed to graph.add_conditional_edges().
        """
        tn = state.get("truenorth", {})
        if tn.get("is_complete") or (tn.get("current_turn", 0) >= self._max_turns):
            return "end"
        return "continue"

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_collected_fields(self, state: dict) -> dict:
        """Extract collected fields from LangGraph state."""
        return state.get("truenorth", {}).get("collected_fields", {})

    def get_final_output(self, state: dict) -> Optional[dict]:
        """Extract final output from LangGraph state (None if not complete)."""
        return state.get("truenorth", {}).get("final_output")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_engine(self) -> "TrueNorthEngine":
        if self._engine is None:
            from truenorth.core.engine import TrueNorthEngine
            self._engine = TrueNorthEngine(
                goal_config  = self._goal_config,
                session_id   = self._session_id,
                router       = self._router,
                cost_tracker = self._cost_tracker,
            )
            await self._engine.start()
        return self._engine


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraphAgent — LangGraph graph as a TrueNorth agent
# ─────────────────────────────────────────────────────────────────────────────

class LangGraphAgent:
    """
    Wrap a compiled LangGraph graph as a TrueNorth BaseAgent.
    Use this to register your existing LangGraph graphs with TrueNorth's
    AgentOrchestrator.
    """

    def __init__(
        self,
        compiled_graph: Any,
        agent_id:    str      = "langgraph_agent",
        role:        str      = "custom",
        capabilities: set    = None,
        input_key:   str     = "messages",
        output_key:  str     = "messages",
        config:      Optional[dict] = None,
    ):
        self.agent_id     = agent_id
        self.capabilities = capabilities or {"langgraph", "process"}
        self._graph       = compiled_graph
        self._input_key   = input_key
        self._output_key  = output_key
        self._config      = config or {}

        from truenorth.agents.messages import AgentRole
        self.role         = AgentRole.CUSTOM

    def is_ready(self) -> bool:
        return self._graph is not None

    def can_handle(self, message: Any) -> bool:
        task_lower = message.task.lower()
        return any(cap in task_lower for cap in self.capabilities)

    async def execute(self, message: Any) -> Any:
        """Run the LangGraph graph with this message's payload."""
        from truenorth.agents.messages import AgentResponse, TaskStatus
        import time as _time

        t0 = _time.perf_counter()
        task_text = message.task
        payload   = message.payload or {}

        if self._input_key == "messages":
            input_state = {
                "messages": [{"role": "user", "content": task_text}],
                **payload,
            }
        else:
            input_state = {self._input_key: task_text, **payload}

        try:
            result_state = await self._graph.ainvoke(input_state)
        except Exception as e:
            logger.error("langgraph_bridge: graph execution failed: %s", e)
            return AgentResponse(
                message_id = message.message_id,
                agent_id   = self.agent_id,
                status     = TaskStatus.FAILED,
                result     = None,
                error      = str(e),
            )

        output = result_state.get(self._output_key)
        if isinstance(output, list):
            for msg in reversed(output):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    output = msg.get("content", "")
                    break
                elif hasattr(msg, "content"):
                    output = msg.content
                    break

        latency = int((_time.perf_counter() - t0) * 1000)
        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.COMPLETED,
            result     = output,
            confidence = 0.80,
            latency_ms = latency,
            metadata   = {"result_state_keys": list(result_state.keys())},
        )

    def health(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "role":     "custom",
            "ready":    self.is_ready(),
            "type":     "langgraph_bridge",
        }


# ─────────────────────────────────────────────────────────────────────────────
#  LangGraphCheckpointer — use TrueNorth session storage as LangGraph memory
# ─────────────────────────────────────────────────────────────────────────────

class TrueNorthCheckpointer:
    """
    Optional: use TrueNorth's SessionManager as the LangGraph checkpointer.
    This means LangGraph state is persisted in TrueNorth's Postgres + Redis.

    Usage:
        checkpointer = TrueNorthCheckpointer(session_manager=sm)
        compiled     = graph.compile(checkpointer=checkpointer)

    Note: implements a subset of the LangGraph BaseCheckpointSaver interface.
    Only put/get/list are needed for most graphs.
    """

    def __init__(self, session_manager: Any):
        self._sm = session_manager

    async def aput(self, config: dict, checkpoint: dict, metadata: dict, new_versions: dict) -> dict:
        """Save a checkpoint (LangGraph calls this after each node)."""
        thread_id  = config.get("configurable", {}).get("thread_id", "default")
        session_id = f"lg:{thread_id}"
        try:
            await self._sm.save(session_id, {
                "checkpoint": checkpoint,
                "metadata":   metadata,
                "lg_config":  config,
            })
        except Exception as e:
            logger.warning("TrueNorthCheckpointer: put failed: %s", e)
        return config

    async def aget(self, config: dict) -> Optional[dict]:
        """Load a checkpoint."""
        thread_id  = config.get("configurable", {}).get("thread_id", "default")
        session_id = f"lg:{thread_id}"
        try:
            state = await self._sm.load(session_id)
            if state:
                return state.get("checkpoint")
        except Exception as e:
            logger.warning("TrueNorthCheckpointer: get failed: %s", e)
        return None

    async def alist(self, config: dict) -> list:
        """List checkpoints for a thread."""
        return []   # minimal implementation