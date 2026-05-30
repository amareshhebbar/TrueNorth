"""
No real HTTP calls. No LangGraph installation required. All external
dependencies mocked.

Classes:
  1.  A2ADataTypes       — A2APart, A2AMessage, A2ATask, AgentCard
  2.  A2ATaskBridge      — TrueNorth ↔ A2A format conversion
  3.  A2AClient_Parsing  — JSON-RPC request/response handling (mock HTTP)
  4.  A2AServer_Handlers — handle_send, handle_get, handle_cancel
  5.  A2AServer_Card     — AgentCard generation from BaseAgent
  6.  FieldMapping       — field-level transform and condition
  7.  FieldMap           — explicit, direct, yaml, auto_infer
  8.  StateTransfer      — extract, seed_engine, coverage
  9.  TransferResult     — coverage_pct, to_dict
  10. GoalChain          — next step routing, condition evaluation
  11. ChainStep          — condition_met
  12. StateAdapter       — truenorth_to_langgraph, langgraph_to_truenorth
  13. TrueNorthNode      — should_continue, get_collected_fields
  14. LangGraphAgent     — execute with mock compiled graph
  15. SectorTransfer     — state transfer across 5 sectors
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.agents.a2a_protocol import (
    A2APart, A2AMessage, A2ATask, A2ATaskState, AgentCard,
    A2ATaskBridge, A2AClient, A2AServer,
)
from truenorth.agents.state_transfer import (
    FieldMapping, FieldMap, StateTransfer, TransferResult,
    GoalChain, ChainStep,
)
from truenorth.agents.langgraph_bridge import (
    StateAdapter, TrueNorthNode, LangGraphAgent,
)
from truenorth.agents.messages import (
    AgentMessage, AgentResponse, AgentRole, TaskStatus, Priority,
)
from truenorth.agents.base import BaseAgent


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _msg(task: str = "test task", payload: dict = None) -> AgentMessage:
    return AgentMessage.create(
        sender="orch", recipient="agent", task=task,
        payload=payload or {}, session_id="test-sess", turn=1,
    )


class _EchoAgent(BaseAgent):
    agent_id     = "echo_agent"
    role         = AgentRole.CUSTOM
    capabilities = {"echo", "test"}

    async def handle(self, message):
        return self.ok(message, {"echo": message.task}, confidence=0.90)


def _make_a2a_task(
    state: A2ATaskState = A2ATaskState.SUBMITTED,
    text:  str          = "run this task",
) -> A2ATask:
    return A2ATask(
        id       = "task-001",
        state    = state,
        messages = [A2AMessage(role="user", parts=[A2APart.text(text)])],
    )


COLLECTED_FITNESS = {
    "name":          "Priya",
    "age":           28,
    "weight_kg":     65.0,
    "height_cm":     163.0,
    "primary_goal":  "lose weight",
    "activity_level":"moderately active",
}

CONFIDENCES_FITNESS = {k: 0.90 for k in COLLECTED_FITNESS}


# ─────────────────────────────────────────────────────────────────────────────
#  1. A2A data types
# ─────────────────────────────────────────────────────────────────────────────

class TestA2ADataTypes:

    def test_a2a_part_text(self):
        p = A2APart.text("hello world")
        assert p.type    == "text"
        assert p.content == "hello world"

    def test_a2a_part_data(self):
        p = A2APart.data({"age": 28})
        assert p.type == "data"
        assert p.content == {"age": 28}

    def test_a2a_part_to_dict_text(self):
        d = A2APart.text("hi").to_dict()
        assert d["type"] == "text"
        assert d["text"] == "hi"

    def test_a2a_part_to_dict_data(self):
        d = A2APart.data({"x": 1}).to_dict()
        assert d["type"] == "data"
        assert "x" in str(d["data"])

    def test_a2a_message_text_content(self):
        msg = A2AMessage(
            role  = "user",
            parts = [A2APart.text("hello"), A2APart.text("world")],
        )
        assert "hello" in msg.text_content()
        assert "world" in msg.text_content()

    def test_a2a_message_to_dict(self):
        msg = A2AMessage(role="agent", parts=[A2APart.text("done")])
        d   = msg.to_dict()
        assert d["role"] == "agent"
        assert len(d["parts"]) == 1

    def test_a2a_message_from_dict(self):
        d   = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
        msg = A2AMessage.from_dict(d)
        assert msg.role           == "user"
        assert msg.parts[0].type  == "text"

    def test_a2a_task_state_enum(self):
        assert A2ATaskState.COMPLETED  == "completed"
        assert A2ATaskState.FAILED     == "failed"
        assert A2ATaskState.SUBMITTED  == "submitted"
        assert A2ATaskState.WORKING    == "working"
        assert A2ATaskState.CANCELLED  == "cancelled"

    def test_a2a_task_to_dict(self):
        task = _make_a2a_task()
        d    = task.to_dict()
        assert "id"       in d
        assert "state"    in d
        assert "messages" in d

    def test_a2a_task_from_dict(self):
        task = _make_a2a_task(text="do something")
        d    = task.to_dict()
        t2   = A2ATask.from_dict(d)
        assert t2.id    == task.id
        assert t2.state == task.state

    def test_agent_card_to_dict(self):
        card = AgentCard(
            name        = "My Agent",
            description = "Does stuff",
            url         = "http://agent.example.com",
            skills      = [{"id": "extract", "name": "extraction"}],
        )
        d = card.to_dict()
        assert d["name"]   == "My Agent"
        assert d["url"]    == "http://agent.example.com"
        assert len(d["skills"]) == 1

    def test_agent_card_from_dict(self):
        d = {
            "name": "A", "description": "B",
            "url": "http://x.com", "version": "2.0",
            "skills": [], "capabilities": {},
        }
        card = AgentCard.from_dict(d)
        assert card.name    == "A"
        assert card.version == "2.0"


# ─────────────────────────────────────────────────────────────────────────────
#  2. A2ATaskBridge
# ─────────────────────────────────────────────────────────────────────────────

class TestA2ATaskBridge:

    def test_agent_message_to_a2a(self):
        msg  = _msg("extract patient age", {"text": "I am 28"})
        task = A2ATaskBridge.agent_message_to_a2a(msg)
        assert isinstance(task, A2ATask)
        assert task.state    == A2ATaskState.SUBMITTED
        assert len(task.messages) == 1
        assert task.messages[0].role == "user"

    def test_a2a_task_to_agent_response_completed(self):
        task = A2ATask(
            id       = "t1",
            state    = A2ATaskState.COMPLETED,
            messages = [
                A2AMessage(role="user",  parts=[A2APart.text("extract age")]),
                A2AMessage(role="agent", parts=[A2APart.text("age is 28")]),
            ],
        )
        resp = A2ATaskBridge.a2a_task_to_agent_response(task)
        assert resp.is_success
        assert "28" in resp.result_text

    def test_a2a_task_to_agent_response_failed(self):
        task = A2ATask(
            id="t2", state=A2ATaskState.FAILED,
            messages=[], error="tool timeout",
        )
        resp = A2ATaskBridge.a2a_task_to_agent_response(task)
        assert resp.status == TaskStatus.FAILED
        assert resp.error  == "tool timeout"

    def test_text_to_a2a_task(self):
        task = A2ATaskBridge.text_to_a2a_task("research BMI formula", "sess-1")
        assert task.session_id == "sess-1"
        assert task.state      == A2ATaskState.SUBMITTED
        assert "BMI" in task.messages[0].text_content()


# ─────────────────────────────────────────────────────────────────────────────
#  3. A2AClient — mock HTTP
# ─────────────────────────────────────────────────────────────────────────────

class TestA2AClientParsing:

    def test_jsonrpc_payload_structure(self):
        payload = A2AClient._jsonrpc("tasks/send", {"task": {}})
        assert payload["jsonrpc"] == "2.0"
        assert payload["method"]  == "tasks/send"
        assert "id"               in payload
        assert payload["params"]  == {"task": {}}

    def test_client_constructed_with_endpoint(self):
        client = A2AClient(endpoint="http://agent.example.com/a2a")
        assert "agent.example.com" in client._endpoint

    def test_client_strips_trailing_slash(self):
        client = A2AClient(endpoint="http://agent.example.com/a2a/")
        assert not client._endpoint.endswith("/")

    def test_api_key_added_to_headers(self):
        client = A2AClient(endpoint="http://x.com", api_key="secret-123")
        assert "Authorization" in client._headers
        assert "secret-123" in client._headers["Authorization"]

    @pytest.mark.asyncio
    async def test_send_task_mock_completed(self):
        """send_task returns a completed A2ATask when mock HTTP returns success."""
        client = A2AClient(endpoint="http://mock.agent/a2a", poll_interval=0.01)

        completed_task = A2ATask(
            id       = "task-123",
            state    = A2ATaskState.COMPLETED,
            messages = [
                A2AMessage(role="user",  parts=[A2APart.text("hello")]),
                A2AMessage(role="agent", parts=[A2APart.text("world")]),
            ],
        )

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "jsonrpc": "2.0", "id": "1",
            "result": completed_task.to_dict(),
        }

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        result = await client._send_and_poll(_make_a2a_task())
        assert result.state == A2ATaskState.COMPLETED

    @pytest.mark.asyncio
    async def test_send_task_http_failure_returns_failed(self):
        client = A2AClient(endpoint="http://down.agent/a2a", poll_interval=0.01)

        mock_http = AsyncMock()
        mock_http.post = AsyncMock(side_effect=ConnectionRefusedError("refused"))
        client._client = mock_http

        result = await client._send_and_poll(_make_a2a_task())
        assert result.state == A2ATaskState.FAILED
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_cancel_task_mock(self):
        client = A2AClient(endpoint="http://mock.agent/a2a")
        mock_resp = MagicMock(status_code=200)
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        result = await client.cancel_task("task-999")
        assert result is True

    @pytest.mark.asyncio
    async def test_get_agent_card_mock(self):
        client = A2AClient(endpoint="http://mock.agent")
        card_data = {
            "name": "Test Agent", "description": "desc",
            "url": "http://mock.agent", "version": "1.0",
            "skills": [], "capabilities": {},
        }
        mock_resp = MagicMock(status_code=200)
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = card_data
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        client._client = mock_http

        card = await client.get_agent_card()
        assert card is not None
        assert card.name == "Test Agent"

    @pytest.mark.asyncio
    async def test_get_agent_card_returns_none_on_failure(self):
        client = A2AClient(endpoint="http://down.agent")
        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=Exception("unreachable"))
        client._client = mock_http

        card = await client.get_agent_card()
        assert card is None


# ─────────────────────────────────────────────────────────────────────────────
#  4. A2AServer — handle_send, handle_get, handle_cancel
# ─────────────────────────────────────────────────────────────────────────────

class TestA2AServerHandlers:

    @pytest.mark.asyncio
    async def test_handle_send_runs_agent(self):
        agent  = _EchoAgent()
        server = A2AServer(agent=agent)
        task   = _make_a2a_task(text="echo this back")
        result = await server.handle_send(task.to_dict())
        assert "result" in result
        returned_task = A2ATask.from_dict(result["result"])
        assert returned_task.state in (A2ATaskState.COMPLETED, A2ATaskState.FAILED)

    @pytest.mark.asyncio
    async def test_handle_send_stores_task(self):
        agent  = _EchoAgent()
        server = A2AServer(agent=agent)
        task   = _make_a2a_task(text="store me")
        await server.handle_send(task.to_dict())
        assert task.id in server._tasks

    @pytest.mark.asyncio
    async def test_handle_get_returns_task(self):
        agent  = _EchoAgent()
        server = A2AServer(agent=agent)
        task   = _make_a2a_task()
        await server.handle_send(task.to_dict())
        result = await server.handle_get(task.id)
        assert "result" in result
        assert result["result"]["id"] == task.id

    @pytest.mark.asyncio
    async def test_handle_get_unknown_task_returns_error(self):
        server = A2AServer(agent=_EchoAgent())
        result = await server.handle_get("nonexistent-id")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_handle_cancel_changes_state(self):
        agent  = _EchoAgent()
        server = A2AServer(agent=agent)
        task   = _make_a2a_task()
        task.state = A2ATaskState.WORKING
        server._tasks[task.id] = task
        await server.handle_cancel(task.id)
        assert server._tasks[task.id].state == A2ATaskState.CANCELLED

    @pytest.mark.asyncio
    async def test_agent_card_generated(self):
        agent  = _EchoAgent()
        server = A2AServer(agent=agent)
        card   = server.agent_card_dict()
        assert card["name"] == "echo_agent"
        assert "skills"     in card


# ─────────────────────────────────────────────────────────────────────────────
#  5. A2AServer — AgentCard from BaseAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestA2AServerCard:

    def test_default_card_has_name(self):
        card = A2AServer._default_card(_EchoAgent())
        assert card.name == "echo_agent"

    def test_default_card_has_skills(self):
        card = A2AServer._default_card(_EchoAgent())
        assert len(card.skills) >= 1

    def test_custom_card_used_when_provided(self):
        agent  = _EchoAgent()
        custom = AgentCard(name="Custom", description="d", url="http://x.com")
        server = A2AServer(agent=agent, card=custom)
        assert server.agent_card_dict()["name"] == "Custom"


# ─────────────────────────────────────────────────────────────────────────────
#  6. FieldMapping
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldMapping:

    def test_basic_mapping_apply(self):
        fm = FieldMapping(source_field="age", target_field="user_age")
        ok, val = fm.apply(28)
        assert ok  is True
        assert val == 28

    def test_transform_applied(self):
        fm = FieldMapping(
            source_field="weight_lbs", target_field="weight_kg",
            transform=lambda x: round(x / 2.205, 1),
        )
        ok, val = fm.apply(143.0)
        assert ok  is True
        assert val == pytest.approx(64.9, abs=0.2)

    def test_condition_blocks_carry(self):
        fm = FieldMapping(
            source_field="goal", target_field="goal",
            condition=lambda v: v == "lose weight",
        )
        ok_yes, _ = fm.apply("lose weight")
        ok_no,  _ = fm.apply("build muscle")
        assert ok_yes is True
        assert ok_no  is False

    def test_transform_exception_returns_false(self):
        fm = FieldMapping(
            source_field="x", target_field="y",
            transform=lambda v: 1 / v, 
        )
        ok, _ = fm.apply(0)
        assert ok is False


# ─────────────────────────────────────────────────────────────────────────────
#  7. FieldMap construction
# ─────────────────────────────────────────────────────────────────────────────

class TestFieldMap:

    def test_add_mapping(self):
        fm = FieldMap().add("age", "user_age")
        assert len(fm.mappings) == 1
        assert fm.mappings[0].source_field == "age"
        assert fm.mappings[0].target_field == "user_age"

    def test_add_direct_multiple(self):
        fm = FieldMap().add_direct("name", "age", "weight_kg")
        assert len(fm.mappings) == 3
        for m in fm.mappings:
            assert m.source_field == m.target_field

    def test_from_yaml_strings(self):
        fm = FieldMap.from_yaml(["age", "weight_kg", "name"])
        names = [(m.source_field, m.target_field) for m in fm.mappings]
        assert ("age", "age")           in names
        assert ("weight_kg", "weight_kg") in names

    def test_from_yaml_dicts(self):
        fm = FieldMap.from_yaml([{"age": "user_age"}, {"weight_kg": "starting_weight"}])
        names = {m.source_field: m.target_field for m in fm.mappings}
        assert names["age"]       == "user_age"
        assert names["weight_kg"] == "starting_weight"

    def test_from_yaml_mixed(self):
        fm = FieldMap.from_yaml(["name", {"age": "user_age"}])
        assert len(fm.mappings) == 2

    def test_auto_infer_common_fields(self):
        fm = FieldMap.auto_infer(
            source_fields = ["age", "weight_kg", "goal", "streak"],
            target_fields = ["age", "weight_kg", "calories", "goal"],
        )
        names = {m.source_field for m in fm.mappings}
        assert "age"       in names
        assert "weight_kg" in names
        assert "goal"      in names
        assert "streak"    not in names   # not in target
        assert "calories"  not in names   # not in source


# ─────────────────────────────────────────────────────────────────────────────
#  8. StateTransfer — extract and seed
# ─────────────────────────────────────────────────────────────────────────────

class TestStateTransfer:

    def _source_state(self) -> dict:
        import copy
        return {
            "collected_fields": copy.deepcopy(COLLECTED_FITNESS),
            "field_confidences": copy.deepcopy(CONFIDENCES_FITNESS),
            "session_id": "fitness-sess",
            "goal_id":    "fitness_plan",
        }

    def test_extract_with_auto_infer(self):
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state           = self._source_state(),
            source_goal_id         = "fitness_plan",
            target_goal_id         = "nutrition_plan",
            target_required_fields = ["age", "weight_kg", "name"],
        )
        assert "age"       in result.carried_fields
        assert "weight_kg" in result.carried_fields
        assert "name"      in result.carried_fields

    def test_extract_with_explicit_map(self):
        fm = FieldMap().add("age", "user_age").add("weight_kg", "starting_weight")
        transfer = StateTransfer(field_map=fm)
        result   = transfer.extract(
            source_state   = self._source_state(),
            source_goal_id = "fitness_plan",
            target_goal_id = "nutrition_plan",
        )
        assert "user_age"        in result.carried_fields
        assert "starting_weight" in result.carried_fields
        assert result.carried_fields["user_age"] == 28

    def test_low_confidence_skipped(self):
        source = self._source_state()
        source["field_confidences"]["age"] = 0.30   # below threshold
        transfer = StateTransfer(auto_infer=True, confidence_threshold=0.70)
        result   = transfer.extract(
            source_state           = source,
            source_goal_id         = "fitness_plan",
            target_goal_id         = "nutrition_plan",
            target_required_fields = ["age", "weight_kg"],
        )
        assert "age" not in result.carried_fields
        assert "age" in result.skipped_fields

    def test_missing_fields_identified(self):
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state           = self._source_state(),
            source_goal_id         = "fitness_plan",
            target_goal_id         = "nutrition_plan",
            target_required_fields = ["age", "calorie_goal", "meal_preference"],
        )
        # calorie_goal and meal_preference not in source → missing
        assert "calorie_goal"   in result.missing_fields
        assert "meal_preference" in result.missing_fields

    def test_coverage_pct_all_carried(self):
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state           = self._source_state(),
            source_goal_id         = "a",
            target_goal_id         = "b",
            target_required_fields = ["age", "weight_kg"],
        )
        assert "age"       in result.carried_fields
        assert "weight_kg" in result.carried_fields
        assert len(result.missing_fields) == 0

    def test_coverage_pct_partial(self):
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state           = self._source_state(),
            source_goal_id         = "a",
            target_goal_id         = "b",
            target_required_fields = ["age", "missing_field_xyz"],
        )
        assert "age"  in result.carried_fields
        assert "missing_field_xyz" in result.missing_fields

    def test_to_dict_has_required_keys(self):
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state   = self._source_state(),
            source_goal_id = "a",
            target_goal_id = "b",
        )
        d = result.to_dict()
        for key in ["source_goal_id", "target_goal_id", "carried_count",
                    "coverage_pct", "carried_fields"]:
            assert key in d


# ─────────────────────────────────────────────────────────────────────────────
#  9. TransferResult
# ─────────────────────────────────────────────────────────────────────────────

class TestTransferResult:

    def test_full_coverage(self):
        r = TransferResult(
            source_goal_id = "a", target_goal_id = "b",
            carried_fields = {"age": 28, "weight": 65},
            skipped_fields = [],
            missing_fields = [],
        )
        assert r.coverage_pct == pytest.approx(100.0)

    def test_no_coverage(self):
        r = TransferResult(
            source_goal_id = "a", target_goal_id = "b",
            carried_fields = {},
            skipped_fields = [],
            missing_fields = ["age", "weight"],
        )
        assert r.coverage_pct == pytest.approx(0.0)

    def test_half_coverage(self):
        r = TransferResult(
            source_goal_id = "a", target_goal_id = "b",
            carried_fields = {"age": 28},
            skipped_fields = [],
            missing_fields = ["weight"],
        )
        assert r.coverage_pct == pytest.approx(50.0)


# ─────────────────────────────────────────────────────────────────────────────
#  10. GoalChain
# ─────────────────────────────────────────────────────────────────────────────

class TestGoalChain:

    def test_next_returns_matching_step(self):
        chain = GoalChain([
            ChainStep("nutrition_plan", condition={"primary_goal": "lose weight"}),
            ChainStep("muscle_plan",    condition={"primary_goal": "build muscle"}),
        ])
        step = chain.next("fitness_plan", {"primary_goal": "lose weight"})
        assert step is not None
        assert step.goal_id == "nutrition_plan"

    def test_next_returns_else_step(self):
        chain = GoalChain([
            ChainStep("special_plan", condition={"vip": True}),
            ChainStep("default_plan"),  # no condition = else
        ])
        step = chain.next("start", {"vip": False})
        assert step is not None
        assert step.goal_id == "default_plan"

    def test_next_returns_none_when_no_match(self):
        chain = GoalChain([
            ChainStep("x", condition={"impossible_field": "impossible_value"}),
        ])
        step = chain.next("start", {})
        assert step is None

    def test_next_skips_current_goal(self):
        chain = GoalChain([
            ChainStep("fitness_plan"),
            ChainStep("nutrition_plan"),
        ])
        step = chain.next("fitness_plan", {})
        assert step is not None
        assert step.goal_id == "nutrition_plan"

    def test_from_yaml(self):
        config = {
            "on_complete": [
                {"if": {"primary_goal": "lose weight"}, "then": "nutrition_plan",
                 "carry_fields": ["age", {"weight_kg": "start_weight"}]},
                {"else": "maintenance_plan"},
            ]
        }
        chain = GoalChain.from_yaml(config)
        assert len(chain._steps) == 2
        assert chain._steps[0].goal_id == "nutrition_plan"
        assert chain._steps[1].goal_id == "maintenance_plan"

    def test_all_goals_list(self):
        chain = GoalChain([
            ChainStep("a"), ChainStep("b"), ChainStep("c"),
        ])
        assert chain.all_goals() == ["a", "b", "c"]

    def test_field_map_for_step(self):
        step  = ChainStep("next", carry_fields=["age", {"weight_kg": "start_weight"}])
        chain = GoalChain([step])
        fm    = chain.field_map_for(step)
        names = {m.source_field: m.target_field for m in fm.mappings}
        assert "age" in names
        assert names["weight_kg"] == "start_weight"


# ─────────────────────────────────────────────────────────────────────────────
#  11. ChainStep — condition_met
# ─────────────────────────────────────────────────────────────────────────────

class TestChainStep:

    def test_no_condition_always_true(self):
        step = ChainStep("x")
        assert step.condition_met({"anything": "here"}) is True
        assert step.condition_met({}) is True

    def test_condition_matches(self):
        step = ChainStep("x", condition={"goal": "lose weight"})
        assert step.condition_met({"goal": "lose weight"}) is True

    def test_condition_not_matches(self):
        step = ChainStep("x", condition={"goal": "lose weight"})
        assert step.condition_met({"goal": "build muscle"}) is False

    def test_condition_missing_field(self):
        step = ChainStep("x", condition={"goal": "lose weight"})
        assert step.condition_met({}) is False   # field doesn't exist → no match

    def test_multi_condition_all_must_match(self):
        step = ChainStep("x", condition={"goal": "lose weight", "level": "beginner"})
        assert step.condition_met({"goal": "lose weight", "level": "beginner"}) is True
        assert step.condition_met({"goal": "lose weight", "level": "advanced"}) is False


# ─────────────────────────────────────────────────────────────────────────────
#  12. StateAdapter
# ─────────────────────────────────────────────────────────────────────────────

class TestStateAdapter:

    def test_truenorth_to_langgraph(self):
        tn_state = {
            "session_id":       "sess-1",
            "goal_id":          "fitness_plan",
            "collected_fields": {"age": 28, "weight_kg": 65},
            "completion_pct":   50.0,
            "final_output":     None,
            "total_cost_usd":   0.001,
            "detected_language":"en",
            "current_turn":     3,
        }
        existing = {"messages": []}
        result   = StateAdapter.truenorth_to_langgraph(tn_state, existing)
        assert "truenorth"     in result
        assert result["truenorth"]["session_id"] == "sess-1"
        assert result["truenorth"]["collected_fields"]["age"] == 28
        assert result["truenorth"]["is_complete"] is False
        assert result.get("age") == 28

    def test_is_complete_when_final_output_set(self):
        tn_state = {
            "session_id": "s", "goal_id": "g",
            "collected_fields": {}, "completion_pct": 100.0,
            "final_output": {"format": "json", "content": "{}"},
            "total_cost_usd": 0.0, "detected_language": "en", "current_turn": 5,
        }
        result = StateAdapter.truenorth_to_langgraph(tn_state, {})
        assert result["truenorth"]["is_complete"] is True

    def test_langgraph_to_truenorth(self):
        lg_state = {
            "messages": [{"role": "user", "content": "hi"}],
            "truenorth": {
                "session_id":       "sess-2",
                "collected_fields": {"age": 30},
            },
        }
        tn_seed = StateAdapter.langgraph_to_truenorth(lg_state)
        assert tn_seed["session_id"]              == "sess-2"
        assert tn_seed["collected_fields"]["age"] == 30

    def test_langgraph_to_truenorth_no_namespace(self):
        """Works even if there's no 'truenorth' key."""
        tn_seed = StateAdapter.langgraph_to_truenorth({"messages": []})
        assert tn_seed["collected_fields"] == {}


# ─────────────────────────────────────────────────────────────────────────────
#  13. TrueNorthNode
# ─────────────────────────────────────────────────────────────────────────────

class TestTrueNorthNode:

    def _goal_config(self):
        return {
            "id": "test_goal",
            "fields": [{"name": "age", "type": "integer", "required": True, "question": "Age?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output": {"format": "json"},
        }

    def test_should_continue_not_complete(self):
        node  = TrueNorthNode(goal_config=self._goal_config())
        state = {"truenorth": {"is_complete": False, "current_turn": 2}}
        assert node.should_continue(state) == "continue"

    def test_should_continue_complete(self):
        node  = TrueNorthNode(goal_config=self._goal_config())
        state = {"truenorth": {"is_complete": True, "current_turn": 5}}
        assert node.should_continue(state) == "end"

    def test_should_continue_max_turns(self):
        node  = TrueNorthNode(goal_config=self._goal_config(), max_turns=3)
        state = {"truenorth": {"is_complete": False, "current_turn": 4}}
        assert node.should_continue(state) == "end"

    def test_get_collected_fields(self):
        node  = TrueNorthNode(goal_config=self._goal_config())
        state = {"truenorth": {"collected_fields": {"age": 28}}}
        fields = node.get_collected_fields(state)
        assert fields["age"] == 28

    def test_get_final_output_none_when_incomplete(self):
        node  = TrueNorthNode(goal_config=self._goal_config())
        state = {"truenorth": {"is_complete": False, "final_output": None}}
        assert node.get_final_output(state) is None

    def test_get_final_output_when_complete(self):
        node  = TrueNorthNode(goal_config=self._goal_config())
        output = {"format": "json", "content": '{"age": 28}'}
        state  = {"truenorth": {"is_complete": True, "final_output": output}}
        assert node.get_final_output(state) == output


# ─────────────────────────────────────────────────────────────────────────────
#  14. LangGraphAgent — mock compiled graph
# ─────────────────────────────────────────────────────────────────────────────

class TestLangGraphAgent:

    def _mock_graph(self, output: str = "result text") -> Any:
        """Create a mock compiled LangGraph graph."""
        graph = AsyncMock()
        graph.ainvoke = AsyncMock(return_value={
            "messages": [
                {"role": "user",  "content": "input"},
                {"role": "assistant", "content": output},
            ]
        })
        return graph

    @pytest.mark.asyncio
    async def test_execute_returns_success(self):
        graph = self._mock_graph("research complete")
        agent = LangGraphAgent(compiled_graph=graph, agent_id="lg_test")
        msg   = _msg("research this topic")
        resp  = await agent.execute(msg)
        assert resp.is_success
        assert "research complete" in resp.result_text

    @pytest.mark.asyncio
    async def test_execute_on_failure(self):
        graph = AsyncMock()
        graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph crashed"))
        agent = LangGraphAgent(compiled_graph=graph, agent_id="fail_lg")
        msg   = _msg("do something")
        resp  = await agent.execute(msg)
        assert resp.status == TaskStatus.FAILED
        assert resp.error  is not None

    def test_is_ready_with_graph(self):
        agent = LangGraphAgent(compiled_graph=AsyncMock())
        assert agent.is_ready() is True

    def test_is_ready_without_graph(self):
        agent = LangGraphAgent(compiled_graph=None)
        assert agent.is_ready() is False

    def test_can_handle_by_capability(self):
        agent   = LangGraphAgent(compiled_graph=AsyncMock(), capabilities={"research"})
        msg_yes = _msg("research clinical trials")
        msg_no  = _msg("book a flight")
        assert agent.can_handle(msg_yes) is True
        assert agent.can_handle(msg_no)  is False

    def test_health_has_required_keys(self):
        agent = LangGraphAgent(compiled_graph=AsyncMock(), agent_id="my_lg")
        h     = agent.health()
        assert h["agent_id"] == "my_lg"
        assert "ready"       in h
        assert "type"        in h


# ─────────────────────────────────────────────────────────────────────────────
#  15. Sector agnosticism — state transfer across domains
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorTransfer:

    SECTORS = [
        ("healthcare → nutrition", {
            "chief_complaint": "lower back pain",
            "pain_scale":      7,
            "age":             35,
            "weight_kg":       80.0,
        }, ["age", "weight_kg"]),
        ("legal intake → billing", {
            "case_type":       "personal_injury",
            "client_name":     "Alex Kumar",
            "incident_date":   "2024-03-15",
        }, ["client_name", "case_type"]),
        ("hr screening → onboarding", {
            "candidate_name":  "Priya Mehta",
            "start_date":      "2025-02-01",
            "department":      "engineering",
        }, ["candidate_name", "start_date"]),
        ("financial → investment", {
            "annual_income":   1_500_000,
            "risk_tolerance":  "moderate",
            "age":             32,
        }, ["annual_income", "risk_tolerance", "age"]),
        ("fitness → nutrition", COLLECTED_FITNESS, ["age", "weight_kg", "primary_goal"]),
    ]

    @pytest.mark.parametrize("scenario,collected,expected_carry", SECTORS)
    def test_state_transfer_across_sectors(self, scenario, collected, expected_carry):
        source_state = {
            "collected_fields":  collected,
            "field_confidences": {k: 0.90 for k in collected},
        }
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state           = source_state,
            source_goal_id         = "source",
            target_goal_id         = "target",
            target_required_fields = expected_carry,
        )
        for field in expected_carry:
            assert field in result.carried_fields, \
                f"{scenario}: expected {field!r} in carried_fields"

    @pytest.mark.parametrize("scenario,collected,expected_carry", SECTORS)
    def test_coverage_at_least_50pct(self, scenario, collected, expected_carry):
        source_state = {
            "collected_fields":  collected,
            "field_confidences": {k: 0.90 for k in collected},
        }
        transfer = StateTransfer(auto_infer=True)
        result   = transfer.extract(
            source_state           = source_state,
            source_goal_id         = "source",
            target_goal_id         = "target",
            target_required_fields = expected_carry,
        )
        assert result.coverage_pct >= 50.0, \
            f"{scenario}: coverage {result.coverage_pct:.1f}% < 50%"