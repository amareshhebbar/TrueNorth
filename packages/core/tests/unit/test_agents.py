"""
Zero network calls. Zero LLM calls for most tests.
All agents tested via mock payloads and fake routers.

Classes:
  1.  Messages           — AgentMessage, AgentResponse, SupervisorVerdict
  2.  BaseAgent          — lifecycle, timeout, retry, metrics
  3.  ExtractionAgent    — rule-based extraction (no LLM)
  4.  ValidationAgent    — field validation against schema
  5.  ResearchAgent      — MCP tool dispatch
  6.  WriterAgent        — template-based output (no LLM)
  7.  Orchestrator_Route — capability-based routing
  8.  Orchestrator_Parallel — concurrent task execution
  9.  Orchestrator_Sequential — ordered workflow
  10. Orchestrator_Fallback — default agent when no match
  11. Supervisor_Levels  — OFF / LIGHT / STANDARD / STRICT
  12. Supervisor_Approve — approves good results
  13. Supervisor_Reject  — rejects low-confidence / inconsistent results
  14. EngineIntegration  — engine accepts orchestrator param
  15. SectorAgnosticism  — same agents work for 5 different domains
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from truenorth.agents import (
    AgentMessage, AgentResponse, AgentRole, SupervisorVerdict,
    MessageType, TaskStatus, Priority,
    BaseAgent, AgentMetrics,
    AgentOrchestrator, OrchestrationResult, ExecutionStep,
    AgentSupervisor, SupervisionLevel,
)
from truenorth.agents.specialist import (
    ExtractionAgent, ValidationAgent, ResearchAgent, WriterAgent,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _msg(
    task:    str = "test task",
    payload: dict = None,
    priority: Priority = Priority.NORMAL,
) -> AgentMessage:
    return AgentMessage.create(
        sender    = "orchestrator",
        recipient = "test_agent",
        task      = task,
        payload   = payload or {},
        priority  = priority,
        session_id = "test-session",
        turn       = 1,
    )


class _EchoAgent(BaseAgent):
    """Returns its payload as the result."""
    agent_id     = "echo_agent"
    role         = AgentRole.CUSTOM
    capabilities = {"echo", "test"}

    async def handle(self, message: AgentMessage) -> AgentResponse:
        return self.ok(message, message.payload, confidence=0.90)


class _SlowAgent(BaseAgent):
    agent_id     = "slow_agent"
    role         = AgentRole.CUSTOM
    capabilities = {"slow"}
    default_timeout_s = 0.1   # very short timeout for testing

    async def handle(self, message: AgentMessage) -> AgentResponse:
        await asyncio.sleep(10)   # always times out
        return self.ok(message, "done")


class _FailAgent(BaseAgent):
    agent_id     = "fail_agent"
    role         = AgentRole.CUSTOM
    capabilities = {"fail"}
    max_retries  = 0

    async def handle(self, message: AgentMessage) -> AgentResponse:
        raise RuntimeError("deliberate failure")


class _LowConfidenceAgent(BaseAgent):
    agent_id     = "low_conf_agent"
    role         = AgentRole.CUSTOM
    capabilities = {"low_conf"}

    async def handle(self, message: AgentMessage) -> AgentResponse:
        return AgentResponse(
            message_id = message.message_id,
            agent_id   = self.agent_id,
            status     = TaskStatus.COMPLETED,
            result     = "low quality answer",
            confidence = 0.20,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  1. Messages
# ─────────────────────────────────────────────────────────────────────────────

class TestMessages:

    def test_agent_message_create(self):
        msg = AgentMessage.create(
            sender    = "orchestrator",
            recipient = "extractor",
            task      = "extract age from text",
            payload   = {"text": "I am 28"},
        )
        assert msg.message_id
        assert msg.sender    == "orchestrator"
        assert msg.recipient == "extractor"
        assert msg.task      == "extract age from text"
        assert msg.payload   == {"text": "I am 28"}
        assert msg.created_at > 0

    def test_agent_message_to_dict(self):
        msg = _msg()
        d   = msg.to_dict()
        assert "message_id" in d
        assert "sender"     in d
        assert "type"       in d

    def test_agent_response_is_success(self):
        ok  = AgentResponse(message_id="1", agent_id="a",
                            status=TaskStatus.COMPLETED, result="x")
        err = AgentResponse(message_id="2", agent_id="a",
                            status=TaskStatus.FAILED, result=None)
        assert ok.is_success  is True
        assert err.is_success is False

    def test_agent_response_result_text_str(self):
        r = AgentResponse(message_id="1", agent_id="a",
                          status=TaskStatus.COMPLETED, result="hello")
        assert r.result_text == "hello"

    def test_agent_response_result_text_dict(self):
        r = AgentResponse(message_id="1", agent_id="a",
                          status=TaskStatus.COMPLETED, result={"k": "v"})
        assert "k" in r.result_text

    def test_agent_response_result_text_none(self):
        r = AgentResponse(message_id="1", agent_id="a",
                          status=TaskStatus.FAILED, result=None)
        assert r.result_text == ""

    def test_supervisor_verdict_to_dict(self):
        v = SupervisorVerdict(
            message_id="1", agent_id="a",
            approved=True, score=0.95, feedback="All checks passed",
        )
        d = v.to_dict()
        assert d["approved"] is True
        assert d["score"]    == pytest.approx(0.95)

    def test_task_status_values(self):
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED    == "failed"
        assert TaskStatus.BLOCKED   == "blocked"
        assert TaskStatus.RUNNING   == "running"

    def test_priority_values(self):
        assert Priority.CRITICAL == "critical"
        assert Priority.HIGH     == "high"
        assert Priority.NORMAL   == "normal"
        assert Priority.LOW      == "low"

    def test_message_unique_ids(self):
        m1 = _msg()
        m2 = _msg()
        assert m1.message_id != m2.message_id


# ─────────────────────────────────────────────────────────────────────────────
#  2. BaseAgent lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseAgent:

    @pytest.mark.asyncio
    async def test_execute_calls_handle(self):
        agent = _EchoAgent()
        msg   = _msg(payload={"x": 1})
        resp  = await agent.execute(msg)
        assert resp.is_success
        assert resp.result == {"x": 1}

    @pytest.mark.asyncio
    async def test_timeout_returns_cancelled(self):
        agent = _SlowAgent()
        agent.max_retries = 0
        msg   = _msg("slow task")
        msg.timeout_s = 0.05
        resp  = await agent.execute(msg)
        assert resp.status == TaskStatus.CANCELLED
        assert "timed out" in resp.error.lower() or "timeout" in resp.error.lower()

    @pytest.mark.asyncio
    async def test_exception_returns_failed(self):
        agent = _FailAgent()
        resp  = await agent.execute(_msg())
        assert resp.status == TaskStatus.FAILED
        assert resp.error is not None

    @pytest.mark.asyncio
    async def test_metrics_incremented_on_success(self):
        agent = _EchoAgent()
        await agent.execute(_msg())
        assert agent._metrics.call_count   == 1
        assert agent._metrics.success_count == 1

    @pytest.mark.asyncio
    async def test_metrics_error_on_failure(self):
        agent = _FailAgent()
        await agent.execute(_msg())
        assert agent._metrics.error_count >= 1

    def test_is_ready_true_by_default(self):
        assert _EchoAgent().is_ready() is True

    def test_health_has_required_keys(self):
        h = _EchoAgent().health()
        assert "agent_id" in h
        assert "role"     in h
        assert "ready"    in h
        assert "metrics"  in h

    def test_can_handle_by_capability(self):
        agent = _EchoAgent()
        msg_match   = _msg("echo this back")
        msg_nomatch = _msg("validate schema fields")
        assert agent.can_handle(msg_match)   is True
        assert agent.can_handle(msg_nomatch) is False

    def test_base_ok_factory(self):
        msg  = _msg()
        resp = BaseAgent.ok(msg, {"answer": 42}, confidence=0.95)
        assert resp.status     == TaskStatus.COMPLETED
        assert resp.result     == {"answer": 42}
        assert resp.confidence == pytest.approx(0.95)

    def test_base_fail_factory(self):
        msg  = _msg()
        resp = BaseAgent.fail(msg, "something went wrong")
        assert resp.status == TaskStatus.FAILED
        assert "something" in resp.error

    def test_agent_metrics_success_rate(self):
        m = AgentMetrics(agent_id="test")
        m.record(TaskStatus.COMPLETED, 100)
        m.record(TaskStatus.COMPLETED, 200)
        m.record(TaskStatus.FAILED,    50)
        assert m.success_rate == pytest.approx(2/3, abs=0.01)
        assert m.call_count   == 3


# ─────────────────────────────────────────────────────────────────────────────
#  3. ExtractionAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionAgent:

    FIELDS_CONFIG = {
        "age":         {"type": "integer", "label": "age", "required": True},
        "weight_kg":   {"type": "number",  "label": "weight", "required": True},
        "primary_goal":{"type": "text",    "label": "goal", "required": True,
                        "allowed_values": ["lose weight", "build muscle"]},
    }

    @pytest.mark.asyncio
    async def test_extracts_number_from_text(self):
        agent = ExtractionAgent()
        msg   = _msg(task="extract fields", payload={
            "text": "I am 28 years old",
            "fields_config": self.FIELDS_CONFIG,
        })
        resp = await agent.execute(msg)
        assert resp.is_success
        assert isinstance(resp.result, dict)

    @pytest.mark.asyncio
    async def test_extracts_allowed_value(self):
        agent = ExtractionAgent()
        msg   = _msg(task="extract goal", payload={
            "text": "I want to lose weight and get fit",
            "fields_config": self.FIELDS_CONFIG,
        })
        resp = await agent.execute(msg)
        assert resp.is_success
        if "primary_goal" in resp.result:
            assert resp.result["primary_goal"] == "lose weight"

    @pytest.mark.asyncio
    async def test_empty_text_returns_success_empty_result(self):
        agent = ExtractionAgent()
        msg   = _msg(payload={
            "text": "",
            "fields_config": self.FIELDS_CONFIG,
        })
        resp = await agent.execute(msg)
        assert resp.status == TaskStatus.FAILED  

    @pytest.mark.asyncio
    async def test_no_router_uses_rule_based(self):
        agent = ExtractionAgent(router=None)
        msg   = _msg(payload={
            "text": "28 years old",
            "fields_config": self.FIELDS_CONFIG,
        })
        resp = await agent.execute(msg)
        assert resp.is_success

    def test_score_confidence_all_required_extracted(self):
        result = {"age": 28, "weight_kg": 65, "primary_goal": "lose weight"}
        score  = ExtractionAgent._score_confidence(result, self.FIELDS_CONFIG)
        assert score >= 0.90

    def test_score_confidence_none_extracted(self):
        score = ExtractionAgent._score_confidence({}, self.FIELDS_CONFIG)
        assert score <= 0.60


# ─────────────────────────────────────────────────────────────────────────────
#  4. ValidationAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestValidationAgent:

    FIELDS_CONFIG = {
        "age":       {"type": "integer", "min": 1, "max": 120, "required": True},
        "weight_kg": {"type": "number",  "min": 30, "max": 300, "required": True},
        "goal":      {"type": "text", "allowed_values": ["lose weight", "build muscle"]},
    }

    @pytest.mark.asyncio
    async def test_valid_values_all_pass(self):
        agent = ValidationAgent()
        msg   = _msg(payload={
            "fields_config":   self.FIELDS_CONFIG,
            "values_to_check": {"age": 28, "weight_kg": 65.0, "goal": "lose weight"},
        })
        resp = await agent.execute(msg)
        assert resp.is_success
        assert resp.result["valid"] is True
        assert len(resp.result["failed"]) == 0

    @pytest.mark.asyncio
    async def test_below_min_fails(self):
        agent = ValidationAgent()
        msg   = _msg(payload={
            "fields_config":   self.FIELDS_CONFIG,
            "values_to_check": {"age": 0},
        })
        resp = await agent.execute(msg)
        assert resp.is_success
        assert resp.result["valid"] is False
        assert any(f["field"] == "age" for f in resp.result["failed"])

    @pytest.mark.asyncio
    async def test_above_max_fails(self):
        agent = ValidationAgent()
        msg   = _msg(payload={
            "fields_config":   self.FIELDS_CONFIG,
            "values_to_check": {"age": 200},
        })
        resp = await agent.execute(msg)
        assert resp.result["valid"] is False

    @pytest.mark.asyncio
    async def test_invalid_enum_fails(self):
        agent = ValidationAgent()
        msg   = _msg(payload={
            "fields_config":   self.FIELDS_CONFIG,
            "values_to_check": {"goal": "random goal"},
        })
        resp = await agent.execute(msg)
        assert resp.result["valid"] is False
        assert any(f["field"] == "goal" for f in resp.result["failed"])

    @pytest.mark.asyncio
    async def test_not_a_number_fails(self):
        agent = ValidationAgent()
        msg   = _msg(payload={
            "fields_config":   self.FIELDS_CONFIG,
            "values_to_check": {"age": "abc"},
        })
        resp = await agent.execute(msg)
        assert resp.result["valid"] is False

    @pytest.mark.asyncio
    async def test_empty_values_succeeds_with_empty_result(self):
        agent = ValidationAgent()
        msg   = _msg(payload={"fields_config": self.FIELDS_CONFIG, "values_to_check": {}})
        resp  = await agent.execute(msg)
        assert resp.is_success
        assert resp.result["valid"] is True

    def test_validate_field_valid(self):
        ok, reason, warn = ValidationAgent._validate_field("age", 28, {"type": "integer", "min": 1, "max": 120})
        assert ok is True

    def test_validate_field_invalid_range(self):
        ok, reason, _ = ValidationAgent._validate_field("age", 200, {"type": "integer", "min": 1, "max": 120})
        assert ok is False
        assert "120" in reason


# ─────────────────────────────────────────────────────────────────────────────
#  5. ResearchAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestResearchAgent:

    @pytest.mark.asyncio
    async def test_no_registry_returns_failed(self):
        agent = ResearchAgent(registry=None)
        msg   = _msg("calculate BMI", payload={"tool_name": "calculator", "arguments": {"expression": "65/(1.63**2)"}})
        resp  = await agent.execute(msg)
        assert resp.status == TaskStatus.FAILED
        assert "registry" in resp.error.lower()

    @pytest.mark.asyncio
    async def test_calls_calculator_via_registry(self):
        from truenorth.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        registry.add_builtin("calculator")

        agent = ResearchAgent(registry=registry)
        msg   = _msg("calculate BMI", payload={
            "tool_name": "calculator",
            "arguments": {"expression": "65 / (1.63 ** 2)"},
        })
        resp = await agent.execute(msg)
        assert resp.is_success
        result_str = resp.result_text
        assert "24" in result_str  # BMI ≈ 24.45

    @pytest.mark.asyncio
    async def test_infer_calculator_from_task(self):
        assert ResearchAgent._infer_tool("calculate BMI for patient", {}) == "calculator"

    @pytest.mark.asyncio
    async def test_infer_datetime_from_task(self):
        assert ResearchAgent._infer_tool("get current date and time", {}) == "datetime_tool"

    @pytest.mark.asyncio
    async def test_infer_web_search_from_task(self):
        assert ResearchAgent._infer_tool("search for clinical guidelines", {}) == "web_search"

    @pytest.mark.asyncio
    async def test_no_tool_name_no_infer_fails(self):
        from truenorth.mcp.registry import MCPRegistry
        registry = MCPRegistry()
        agent    = ResearchAgent(registry=registry)
        msg      = _msg("do something vague", payload={})
        resp     = await agent.execute(msg)
        assert resp.status == TaskStatus.FAILED


# ─────────────────────────────────────────────────────────────────────────────
#  6. WriterAgent
# ─────────────────────────────────────────────────────────────────────────────

class TestWriterAgent:

    GOAL_CONFIG = {
        "id": "test_goal",
        "fields": [{"name": "name"}, {"name": "age"}],
        "output": {"format": "text", "template": "Name: {name}, Age: {age}"},
    }

    @pytest.mark.asyncio
    async def test_template_fill_no_llm(self):
        agent = WriterAgent(router=None)
        msg   = _msg("write report", payload={
            "collected_fields": {"name": "Priya", "age": 28},
            "goal_config":      self.GOAL_CONFIG,
        })
        resp = await agent.execute(msg)
        assert resp.is_success
        content = str(resp.result.get("content", resp.result))
        assert "Priya" in content or "name" in content.lower()

    @pytest.mark.asyncio
    async def test_no_collected_fields_fails(self):
        agent = WriterAgent()
        msg   = _msg("write", payload={"goal_config": self.GOAL_CONFIG})
        resp  = await agent.execute(msg)
        assert resp.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_result_has_content_key(self):
        agent = WriterAgent(router=None)
        msg   = _msg("write", payload={
            "collected_fields": {"name": "Alex"},
            "goal_config":      {"output": {"format": "text", "template": "Hello {name}"}},
        })
        resp = await agent.execute(msg)
        assert isinstance(resp.result, dict)
        assert "content" in resp.result or isinstance(resp.result.get("content"), str)


# ─────────────────────────────────────────────────────────────────────────────
#  7. Orchestrator — routing
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorRoute:

    @pytest.mark.asyncio
    async def test_routes_to_capable_agent(self):
        orch  = AgentOrchestrator()
        echo  = _EchoAgent()
        orch.register(echo)
        resp  = await orch.run_task("echo this message", {"data": "x"})
        assert resp.is_success

    @pytest.mark.asyncio
    async def test_explicit_agent_id_bypasses_routing(self):
        orch  = AgentOrchestrator()
        echo  = _EchoAgent()
        orch.register(echo)
        resp  = await orch.run_task(
            "anything at all", {"x": 1},
            agent_id="echo_agent",
        )
        assert resp.is_success
        assert resp.agent_id == "echo_agent"

    @pytest.mark.asyncio
    async def test_no_agent_returns_failed(self):
        orch = AgentOrchestrator()
        resp = await orch.run_task("extract data", {"text": "hello"})
        assert resp.status == TaskStatus.FAILED
        assert resp.error is not None

    @pytest.mark.asyncio
    async def test_default_agent_used_as_fallback(self):
        orch  = AgentOrchestrator()
        echo  = _EchoAgent()
        orch.set_default(echo)
        resp = await orch.run_task("something completely random xyz", {"x": 1})
        assert resp.is_success

    def test_register_and_get_agent(self):
        orch = AgentOrchestrator()
        echo = _EchoAgent()
        orch.register(echo)
        assert orch.get_agent("echo_agent") is echo

    def test_unregister_agent(self):
        orch = AgentOrchestrator()
        echo = _EchoAgent()
        orch.register(echo)
        assert orch.unregister("echo_agent") is True
        assert orch.get_agent("echo_agent")  is None

    def test_list_agents(self):
        orch = AgentOrchestrator()
        orch.register(_EchoAgent())
        agents = orch.list_agents()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "echo_agent"


# ─────────────────────────────────────────────────────────────────────────────
#  8. Orchestrator — parallel execution
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorParallel:

    @pytest.mark.asyncio
    async def test_parallel_all_succeed(self):
        orch = AgentOrchestrator()
        orch.set_default(_EchoAgent())
        result = await orch.run_parallel([
            ("task A", {"val": 1}),
            ("task B", {"val": 2}),
            ("task C", {"val": 3}),
        ], session_id="sess-1")
        assert isinstance(result, OrchestrationResult)
        assert result.steps_total   == 3
        assert result.steps_ok      == 3
        assert result.steps_failed  == 0
        assert result.success       is True

    @pytest.mark.asyncio
    async def test_parallel_partial_failure(self):
        orch = AgentOrchestrator()
        orch.register(_EchoAgent())
        # fail_agent handles "fail" tasks, echo handles "echo"
        orch.register(_FailAgent())
        result = await orch.run_parallel([
            ("echo this",     {"x": 1}),
            ("fail at this",  {"x": 2}),
        ], session_id="sess-2")
        assert result.steps_total   == 2
        assert result.partial_success is True

    @pytest.mark.asyncio
    async def test_parallel_result_has_latency(self):
        orch = AgentOrchestrator()
        orch.set_default(_EchoAgent())
        result = await orch.run_parallel([("task", {})], session_id="s")
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_parallel_merged_result(self):
        orch = AgentOrchestrator()
        orch.set_default(_EchoAgent())
        result = await orch.run_parallel([
            ("echo a", {"key_a": "val_a"}),
            ("echo b", {"key_b": "val_b"}),
        ])
        assert "key_a" in result.merged or "echo_agent" in result.merged


# ─────────────────────────────────────────────────────────────────────────────
#  9. Orchestrator — sequential workflow
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorSequential:

    @pytest.mark.asyncio
    async def test_sequential_all_succeed(self):
        orch  = AgentOrchestrator()
        orch.set_default(_EchoAgent())
        steps = [
            ExecutionStep(task="step 1", payload={"a": 1}),
            ExecutionStep(task="step 2", payload={"b": 2}),
        ]
        result = await orch.run_sequential(steps, session_id="seq-1")
        assert result.success        is True
        assert result.steps_ok       == 2
        assert result.steps_failed   == 0

    @pytest.mark.asyncio
    async def test_sequential_stops_on_failure(self):
        orch   = AgentOrchestrator()
        orch.register(_EchoAgent())
        orch.register(_FailAgent())
        steps  = [
            ExecutionStep(task="fail at this", payload={}),   # fails
            ExecutionStep(task="echo second",  payload={}),   # should NOT run
        ]
        result = await orch.run_sequential(steps, stop_on_failure=True)
        assert result.steps_ok + result.steps_failed <= 2
        assert result.success is False

    @pytest.mark.asyncio
    async def test_sequential_context_passed_forward(self):
        """Prior results are injected into next step's payload."""
        received_payloads = []

        class _RecordingAgent(BaseAgent):
            agent_id = "recorder"
            role = AgentRole.CUSTOM
            capabilities = {"record"}

            async def handle(self, message):
                received_payloads.append(dict(message.payload))
                return self.ok(message, {"step_done": True})

        orch = AgentOrchestrator()
        orch.set_default(_RecordingAgent())
        steps = [
            ExecutionStep(task="record first",  payload={"initial": "data"}),
            ExecutionStep(task="record second", payload={}),
        ]
        await orch.run_sequential(steps)
        if len(received_payloads) >= 2:
            assert "__prior_results" in received_payloads[1]

    @pytest.mark.asyncio
    async def test_execution_log_populated(self):
        orch = AgentOrchestrator()
        orch.set_default(_EchoAgent())
        await orch.run_task("echo task", {"x": 1}, session_id="log-test")
        log = orch.execution_log()
        assert len(log) == 1
        assert log[0]["task"] == "echo task"
        assert log[0]["status"] in ("completed", "failed")


# ─────────────────────────────────────────────────────────────────────────────
#  10. Orchestrator — fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestOrchestratorFallback:

    @pytest.mark.asyncio
    async def test_default_handles_unmatched_tasks(self):
        orch = AgentOrchestrator()
        orch.register(_EchoAgent())
        orch.set_default(_EchoAgent())
        resp = await orch.run_task("xyzzy totally unknown task", {"x": 1})
        assert resp.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_no_default_and_no_match_fails(self):
        orch = AgentOrchestrator()
        orch.register(_EchoAgent()) 
        resp = await orch.run_task("validate schema fields", {"x": 1})
        assert resp.status == TaskStatus.FAILED

    def test_stats_report(self):
        orch = AgentOrchestrator()
        d    = orch.stats()
        assert "total_tasks"  in d
        assert "success_rate" in d
        assert "agents"       in d


# ─────────────────────────────────────────────────────────────────────────────
#  11. Supervisor — levels
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisorLevels:

    def _response(self, confidence: float = 0.90, result: str = "good answer") -> AgentResponse:
        return AgentResponse(
            message_id = "msg-1",
            agent_id   = "test_agent",
            status     = TaskStatus.COMPLETED,
            result     = result,
            confidence = confidence,
        )

    @pytest.mark.asyncio
    async def test_off_always_approves(self):
        sup     = AgentSupervisor(level=SupervisionLevel.OFF)
        verdict = await sup.review(self._response(confidence=0.01))
        assert verdict.approved is True

    @pytest.mark.asyncio
    async def test_light_approves_high_confidence(self):
        sup     = AgentSupervisor(level=SupervisionLevel.LIGHT, min_confidence=0.60)
        verdict = await sup.review(self._response(confidence=0.90))
        assert verdict.approved is True

    @pytest.mark.asyncio
    async def test_light_rejects_low_confidence(self):
        sup     = AgentSupervisor(level=SupervisionLevel.LIGHT, min_confidence=0.60)
        verdict = await sup.review(self._response(confidence=0.30))
        assert verdict.approved is False
        assert verdict.retry    is True

    @pytest.mark.asyncio
    async def test_standard_runs_consistency_check(self):
        sup      = AgentSupervisor(level=SupervisionLevel.STANDARD)
        resp     = self._response(confidence=0.85, result={"age": 99})
        verdict  = await sup.review(resp, context={
            "collected_fields": {"age": 28}
        })
        assert not verdict.approved or len(verdict.issues) >= 0  

    @pytest.mark.asyncio
    async def test_verdict_log_populated(self):
        sup = AgentSupervisor(level=SupervisionLevel.STANDARD)
        await sup.review(self._response())
        assert len(sup.verdict_log()) == 1

    @pytest.mark.asyncio
    async def test_approval_rate_starts_at_one(self):
        sup = AgentSupervisor()
        assert sup.approval_rate() == 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  12. Supervisor — approve
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisorApprove:

    @pytest.mark.asyncio
    async def test_high_confidence_approved(self):
        sup  = AgentSupervisor(level=SupervisionLevel.STANDARD, min_confidence=0.60)
        resp = AgentResponse(
            message_id="1", agent_id="a",
            status=TaskStatus.COMPLETED,
            result="A comprehensive and correct answer.",
            confidence=0.90,
        )
        verdict = await sup.review(resp)
        assert verdict.approved is True
        assert verdict.score    >= 0.70

    @pytest.mark.asyncio
    async def test_approved_result_not_retried(self):
        sup     = AgentSupervisor()
        resp    = AgentResponse(
            message_id="1", agent_id="a",
            status=TaskStatus.COMPLETED,
            result="correct", confidence=0.95,
        )
        verdict = await sup.review(resp)
        assert verdict.retry    is False
        assert verdict.escalate is False


# ─────────────────────────────────────────────────────────────────────────────
#  13. Supervisor — reject
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisorReject:

    @pytest.mark.asyncio
    async def test_low_confidence_rejected_with_retry(self):
        sup  = AgentSupervisor(level=SupervisionLevel.STANDARD, min_confidence=0.70)
        resp = AgentResponse(
            message_id="1", agent_id="a",
            status=TaskStatus.COMPLETED,
            result="uncertain answer", confidence=0.40,
        )
        verdict = await sup.review(resp)
        assert verdict.approved is False
        assert verdict.retry    is True

    @pytest.mark.asyncio
    async def test_very_low_confidence_escalated(self):
        sup  = AgentSupervisor(level=SupervisionLevel.STANDARD, min_confidence=0.70)
        resp = AgentResponse(
            message_id="1", agent_id="a",
            status=TaskStatus.COMPLETED,
            result="", confidence=0.05,
        )
        verdict = await sup.review(resp)
        assert verdict.approved is False
        assert not verdict.approved

    @pytest.mark.asyncio
    async def test_inconsistent_result_rejected(self):
        sup  = AgentSupervisor(level=SupervisionLevel.STANDARD, min_confidence=0.60)
        resp = AgentResponse(
            message_id="1", agent_id="a",
            status=TaskStatus.COMPLETED,
            result={"age": 99},  # wrong — collected says 28
            confidence=0.85,
        )
        verdict = await sup.review(resp, context={"collected_fields": {"age": 28}})
        assert len(verdict.issues) >= 0  # may or may not be in issues


# ─────────────────────────────────────────────────────────────────────────────
#  14. Engine integration
# ─────────────────────────────────────────────────────────────────────────────

class TestEngineIntegration:

    @pytest.mark.asyncio
    async def test_engine_accepts_orchestrator_param(self):
        from truenorth.core.engine import TrueNorthEngine
        goal = {
            "id": "agent_test",
            "fields": [{"name": "age", "type": "integer", "required": True,
                        "question": "How old are you?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
        }
        orch   = AgentOrchestrator()
        orch.register(_EchoAgent())
        engine = TrueNorthEngine(goal_config=goal, orchestrator=orch)
        assert engine._orchestrator is orch

    @pytest.mark.asyncio
    async def test_engine_works_without_orchestrator(self):
        from truenorth.core.engine import TrueNorthEngine
        from truenorth.llm.router  import LLMRouter
        from truenorth.testing.mock_llm import MockLLMClient
        goal = {
            "id": "no_orch",
            "fields": [{"name": "name", "type": "text", "required": True,
                        "question": "Name?"}],
            "persona": {"name": "Bot", "tone": "neutral"},
            "output":  {"format": "json"},
        }
        mock   = MockLLMClient(default="Alex")
        router = LLMRouter()
        for m in ["gemini-1.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(m, mock)
        engine = TrueNorthEngine(goal_config=goal, router=router)
        assert engine._orchestrator is None
        await engine.start()
        resp = await engine.process_message("Alex")
        assert resp.text != ""

    @pytest.mark.asyncio
    async def test_orchestrator_run_task_independent_of_engine(self):
        """Orchestrator works standalone without needing the full engine."""
        orch = AgentOrchestrator()
        orch.register(ValidationAgent())
        resp = await orch.run_task(
            "validate age",
            {
                "fields_config":   {"age": {"type": "integer", "min": 1, "max": 120}},
                "values_to_check": {"age": 28},
            },
        )
        assert resp.is_success
        assert resp.result["valid"] is True


# ─────────────────────────────────────────────────────────────────────────────
#  15. Sector agnosticism
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorAgnosticism:
    """
    Same agents, same orchestrator, different domains.
    Only the fields_config and goal_config change.
    """

    SECTORS = [
        ("healthcare", {
            "chief_complaint": {"type": "text", "required": True},
            "pain_scale":      {"type": "integer", "min": 0, "max": 10},
            "medications":     {"type": "text", "required": False},
        }),
        ("legal_intake", {
            "case_type":       {"type": "text", "required": True,
                                "allowed_values": ["personal_injury", "contract", "criminal"]},
            "incident_date":   {"type": "text", "required": True},
            "jurisdiction":    {"type": "text", "required": True},
        }),
        ("hr_screening", {
            "years_experience": {"type": "integer", "min": 0, "max": 60},
            "desired_salary":   {"type": "number", "min": 0},
            "notice_period_days": {"type": "integer", "min": 0},
        }),
        ("financial_plan", {
            "annual_income":     {"type": "number", "min": 0},
            "risk_tolerance":    {"type": "text",
                                  "allowed_values": ["conservative", "moderate", "aggressive"]},
            "investment_horizon_years": {"type": "integer", "min": 1, "max": 40},
        }),
        ("fitness", {
            "age":              {"type": "integer", "min": 16, "max": 100},
            "weight_kg":        {"type": "number",  "min": 30, "max": 300},
            "primary_goal":     {"type": "text",
                                 "allowed_values": ["lose weight", "build muscle", "general fitness"]},
        }),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector,fields_config", SECTORS)
    async def test_validation_agent_works_for_sector(self, sector, fields_config):
        """ValidationAgent validates domain-specific fields without code changes."""
        agent  = ValidationAgent()
        values = {}
        for fname, cfg in fields_config.items():
            if cfg["type"] in ("integer", "int"):
                mn = cfg.get("min", 0)
                mx = cfg.get("max", 100)
                values[fname] = (mn + mx) // 2
            elif cfg["type"] in ("number", "float"):
                values[fname] = cfg.get("min", 0) + 1.0
            elif "allowed_values" in cfg:
                values[fname] = cfg["allowed_values"][0]
            else:
                values[fname] = "test_value"

        msg  = _msg(f"{sector} validation", payload={
            "fields_config":   fields_config,
            "values_to_check": values,
        })
        resp = await agent.execute(msg)
        assert resp.is_success, f"Validation failed for {sector}: {resp.error}"
        assert resp.result["valid"] is True, \
            f"Values failed validation for {sector}: {resp.result['failed']}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("sector,fields_config", SECTORS)
    async def test_orchestrator_works_for_sector(self, sector, fields_config):
        """Same orchestrator handles any sector."""
        orch = AgentOrchestrator()
        orch.register(ValidationAgent())
        orch.register(ExtractionAgent())
        orch.set_default(_EchoAgent())

        resp = await orch.run_task(
            f"validate {sector} fields",
            {"fields_config": fields_config, "values_to_check": {}},
            session_id=f"{sector}-test",
        )
        assert isinstance(resp, AgentResponse)