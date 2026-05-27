"""Unit tests for the TrueNorthEngine using MockLLMClient."""

import pytest
import uuid
from truenorth.core.engine import TrueNorthEngine
from truenorth.core.yaml_loader import YamlLoader, GoalConfig
from truenorth.llm.router import LLMRouter
from truenorth.testing.mock_llm import MockLLMClient


SIMPLE_GOAL_YAML = """
goal_id: test_goal
persona:
  base: test assistant
required_fields:
  - name: age
    type: integer
  - name: name
    type: text
"""


@pytest.fixture
def config():
    return YamlLoader().load(SIMPLE_GOAL_YAML)


@pytest.fixture
def mock_router():
    mock = MockLLMClient()
    router = LLMRouter.__new__(LLMRouter)
    router._clients = {"mock": mock}
    router._session_cost = 0.0
    router._session_tokens = 0
    router.max_cost_usd = 99.0

    import json as _json

    async def _complete(task, prompt, **kwargs):
        return await mock.complete(prompt, **{k:v for k,v in kwargs.items()
                                              if k in ('system','model','temperature','max_tokens')})

    async def _complete_json(task, prompt, **kwargs):
        resp = await mock.complete(prompt)
        try:
            return _json.loads(resp.content), resp
        except:
            return {}, resp

    router.complete = _complete
    router.complete_json = _complete_json
    return router


def test_initial_state(config, mock_router):
    engine = TrueNorthEngine(config, mock_router)
    state = engine.create_initial_state("test-session-1")

    assert state.session_id == "test-session-1"
    assert state.goal_id == "test_goal"
    assert state.missing_required == ["age", "name"]
    assert not state.completed
    assert state.cost_usd == 0.0


def test_field_set(config, mock_router):
    from truenorth.core.graph_state import FieldValue
    engine = TrueNorthEngine(config, mock_router)
    state = engine.create_initial_state("test-session-2")

    fv = FieldValue(value=25, confidence=0.9, source="user_stated", raw_text="I'm 25")
    new_state = state.set_field("age", fv)

    assert "age" in new_state.profile
    assert new_state.profile["age"].value == 25
    assert "age" not in new_state.missing_required


@pytest.mark.asyncio
async def test_process_turn_extracts_nothing_gracefully(config, mock_router):
    engine = TrueNorthEngine(config, mock_router)
    state = engine.create_initial_state("test-session-3")
    new_state, response = await engine.process_turn(state, "Hello!")
    assert response  # Should always get some response
    assert not new_state.completed  # Still missing required fields
