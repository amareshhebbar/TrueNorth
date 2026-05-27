"""
Dry-run engine: run a full agent scenario with zero LLM API calls.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from truenorth.core.engine import TrueNorthEngine
from truenorth.core.yaml_loader import YamlLoader
from truenorth.core.graph_state import GraphState
from truenorth.testing.mock_llm import MockLLMClient
from truenorth.llm.router import LLMRouter
import uuid


@dataclass
class DryRunResult:
    turns: list[dict]
    final_profile: dict
    missing_required: list[str]
    missing_optional: list[str]
    api_calls_made: int
    completed: bool
    bugs: list[str]

    def print_report(self):
        from rich.console import Console
        from rich.table import Table
        from rich import box

        c = Console()
        c.print(f"\n[bold]TrueNorth Dry Run Report[/bold]")

        for i, turn in enumerate(self.turns):
            c.print(f"\n[dim]Turn {i+1}[/dim]")
            if turn["role"] == "user":
                c.print(f"  [cyan]USER:[/cyan] {turn['content']}")
            else:
                c.print(f"  [green]AGENT:[/green] {turn['content']}")

        c.print(f"\n[bold]Summary[/bold]")
        c.print(f"  Turns: {len(self.turns)}")
        c.print(f"  API calls: {self.api_calls_made} [green](mock — $0.00)[/green]")
        c.print(f"  Completed: {'✓' if self.completed else '✗'}")

        if self.final_profile:
            c.print(f"\n[bold]Collected fields:[/bold]")
            for k, v in self.final_profile.items():
                c.print(f"  ✓ {k}: {v}")

        if self.missing_required:
            c.print(f"\n[bold red]Missing required:[/bold red]")
            for f in self.missing_required:
                c.print(f"  ✗ {f}")

        if self.bugs:
            c.print(f"\n[bold red]Bugs detected:[/bold red]")
            for bug in self.bugs:
                c.print(f"  ⚠ {bug}")


class DryRunner:
    def __init__(self, goal_yaml: str, scenario_path: str):
        self.config = YamlLoader().load(goal_yaml)
        self.scenario_path = scenario_path
        scenario = json.loads(Path(scenario_path).read_text())
        self.user_turns = [t["user"] for t in scenario.get("turns", [])]
        self.expected_profile = scenario.get("expected_profile", {})

    async def run(self) -> DryRunResult:
        mock_llm = MockLLMClient(self.scenario_path)

        class MockRouter(LLMRouter):
            def __init__(self_r):
                self_r._clients = {"mock": mock_llm}
                self_r._session_cost = 0.0
                self_r._session_tokens = 0
                self_r.max_cost_usd = 99.0
                self_r._call_count = 0

            async def complete(self_r, task, prompt, **kwargs):
                self_r._call_count += 1
                return await mock_llm.complete(prompt, **{k:v for k,v in kwargs.items() if k in ('system','model','temperature','max_tokens')})

            async def complete_json(self_r, task, prompt, **kwargs):
                self_r._call_count += 1
                resp = await mock_llm.complete(prompt, **{k:v for k,v in kwargs.items() if k in ('system','model','temperature','max_tokens')})
                import json as _json
                try:
                    return _json.loads(resp.content), resp
                except:
                    return {}, resp

        mock_router = MockRouter()
        engine = TrueNorthEngine(self.config, mock_router)
        state = engine.create_initial_state(str(uuid.uuid4()))

        turns = []
        welcome = await engine.generate_welcome_message(state)
        turns.append({"role": "assistant", "content": welcome})

        for user_msg in self.user_turns:
            turns.append({"role": "user", "content": user_msg})
            state, response = await engine.process_turn(state, user_msg)
            turns.append({"role": "assistant", "content": response})

            if state.completed or state.escalated:
                break

        # Detect bugs
        bugs = []
        for required_field in self.config.required_fields:
            if required_field.name not in state.profile:
                bugs.append(f"Required field '{required_field.name}' was never collected")

        return DryRunResult(
            turns=turns,
            final_profile=state.collected_fields,
            missing_required=state.missing_required,
            missing_optional=state.missing_optional,
            api_calls_made=mock_router._call_count,
            completed=state.completed,
            bugs=bugs,
        )
