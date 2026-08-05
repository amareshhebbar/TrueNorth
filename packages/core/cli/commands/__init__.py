"""
cli/commands/__init__.py

All TrueNorth CLI command implementations.
Each async function is wrapped by Click in cli/main.py.

Commands:
  chat      — interactive conversation in the terminal
  dry-run   — automated test run from scenario file or auto-answers
  validate  — validate a goal YAML against the JSON Schema
  replay    — replay a past session turn-by-turn
  cost      — show cost breakdown for a session
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_BLUE   = "\033[34m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"

def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}"

def _header(title: str) -> None:
    print(f"\n{_BOLD}{'─'*58}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'─'*58}{_RESET}")

def _agent(name: str, text: str) -> None:
    print(f"\n  {_c(_CYAN+_BOLD, f'🤖 {name}:')} {text}")

def _user_prompt(name: str = "You") -> str:
    try:
        text = input(f"\n  {_c(_GREEN+_BOLD, f'👤 {name}:')} ")
        return text.strip()
    except (EOFError, KeyboardInterrupt):
        return "/quit"

def _info(label: str, value: str) -> None:
    print(f"  {_c(_DIM, label):<30} {value}")

def _success(text: str) -> None:
    print(f"\n  {_c(_GREEN, '✓')} {text}")

def _error(text: str) -> None:
    print(f"\n  {_c(_RED, '✗')} {text}", file=sys.stderr)

def _warn(text: str) -> None:
    print(f"\n  {_c(_YELLOW, '!')} {text}")

def _progress_bar(collected: int, total: int, width: int = 28) -> str:
    pct   = collected / max(total, 1)
    filled = int(pct * width)
    bar   = "█" * filled + "░" * (width - filled)
    return f"{_c(_GREEN, bar)} {collected}/{total}"

async def cmd_chat(
    goal:       str,
    mock:       bool,
    session_id: Optional[str],
    user_name:  str,
    no_color:   bool,
) -> None:
    """
    Run an interactive conversation in the terminal.

    Special commands (type during chat):
      /quit     — exit without saving
      /status   — show collected fields so far
      /cost     — show current cost
      /skip     — skip the current question (marks field optional)
      /save     — save progress and exit (resumable)
    """

    _add_package_to_path()
    from truenorth.core.engine import TrueNorthEngine
    from truenorth.llm.router import LLMRouter

    goal_path = _resolve_goal_path(goal)
    if not goal_path:
        _error(f"Goal YAML not found: {goal}")
        _warn("Tried: current dir, examples/goals/, packages/core/examples/goals/")
        sys.exit(1)

    router = _build_router(mock)

    try:
        engine = await TrueNorthEngine.from_yaml(
            str(goal_path),
            session_id = session_id,
            router     = router,
        )
    except Exception as e:
        _error(f"Failed to load goal: {e}")
        sys.exit(1)

    goal_config = engine.state.goal_config
    persona     = goal_config.get("persona", {})
    agent_name  = persona.get("name", "TrueNorth")
    total_req   = len(engine.state.required_fields)

    _header(f"TrueNorth Chat — {goal_config.get('name', goal_path.stem)}")
    _info("Goal YAML",    str(goal_path))
    _info("Session ID",   engine.get_session_id())
    _info("Agent",        agent_name)
    _info("LLM mode",     "mock (no API cost)" if mock else "live")
    _info("Required fields", str(total_req))
    print(f"\n  {_c(_DIM, 'Type /quit to exit  •  /status to see progress  •  /skip to skip a question')}")

    try:
        start_resp = await engine.start()
        _agent(agent_name, start_resp.text)
    except Exception as e:
        _error(f"Engine failed to start: {e}")
        sys.exit(1)

    while not engine.state.is_complete:

        collected = len(engine.state.collected_fields)
        print(f"  {_c(_DIM, _progress_bar(collected, total_req))}", end="\r")

        user_input = _user_prompt(user_name)

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print(f"\n  {_c(_YELLOW, 'Session paused. Resume with:')} "
                  f"truenorth chat --session-id {engine.get_session_id()}")
            break

        if user_input.lower() == "/status":
            _print_status(engine)
            continue

        if user_input.lower() == "/cost":
            _print_cost(engine)
            continue

        if user_input.lower() == "/skip":
            user_input = "[skip]"
        try:
            response = await engine.process_message(user_input)
        except KeyboardInterrupt:
            print(f"\n\n  {_c(_YELLOW, 'Interrupted.')}")
            break
        except Exception as e:
            _error(f"Error processing message: {e}")
            _warn("The conversation can continue — try rephrasing.")
            continue

        _agent(agent_name, response.text)

        if response.is_complete and response.final_output:
            _print_final_output(response.final_output, goal_config)
            break

    _print_session_summary(engine)

async def cmd_dry_run(
    goal:     str,
    scenario: Optional[str],
    mock:     bool,
    verbose:  bool,
    output:   Optional[str],
) -> None:
    """Run an automated test conversation — no human input needed."""
    _add_package_to_path()
    from truenorth.testing.dry_runner import DryRunner

    goal_path = _resolve_goal_path(goal)
    if not goal_path:
        _error(f"Goal YAML not found: {goal}")
        sys.exit(1)

    _header(f"DRY RUN — {goal_path.stem}")
    _info("Goal",     str(goal_path))
    _info("Scenario", scenario or "auto-generate answers")
    _info("LLM mode", "mock" if mock else "live")
    print()

    runner = DryRunner(
        goal_path     = str(goal_path),
        scenario_path = scenario,
        mock          = mock,
        verbose       = verbose,
    )

    t0     = time.perf_counter()
    report = await runner.run()
    elapsed = time.perf_counter() - t0

    if not verbose:
        print(report.summary())

    if output:
        Path(output).write_text(json.dumps(report.to_dict(), indent=2))
        _success(f"Report saved to {output}")

    sys.exit(0 if report.passed else 1)

def cmd_validate(goal: str) -> None:
    """Validate a goal YAML file against the JSON Schema."""
    _add_package_to_path()
    from truenorth.core.yaml_loader import YAMLLoader, YAMLLoaderError

    goal_path = _resolve_goal_path(goal)
    if not goal_path:
        _error(f"Goal YAML not found: {goal}")
        sys.exit(1)

    _header(f"VALIDATE — {goal_path.stem}")

    try:
        config = YAMLLoader.load(str(goal_path))
        _success(f"YAML is valid: {goal_path.name}")
        print()
        _info("Goal ID",       config.get("id", "?"))
        _info("Goal name",     config.get("name", "?"))
        _info("Total fields",  str(len(config.get("fields", []))))
        required = [f["name"] for f in config.get("fields", []) if f.get("required", True)]
        optional = [f["name"] for f in config.get("fields", []) if not f.get("required", True)]
        _info("Required",      ", ".join(required) or "none")
        _info("Optional",      ", ".join(optional) or "none")
        _info("Output format", config.get("output", {}).get("format", "text"))
        pii_fields = [f["name"] for f in config.get("fields", []) if f.get("pii", False)]
        if pii_fields:
            _info("PII fields",  ", ".join(pii_fields))
        print()
    except YAMLLoaderError as e:
        _error(f"Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        _error(f"Unexpected error: {e}")
        sys.exit(1)

async def cmd_cost(session_id: str) -> None:
    """Show cost breakdown for a session ID."""
    _add_package_to_path()
    from truenorth.llm.cost_tracker import CostTracker

    tracker = CostTracker()
    summary = tracker.summary(session_id)

    _header(f"COST — {session_id[:16]}...")
    _info("Total cost",     summary.get("cost_formatted", "$0.0000"))
    _info("Total tokens",   str(summary.get("total_tokens", 0)))
    _info("LLM calls",      str(summary.get("call_count", 0)))
    if summary.get("budget_usd"):
        _info("Budget",     f"${summary['budget_usd']:.4f}")
        _info("Budget used", f"{summary.get('budget_used_pct', 0):.1f}%")
    print()
    by_task = summary.get("by_task", {})
    if by_task:
        print(f"  {_c(_DIM, 'By task:')}")
        for task, cost in by_task.items():
            print(f"    {task:<20} ${cost:.6f}")
    by_model = summary.get("by_model", {})
    if by_model:
        print(f"\n  {_c(_DIM, 'By model:')}")
        for model, cost in by_model.items():
            print(f"    {model:<36} ${cost:.6f}")
    print()

def _print_status(engine) -> None:
    state = engine.state
    total = len(state.required_fields)
    coll  = len(state.collected_fields)
    print()
    _header("SESSION STATUS")
    _info("Session ID",  state.session_id)
    _info("Turn",        str(state.current_turn))
    _info("Progress",    f"{coll}/{total} required fields")
    _info("Cost so far", f"${state.total_cost_usd:.4f}")
    _info("Language",    state.detected_language)
    if state.collected_fields:
        print(f"\n  {_c(_DIM, 'Collected:')}")
        for k, v in state.collected_fields.items():
            conf = state.field_confidences.get(k, 0)
            print(f"    {_c(_GREEN, '✓')} {k:<28} = {v!r}  (conf {conf:.2f})")
    if state.missing_required:
        print(f"\n  {_c(_DIM, 'Still needed:')}")
        for m in state.missing_required:
            print(f"    {_c(_YELLOW, '○')} {m}")
    print()

def _print_cost(engine) -> None:
    state = engine.state
    print()
    _info("Cost so far",  f"${state.total_cost_usd:.6f}")
    budget = state.cost_budget_usd
    if budget:
        remaining = max(0.0, budget - state.total_cost_usd)
        _info("Budget",       f"${budget:.4f}")
        _info("Remaining",    f"${remaining:.4f}")
    print()

def _print_final_output(final_output: dict, goal_config: dict) -> None:
    fmt     = final_output.get("format", "text")
    content = final_output.get("content", "")
    print()
    _header(f"YOUR RESULTS — {goal_config.get('name', '')}")
    if fmt == "json":
        print(json.dumps(content, indent=2))
    else:
        print(content)
    print()

def _print_session_summary(engine) -> None:
    state = engine.state
    print()
    _header("SESSION SUMMARY")
    _info("Session ID",    state.session_id)
    _info("Total turns",   str(state.current_turn))
    _info("Fields collected", f"{len(state.collected_fields)}/{len(state.required_fields)}")
    _info("Total cost",    f"${state.total_cost_usd:.4f}")
    _info("Language",      state.detected_language)
    print()

def _add_package_to_path() -> None:
    """Ensure the packages/core directory is on sys.path."""
    candidates = [
        Path(__file__).parent.parent.parent,
        Path.cwd() / "packages" / "core",
        Path.cwd() / "core",
        Path.cwd(),
    ]
    for p in candidates:
        if (p / "truenorth").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return

def _resolve_goal_path(goal: str) -> Optional[Path]:
    """
    Resolve goal arg to an actual file path.
    Tries: as-is, examples/goals/<goal>.yaml, packages/core/examples/goals/<goal>.yaml
    """
    candidates = [
        Path(goal),
        Path("examples") / "goals" / goal,
        Path("examples") / "goals" / f"{goal}.yaml",
        Path("packages") / "core" / "examples" / "goals" / goal,
        Path("packages") / "core" / "examples" / "goals" / f"{goal}.yaml",
        Path(__file__).parent.parent.parent / "examples" / "goals" / goal,
        Path(__file__).parent.parent.parent / "examples" / "goals" / f"{goal}.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return None

def _build_router(mock: bool):
    """Build LLMRouter — mock for testing, real for production."""
    _add_package_to_path()
    from truenorth.llm.router import LLMRouter

    if mock:
        from truenorth.testing.mock_llm import MockLLMClient

        class _SmartMock(MockLLMClient):
            """Smart mock that returns realistic-looking extraction JSON."""

            async def generate(self, messages, system=None, max_tokens=1024, temperature=0.7, **kw):
                import json as _json
                resp = await super().generate(messages, system, max_tokens, temperature, **kw)

                content = " ".join(m.content for m in messages).lower()
                if "extract field values" in content or "extractions" in content:
                    resp.content = '{"extractions": []}'
                elif "classify" in content or "emotional" in content:
                    resp.content = '{"label": "neutral", "score": 0.6}'
                else:
                    resp.content = "Got it! And next — "
                return resp

        mock_client = _SmartMock(default="Understood, thank you.")
        router = LLMRouter()
        for model in ["gemini-3.5-flash", "claude-haiku-4-5-20251001", "claude-sonnet-4-20250514"]:
            router.register_client(model, mock_client)
        return router

    _warn_missing_keys()
    return LLMRouter.from_env()

def _warn_missing_keys() -> None:
    """Warn about missing API keys before real LLM calls."""
    missing = []
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GEMINI_API_KEY")

    if missing:
        _warn(f"Missing API keys: {', '.join(missing)}")
        _warn("Set them in .env or use --mock for a free test run.")
