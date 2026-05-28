"""
truenorth/cli/main.py  (also mounted as packages/core/cli/main.py)

TrueNorth CLI — entry point for the `truenorth` command.

Commands:
  truenorth chat        [--goal YAML] [--mock] [--session-id ID]
  truenorth dry-run     [--goal YAML] [--scenario JSON] [--mock] [--verbose]
  truenorth validate    [--goal YAML]
  truenorth cost        [--session-id ID]
  truenorth version

Registered in pyproject.toml as:
  [tool.poetry.scripts]
  truenorth = "truenorth.cli.main:cli"
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import click

# ── Make sure the package root is importable when running from any directory ──
_HERE = Path(__file__).resolve()
for _candidate in [_HERE.parent.parent.parent, _HERE.parent.parent]:
    if (_candidate / "truenorth").exists():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="truenorth")
def cli() -> None:
    """
    \b
    TrueNorth — AI agent framework for real humans.

    \b
    Quick start:
      truenorth chat --mock                      # test with no API key
      truenorth chat --goal fitness_plan         # real conversation
      truenorth dry-run --goal fitness_plan      # automated test run
      truenorth validate --goal fitness_plan     # validate YAML
    """
    # Load .env if present (silent fail if python-dotenv not installed)
    _load_dotenv()


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--goal", "-g",
    default="fitness_plan",
    show_default=True,
    help="Goal YAML path or name (e.g. fitness_plan, medical_intake, path/to/goal.yaml)",
)
@click.option(
    "--mock", "-m",
    is_flag=True,
    default=False,
    help="Use mock LLM — no API key needed, free to run. "
         "Good for testing goal YAML and conversation flow.",
)
@click.option(
    "--session-id", "-s",
    default=None,
    help="Resume an existing session by ID.",
)
@click.option(
    "--name", "-n",
    default="You",
    show_default=True,
    help="Your name (shown in the chat prompt).",
)
@click.option(
    "--no-color",
    is_flag=True,
    default=False,
    help="Disable ANSI color output.",
)
def chat(goal: str, mock: bool, session_id: str, name: str, no_color: bool) -> None:
    """
    Start an interactive conversation with a TrueNorth agent.

    \b
    Examples:
      truenorth chat --mock
      truenorth chat --goal fitness_plan
      truenorth chat --goal path/to/my_goal.yaml
      truenorth chat --goal fitness_plan --session-id abc-123   (resume)

    \b
    Special commands during chat:
      /status   show collected fields
      /cost     show current cost
      /skip     skip the current question
      /quit     exit the conversation
    """
    from cli.commands import cmd_chat
    asyncio.run(cmd_chat(
        goal       = goal,
        mock       = mock,
        session_id = session_id,
        user_name  = name,
        no_color   = no_color,
    ))


# ---------------------------------------------------------------------------
# dry-run
# ---------------------------------------------------------------------------

@cli.command("dry-run")
@click.option(
    "--goal", "-g",
    default="fitness_plan",
    show_default=True,
    help="Goal YAML path or name.",
)
@click.option(
    "--scenario", "-s",
    default=None,
    help="Path to a JSON scenario file with predefined user answers.",
)
@click.option(
    "--mock/--no-mock",
    default=True,
    show_default=True,
    help="Use mock LLM (default: True — dry-run is free).",
)
@click.option(
    "--verbose/--quiet",
    default=True,
    show_default=True,
    help="Print each turn during the run.",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Save JSON report to this file path.",
)
def dry_run(goal: str, scenario: str, mock: bool, verbose: bool, output: str) -> None:
    """
    Run an automated test conversation — no human input needed.

    \b
    Loads a goal YAML, auto-generates user answers (or replays a scenario file),
    and verifies that all required fields are collected.

    \b
    Examples:
      truenorth dry-run --goal fitness_plan
      truenorth dry-run --goal fitness_plan --scenario tests/fixtures/scenarios/happy_path.json
      truenorth dry-run --goal fitness_plan --output report.json

    \b
    Exit code: 0 = passed, 1 = failed (missing fields or errors).
    """
    from cli.commands import cmd_dry_run
    asyncio.run(cmd_dry_run(
        goal     = goal,
        scenario = scenario,
        mock     = mock,
        verbose  = verbose,
        output   = output,
    ))


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--goal", "-g",
    required=True,
    help="Goal YAML path or name to validate.",
)
def validate(goal: str) -> None:
    """
    Validate a goal YAML file against the TrueNorth schema.

    \b
    Checks:
      - Valid YAML syntax
      - Required top-level keys (id, name, fields, output)
      - Field definitions (name, type, question)
      - JSON Schema validation (if schema file found)

    \b
    Examples:
      truenorth validate --goal fitness_plan
      truenorth validate --goal path/to/custom_goal.yaml
    """
    from cli.commands import cmd_validate
    cmd_validate(goal=goal)


# ---------------------------------------------------------------------------
# cost
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--session-id", "-s",
    required=True,
    help="Session ID to show cost for.",
)
def cost(session_id: str) -> None:
    """
    Show token usage and cost breakdown for a session.

    \b
    Example:
      truenorth cost --session-id abc-123-def-456
    """
    from cli.commands import cmd_cost
    asyncio.run(cmd_cost(session_id=session_id))


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

@cli.command()
def version() -> None:
    """Show TrueNorth version and environment info."""
    import platform
    click.echo("TrueNorth 0.1.0")
    click.echo(f"Python {platform.python_version()} on {platform.system()}")
    click.echo(f"Path: {Path(__file__).parent.parent}")
    # Check API keys
    keys = {
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "GEMINI_API_KEY":    os.environ.get("GEMINI_API_KEY", ""),
        "OPENAI_API_KEY":    os.environ.get("OPENAI_API_KEY", ""),
    }
    for name, val in keys.items():
        status = "✓ set" if val else "✗ not set"
        click.echo(f"  {name}: {status}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_dotenv() -> None:
    """Load .env file silently if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        # Try .env in cwd, then parent dirs
        for p in [Path.cwd(), Path.cwd().parent]:
            env_file = p / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()