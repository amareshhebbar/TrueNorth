"""
TrueNorth CLI entry point.

Commands:
  truenorth cost     --session SESSION_ID [--format json|table|csv]
  truenorth pricing  [--provider anthropic|google|openai|...] [--format table|json]
  truenorth estimate --model MODEL --tokens N [--output N]
  truenorth version

Usage:
  pip install truenorth
  truenorth cost --session sess-abc123
  truenorth pricing --provider anthropic
  truenorth estimate --model claude-haiku-4-5-20251001 --tokens 1000 --output 500
"""

from __future__ import annotations

import json
import sys

import click

try:
    from rich.console import Console
    from rich.table   import Table
    from rich.text    import Text
    from rich         import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False

_console = Console() if _RICH else None

VERSION = "0.1.1"

@click.group()
@click.version_option(VERSION, "--version", "-v", prog_name="truenorth")
def cli():
    """TrueNorth — AI agent framework CLI."""

@cli.command("cost")
@click.option("--session", "-s", required=False,
              help="Session ID to display cost breakdown for.")
@click.option("--goal",    "-g", required=False,
              help="Goal ID — aggregate cost across all sessions.")
@click.option("--format",  "-f", "fmt",
              type=click.Choice(["table", "json", "csv"]), default="table",
              help="Output format (default: table).")
@click.option("--top", "-t", default=5,
              help="Show top N most expensive calls (default: 5).")
@click.option("--db", envvar="TRUENORTH_DB_URL",
              help="Postgres URL (reads TRUENORTH_DB_URL env var).")
def cost_cmd(session: str, goal: str, fmt: str, top: int, db: str):
    """
    Display cost breakdown for a session or goal.

    Examples:
      truenorth cost --session sess-abc123
      truenorth cost --session sess-abc123 --format json
      truenorth cost --goal fitness_plan --format table
    """
    from truenorth.llm.cost_tracker import CostTracker

    if not session and not goal:
        click.echo("Error: provide --session SESSION_ID or --goal GOAL_ID", err=True)
        sys.exit(1)

    ct = CostTracker()

    if goal:
        _display_goal_cost(ct, goal, fmt)
    else:
        _display_session_cost(ct, session, fmt, top)

def _display_session_cost(ct, session_id: str, fmt: str, top: int):
    """Display per-session cost."""

    s   = ct.get_session_cost(session_id)
    bd  = ct.task_breakdown(session_id)
    top_calls = ct.top_expensive_calls(session_id, limit=top)
    turns = ct.get_all_turns(session_id)

    data = {
        "session":        s.to_dict(),
        "task_breakdown": bd,
        "top_calls":      top_calls,
        "turns":          [t.to_dict() for t in turns],
    }

    if fmt == "json":
        click.echo(json.dumps(data, indent=2, default=str))
        return

    if fmt == "csv":
        click.echo(ct.export_csv(session_id))
        return

    if _RICH:
        _rich_session_cost(session_id, s, bd, top_calls, turns)
    else:
        _plain_session_cost(session_id, s, bd, top_calls)

def _display_goal_cost(ct, goal_id: str, fmt: str):
    """Display per-goal aggregated cost."""
    gc = ct.goal_cost(goal_id)
    if fmt == "json":
        click.echo(json.dumps(gc.to_dict(), indent=2, default=str))
        return
    if _RICH:
        _rich_goal_cost(goal_id, gc)
    else:
        d = gc.to_dict()
        click.echo(f"Goal: {goal_id}")
        click.echo(f"  Sessions:   {d['session_count']}")
        click.echo(f"  Total cost: ${d['total_cost_usd']:.4f}")
        click.echo(f"  Avg/session: ${d['avg_cost_per_session']:.4f}")

def _rich_session_cost(session_id, s, bd, top_calls, turns):
    con = _console

    con.print()
    con.rule(f"[bold]Cost Report — {session_id}[/bold]")
    con.print()

    budget_str = ""
    if s.budget_usd:
        pct   = s.budget_used_pct or 0
        color = {"ok":"green","warning":"yellow","critical":"red","exceeded":"red bold"}.get(
            s.budget_status.value, "white")
        budget_str = f"  Budget: [bold]${s.total_cost_usd:.4f}[/bold] / ${s.budget_usd:.2f} [{color}]{pct:.1f}%[/{color}]"

    con.print(
        f"  Total cost:   [bold green]${s.total_cost_usd:.4f}[/bold green]"
        f"   Tokens: {s.total_tokens:,}"
        f"   Calls: {s.call_count}"
        f"   Turns: {s.turn_count}"
        + budget_str
    )
    con.print()

    if bd:
        t = Table(title="Cost by Task", box=rich_box.SIMPLE_HEAVY, show_header=True)
        t.add_column("Task",        style="cyan",  no_wrap=True)
        t.add_column("Cost (USD)",  justify="right")
        t.add_column("% of Total",  justify="right")
        t.add_column("Calls",       justify="right")
        for task in sorted(bd, key=lambda x: bd[x]["cost_usd"], reverse=True):
            v = bd[task]
            t.add_row(task, f"${v['cost_usd']:.6f}", f"{v['pct']:.1f}%", str(v["call_count"]))
        con.print(t)
        con.print()

    if top_calls:
        t2 = Table(title=f"Top {len(top_calls)} Most Expensive Calls",
                   box=rich_box.SIMPLE_HEAVY, show_header=True)
        t2.add_column("Turn",   justify="right")
        t2.add_column("Model",  style="dim")
        t2.add_column("Task",   style="cyan")
        t2.add_column("In",     justify="right")
        t2.add_column("Out",    justify="right")
        t2.add_column("Cost",   justify="right", style="bold")
        t2.add_column("ms",     justify="right", style="dim")
        for c in top_calls:
            t2.add_row(
                str(c.get("turn", 0)),
                c["model"][:32],
                c["task_type"],
                f"{c['input_tokens']:,}",
                f"{c['output_tokens']:,}",
                f"${c['cost_usd']:.6f}",
                str(c.get("latency_ms", 0)),
            )
        con.print(t2)

    con.print()

def _rich_goal_cost(goal_id, gc):
    con = _console
    con.print()
    con.rule(f"[bold]Goal Cost Report — {goal_id}[/bold]")
    con.print(
        f"  Sessions:     [bold]{gc.session_count}[/bold]\n"
        f"  Total cost:   [bold green]${gc.total_cost_usd:.4f}[/bold green]\n"
        f"  Avg/session:  ${gc.avg_cost_per_session:.4f}\n"
        f"  Total tokens: {gc.total_tokens:,}"
    )
    con.print()

    if gc.by_task:
        t = Table(title="Cost by Task", box=rich_box.SIMPLE_HEAVY)
        t.add_column("Task",       style="cyan")
        t.add_column("Cost (USD)", justify="right")
        for task, cost in sorted(gc.by_task.items(), key=lambda x: x[1], reverse=True):
            t.add_row(task, f"${cost:.4f}")
        con.print(t)

def _plain_session_cost(session_id, s, bd, top_calls):
    click.echo(f"Session: {session_id}")
    click.echo(f"  Total cost:  ${s.total_cost_usd:.4f}")
    click.echo(f"  Tokens:      {s.total_tokens:,}")
    click.echo(f"  Calls:       {s.call_count}")
    if bd:
        click.echo("  By task:")
        for task, v in sorted(bd.items(), key=lambda x: x[1]["cost_usd"], reverse=True):
            click.echo(f"    {task:<14} ${v['cost_usd']:.6f}  ({v['pct']:.1f}%)")

@cli.command("pricing")
@click.option("--provider", "-p",
              type=click.Choice(["anthropic","google","openai","cohere",
                                 "groq","together","mistral","local","all"],
                                case_sensitive=False),
              default="all",
              help="Filter by provider (default: all).")
@click.option("--format", "-f", "fmt",
              type=click.Choice(["table","json"]), default="table",
              help="Output format (default: table).")
@click.option("--include-local", is_flag=True, default=False,
              help="Include local/free models (hidden by default in table).")
def pricing_cmd(provider: str, fmt: str, include_local: bool):
    """
    Display token pricing for all supported models.

    Examples:
      truenorth pricing
      truenorth pricing --provider anthropic
      truenorth pricing --provider openai --format json
    """
    from truenorth.llm.pricing import list_models

    prov_filter = None if provider == "all" else provider
    models = list_models(provider=prov_filter)

    if not include_local:
        models = [m for m in models if not m["free"]]

    if fmt == "json":
        click.echo(json.dumps(models, indent=2))
        return

    if _RICH:
        _rich_pricing_table(models, provider, include_local)
    else:
        _plain_pricing_table(models)

def _rich_pricing_table(models, provider, include_local):
    con = _console
    con.print()
    title = f"Pricing Table — {provider.title()}" if provider != "all" else "Pricing Table — All Providers"
    t = Table(title=title, box=rich_box.SIMPLE_HEAVY, show_header=True)
    t.add_column("Provider",        style="dim",    no_wrap=True)
    t.add_column("Model",           style="cyan",   no_wrap=True)
    t.add_column("Input / 1M",      justify="right")
    t.add_column("Output / 1M",     justify="right")
    t.add_column("Typical / 1K",    justify="right", style="bold")

    current_prov = None
    for m in models:
        prov = m["provider"]
        if prov != current_prov:
            if current_prov is not None:
                t.add_row("", "", "", "", "", style="dim")
            current_prov = prov
        t.add_row(
            prov if prov != current_prov else "",
            m["model"][:45],
            f"${m['input_per_1m']:.4f}" if m["input_per_1m"] else "[green]free[/green]",
            f"${m['output_per_1m']:.4f}" if m["output_per_1m"] else "[green]free[/green]",
            f"${m['cost_1k_tokens']:.6f}" if m["cost_1k_tokens"] else "[green]$0.00[/green]",
        )
    con.print(t)
    con.print(
        f"  [dim]{len(models)} models shown. "
        f"{'Pass --include-local to see free/local models.' if not include_local else ''}"
        f"  Prices as of mid-2025.[/dim]"
    )
    con.print()

def _plain_pricing_table(models):
    click.echo(f"{'Provider':<12} {'Model':<45} {'Input/1M':>10} {'Output/1M':>11} {'Typical/1K':>12}")
    click.echo("-" * 96)
    for m in models:
        inp  = f"${m['input_per_1m']:.4f}"  if m["input_per_1m"]  else "free"
        outp = f"${m['output_per_1m']:.4f}" if m["output_per_1m"] else "free"
        typ  = f"${m['cost_1k_tokens']:.6f}"if m["cost_1k_tokens"] else "$0.000000"
        click.echo(
            f"{m['provider']:<12} {m['model']:<45} {inp:>10} {outp:>11} {typ:>12}"
        )

@cli.command("estimate")
@click.option("--model",  "-m", required=True, help="Model name.")
@click.option("--tokens", "-t", default=500, show_default=True,
              help="Input tokens.")
@click.option("--output", "-o", default=500, show_default=True,
              help="Output tokens.")
@click.option("--sessions", "-n", default=1, show_default=True,
              help="Number of sessions to estimate for.")
def estimate_cmd(model: str, tokens: int, output: int, sessions: int):
    """
    Estimate the cost of a single call or batch of sessions.

    Examples:
      truenorth estimate --model claude-haiku-4-5-20251001 --tokens 1000 --output 500
      truenorth estimate --model gpt-4o --tokens 2000 --output 1000 --sessions 1000
    """
    from truenorth.llm.pricing import cost_usd, get_model_price, get_provider

    pin, pout = get_model_price(model)
    one_call  = cost_usd(model, tokens, output)
    total     = one_call * sessions

    if _RICH:
        con = _console
        con.print()
        con.rule(f"[bold]Cost Estimate — {model}[/bold]")
        con.print(f"  Provider:      {get_provider(model)}")
        con.print(f"  Input rate:    ${pin:.4f} / 1M tokens")
        con.print(f"  Output rate:   ${pout:.4f} / 1M tokens")
        con.print()
        con.print(f"  Input tokens:  {tokens:,}")
        con.print(f"  Output tokens: {output:,}")
        con.print(f"  Cost per call: [bold green]${one_call:.6f}[/bold green]")
        if sessions > 1:
            con.print(f"  Sessions:      {sessions:,}")
            con.print(f"  Total:         [bold green]${total:.4f}[/bold green]")
        con.print()
    else:
        click.echo(f"Model:    {model}")
        click.echo(f"Per call: ${one_call:.6f}")
        if sessions > 1:
            click.echo(f"Total ({sessions} sessions): ${total:.4f}")

@cli.command("version")
def version_cmd():
    """Show TrueNorth version."""
    click.echo(f"TrueNorth v{VERSION}")

if __name__ == "__main__":
    cli()
