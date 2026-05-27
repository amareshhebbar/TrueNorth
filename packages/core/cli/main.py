"""TrueNorth CLI."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option("0.1.0", prog_name="truenorth")
def cli():
    """TrueNorth — Conversation-first AI agent framework."""
    pass


@cli.command("dry-run")
@click.argument("goal_yaml")
@click.option("--scenario", "-s", required=True, help="Path to scenario JSON file")
@click.option("--assert-complete", is_flag=True, help="Fail if not all required fields collected")
def dry_run(goal_yaml, scenario, assert_complete):
    """Run a full agent flow with zero API calls."""
    import asyncio
    from truenorth.testing.dry_runner import DryRunner

    async def _run():
        runner = DryRunner(goal_yaml, scenario)
        result = await runner.run()
        result.print_report()

        if assert_complete and result.missing_required:
            raise SystemExit(1)

    asyncio.run(_run())


@cli.command("validate")
@click.argument("goal_yaml")
def validate(goal_yaml):
    """Validate a goal YAML config."""
    from truenorth.core.yaml_loader import YamlLoader
    try:
        config = YamlLoader().load(goal_yaml)
        console.print(f"[green]✓[/green] {goal_yaml} is valid")
        console.print(f"  goal_id: {config.goal_id}")
        console.print(f"  required fields: {len(config.required_fields)}")
        console.print(f"  optional fields: {len(config.optional_fields)}")
        console.print(f"  compliance: {config.compliance.mode}")
    except Exception as e:
        console.print(f"[red]✗[/red] Invalid: {e}")
        raise SystemExit(1)


@cli.command("replay")
@click.option("--session-id", required=True)
@click.option("--step", is_flag=True, help="Step through turn by turn")
def replay(session_id, step):
    """Replay a past conversation from the database."""
    import asyncio, os
    from truenorth.storage.postgres import PostgresStore

    async def _run():
        db_url = os.getenv("DATABASE_URL", "")
        if not db_url:
            console.print("[red]DATABASE_URL not set[/red]")
            return

        store = PostgresStore(db_url)
        async with store.session_factory() as db:
            from truenorth.storage.models import Session
            row = await db.get(Session, session_id)

        if not row:
            console.print(f"[red]Session {session_id} not found[/red]")
            return

        turns = row.conversation or []
        console.print(f"\n[bold]Replaying session {session_id}[/bold] ({len(turns)} turns)\n")

        for i, turn in enumerate(turns):
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                console.print(f"[cyan]USER:[/cyan] {content}")
            else:
                console.print(f"[green]AGENT:[/green] {content}")

            if step and i < len(turns) - 1:
                input("  [Enter for next turn...]")

        console.print(f"\n[bold]Final profile:[/bold]")
        for k, v in (row.profile or {}).items():
            console.print(f"  {k}: {v.get('value')}")

    asyncio.run(_run())


@cli.command("chat")
@click.argument("goal_yaml")
@click.option("--session-id", default=None)
def chat(goal_yaml, session_id):
    """Start an interactive chat session in the terminal (uses real LLMs)."""
    import asyncio, uuid
    from truenorth.core.engine import TrueNorthEngine
    from truenorth.llm.router import LLMRouter

    async def _run():
        engine = TrueNorthEngine.from_yaml(goal_yaml)
        state = engine.create_initial_state(session_id or str(uuid.uuid4()))

        welcome = await engine.generate_welcome_message(state)
        console.print(f"\n[green]AGENT:[/green] {welcome}\n")

        while not state.completed and not state.escalated:
            user_input = input("YOU: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            state, response = await engine.process_turn(state, user_input)
            console.print(f"\n[green]AGENT:[/green] {response}")
            console.print(f"[dim]  (emotion: {state.emotion_state}, cost: ${state.cost_usd:.4f})[/dim]\n")

        if state.completed and state.output:
            console.print("\n[bold]Generated Output:[/bold]")
            console.print(state.output.get("content", ""))

    asyncio.run(_run())


if __name__ == "__main__":
    cli()
