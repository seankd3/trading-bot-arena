"""CLI interface for the trading bot."""

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Initialize Typer app
app = typer.Typer(
    name="trading-bot",
    help="LLM-powered deep research trading bot for Alpaca",
    no_args_is_help=True,
)

console = Console()

# Sub-commands
strategies_app = typer.Typer(help="Manage trading strategies")
app.add_typer(strategies_app, name="strategies")


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def run(
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Strategy name to use"),
    ] = "value_investor",
    symbols: Annotated[
        str | None,
        typer.Option("--symbols", help="Comma-separated list of symbols to analyze"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Don't execute actual trades"),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose output"),
    ] = False,
) -> None:
    """Run the trading bot with a strategy."""
    setup_logging("DEBUG" if verbose else "INFO")

    from trading_bot.config import get_settings
    from trading_bot.execution import Trader
    from trading_bot.strategies import StrategyLoader

    settings = get_settings()
    loader = StrategyLoader()

    # Load strategy
    try:
        strat = loader.create_strategy(strategy)
        rprint(f"[green]Loaded strategy:[/green] {strat.name}")
        rprint(f"[dim]{strat.description}[/dim]\n")
    except FileNotFoundError:
        rprint(f"[red]Strategy not found:[/red] {strategy}")
        rprint("Use 'trading-bot strategies list' to see available strategies")
        raise typer.Exit(1)

    # Determine symbols to analyze
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    elif strat.config.watchlist:
        symbol_list = strat.config.watchlist
    else:
        rprint("[yellow]No symbols specified. Use --symbols or add a watchlist to strategy.[/yellow]")
        raise typer.Exit(1)

    rprint(f"[blue]Analyzing {len(symbol_list)} symbols:[/blue] {', '.join(symbol_list)}")
    rprint(f"[blue]Mode:[/blue] {'DRY RUN' if dry_run else 'LIVE TRADING'}")
    if not settings.alpaca_paper:
        rprint("[red bold]WARNING: Live trading mode![/red bold]")
    rprint()

    # Initialize trader
    trader = Trader(dry_run=dry_run)

    # Analyze each symbol
    results_table = Table(title="Analysis Results")
    results_table.add_column("Symbol", style="cyan")
    results_table.add_column("Action", style="bold")
    results_table.add_column("Confidence")
    results_table.add_column("Reasoning")
    results_table.add_column("Executed")

    for symbol in symbol_list:
        try:
            with console.status(f"[bold blue]Analyzing {symbol}..."):
                decision = strat.research_and_decide(symbol)

            # Style based on action
            action_style = {
                "buy": "[green]BUY[/green]",
                "sell": "[red]SELL[/red]",
                "hold": "[yellow]HOLD[/yellow]",
            }.get(decision.action.value, decision.action.value)

            # Execute if actionable
            executed = "-"
            if decision.action.value != "hold" and decision.confidence >= strat.config.risk.min_confidence:
                result = trader.execute(decision)
                executed = "[green]Yes[/green]" if result.success else f"[red]No: {result.error}[/red]"

            # Truncate reasoning for table
            reasoning = decision.reasoning[:50] + "..." if len(decision.reasoning) > 50 else decision.reasoning

            results_table.add_row(
                symbol,
                action_style,
                f"{decision.confidence:.2f}",
                reasoning,
                executed,
            )

            # Show full analysis if verbose
            if verbose and decision.analysis:
                rprint(Panel(decision.analysis, title=f"Analysis: {symbol}", border_style="blue"))

        except Exception as e:
            results_table.add_row(
                symbol,
                "[red]ERROR[/red]",
                "-",
                str(e)[:50],
                "-",
            )

    console.print(results_table)


@app.command()
def research(
    symbol: Annotated[str, typer.Argument(help="Stock symbol to research")],
    strategy: Annotated[
        str,
        typer.Option("--strategy", "-s", help="Strategy to use for analysis"),
    ] = "value_investor",
) -> None:
    """Research a specific stock symbol."""
    setup_logging("INFO")

    from trading_bot.strategies import StrategyLoader

    loader = StrategyLoader()

    try:
        strat = loader.create_strategy(strategy)
    except FileNotFoundError:
        rprint(f"[red]Strategy not found:[/red] {strategy}")
        raise typer.Exit(1)

    symbol = symbol.upper()
    rprint(f"[blue]Researching {symbol} using {strat.name}...[/blue]\n")

    with console.status(f"[bold blue]Gathering data and analyzing {symbol}..."):
        decision = strat.research_and_decide(symbol)

    # Display analysis
    rprint(Panel(decision.analysis, title=f"Analysis: {symbol}", border_style="blue"))

    # Display decision
    action_color = {"buy": "green", "sell": "red", "hold": "yellow"}.get(
        decision.action.value, "white"
    )
    rprint(f"\n[bold {action_color}]Decision: {decision.action.value.upper()}[/bold {action_color}]")
    rprint(f"Confidence: {decision.confidence:.2f}")
    rprint(f"Reasoning: {decision.reasoning}")


@app.command()
def account() -> None:
    """Show Alpaca account information."""
    setup_logging("WARNING")

    from trading_bot.alpaca import AlpacaClient

    try:
        client = AlpacaClient()
        acc = client.get_account()

        table = Table(title=f"Account: {acc.account_number}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Status", acc.status)
        table.add_row("Portfolio Value", f"${acc.portfolio_value:,.2f}")
        table.add_row("Cash", f"${acc.cash:,.2f}")
        table.add_row("Buying Power", f"${acc.buying_power:,.2f}")
        table.add_row("Equity", f"${acc.equity:,.2f}")
        table.add_row("Day Trades", str(acc.daytrade_count))
        table.add_row("PDT", "Yes" if acc.pattern_day_trader else "No")
        table.add_row("Mode", "Paper" if client.is_paper else "LIVE")

        console.print(table)

    except Exception as e:
        rprint(f"[red]Failed to get account info:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def positions() -> None:
    """Show current positions."""
    setup_logging("WARNING")

    from trading_bot.alpaca import AlpacaClient

    try:
        client = AlpacaClient()
        positions = client.get_positions()

        if not positions:
            rprint("[yellow]No open positions[/yellow]")
            return

        table = Table(title="Current Positions")
        table.add_column("Symbol", style="cyan")
        table.add_column("Qty", justify="right")
        table.add_column("Avg Entry", justify="right")
        table.add_column("Current", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("P&L %", justify="right")

        for pos in positions:
            pnl_style = "green" if pos.unrealized_pl >= 0 else "red"
            table.add_row(
                pos.symbol,
                str(pos.qty),
                f"${pos.avg_entry_price:.2f}",
                f"${pos.current_price:.2f}",
                f"[{pnl_style}]${pos.unrealized_pl:+,.2f}[/{pnl_style}]",
                f"[{pnl_style}]{pos.pnl_percent:+.2f}%[/{pnl_style}]",
            )

        console.print(table)

    except Exception as e:
        rprint(f"[red]Failed to get positions:[/red] {e}")
        raise typer.Exit(1)


@strategies_app.command("list")
def list_strategies() -> None:
    """List available strategies."""
    setup_logging("WARNING")

    from trading_bot.strategies import StrategyLoader

    loader = StrategyLoader()
    strategies = loader.list_strategies()

    if not strategies:
        rprint("[yellow]No strategies found in strategies/ directory[/yellow]")
        return

    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("File", style="dim")

    for strat in strategies:
        table.add_row(strat["name"], strat["description"], strat["file"])

    console.print(table)


@strategies_app.command("show")
def show_strategy(
    name: Annotated[str, typer.Argument(help="Strategy name to show")],
) -> None:
    """Show details of a specific strategy."""
    setup_logging("WARNING")

    from trading_bot.strategies import StrategyLoader

    loader = StrategyLoader()

    try:
        config = loader.load_config(name)
    except FileNotFoundError:
        rprint(f"[red]Strategy not found:[/red] {name}")
        raise typer.Exit(1)

    rprint(f"[bold cyan]{config.name}[/bold cyan] v{config.version}")
    rprint(f"[dim]{config.description}[/dim]\n")

    rprint("[bold]Risk Settings:[/bold]")
    rprint(f"  Max Position: {config.risk.max_position_pct * 100:.0f}%")
    rprint(f"  Stop Loss: {config.risk.stop_loss_pct * 100:.0f}%")
    rprint(f"  Min Confidence: {config.risk.min_confidence:.0%}")
    rprint(f"  Max Daily Trades: {config.risk.max_daily_trades}")

    if config.watchlist:
        rprint(f"\n[bold]Watchlist:[/bold] {', '.join(config.watchlist)}")

    rprint("\n[bold]System Prompt:[/bold]")
    rprint(Panel(config.system_prompt, border_style="dim"))


@strategies_app.command("create")
def create_strategy(
    name: Annotated[str, typer.Argument(help="Name for the new strategy")],
) -> None:
    """Create a new strategy from template."""
    from trading_bot.config import get_settings

    settings = get_settings()
    strategies_dir = settings.strategies_dir

    if not strategies_dir.exists():
        strategies_dir.mkdir(parents=True)

    path = strategies_dir / f"{name}.yaml"
    if path.exists():
        rprint(f"[red]Strategy already exists:[/red] {path}")
        raise typer.Exit(1)

    template = f'''name: {name}
description: Describe your strategy here
version: "1.0.0"

research:
  system_prompt: |
    You are a trading analyst. Describe your persona and approach here.
    - Key principle 1
    - Key principle 2
    - Key principle 3

  analysis_prompt: |
    Analyze ${{symbol}} for trading opportunity.

    ${{research_context}}

    Please analyze:
    1. Current market conditions
    2. Technical setup
    3. Fundamental factors
    4. Risk/reward assessment

    Provide your analysis and recommendation.

decision:
  prompt: |
    Based on your analysis of ${{symbol}}:

    ${{analysis}}

    Portfolio Context:
    ${{portfolio_context}}
    Available Cash: $${{available_cash}}

    Respond with JSON only:
    {{
      "action": "buy" | "sell" | "hold",
      "confidence": 0.0 to 1.0,
      "quantity": number or null,
      "reasoning": "brief explanation"
    }}

risk:
  max_position_pct: 0.05
  stop_loss_pct: 0.10
  min_confidence: 0.70
  max_daily_trades: 5

watchlist:
  - AAPL
  - GOOGL
  - MSFT

parameters:
  custom_param: value
'''

    path.write_text(template)
    rprint(f"[green]Created strategy:[/green] {path}")
    rprint("Edit the file to customize your strategy prompts and settings.")


@app.command()
def quick(
    symbol: Annotated[str, typer.Argument(help="Stock symbol to analyze")],
    prompt: Annotated[
        str,
        typer.Option("--prompt", "-p", help="Custom analysis prompt"),
    ] = "Analyze this stock and tell me if I should buy, sell, or hold.",
) -> None:
    """Quick analysis with a custom prompt."""
    setup_logging("INFO")

    from trading_bot.research import DataGatherer, LLMEngine

    symbol = symbol.upper()
    rprint(f"[blue]Quick analysis of {symbol}...[/blue]\n")

    with console.status(f"[bold blue]Gathering data for {symbol}..."):
        gatherer = DataGatherer()
        data = gatherer.gather(symbol)

    context = data.to_prompt_context()
    full_prompt = f"{context}\n\n{prompt}"

    with console.status("[bold blue]Running LLM analysis..."):
        llm = LLMEngine()
        response = llm.analyze(full_prompt)

    rprint(Panel(response.content, title=f"Analysis: {symbol}", border_style="green"))
    rprint(f"\n[dim]Tokens: {response.input_tokens} in, {response.output_tokens} out[/dim]")


def main() -> None:
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
