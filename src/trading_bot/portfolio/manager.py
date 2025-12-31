"""Autonomous portfolio manager using GPT-5 deep research."""

import json
import logging
import time
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from trading_bot.alpaca import AlpacaClient, OrderSide
from trading_bot.config import get_settings
from trading_bot.research import DataGatherer, LLMEngine

logger = logging.getLogger(__name__)


class PortfolioState(BaseModel):
    """Current state of the portfolio."""

    total_value: Decimal
    cash: Decimal
    buying_power: Decimal
    positions: dict[str, dict[str, Any]]  # symbol -> position info
    timestamp: datetime


class TradeRecommendation(BaseModel):
    """A trade recommendation from the AI."""

    symbol: str
    action: str  # buy, sell, hold
    quantity: int | None = None
    target_allocation_pct: float | None = None
    reasoning: str
    confidence: float
    urgency: str = "normal"  # low, normal, high


class PortfolioAnalysis(BaseModel):
    """Complete portfolio analysis."""

    summary: str
    health_score: float  # 0-100
    recommendations: list[TradeRecommendation]
    watchlist_additions: list[str]
    watchlist_removals: list[str]
    risk_assessment: str
    market_outlook: str


class PortfolioManager:
    """Autonomous portfolio manager powered by GPT-5 deep research."""

    DEFAULT_WATCHLIST = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK.B", "V", "JNJ", "UNH", "JPM", "XOM", "PG", "MA",
    ]

    def __init__(
        self,
        watchlist: list[str] | None = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the portfolio manager.

        Args:
            watchlist: Symbols to monitor. Defaults to top 15 stocks.
            dry_run: If True, don't execute actual trades.
        """
        self.settings = get_settings()
        self.alpaca = AlpacaClient()
        self.llm = LLMEngine()
        self.data_gatherer = DataGatherer(self.alpaca)
        self.watchlist = watchlist or self.DEFAULT_WATCHLIST
        self.dry_run = dry_run
        self._last_analysis: PortfolioAnalysis | None = None

    def get_portfolio_state(self) -> PortfolioState:
        """Get current portfolio state."""
        account = self.alpaca.get_account()
        positions = self.alpaca.get_positions()

        position_dict = {}
        for pos in positions:
            position_dict[pos.symbol] = {
                "quantity": float(pos.qty),
                "avg_entry_price": float(pos.avg_entry_price),
                "current_price": float(pos.current_price),
                "market_value": float(pos.market_value),
                "unrealized_pl": float(pos.unrealized_pl),
                "unrealized_pl_pct": float(pos.unrealized_plpc) * 100,
                "cost_basis": float(pos.cost_basis),
            }

        return PortfolioState(
            total_value=account.portfolio_value,
            cash=account.cash,
            buying_power=account.buying_power,
            positions=position_dict,
            timestamp=datetime.now(),
        )

    def analyze_portfolio(self) -> PortfolioAnalysis:
        """Perform deep research analysis of the portfolio."""
        logger.info("Starting deep portfolio analysis with GPT-5...")

        # Get current state
        try:
            state = self.get_portfolio_state()
        except Exception as e:
            logger.error(f"Failed to get portfolio state: {e}")
            raise

        # Gather research data for all positions and watchlist
        symbols_to_research = list(state.positions.keys()) + [
            s for s in self.watchlist if s not in state.positions
        ]

        research_summaries = []
        for symbol in symbols_to_research[:10]:  # Limit to avoid rate limits
            try:
                data = self.data_gatherer.gather(symbol, include_news=True)
                research_summaries.append(f"**{symbol}**:\n{data.to_prompt_context()[:2000]}")
            except Exception as e:
                logger.warning(f"Failed to gather data for {symbol}: {e}")
                research_summaries.append(f"**{symbol}**: Data unavailable")

        # Build the analysis prompt
        prompt = self._build_analysis_prompt(state, research_summaries)

        # Run GPT-5 deep analysis
        logger.info("Running GPT-5 deep research analysis...")
        response = self.llm.analyze(
            prompt=prompt,
            system_prompt=self._get_system_prompt(),
            max_tokens=8000,
            temperature=0.7,
        )

        logger.info(f"Analysis complete (tokens: {response.input_tokens}+{response.output_tokens})")

        # Parse the response
        analysis = self._parse_analysis(response.content)
        self._last_analysis = analysis

        return analysis

    def execute_recommendations(
        self,
        analysis: PortfolioAnalysis | None = None,
        min_confidence: float = 0.7,
    ) -> list[dict[str, Any]]:
        """Execute trade recommendations from analysis.

        Args:
            analysis: Portfolio analysis (uses last if not provided)
            min_confidence: Minimum confidence to execute trades

        Returns:
            List of execution results
        """
        if analysis is None:
            analysis = self._last_analysis

        if analysis is None:
            raise ValueError("No analysis available. Run analyze_portfolio first.")

        results = []
        state = self.get_portfolio_state()

        for rec in analysis.recommendations:
            if rec.action == "hold":
                continue

            if rec.confidence < min_confidence:
                logger.info(
                    f"Skipping {rec.symbol}: confidence {rec.confidence:.2f} < {min_confidence}"
                )
                continue

            try:
                result = self._execute_recommendation(rec, state)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to execute {rec.action} {rec.symbol}: {e}")
                results.append({
                    "symbol": rec.symbol,
                    "action": rec.action,
                    "success": False,
                    "error": str(e),
                })

        return results

    def _execute_recommendation(
        self,
        rec: TradeRecommendation,
        state: PortfolioState,
    ) -> dict[str, Any]:
        """Execute a single trade recommendation."""
        symbol = rec.symbol

        if self.dry_run:
            logger.info(f"[DRY RUN] Would {rec.action} {symbol}: {rec.reasoning}")
            return {
                "symbol": symbol,
                "action": rec.action,
                "success": True,
                "dry_run": True,
                "reasoning": rec.reasoning,
            }

        if rec.action == "buy":
            # Calculate quantity
            quantity = rec.quantity
            if quantity is None and rec.target_allocation_pct:
                target_value = float(state.total_value) * (rec.target_allocation_pct / 100)
                quote = self.alpaca.get_quote(symbol)
                price = float(quote.ask_price)
                quantity = int(target_value / price)

            if not quantity or quantity <= 0:
                return {"symbol": symbol, "action": "buy", "success": False, "error": "Invalid quantity"}

            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
            )
            logger.info(f"BUY order placed: {quantity} {symbol}")
            return {
                "symbol": symbol,
                "action": "buy",
                "quantity": quantity,
                "success": True,
                "order_id": order.id,
            }

        elif rec.action == "sell":
            position = state.positions.get(symbol)
            if not position:
                return {"symbol": symbol, "action": "sell", "success": False, "error": "No position"}

            quantity = rec.quantity or int(position["quantity"])
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
            )
            logger.info(f"SELL order placed: {quantity} {symbol}")
            return {
                "symbol": symbol,
                "action": "sell",
                "quantity": quantity,
                "success": True,
                "order_id": order.id,
            }

        return {"symbol": symbol, "action": rec.action, "success": False, "error": "Unknown action"}

    def run_continuous(
        self,
        interval_minutes: int = 60,
        auto_execute: bool = False,
        min_confidence: float = 0.8,
    ) -> None:
        """Run continuous portfolio management loop.

        Args:
            interval_minutes: Minutes between analyses
            auto_execute: Whether to auto-execute high-confidence trades
            min_confidence: Minimum confidence for auto-execution
        """
        logger.info(f"Starting continuous portfolio management (interval: {interval_minutes}m)")
        logger.info(f"Auto-execute: {auto_execute}, Min confidence: {min_confidence}")

        while True:
            try:
                # Run analysis
                analysis = self.analyze_portfolio()

                # Log summary
                logger.info(f"Portfolio Health: {analysis.health_score}/100")
                logger.info(f"Market Outlook: {analysis.market_outlook}")
                logger.info(f"Recommendations: {len(analysis.recommendations)}")

                for rec in analysis.recommendations:
                    logger.info(
                        f"  {rec.action.upper()} {rec.symbol} "
                        f"(confidence: {rec.confidence:.2f}, urgency: {rec.urgency})"
                    )

                # Execute if auto-execute enabled
                if auto_execute:
                    results = self.execute_recommendations(
                        analysis, min_confidence=min_confidence
                    )
                    for r in results:
                        if r.get("success"):
                            logger.info(f"Executed: {r['action']} {r['symbol']}")
                        else:
                            logger.warning(f"Failed: {r}")

            except Exception as e:
                logger.error(f"Error in management loop: {e}")

            # Wait for next iteration
            logger.info(f"Sleeping {interval_minutes} minutes...")
            time.sleep(interval_minutes * 60)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for GPT-5."""
        return """You are an expert portfolio manager and financial analyst powered by GPT-5.
Your role is to manage a stock portfolio with deep research and careful analysis.

Your approach:
- Conduct thorough fundamental and technical analysis
- Consider macro-economic factors and market conditions
- Balance risk and reward appropriately
- Think long-term but capitalize on short-term opportunities when confident
- Diversify across sectors to manage risk
- Cut losses and let winners run

You have access to real market data and can execute real trades.
Be decisive but not reckless. Quality over quantity.

Always provide your analysis in the specified JSON format."""

    def _build_analysis_prompt(
        self,
        state: PortfolioState,
        research_summaries: list[str],
    ) -> str:
        """Build the analysis prompt."""
        positions_str = ""
        if state.positions:
            for symbol, pos in state.positions.items():
                pnl_sign = "+" if pos["unrealized_pl"] >= 0 else ""
                positions_str += (
                    f"  {symbol}: {pos['quantity']} shares @ ${pos['avg_entry_price']:.2f} "
                    f"(Current: ${pos['current_price']:.2f}, "
                    f"P&L: {pnl_sign}${pos['unrealized_pl']:.2f} / "
                    f"{pnl_sign}{pos['unrealized_pl_pct']:.2f}%)\n"
                )
        else:
            positions_str = "  No positions\n"

        research_str = "\n\n".join(research_summaries)

        return f"""Analyze my portfolio and provide actionable recommendations.

## CURRENT PORTFOLIO STATE

Total Value: ${state.total_value:,.2f}
Cash Available: ${state.cash:,.2f}
Buying Power: ${state.buying_power:,.2f}

### Positions:
{positions_str}

### Watchlist: {', '.join(self.watchlist)}

## MARKET RESEARCH DATA

{research_str}

## YOUR TASK

Perform deep research analysis and provide:

1. **Portfolio Health Assessment** - Overall score (0-100) and summary
2. **Trade Recommendations** - What to buy, sell, or hold
3. **Risk Assessment** - Current portfolio risks
4. **Market Outlook** - Your view on near-term market direction
5. **Watchlist Updates** - Stocks to add or remove from monitoring

Respond with JSON in this exact format:
```json
{{
  "summary": "Brief portfolio summary",
  "health_score": 75,
  "recommendations": [
    {{
      "symbol": "AAPL",
      "action": "buy",
      "quantity": 10,
      "target_allocation_pct": 5.0,
      "reasoning": "Strong fundamentals...",
      "confidence": 0.85,
      "urgency": "normal"
    }}
  ],
  "watchlist_additions": ["NEW1"],
  "watchlist_removals": ["OLD1"],
  "risk_assessment": "Current risk level is...",
  "market_outlook": "Near-term outlook..."
}}
```"""

    def _parse_analysis(self, content: str) -> PortfolioAnalysis:
        """Parse LLM response into PortfolioAnalysis."""
        # Try to extract JSON
        try:
            # Find JSON in response
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                json_str = content[start:end].strip()
            else:
                json_str = content

            data = json.loads(json_str)

            recommendations = []
            for rec in data.get("recommendations", []):
                recommendations.append(TradeRecommendation(
                    symbol=rec["symbol"],
                    action=rec["action"],
                    quantity=rec.get("quantity"),
                    target_allocation_pct=rec.get("target_allocation_pct"),
                    reasoning=rec.get("reasoning", ""),
                    confidence=rec.get("confidence", 0.5),
                    urgency=rec.get("urgency", "normal"),
                ))

            return PortfolioAnalysis(
                summary=data.get("summary", "Analysis complete"),
                health_score=data.get("health_score", 50),
                recommendations=recommendations,
                watchlist_additions=data.get("watchlist_additions", []),
                watchlist_removals=data.get("watchlist_removals", []),
                risk_assessment=data.get("risk_assessment", "Unknown"),
                market_outlook=data.get("market_outlook", "Unknown"),
            )

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            # Return minimal analysis
            return PortfolioAnalysis(
                summary=content[:500],
                health_score=50,
                recommendations=[],
                watchlist_additions=[],
                watchlist_removals=[],
                risk_assessment="Could not parse",
                market_outlook="Could not parse",
            )
