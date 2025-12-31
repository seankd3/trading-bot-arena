"""Prompt-based strategy implementation."""

import json
import logging
from decimal import Decimal
from string import Template
from typing import Any

from trading_bot.alpaca import AlpacaClient
from trading_bot.research import DataGatherer, LLMEngine, ResearchData
from trading_bot.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyDecision,
    TradeAction,
)

logger = logging.getLogger(__name__)


class PromptStrategy(Strategy):
    """Strategy that uses configurable prompts for analysis and decisions."""

    def __init__(
        self,
        config: StrategyConfig,
        llm_engine: LLMEngine | None = None,
        data_gatherer: DataGatherer | None = None,
    ) -> None:
        """Initialize the prompt strategy.

        Args:
            config: Strategy configuration
            llm_engine: Optional LLM engine (created if not provided)
            data_gatherer: Optional data gatherer (created if not provided)
        """
        super().__init__(config)
        self.llm = llm_engine or LLMEngine()
        self.data_gatherer = data_gatherer or DataGatherer()
        self._research_cache: dict[str, ResearchData] = {}

    def analyze(self, symbol: str) -> str:
        """Perform deep analysis on a symbol using the configured prompts.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Analysis text from the LLM
        """
        logger.info(f"[{self.name}] Analyzing {symbol}")

        # Gather research data
        research_data = self.data_gatherer.gather(symbol)
        self._research_cache[symbol] = research_data

        # Build the analysis prompt from template
        prompt = self._build_analysis_prompt(symbol, research_data)

        # Run LLM analysis
        response = self.llm.analyze(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
            temperature=0.7,
        )

        logger.info(
            f"[{self.name}] Analysis complete for {symbol} "
            f"(tokens: {response.input_tokens}+{response.output_tokens})"
        )

        return response.content

    def decide(
        self,
        symbol: str,
        analysis: str,
        portfolio_context: str,
    ) -> StrategyDecision:
        """Make a trading decision based on analysis.

        Args:
            symbol: Stock symbol
            analysis: Previous analysis
            portfolio_context: Current portfolio state

        Returns:
            Trading decision
        """
        logger.info(f"[{self.name}] Making decision for {symbol}")

        # Get account info for decision context
        try:
            account = self.data_gatherer.alpaca.get_account()
            available_cash = float(account.cash)
        except Exception:
            available_cash = 0.0

        # Get current position info
        position = self.data_gatherer.alpaca.get_position(symbol)
        position_info = "No current position"
        if position:
            position_info = (
                f"Current position: {position.qty} shares at ${position.avg_entry_price:.2f} "
                f"(P&L: ${position.unrealized_pl:.2f})"
            )

        # Build decision prompt
        prompt = self._build_decision_prompt(
            symbol=symbol,
            analysis=analysis,
            portfolio_context=portfolio_context,
            position_info=position_info,
            available_cash=available_cash,
        )

        # Get structured decision from LLM
        response = self.llm.analyze(
            prompt=prompt,
            system_prompt=self.config.system_prompt,
            temperature=0.3,  # Lower temperature for more consistent decisions
        )

        # Parse the decision
        decision = self._parse_decision(symbol, analysis, response.content)

        logger.info(
            f"[{self.name}] Decision for {symbol}: {decision.action.value} "
            f"(confidence: {decision.confidence:.2f})"
        )

        return decision

    def research_and_decide(self, symbol: str) -> StrategyDecision:
        """Complete research and decision flow for a symbol.

        Args:
            symbol: Stock symbol to evaluate

        Returns:
            Trading decision with full analysis
        """
        # Perform analysis
        analysis = self.analyze(symbol)

        # Get portfolio context
        portfolio_context = self.data_gatherer.get_portfolio_context()

        # Make decision
        decision = self.decide(symbol, analysis, portfolio_context)

        # Attach full analysis to decision
        decision.analysis = analysis

        return decision

    def batch_analyze(self, symbols: list[str]) -> dict[str, StrategyDecision]:
        """Analyze multiple symbols and return decisions.

        Args:
            symbols: List of symbols to analyze

        Returns:
            Dictionary mapping symbols to decisions
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.research_and_decide(symbol)
            except Exception as e:
                logger.error(f"[{self.name}] Failed to analyze {symbol}: {e}")
                results[symbol] = StrategyDecision(
                    symbol=symbol,
                    action=TradeAction.HOLD,
                    confidence=0.0,
                    reasoning=f"Analysis failed: {e}",
                )
        return results

    def _build_analysis_prompt(self, symbol: str, research_data: ResearchData) -> str:
        """Build the analysis prompt from template and data."""
        # Get market data values for template
        template_vars: dict[str, Any] = {
            "symbol": symbol,
            "research_context": research_data.to_prompt_context(),
        }

        if research_data.market_data:
            md = research_data.market_data
            template_vars.update({
                "price": f"{md.current_price:.2f}",
                "change": f"{md.change:.2f}",
                "change_percent": f"{md.change_percent:.2f}",
                "volume": f"{md.volume:,}",
                "high": f"{md.high_price:.2f}",
                "low": f"{md.low_price:.2f}",
            })

        if research_data.company_info:
            ci = research_data.company_info
            if ci.pe_ratio:
                template_vars["pe_ratio"] = f"{ci.pe_ratio:.2f}"
            if ci.market_cap:
                template_vars["market_cap"] = f"{ci.market_cap / Decimal('1e9'):.2f}B"

        # Generate news summary
        from trading_bot.research.news_fetcher import NewsFetcher

        fetcher = NewsFetcher()
        template_vars["news_summary"] = fetcher.summarize_news(research_data.news_articles)

        # Apply template
        template = Template(self.config.analysis_prompt_template)
        try:
            prompt = template.safe_substitute(template_vars)
        except Exception:
            prompt = self.config.analysis_prompt_template.format(**template_vars)

        return prompt

    def _build_decision_prompt(
        self,
        symbol: str,
        analysis: str,
        portfolio_context: str,
        position_info: str,
        available_cash: float,
    ) -> str:
        """Build the decision prompt from template."""
        template_vars = {
            "symbol": symbol,
            "analysis": analysis,
            "portfolio_context": portfolio_context,
            "position_info": position_info,
            "available_cash": f"{available_cash:.2f}",
            "max_position_pct": f"{self.config.risk.max_position_pct * 100:.0f}",
            "min_confidence": f"{self.config.risk.min_confidence:.2f}",
        }

        template = Template(self.config.decision_prompt_template)
        try:
            prompt = template.safe_substitute(template_vars)
        except Exception:
            prompt = self.config.decision_prompt_template.format(**template_vars)

        return prompt

    def _parse_decision(
        self,
        symbol: str,
        analysis: str,
        response: str,
    ) -> StrategyDecision:
        """Parse LLM response into a StrategyDecision."""
        # Try to extract JSON from response
        decision_data = None

        try:
            # Try direct JSON parse
            decision_data = json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in response
            import re

            json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
            if json_match:
                try:
                    decision_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        if decision_data:
            action_str = decision_data.get("action", "hold").lower()
            action = TradeAction.HOLD
            if action_str == "buy":
                action = TradeAction.BUY
            elif action_str == "sell":
                action = TradeAction.SELL

            return StrategyDecision(
                symbol=symbol,
                action=action,
                confidence=float(decision_data.get("confidence", 0.5)),
                quantity=decision_data.get("quantity"),
                reasoning=decision_data.get("reasoning", ""),
                analysis=analysis,
                raw_response=response,
            )

        # Fallback: try to infer from text
        response_lower = response.lower()
        if "strong buy" in response_lower or "definitely buy" in response_lower:
            return StrategyDecision(
                symbol=symbol,
                action=TradeAction.BUY,
                confidence=0.8,
                reasoning="Inferred from response text",
                analysis=analysis,
                raw_response=response,
            )
        elif "buy" in response_lower and "don't buy" not in response_lower:
            return StrategyDecision(
                symbol=symbol,
                action=TradeAction.BUY,
                confidence=0.6,
                reasoning="Inferred from response text",
                analysis=analysis,
                raw_response=response,
            )
        elif "sell" in response_lower and "don't sell" not in response_lower:
            return StrategyDecision(
                symbol=symbol,
                action=TradeAction.SELL,
                confidence=0.6,
                reasoning="Inferred from response text",
                analysis=analysis,
                raw_response=response,
            )

        # Default to hold
        return StrategyDecision(
            symbol=symbol,
            action=TradeAction.HOLD,
            confidence=0.5,
            reasoning="Could not parse clear signal from analysis",
            analysis=analysis,
            raw_response=response,
        )
