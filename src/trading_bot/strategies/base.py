"""Base classes for trading strategies."""

from abc import ABC, abstractmethod
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TradeAction(str, Enum):
    """Possible trade actions."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class StrategyDecision(BaseModel):
    """A trading decision made by a strategy."""

    symbol: str
    action: TradeAction
    confidence: float = Field(ge=0.0, le=1.0)
    quantity: int | None = None
    limit_price: Decimal | None = None
    reasoning: str = ""
    analysis: str = ""
    raw_response: str = ""

    @property
    def should_act(self) -> bool:
        """Check if this decision should result in a trade."""
        return self.action != TradeAction.HOLD and self.confidence > 0


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Maximum position size as fraction of portfolio",
    )
    stop_loss_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=0.50,
        description="Stop loss percentage",
    )
    take_profit_pct: float | None = Field(
        default=None,
        ge=0.01,
        le=1.0,
        description="Take profit percentage (optional)",
    )
    min_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence to act on a signal",
    )
    max_daily_trades: int = Field(
        default=10,
        ge=1,
        description="Maximum trades per day",
    )


class ResearchConfig(BaseModel):
    """Research prompt configuration."""

    system_prompt: str = Field(
        description="System prompt defining the researcher's persona and approach"
    )
    analysis_prompt: str = Field(
        description="Template for analyzing a specific symbol"
    )


class DecisionConfig(BaseModel):
    """Decision prompt configuration."""

    prompt: str = Field(
        description="Template for making buy/sell/hold decisions"
    )


class StrategyConfig(BaseModel):
    """Complete strategy configuration loaded from YAML."""

    name: str
    description: str = ""
    version: str = "1.0.0"

    research: ResearchConfig
    decision: DecisionConfig
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # Optional: symbols to focus on
    watchlist: list[str] = Field(default_factory=list)

    # Optional: custom parameters
    parameters: dict[str, Any] = Field(default_factory=dict)

    @property
    def system_prompt(self) -> str:
        """Get the research system prompt."""
        return self.research.system_prompt

    @property
    def analysis_prompt_template(self) -> str:
        """Get the analysis prompt template."""
        return self.research.analysis_prompt

    @property
    def decision_prompt_template(self) -> str:
        """Get the decision prompt template."""
        return self.decision.prompt


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, config: StrategyConfig) -> None:
        """Initialize the strategy with configuration."""
        self.config = config

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self.config.name

    @property
    def description(self) -> str:
        """Get strategy description."""
        return self.config.description

    @abstractmethod
    def analyze(self, symbol: str) -> str:
        """Perform analysis on a symbol.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Analysis text from the LLM
        """
        pass

    @abstractmethod
    def decide(
        self,
        symbol: str,
        analysis: str,
        portfolio_context: str,
    ) -> StrategyDecision:
        """Make a trading decision based on analysis.

        Args:
            symbol: Stock symbol
            analysis: Previous analysis from analyze()
            portfolio_context: Current portfolio state

        Returns:
            Trading decision
        """
        pass

    @abstractmethod
    def research_and_decide(self, symbol: str) -> StrategyDecision:
        """Complete research and decision flow for a symbol.

        Args:
            symbol: Stock symbol to evaluate

        Returns:
            Trading decision with full analysis
        """
        pass
