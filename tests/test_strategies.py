"""Tests for strategy loading and configuration."""

from pathlib import Path

import pytest

from trading_bot.strategies.base import (
    DecisionConfig,
    ResearchConfig,
    RiskConfig,
    StrategyConfig,
    StrategyDecision,
    TradeAction,
)
from trading_bot.strategies.loader import StrategyLoader, quick_strategy


class TestStrategyConfig:
    """Tests for StrategyConfig."""

    def test_create_config(self) -> None:
        """Test creating a strategy configuration."""
        config = StrategyConfig(
            name="test_strategy",
            description="A test strategy",
            research=ResearchConfig(
                system_prompt="You are a test analyst.",
                analysis_prompt="Analyze {symbol}.",
            ),
            decision=DecisionConfig(
                prompt="Decide on {symbol}.",
            ),
        )

        assert config.name == "test_strategy"
        assert config.description == "A test strategy"
        assert config.system_prompt == "You are a test analyst."
        assert config.risk.max_position_pct == 0.10  # Default

    def test_risk_defaults(self) -> None:
        """Test default risk settings."""
        risk = RiskConfig()

        assert risk.max_position_pct == 0.10
        assert risk.stop_loss_pct == 0.10
        assert risk.min_confidence == 0.7
        assert risk.max_daily_trades == 10


class TestStrategyDecision:
    """Tests for StrategyDecision."""

    def test_should_act_buy(self) -> None:
        """Test buy decision should act."""
        decision = StrategyDecision(
            symbol="AAPL",
            action=TradeAction.BUY,
            confidence=0.8,
            reasoning="Strong buy signal",
        )

        assert decision.should_act is True

    def test_should_act_hold(self) -> None:
        """Test hold decision should not act."""
        decision = StrategyDecision(
            symbol="AAPL",
            action=TradeAction.HOLD,
            confidence=0.9,
            reasoning="No action needed",
        )

        assert decision.should_act is False

    def test_should_not_act_zero_confidence(self) -> None:
        """Test zero confidence should not act."""
        decision = StrategyDecision(
            symbol="AAPL",
            action=TradeAction.BUY,
            confidence=0.0,
            reasoning="Low confidence",
        )

        assert decision.should_act is False


class TestQuickStrategy:
    """Tests for quick_strategy helper."""

    def test_create_quick_strategy(self) -> None:
        """Test creating a strategy without YAML."""
        strategy = quick_strategy(
            name="quick_test",
            system_prompt="You are a test.",
            analysis_prompt="Analyze {symbol}.",
            decision_prompt="Decide on {symbol}.",
            max_position_pct=0.05,
        )

        assert strategy.name == "quick_test"
        assert strategy.config.risk.max_position_pct == 0.05


class TestStrategyLoader:
    """Tests for StrategyLoader."""

    def test_load_from_dict(self) -> None:
        """Test loading strategy from dictionary."""
        loader = StrategyLoader()

        data = {
            "name": "dict_strategy",
            "description": "Loaded from dict",
            "research": {
                "system_prompt": "You are an analyst.",
                "analysis_prompt": "Analyze {symbol}.",
            },
            "decision": {
                "prompt": "Decide on {symbol}.",
            },
            "risk": {
                "max_position_pct": 0.08,
                "stop_loss_pct": 0.12,
            },
        }

        config = loader.load_from_dict(data)

        assert config.name == "dict_strategy"
        assert config.risk.max_position_pct == 0.08
        assert config.risk.stop_loss_pct == 0.12
