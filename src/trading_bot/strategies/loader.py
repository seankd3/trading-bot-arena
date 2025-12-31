"""Strategy loader for YAML configuration files."""

import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from trading_bot.config import get_settings
from trading_bot.strategies.base import (
    DecisionConfig,
    ResearchConfig,
    RiskConfig,
    StrategyConfig,
)
from trading_bot.strategies.prompt_strategy import PromptStrategy

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads strategy configurations from YAML files."""

    def __init__(self, strategies_dir: Path | None = None) -> None:
        """Initialize the strategy loader.

        Args:
            strategies_dir: Directory containing strategy YAML files
        """
        if strategies_dir is None:
            strategies_dir = get_settings().strategies_dir
        self.strategies_dir = strategies_dir

    def list_strategies(self) -> list[dict[str, str]]:
        """List all available strategies.

        Returns:
            List of dicts with name and description
        """
        strategies = []

        if not self.strategies_dir.exists():
            logger.warning(f"Strategies directory not found: {self.strategies_dir}")
            return strategies

        for path in self.strategies_dir.glob("*.yaml"):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                    strategies.append({
                        "name": data.get("name", path.stem),
                        "description": data.get("description", ""),
                        "file": path.name,
                    })
            except Exception as e:
                logger.warning(f"Failed to load strategy {path}: {e}")

        return strategies

    def load_config(self, name: str) -> StrategyConfig:
        """Load a strategy configuration by name.

        Args:
            name: Strategy name (without .yaml extension)

        Returns:
            StrategyConfig instance

        Raises:
            FileNotFoundError: If strategy file not found
            ValidationError: If configuration is invalid
        """
        # Try exact filename first
        path = self.strategies_dir / f"{name}.yaml"
        if not path.exists():
            # Try without extension
            path = self.strategies_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Strategy not found: {name}")

        logger.info(f"Loading strategy from {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def load_from_file(self, path: Path | str) -> StrategyConfig:
        """Load a strategy configuration from a specific file.

        Args:
            path: Path to the YAML file

        Returns:
            StrategyConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def load_from_dict(self, data: dict[str, Any]) -> StrategyConfig:
        """Load a strategy configuration from a dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            StrategyConfig instance
        """
        return self._parse_config(data)

    def _parse_config(self, data: dict[str, Any]) -> StrategyConfig:
        """Parse configuration data into StrategyConfig.

        Args:
            data: Raw configuration dictionary

        Returns:
            StrategyConfig instance
        """
        # Parse nested configs
        research_data = data.get("research", {})
        research = ResearchConfig(
            system_prompt=research_data.get("system_prompt", ""),
            analysis_prompt=research_data.get("analysis_prompt", ""),
        )

        decision_data = data.get("decision", {})
        decision = DecisionConfig(
            prompt=decision_data.get("prompt", ""),
        )

        risk_data = data.get("risk", {})
        risk = RiskConfig(
            max_position_pct=risk_data.get("max_position_pct", 0.10),
            stop_loss_pct=risk_data.get("stop_loss_pct", 0.10),
            take_profit_pct=risk_data.get("take_profit_pct"),
            min_confidence=risk_data.get("min_confidence", 0.7),
            max_daily_trades=risk_data.get("max_daily_trades", 10),
        )

        return StrategyConfig(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            research=research,
            decision=decision,
            risk=risk,
            watchlist=data.get("watchlist", []),
            parameters=data.get("parameters", {}),
        )

    def create_strategy(self, name: str) -> PromptStrategy:
        """Load and instantiate a strategy by name.

        Args:
            name: Strategy name

        Returns:
            PromptStrategy instance ready to use
        """
        config = self.load_config(name)
        return PromptStrategy(config)

    def create_from_config(self, config: StrategyConfig) -> PromptStrategy:
        """Create a strategy from a configuration object.

        Args:
            config: Strategy configuration

        Returns:
            PromptStrategy instance
        """
        return PromptStrategy(config)


def quick_strategy(
    name: str,
    system_prompt: str,
    analysis_prompt: str,
    decision_prompt: str,
    **kwargs: Any,
) -> PromptStrategy:
    """Quickly create a strategy without a YAML file.

    Args:
        name: Strategy name
        system_prompt: System prompt for the LLM
        analysis_prompt: Prompt template for analysis
        decision_prompt: Prompt template for decisions
        **kwargs: Additional config options (risk settings, watchlist, etc.)

    Returns:
        PromptStrategy instance

    Example:
        strategy = quick_strategy(
            name="my_strategy",
            system_prompt="You are a value investor...",
            analysis_prompt="Analyze {symbol} with price ${price}...",
            decision_prompt="Based on analysis: {analysis}...",
            max_position_pct=0.05,
        )
    """
    risk_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ["max_position_pct", "stop_loss_pct", "take_profit_pct",
                 "min_confidence", "max_daily_trades"]
    }

    config = StrategyConfig(
        name=name,
        description=kwargs.get("description", ""),
        research=ResearchConfig(
            system_prompt=system_prompt,
            analysis_prompt=analysis_prompt,
        ),
        decision=DecisionConfig(
            prompt=decision_prompt,
        ),
        risk=RiskConfig(**risk_kwargs) if risk_kwargs else RiskConfig(),
        watchlist=kwargs.get("watchlist", []),
        parameters=kwargs.get("parameters", {}),
    )

    return PromptStrategy(config)
