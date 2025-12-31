"""Configuration management for the trading bot."""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class RiskLevel(str, Enum):
    """Risk tolerance levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ResearchDepth(str, Enum):
    """Depth of research analysis."""

    SHALLOW = "shallow"
    MODERATE = "moderate"
    DEEP = "deep"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Alpaca Configuration
    alpaca_api_key: SecretStr = Field(description="Alpaca API key")
    alpaca_secret_key: SecretStr = Field(description="Alpaca secret key")
    alpaca_paper: bool = Field(default=True, description="Use paper trading")

    # LLM Configuration
    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC, description="LLM provider to use"
    )
    anthropic_api_key: SecretStr | None = Field(default=None, description="Anthropic API key")
    openai_api_key: SecretStr | None = Field(default=None, description="OpenAI API key")

    # Model Configuration
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514", description="Anthropic model to use"
    )
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")

    # News API
    news_api_key: SecretStr | None = Field(default=None, description="NewsAPI key")

    # Trading Configuration
    max_position_size: float = Field(
        default=0.1, ge=0.01, le=0.5, description="Max position size as fraction of portfolio"
    )
    max_daily_trades: int = Field(default=10, ge=1, le=100, description="Maximum trades per day")
    risk_level: RiskLevel = Field(default=RiskLevel.MODERATE, description="Risk tolerance")

    # Research Configuration
    research_depth: ResearchDepth = Field(
        default=ResearchDepth.DEEP, description="Depth of research analysis"
    )
    cache_duration_hours: int = Field(default=1, ge=0, le=24, description="Cache duration in hours")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="Logging level"
    )

    @property
    def strategies_dir(self) -> Path:
        """Get the strategies directory path."""
        # First check current working directory
        cwd_strategies = Path.cwd() / "strategies"
        if cwd_strategies.exists():
            return cwd_strategies
        # Fall back to package location
        return Path(__file__).parent.parent.parent.parent / "strategies"

    @property
    def active_llm_key(self) -> SecretStr:
        """Get the API key for the active LLM provider."""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key not configured")
            return self.anthropic_api_key
        else:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key not configured")
            return self.openai_api_key

    @property
    def active_model(self) -> str:
        """Get the model name for the active LLM provider."""
        if self.llm_provider == LLMProvider.ANTHROPIC:
            return self.anthropic_model
        return self.openai_model


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
