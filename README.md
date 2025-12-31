# TradingBotArena - LLM Deep Research Trading Bot

An AI-powered trading bot that uses Large Language Models (LLMs) for deep market research and analysis. Experiment with different prompts as trading strategies on Alpaca.

## Features

- **LLM-Powered Research**: Use Claude or GPT models for deep market analysis
- **Prompt-Based Strategies**: Define trading strategies as YAML configuration files with customizable prompts
- **Multi-Source Data**: Gather data from market feeds, news sources, and SEC filings
- **Alpaca Integration**: Trade stocks via Alpaca's paper or live trading API
- **Strategy Arena**: Compare multiple strategies and their performance
- **Risk Management**: Built-in position sizing and risk controls

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Sean-Kenneth-Doherty/TradingBotArena.git
cd TradingBotArena

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# - Alpaca API credentials (required)
# - Anthropic or OpenAI API key (required)
# - News API key (optional)
```

### 3. Run the Bot

```bash
# Run with default strategy
trading-bot run

# Run with a specific strategy
trading-bot run --strategy value_investor

# List available strategies
trading-bot strategies list

# Research a specific stock
trading-bot research AAPL

# Backtest a strategy (coming soon)
trading-bot backtest --strategy momentum_trader --days 30
```

## Creating Custom Strategies

Strategies are defined as YAML files in the `strategies/` directory. Each strategy consists of:

1. **Research Prompts**: How the LLM should analyze market data
2. **Decision Prompts**: How to make buy/sell/hold decisions
3. **Risk Parameters**: Position sizing and stop-loss rules

### Example Strategy

```yaml
name: value_investor
description: Warren Buffett-style value investing approach

research:
  system_prompt: |
    You are a value investor following Warren Buffett's principles.
    Focus on companies with strong moats, good management, and
    trading below intrinsic value.

  analysis_prompt: |
    Analyze {symbol} for value investing potential:

    Current Price: ${price}
    P/E Ratio: {pe_ratio}
    Market Cap: ${market_cap}

    Recent News:
    {news_summary}

    Evaluate:
    1. Competitive moat and market position
    2. Management quality and capital allocation
    3. Intrinsic value estimate vs current price
    4. Long-term growth prospects

    Provide a detailed analysis and investment thesis.

decision:
  prompt: |
    Based on your analysis of {symbol}:

    {analysis}

    Current portfolio allocation: {portfolio_allocation}%
    Available cash: ${available_cash}

    What action should we take? Respond with JSON:
    {
      "action": "buy" | "sell" | "hold",
      "confidence": 0.0-1.0,
      "quantity": number or null,
      "reasoning": "brief explanation"
    }

risk:
  max_position_pct: 0.05  # Max 5% per position
  stop_loss_pct: 0.15     # 15% stop loss
  min_confidence: 0.7     # Minimum confidence to act
```

## Architecture

```
TradingBotArena/
├── src/trading_bot/
│   ├── alpaca/          # Alpaca API integration
│   ├── research/        # LLM research engine
│   ├── strategies/      # Strategy framework
│   └── execution/       # Trade execution
├── strategies/          # Strategy YAML files
└── tests/              # Test suite
```

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `value_investor` | Long-term value investing based on fundamentals |
| `momentum_trader` | Technical momentum and trend following |
| `news_sentiment` | News-driven sentiment analysis |
| `contrarian` | Contrarian plays on oversold conditions |

## Safety Features

- **Paper Trading Default**: Always starts in paper trading mode
- **Position Limits**: Configurable maximum position sizes
- **Daily Trade Limits**: Prevents overtrading
- **Confidence Thresholds**: Only acts on high-confidence signals
- **Human Override**: Requires confirmation for large trades

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/

# Format code
black src/
```

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and never trade with money you cannot afford to lose.

## License

MIT License - see LICENSE file for details.
