"""Data gathering module for research."""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from trading_bot.alpaca import AlpacaClient, MarketData
from trading_bot.research.news_fetcher import NewsFetcher, NewsArticle

logger = logging.getLogger(__name__)


class CompanyInfo(BaseModel):
    """Basic company information."""

    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: Decimal | None = None
    pe_ratio: Decimal | None = None
    dividend_yield: Decimal | None = None
    beta: Decimal | None = None
    fifty_two_week_high: Decimal | None = None
    fifty_two_week_low: Decimal | None = None


class ResearchData(BaseModel):
    """Aggregated research data for a symbol."""

    symbol: str
    gathered_at: datetime
    market_data: MarketData | None = None
    company_info: CompanyInfo | None = None
    news_articles: list[NewsArticle] = []
    additional_data: dict[str, Any] = {}

    def to_prompt_context(self) -> str:
        """Convert research data to text for LLM prompt."""
        lines = [f"=== Research Data for {self.symbol} ===\n"]

        # Market Data
        if self.market_data:
            md = self.market_data
            lines.append("Market Data:")
            lines.append(f"  Current Price: ${md.current_price:.2f}")
            lines.append(f"  Previous Close: ${md.previous_close:.2f}")
            lines.append(f"  Change: ${md.change:.2f} ({md.change_percent:.2f}%)")
            lines.append(f"  Day Range: ${md.low_price:.2f} - ${md.high_price:.2f}")
            lines.append(f"  Volume: {md.volume:,}")
            lines.append("")

        # Company Info
        if self.company_info:
            ci = self.company_info
            lines.append("Company Information:")
            if ci.name:
                lines.append(f"  Name: {ci.name}")
            if ci.sector:
                lines.append(f"  Sector: {ci.sector}")
            if ci.industry:
                lines.append(f"  Industry: {ci.industry}")
            if ci.market_cap:
                cap_b = ci.market_cap / Decimal("1000000000")
                lines.append(f"  Market Cap: ${cap_b:.2f}B")
            if ci.pe_ratio:
                lines.append(f"  P/E Ratio: {ci.pe_ratio:.2f}")
            if ci.dividend_yield:
                lines.append(f"  Dividend Yield: {ci.dividend_yield:.2f}%")
            if ci.beta:
                lines.append(f"  Beta: {ci.beta:.2f}")
            if ci.fifty_two_week_high and ci.fifty_two_week_low:
                lines.append(
                    f"  52-Week Range: ${ci.fifty_two_week_low:.2f} - ${ci.fifty_two_week_high:.2f}"
                )
            lines.append("")

        # Price History Summary
        if self.market_data and self.market_data.bars:
            bars = self.market_data.bars
            lines.append(f"Price History ({len(bars)} days):")

            # Calculate simple metrics
            closes = [float(b.close) for b in bars]
            if len(closes) >= 5:
                recent_avg = sum(closes[-5:]) / 5
                lines.append(f"  5-Day Avg: ${recent_avg:.2f}")
            if len(closes) >= 20:
                sma_20 = sum(closes[-20:]) / 20
                lines.append(f"  20-Day SMA: ${sma_20:.2f}")
            if len(closes) >= 2:
                first = closes[0]
                last = closes[-1]
                period_change = ((last - first) / first) * 100
                lines.append(f"  Period Change: {period_change:.2f}%")
            lines.append("")

        # News Summary
        if self.news_articles:
            lines.append(f"Recent News ({len(self.news_articles)} articles):")
            for i, article in enumerate(self.news_articles[:5], 1):
                age = ""
                if article.age_hours is not None:
                    if article.age_hours < 24:
                        age = f" ({int(article.age_hours)}h ago)"
                    else:
                        age = f" ({int(article.age_hours / 24)}d ago)"
                lines.append(f"  {i}. {article.title}{age}")
                if article.summary:
                    summary = (
                        article.summary[:150] + "..."
                        if len(article.summary) > 150
                        else article.summary
                    )
                    lines.append(f"     {summary}")
            lines.append("")

        # Additional data
        if self.additional_data:
            lines.append("Additional Data:")
            for key, value in self.additional_data.items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        return "\n".join(lines)


class DataGatherer:
    """Gathers research data from multiple sources."""

    def __init__(self, alpaca_client: AlpacaClient | None = None) -> None:
        """Initialize the data gatherer."""
        self.alpaca = alpaca_client or AlpacaClient()
        self.news_fetcher = NewsFetcher()

    def gather(
        self,
        symbol: str,
        include_news: bool = True,
        news_days: int = 7,
        price_history_days: int = 30,
    ) -> ResearchData:
        """Gather all research data for a symbol.

        Args:
            symbol: Stock symbol to research
            include_news: Whether to fetch news articles
            news_days: How many days of news to fetch
            price_history_days: How many days of price history

        Returns:
            ResearchData with all gathered information
        """
        logger.info(f"Gathering research data for {symbol}")

        # Get market data
        market_data = None
        try:
            market_data = self.alpaca.get_market_data(symbol, days=price_history_days)
        except Exception as e:
            logger.warning(f"Failed to get market data for {symbol}: {e}")

        # Get company info (basic from market data)
        company_info = CompanyInfo(symbol=symbol)

        # Get news
        news_articles: list[NewsArticle] = []
        if include_news:
            try:
                news_articles = self.news_fetcher.fetch_for_symbol_sync(
                    symbol, days=news_days
                )
            except Exception as e:
                logger.warning(f"Failed to fetch news for {symbol}: {e}")

        return ResearchData(
            symbol=symbol,
            gathered_at=datetime.now(),
            market_data=market_data,
            company_info=company_info,
            news_articles=news_articles,
        )

    def gather_multiple(
        self,
        symbols: list[str],
        include_news: bool = True,
    ) -> dict[str, ResearchData]:
        """Gather research data for multiple symbols."""
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.gather(symbol, include_news=include_news)
            except Exception as e:
                logger.error(f"Failed to gather data for {symbol}: {e}")
        return results

    def get_portfolio_context(self) -> str:
        """Get current portfolio context for LLM."""
        lines = ["=== Portfolio Context ===\n"]

        try:
            account = self.alpaca.get_account()
            lines.append("Account:")
            lines.append(f"  Portfolio Value: ${account.portfolio_value:.2f}")
            lines.append(f"  Cash: ${account.cash:.2f}")
            lines.append(f"  Buying Power: ${account.buying_power:.2f}")
            lines.append("")
        except Exception as e:
            logger.warning(f"Failed to get account info: {e}")
            lines.append("Account: Unable to fetch\n")

        try:
            positions = self.alpaca.get_positions()
            if positions:
                lines.append("Positions:")
                for pos in positions:
                    pnl_sign = "+" if pos.unrealized_pl >= 0 else ""
                    lines.append(
                        f"  {pos.symbol}: {pos.qty} shares @ ${pos.avg_entry_price:.2f} "
                        f"(P&L: {pnl_sign}${pos.unrealized_pl:.2f} / {pnl_sign}{pos.pnl_percent:.2f}%)"
                    )
                lines.append("")
            else:
                lines.append("Positions: None\n")
        except Exception as e:
            logger.warning(f"Failed to get positions: {e}")
            lines.append("Positions: Unable to fetch\n")

        return "\n".join(lines)
