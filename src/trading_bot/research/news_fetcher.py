"""News fetching and aggregation module."""

import logging
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import feedparser
from bs4 import BeautifulSoup
from pydantic import BaseModel

from trading_bot.config import get_settings

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    """A news article."""

    title: str
    summary: str
    source: str
    url: str
    published: datetime | None = None
    symbols: list[str] = []

    @property
    def age_hours(self) -> float | None:
        """Get age of article in hours."""
        if self.published:
            delta = datetime.now() - self.published
            return delta.total_seconds() / 3600
        return None


class NewsFetcher:
    """Fetches news from multiple sources."""

    # Free RSS feeds for financial news
    RSS_FEEDS = {
        "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
        "seeking_alpha": "https://seekingalpha.com/market_currents.xml",
        "reuters_business": "https://www.reutersagency.com/feed/?best-topics=business-finance",
        "marketwatch": "http://feeds.marketwatch.com/marketwatch/topstories/",
        "cnbc": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    }

    def __init__(self) -> None:
        """Initialize the news fetcher."""
        self.settings = get_settings()
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "NewsFetcher":
        """Async context manager entry."""
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    def fetch_rss_sync(self, feed_name: str | None = None) -> list[NewsArticle]:
        """Synchronously fetch news from RSS feeds.

        Args:
            feed_name: Specific feed to fetch, or None for all feeds

        Returns:
            List of news articles
        """
        articles = []
        feeds_to_fetch = (
            {feed_name: self.RSS_FEEDS[feed_name]}
            if feed_name and feed_name in self.RSS_FEEDS
            else self.RSS_FEEDS
        )

        for source, url in feeds_to_fetch.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:  # Limit to 10 per feed
                    published = None
                    if hasattr(entry, "published_parsed") and entry.published_parsed:
                        published = datetime(*entry.published_parsed[:6])

                    articles.append(
                        NewsArticle(
                            title=entry.get("title", ""),
                            summary=self._clean_html(entry.get("summary", "")),
                            source=source,
                            url=entry.get("link", ""),
                            published=published,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch RSS feed {source}: {e}")

        return articles

    def fetch_for_symbol_sync(
        self,
        symbol: str,
        days: int = 7,
    ) -> list[NewsArticle]:
        """Fetch news related to a specific symbol.

        Uses NewsAPI if available, falls back to RSS with filtering.
        """
        articles = []

        # Try NewsAPI first if configured
        if self.settings.news_api_key:
            try:
                articles = self._fetch_newsapi_sync(symbol, days)
            except Exception as e:
                logger.warning(f"NewsAPI fetch failed: {e}")

        # Supplement with RSS feeds
        rss_articles = self.fetch_rss_sync()

        # Filter RSS articles for symbol mentions
        symbol_lower = symbol.lower()
        for article in rss_articles:
            text = f"{article.title} {article.summary}".lower()
            if symbol_lower in text:
                article.symbols.append(symbol)
                articles.append(article)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_articles = []
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)

        # Sort by date, newest first
        unique_articles.sort(
            key=lambda a: a.published or datetime.min,
            reverse=True,
        )

        return unique_articles

    def _fetch_newsapi_sync(self, symbol: str, days: int) -> list[NewsArticle]:
        """Fetch from NewsAPI synchronously."""
        import httpx

        api_key = self.settings.news_api_key.get_secret_value()
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": api_key,
        }

        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("articles", [])[:20]:
            published = None
            if item.get("publishedAt"):
                try:
                    published = datetime.fromisoformat(
                        item["publishedAt"].replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except ValueError:
                    pass

            articles.append(
                NewsArticle(
                    title=item.get("title", ""),
                    summary=item.get("description", "") or "",
                    source=item.get("source", {}).get("name", "NewsAPI"),
                    url=item.get("url", ""),
                    published=published,
                    symbols=[symbol],
                )
            )

        return articles

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        if not text:
            return ""
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ", strip=True)

    def summarize_news(self, articles: list[NewsArticle], max_articles: int = 10) -> str:
        """Create a text summary of news articles for LLM consumption."""
        if not articles:
            return "No recent news articles found."

        lines = [f"Recent News ({len(articles[:max_articles])} articles):\n"]

        for i, article in enumerate(articles[:max_articles], 1):
            age = ""
            if article.age_hours is not None:
                if article.age_hours < 1:
                    age = f" ({int(article.age_hours * 60)}m ago)"
                elif article.age_hours < 24:
                    age = f" ({int(article.age_hours)}h ago)"
                else:
                    age = f" ({int(article.age_hours / 24)}d ago)"

            lines.append(f"{i}. [{article.source}]{age}")
            lines.append(f"   Title: {article.title}")
            if article.summary:
                summary = article.summary[:300] + "..." if len(article.summary) > 300 else article.summary
                lines.append(f"   Summary: {summary}")
            lines.append("")

        return "\n".join(lines)
