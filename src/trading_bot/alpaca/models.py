"""Data models for Alpaca API responses."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    """Order side (buy or sell)."""

    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(str, Enum):
    """Order status."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"
    ACCEPTED = "accepted"
    PENDING_NEW = "pending_new"
    ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STOPPED = "stopped"
    REJECTED = "rejected"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"


class TimeInForce(str, Enum):
    """Time in force for orders."""

    DAY = "day"
    GTC = "gtc"  # Good till canceled
    OPG = "opg"  # Market on open
    CLS = "cls"  # Market on close
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class AccountInfo(BaseModel):
    """Account information from Alpaca."""

    id: str
    account_number: str
    status: str
    currency: str = "USD"
    cash: Decimal
    portfolio_value: Decimal
    buying_power: Decimal
    equity: Decimal
    last_equity: Decimal
    long_market_value: Decimal
    short_market_value: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    daytrade_count: int
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    created_at: datetime

    @property
    def available_cash(self) -> Decimal:
        """Get available cash for trading."""
        return self.cash

    @property
    def total_value(self) -> Decimal:
        """Get total portfolio value."""
        return self.portfolio_value


class Position(BaseModel):
    """A position in the portfolio."""

    asset_id: str
    symbol: str
    exchange: str
    asset_class: str
    qty: Decimal
    qty_available: Decimal
    side: str
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pl: Decimal
    unrealized_plpc: Decimal  # Percentage
    unrealized_intraday_pl: Decimal
    unrealized_intraday_plpc: Decimal
    current_price: Decimal
    lastday_price: Decimal
    change_today: Decimal
    avg_entry_price: Decimal

    @property
    def quantity(self) -> Decimal:
        """Get position quantity."""
        return self.qty

    @property
    def pnl_percent(self) -> Decimal:
        """Get P&L as percentage."""
        return self.unrealized_plpc * 100


class Order(BaseModel):
    """An order submitted to Alpaca."""

    id: str
    client_order_id: str
    created_at: datetime
    updated_at: datetime | None = None
    submitted_at: datetime | None = None
    filled_at: datetime | None = None
    expired_at: datetime | None = None
    canceled_at: datetime | None = None
    failed_at: datetime | None = None
    asset_id: str
    symbol: str
    asset_class: str
    qty: Decimal | None = None
    filled_qty: Decimal
    notional: Decimal | None = None
    filled_avg_price: Decimal | None = None
    order_class: str = ""
    order_type: OrderType = Field(alias="type")
    side: OrderSide
    time_in_force: TimeInForce
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus
    extended_hours: bool = False
    legs: list[Any] | None = None

    class Config:
        populate_by_name = True


class Quote(BaseModel):
    """A price quote for a symbol."""

    symbol: str
    bid_price: Decimal
    bid_size: int
    ask_price: Decimal
    ask_size: int
    timestamp: datetime

    @property
    def mid_price(self) -> Decimal:
        """Get the mid-point price."""
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> Decimal:
        """Get the bid-ask spread."""
        return self.ask_price - self.bid_price


class Bar(BaseModel):
    """OHLCV bar data."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    trade_count: int | None = None
    vwap: Decimal | None = None


class MarketData(BaseModel):
    """Aggregated market data for a symbol."""

    symbol: str
    current_price: Decimal
    previous_close: Decimal
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    volume: int
    change: Decimal
    change_percent: Decimal
    timestamp: datetime
    bars: list[Bar] = Field(default_factory=list)

    @classmethod
    def from_bars(cls, symbol: str, bars: list[Bar]) -> "MarketData":
        """Create MarketData from a list of bars."""
        if not bars:
            raise ValueError("Cannot create MarketData from empty bars list")

        latest = bars[-1]
        previous = bars[-2] if len(bars) > 1 else bars[0]

        change = latest.close - previous.close
        change_pct = (change / previous.close * 100) if previous.close else Decimal(0)

        return cls(
            symbol=symbol,
            current_price=latest.close,
            previous_close=previous.close,
            open_price=bars[0].open,
            high_price=max(b.high for b in bars),
            low_price=min(b.low for b in bars),
            volume=sum(b.volume for b in bars),
            change=change,
            change_percent=change_pct,
            timestamp=latest.timestamp,
            bars=bars,
        )
