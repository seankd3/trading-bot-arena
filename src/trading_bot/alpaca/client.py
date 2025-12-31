"""Alpaca API client wrapper."""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import OrderType as AlpacaOrderType
from alpaca.trading.enums import TimeInForce as AlpacaTimeInForce
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest

from trading_bot.alpaca.models import (
    AccountInfo,
    Bar,
    MarketData,
    Order,
    OrderSide,
    OrderType,
    Position,
    Quote,
    TimeInForce,
)
from trading_bot.config import get_settings

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Client for interacting with Alpaca Trading API."""

    def __init__(self) -> None:
        """Initialize the Alpaca client."""
        settings = get_settings()

        api_key = settings.alpaca_api_key.get_secret_value()
        secret_key = settings.alpaca_secret_key.get_secret_value()

        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=settings.alpaca_paper,
        )

        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key,
        )

        self._paper = settings.alpaca_paper
        logger.info(f"Alpaca client initialized (paper={self._paper})")

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self._paper

    def get_account(self) -> AccountInfo:
        """Get account information."""
        account = self.trading_client.get_account()

        return AccountInfo(
            id=str(account.id),
            account_number=account.account_number,
            status=account.status.value,
            currency=account.currency,
            cash=Decimal(str(account.cash)),
            portfolio_value=Decimal(str(account.portfolio_value)),
            buying_power=Decimal(str(account.buying_power)),
            equity=Decimal(str(account.equity)),
            last_equity=Decimal(str(account.last_equity)),
            long_market_value=Decimal(str(account.long_market_value)),
            short_market_value=Decimal(str(account.short_market_value)),
            initial_margin=Decimal(str(account.initial_margin)),
            maintenance_margin=Decimal(str(account.maintenance_margin)),
            daytrade_count=account.daytrade_count,
            pattern_day_trader=account.pattern_day_trader,
            trading_blocked=account.trading_blocked,
            transfers_blocked=account.transfers_blocked,
            account_blocked=account.account_blocked,
            created_at=account.created_at,
        )

    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        positions = self.trading_client.get_all_positions()

        return [
            Position(
                asset_id=str(p.asset_id),
                symbol=p.symbol,
                exchange=p.exchange.value if p.exchange else "UNKNOWN",
                asset_class=p.asset_class.value if p.asset_class else "us_equity",
                qty=Decimal(str(p.qty)),
                qty_available=Decimal(str(p.qty_available)),
                side=p.side.value,
                market_value=Decimal(str(p.market_value)),
                cost_basis=Decimal(str(p.cost_basis)),
                unrealized_pl=Decimal(str(p.unrealized_pl)),
                unrealized_plpc=Decimal(str(p.unrealized_plpc)),
                unrealized_intraday_pl=Decimal(str(p.unrealized_intraday_pl)),
                unrealized_intraday_plpc=Decimal(str(p.unrealized_intraday_plpc)),
                current_price=Decimal(str(p.current_price)),
                lastday_price=Decimal(str(p.lastday_price)),
                change_today=Decimal(str(p.change_today)),
                avg_entry_price=Decimal(str(p.avg_entry_price)),
            )
            for p in positions
        ]

    def get_position(self, symbol: str) -> Position | None:
        """Get position for a specific symbol."""
        try:
            p = self.trading_client.get_open_position(symbol)
            return Position(
                asset_id=str(p.asset_id),
                symbol=p.symbol,
                exchange=p.exchange.value if p.exchange else "UNKNOWN",
                asset_class=p.asset_class.value if p.asset_class else "us_equity",
                qty=Decimal(str(p.qty)),
                qty_available=Decimal(str(p.qty_available)),
                side=p.side.value,
                market_value=Decimal(str(p.market_value)),
                cost_basis=Decimal(str(p.cost_basis)),
                unrealized_pl=Decimal(str(p.unrealized_pl)),
                unrealized_plpc=Decimal(str(p.unrealized_plpc)),
                unrealized_intraday_pl=Decimal(str(p.unrealized_intraday_pl)),
                unrealized_intraday_plpc=Decimal(str(p.unrealized_intraday_plpc)),
                current_price=Decimal(str(p.current_price)),
                lastday_price=Decimal(str(p.lastday_price)),
                change_today=Decimal(str(p.change_today)),
                avg_entry_price=Decimal(str(p.avg_entry_price)),
            )
        except Exception:
            return None

    def get_quote(self, symbol: str) -> Quote:
        """Get latest quote for a symbol."""
        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quotes = self.data_client.get_stock_latest_quote(request)
        q = quotes[symbol]

        return Quote(
            symbol=symbol,
            bid_price=Decimal(str(q.bid_price)),
            bid_size=q.bid_size,
            ask_price=Decimal(str(q.ask_price)),
            ask_size=q.ask_size,
            timestamp=q.timestamp,
        )

    def get_bars(
        self,
        symbol: str,
        timeframe: TimeFrame = TimeFrame.Day,
        days: int = 30,
    ) -> list[Bar]:
        """Get historical bars for a symbol."""
        end = datetime.now()
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

        bars_data = self.data_client.get_stock_bars(request)
        bars = bars_data[symbol]

        return [
            Bar(
                symbol=symbol,
                timestamp=b.timestamp,
                open=Decimal(str(b.open)),
                high=Decimal(str(b.high)),
                low=Decimal(str(b.low)),
                close=Decimal(str(b.close)),
                volume=b.volume,
                trade_count=b.trade_count,
                vwap=Decimal(str(b.vwap)) if b.vwap else None,
            )
            for b in bars
        ]

    def get_market_data(self, symbol: str, days: int = 30) -> MarketData:
        """Get aggregated market data for a symbol."""
        bars = self.get_bars(symbol, days=days)
        return MarketData.from_bars(symbol, bars)

    def submit_order(
        self,
        symbol: str,
        qty: Decimal | int,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Decimal | None = None,
    ) -> Order:
        """Submit an order to Alpaca."""
        alpaca_side = AlpacaOrderSide.BUY if side == OrderSide.BUY else AlpacaOrderSide.SELL
        alpaca_tif = AlpacaTimeInForce(time_in_force.value)

        if order_type == OrderType.MARKET:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=alpaca_side,
                time_in_force=alpaca_tif,
            )
        elif order_type == OrderType.LIMIT:
            if limit_price is None:
                raise ValueError("Limit price required for limit orders")
            request = LimitOrderRequest(
                symbol=symbol,
                qty=float(qty),
                side=alpaca_side,
                time_in_force=alpaca_tif,
                limit_price=float(limit_price),
            )
        else:
            raise ValueError(f"Order type {order_type} not yet supported")

        order = self.trading_client.submit_order(request)

        logger.info(f"Order submitted: {side.value} {qty} {symbol} ({order_type.value})")

        return self._convert_order(order)

    def get_order(self, order_id: str) -> Order:
        """Get order by ID."""
        order = self.trading_client.get_order_by_id(order_id)
        return self._convert_order(order)

    def get_orders(self, status: str = "open") -> list[Order]:
        """Get orders by status."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        status_map = {
            "open": QueryOrderStatus.OPEN,
            "closed": QueryOrderStatus.CLOSED,
            "all": QueryOrderStatus.ALL,
        }

        request = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.OPEN))
        orders = self.trading_client.get_orders(request)

        return [self._convert_order(o) for o in orders]

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        self.trading_client.cancel_order_by_id(order_id)
        logger.info(f"Order cancelled: {order_id}")

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self.trading_client.cancel_orders()
        logger.info("All orders cancelled")

    def close_position(self, symbol: str) -> Order | None:
        """Close a position for a symbol."""
        try:
            order = self.trading_client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return self._convert_order(order)
        except Exception as e:
            logger.warning(f"Failed to close position {symbol}: {e}")
            return None

    def close_all_positions(self) -> list[Order]:
        """Close all positions."""
        results = self.trading_client.close_all_positions()
        logger.info(f"Closed {len(results)} positions")
        return [self._convert_order(r) for r in results if r]

    def _convert_order(self, order: Any) -> Order:
        """Convert Alpaca order to our Order model."""
        return Order(
            id=str(order.id),
            client_order_id=order.client_order_id,
            created_at=order.created_at,
            updated_at=order.updated_at,
            submitted_at=order.submitted_at,
            filled_at=order.filled_at,
            expired_at=order.expired_at,
            canceled_at=order.canceled_at,
            failed_at=order.failed_at,
            asset_id=str(order.asset_id),
            symbol=order.symbol,
            asset_class=order.asset_class.value if order.asset_class else "us_equity",
            qty=Decimal(str(order.qty)) if order.qty else None,
            filled_qty=Decimal(str(order.filled_qty)),
            notional=Decimal(str(order.notional)) if order.notional else None,
            filled_avg_price=(
                Decimal(str(order.filled_avg_price)) if order.filled_avg_price else None
            ),
            order_class=order.order_class.value if order.order_class else "",
            type=order.type.value,
            side=order.side.value,
            time_in_force=order.time_in_force.value,
            limit_price=Decimal(str(order.limit_price)) if order.limit_price else None,
            stop_price=Decimal(str(order.stop_price)) if order.stop_price else None,
            status=order.status.value,
            extended_hours=order.extended_hours,
        )
