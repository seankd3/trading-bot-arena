"""Trade execution engine."""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel

from trading_bot.alpaca import AlpacaClient, Order, OrderSide, OrderType
from trading_bot.config import get_settings
from trading_bot.strategies.base import StrategyDecision, TradeAction

logger = logging.getLogger(__name__)


class TradeResult(BaseModel):
    """Result of a trade execution attempt."""

    success: bool
    symbol: str
    action: TradeAction
    quantity: int | None = None
    order: Order | None = None
    error: str | None = None
    executed_at: datetime = datetime.now()

    @property
    def order_id(self) -> str | None:
        """Get order ID if order was placed."""
        return self.order.id if self.order else None


class Trader:
    """Executes trades based on strategy decisions."""

    def __init__(
        self,
        alpaca_client: AlpacaClient | None = None,
        dry_run: bool = False,
    ) -> None:
        """Initialize the trader.

        Args:
            alpaca_client: Alpaca client (created if not provided)
            dry_run: If True, don't actually execute trades
        """
        self.alpaca = alpaca_client or AlpacaClient()
        self.dry_run = dry_run
        self.settings = get_settings()
        self._daily_trades: list[TradeResult] = []
        self._last_trade_date: datetime | None = None

    @property
    def is_paper(self) -> bool:
        """Check if using paper trading."""
        return self.alpaca.is_paper

    @property
    def daily_trade_count(self) -> int:
        """Get number of trades executed today."""
        today = datetime.now().date()
        if self._last_trade_date != today:
            self._daily_trades = []
            self._last_trade_date = today
        return len(self._daily_trades)

    def execute(self, decision: StrategyDecision) -> TradeResult:
        """Execute a trading decision.

        Args:
            decision: The strategy decision to execute

        Returns:
            TradeResult with execution details
        """
        symbol = decision.symbol

        # Check if we should act
        if decision.action == TradeAction.HOLD:
            logger.info(f"HOLD decision for {symbol} - no action taken")
            return TradeResult(
                success=True,
                symbol=symbol,
                action=decision.action,
            )

        # Check confidence threshold
        min_confidence = self.settings.max_position_size  # Use config
        if decision.confidence < 0.5:  # Hard minimum
            logger.info(
                f"Confidence too low for {symbol}: {decision.confidence:.2f} < 0.50"
            )
            return TradeResult(
                success=False,
                symbol=symbol,
                action=decision.action,
                error=f"Confidence below threshold: {decision.confidence:.2f}",
            )

        # Check daily trade limit
        if self.daily_trade_count >= self.settings.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.daily_trade_count}")
            return TradeResult(
                success=False,
                symbol=symbol,
                action=decision.action,
                error="Daily trade limit reached",
            )

        # Calculate quantity if not specified
        quantity = decision.quantity
        if quantity is None:
            quantity = self._calculate_quantity(symbol, decision.action)

        if quantity <= 0:
            logger.warning(f"Calculated quantity is 0 for {symbol}")
            return TradeResult(
                success=False,
                symbol=symbol,
                action=decision.action,
                error="Calculated quantity is 0",
            )

        # Execute the trade
        if decision.action == TradeAction.BUY:
            return self._execute_buy(symbol, quantity)
        else:  # SELL
            return self._execute_sell(symbol, quantity)

    def _calculate_quantity(self, symbol: str, action: TradeAction) -> int:
        """Calculate the quantity to trade based on position sizing rules."""
        try:
            account = self.alpaca.get_account()
            quote = self.alpaca.get_quote(symbol)
            price = float(quote.ask_price if action == TradeAction.BUY else quote.bid_price)

            if price <= 0:
                return 0

            # Calculate max position value
            max_value = float(account.portfolio_value) * self.settings.max_position_size

            # For sells, check current position
            if action == TradeAction.SELL:
                position = self.alpaca.get_position(symbol)
                if position:
                    return int(position.qty)
                return 0

            # For buys, calculate based on available cash and max position
            available = min(float(account.cash), max_value)
            quantity = int(available / price)

            # Check existing position
            position = self.alpaca.get_position(symbol)
            if position:
                current_value = float(position.market_value)
                remaining_allocation = max_value - current_value
                if remaining_allocation <= 0:
                    logger.info(f"Max position size reached for {symbol}")
                    return 0
                quantity = min(quantity, int(remaining_allocation / price))

            return max(0, quantity)

        except Exception as e:
            logger.error(f"Error calculating quantity for {symbol}: {e}")
            return 0

    def _execute_buy(self, symbol: str, quantity: int) -> TradeResult:
        """Execute a buy order."""
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}BUY {quantity} {symbol}")

        if self.dry_run:
            return TradeResult(
                success=True,
                symbol=symbol,
                action=TradeAction.BUY,
                quantity=quantity,
            )

        try:
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
            )

            result = TradeResult(
                success=True,
                symbol=symbol,
                action=TradeAction.BUY,
                quantity=quantity,
                order=order,
            )
            self._daily_trades.append(result)

            logger.info(f"Buy order submitted: {order.id}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute buy for {symbol}: {e}")
            return TradeResult(
                success=False,
                symbol=symbol,
                action=TradeAction.BUY,
                quantity=quantity,
                error=str(e),
            )

    def _execute_sell(self, symbol: str, quantity: int) -> TradeResult:
        """Execute a sell order."""
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}SELL {quantity} {symbol}")

        if self.dry_run:
            return TradeResult(
                success=True,
                symbol=symbol,
                action=TradeAction.SELL,
                quantity=quantity,
            )

        try:
            order = self.alpaca.submit_order(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
            )

            result = TradeResult(
                success=True,
                symbol=symbol,
                action=TradeAction.SELL,
                quantity=quantity,
                order=order,
            )
            self._daily_trades.append(result)

            logger.info(f"Sell order submitted: {order.id}")
            return result

        except Exception as e:
            logger.error(f"Failed to execute sell for {symbol}: {e}")
            return TradeResult(
                success=False,
                symbol=symbol,
                action=TradeAction.SELL,
                quantity=quantity,
                error=str(e),
            )

    def close_position(self, symbol: str) -> TradeResult:
        """Close an entire position."""
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Closing position: {symbol}")

        if self.dry_run:
            return TradeResult(
                success=True,
                symbol=symbol,
                action=TradeAction.SELL,
            )

        try:
            order = self.alpaca.close_position(symbol)
            if order:
                result = TradeResult(
                    success=True,
                    symbol=symbol,
                    action=TradeAction.SELL,
                    order=order,
                )
                self._daily_trades.append(result)
                return result
            else:
                return TradeResult(
                    success=False,
                    symbol=symbol,
                    action=TradeAction.SELL,
                    error="No position to close",
                )
        except Exception as e:
            logger.error(f"Failed to close position {symbol}: {e}")
            return TradeResult(
                success=False,
                symbol=symbol,
                action=TradeAction.SELL,
                error=str(e),
            )

    def get_order_status(self, order_id: str) -> Order | None:
        """Get the status of an order."""
        try:
            return self.alpaca.get_order(order_id)
        except Exception as e:
            logger.error(f"Failed to get order status {order_id}: {e}")
            return None
