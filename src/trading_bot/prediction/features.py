"""Feature extraction for market prediction."""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MarketFeatures:
    """Extracted features for a single time point."""

    symbol: str
    timestamp: str

    # Price features
    price: float
    returns_1d: float
    returns_5d: float
    returns_20d: float

    # Volatility
    volatility_20d: float
    volatility_ratio: float  # short-term vs long-term

    # Momentum indicators
    rsi_14: float
    macd: float
    macd_signal: float
    macd_histogram: float

    # Trend indicators
    sma_20: float
    sma_50: float
    sma_200: float
    price_to_sma20: float
    price_to_sma50: float
    sma_20_50_cross: float  # 1 if golden cross, -1 if death cross, 0 otherwise

    # Volume features
    volume: float
    volume_sma_20: float
    volume_ratio: float

    # Bollinger bands
    bb_upper: float
    bb_lower: float
    bb_position: float  # Where price is within bands (0-1)

    # Market regime
    trend_strength: float  # ADX-like measure
    regime: str  # trending_up, trending_down, ranging

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.returns_1d,
            self.returns_5d,
            self.returns_20d,
            self.volatility_20d,
            self.volatility_ratio,
            self.rsi_14 / 100,  # Normalize to 0-1
            self.macd,
            self.macd_signal,
            self.macd_histogram,
            self.price_to_sma20,
            self.price_to_sma50,
            self.sma_20_50_cross,
            self.volume_ratio,
            self.bb_position,
            self.trend_strength,
            1.0 if self.regime == "trending_up" else (-1.0 if self.regime == "trending_down" else 0.0),
        ], dtype=np.float32)


class FeatureExtractor:
    """Extract features from price data for prediction models."""

    def __init__(self):
        self.feature_dim = 16  # Number of features in to_array()

    def extract(
        self,
        prices: list[float],
        volumes: list[float],
        timestamps: list[str],
        symbol: str,
    ) -> Optional[MarketFeatures]:
        """Extract features from price/volume history.

        Args:
            prices: List of closing prices (oldest to newest)
            volumes: List of volumes (oldest to newest)
            timestamps: List of timestamps
            symbol: Stock symbol

        Returns:
            MarketFeatures if enough data, None otherwise
        """
        if len(prices) < 200:
            return None

        prices = np.array(prices, dtype=np.float64)
        volumes = np.array(volumes, dtype=np.float64)

        current_price = prices[-1]

        # Returns
        returns_1d = (prices[-1] / prices[-2] - 1) if len(prices) >= 2 else 0
        returns_5d = (prices[-1] / prices[-6] - 1) if len(prices) >= 6 else 0
        returns_20d = (prices[-1] / prices[-21] - 1) if len(prices) >= 21 else 0

        # Volatility
        returns = np.diff(prices) / prices[:-1]
        volatility_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0
        vol_short = np.std(returns[-5:]) if len(returns) >= 5 else volatility_20d
        volatility_ratio = vol_short / volatility_20d if volatility_20d > 0 else 1.0

        # RSI
        rsi_14 = self._calculate_rsi(prices, 14)

        # MACD
        macd, macd_signal, macd_histogram = self._calculate_macd(prices)

        # Moving averages
        sma_20 = np.mean(prices[-20:])
        sma_50 = np.mean(prices[-50:])
        sma_200 = np.mean(prices[-200:])

        price_to_sma20 = current_price / sma_20 - 1
        price_to_sma50 = current_price / sma_50 - 1

        # SMA cross detection
        prev_sma20 = np.mean(prices[-21:-1])
        prev_sma50 = np.mean(prices[-51:-1])
        sma_20_50_cross = 0.0
        if prev_sma20 <= prev_sma50 and sma_20 > sma_50:
            sma_20_50_cross = 1.0  # Golden cross
        elif prev_sma20 >= prev_sma50 and sma_20 < sma_50:
            sma_20_50_cross = -1.0  # Death cross

        # Volume
        current_volume = volumes[-1]
        volume_sma_20 = np.mean(volumes[-20:])
        volume_ratio = current_volume / volume_sma_20 if volume_sma_20 > 0 else 1.0

        # Bollinger Bands
        bb_std = np.std(prices[-20:])
        bb_upper = sma_20 + 2 * bb_std
        bb_lower = sma_20 - 2 * bb_std
        bb_range = bb_upper - bb_lower
        bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        bb_position = max(0, min(1, bb_position))

        # Trend strength (simplified ADX-like)
        trend_strength = self._calculate_trend_strength(prices)

        # Regime detection
        if trend_strength > 0.25:
            if sma_20 > sma_50:
                regime = "trending_up"
            else:
                regime = "trending_down"
        else:
            regime = "ranging"

        return MarketFeatures(
            symbol=symbol,
            timestamp=timestamps[-1],
            price=current_price,
            returns_1d=returns_1d,
            returns_5d=returns_5d,
            returns_20d=returns_20d,
            volatility_20d=volatility_20d,
            volatility_ratio=volatility_ratio,
            rsi_14=rsi_14,
            macd=macd,
            macd_signal=macd_signal,
            macd_histogram=macd_histogram,
            sma_20=sma_20,
            sma_50=sma_50,
            sma_200=sma_200,
            price_to_sma20=price_to_sma20,
            price_to_sma50=price_to_sma50,
            sma_20_50_cross=sma_20_50_cross,
            volume=current_volume,
            volume_sma_20=volume_sma_20,
            volume_ratio=volume_ratio,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
            bb_position=bb_position,
            trend_strength=trend_strength,
            regime=regime,
        )

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[float, float, float]:
        """Calculate MACD indicator."""
        if len(prices) < slow + signal:
            return 0.0, 0.0, 0.0

        # EMA calculation
        def ema(data, period):
            alpha = 2 / (period + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result

        ema_fast = ema(prices, fast)
        ema_slow = ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)

        macd = macd_line[-1]
        macd_sig = signal_line[-1]
        histogram = macd - macd_sig

        # Normalize by price
        current_price = prices[-1]
        macd /= current_price
        macd_sig /= current_price
        histogram /= current_price

        return macd, macd_sig, histogram

    def _calculate_trend_strength(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate trend strength (0-1 scale)."""
        if len(prices) < period * 2:
            return 0.0

        # Use directional movement
        highs = prices  # Simplified - using closes as proxy
        lows = prices

        plus_dm = []
        minus_dm = []
        tr = []

        for i in range(1, len(prices)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]

            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)

            tr_val = max(
                highs[i] - lows[i],
                abs(highs[i] - prices[i-1]),
                abs(lows[i] - prices[i-1])
            )
            tr.append(tr_val)

        if not tr or sum(tr[-period:]) == 0:
            return 0.0

        atr = np.mean(tr[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0

        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0.0

        dx = abs(plus_di - minus_di) / di_sum

        return min(1.0, dx)
