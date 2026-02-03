"""Market prediction system using knowledge distillation from LLMs."""

from trading_bot.prediction.features import FeatureExtractor
from trading_bot.prediction.model import MarketPredictor
from trading_bot.prediction.oracle import LLMOracle
from trading_bot.prediction.trainer import PredictionTrainer

__all__ = ["FeatureExtractor", "MarketPredictor", "LLMOracle", "PredictionTrainer"]
