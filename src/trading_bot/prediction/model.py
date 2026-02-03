"""Neural network model for market prediction."""

import json
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class TransformerBlock:
    """Single transformer block with self-attention."""

    def __init__(self, dim: int, num_heads: int = 4, ff_dim: int = None):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ff_dim = ff_dim or dim * 4

        # Initialize weights with Xavier/Glorot
        scale = math.sqrt(2.0 / dim)

        # Attention weights
        self.W_q = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_k = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_v = np.random.randn(dim, dim).astype(np.float32) * scale
        self.W_o = np.random.randn(dim, dim).astype(np.float32) * scale

        # FFN weights
        self.W_ff1 = np.random.randn(dim, self.ff_dim).astype(np.float32) * scale
        self.b_ff1 = np.zeros(self.ff_dim, dtype=np.float32)
        self.W_ff2 = np.random.randn(self.ff_dim, dim).astype(np.float32) * scale
        self.b_ff2 = np.zeros(dim, dtype=np.float32)

        # Layer norm parameters
        self.ln1_gamma = np.ones(dim, dtype=np.float32)
        self.ln1_beta = np.zeros(dim, dtype=np.float32)
        self.ln2_gamma = np.ones(dim, dtype=np.float32)
        self.ln2_beta = np.zeros(dim, dtype=np.float32)

        # Cache for backprop
        self._cache = {}

    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Apply layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Forward pass through transformer block.

        Args:
            x: Input tensor of shape (batch, seq_len, dim) or (seq_len, dim)

        Returns:
            Output tensor of same shape
        """
        # Handle 2D input
        squeeze = False
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
            squeeze = True

        batch_size, seq_len, _ = x.shape

        # Self-attention with pre-norm
        x_norm = self.layer_norm(x, self.ln1_gamma, self.ln1_beta)

        # Compute Q, K, V
        Q = x_norm @ self.W_q  # (batch, seq, dim)
        K = x_norm @ self.W_k
        V = x_norm @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = (Q @ K.transpose(0, 1, 3, 2)) / scale
        attn_probs = self._softmax(attn_scores, axis=-1)

        # Apply attention to values
        attn_out = attn_probs @ V  # (batch, heads, seq, head_dim)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.dim)

        # Output projection
        attn_out = attn_out @ self.W_o

        # Residual connection
        x = x + attn_out

        # FFN with pre-norm
        x_norm = self.layer_norm(x, self.ln2_gamma, self.ln2_beta)

        # FFN: dim -> ff_dim -> dim with GELU activation
        ff_out = x_norm @ self.W_ff1 + self.b_ff1
        ff_out = self._gelu(ff_out)
        ff_out = ff_out @ self.W_ff2 + self.b_ff2

        # Residual connection
        x = x + ff_out

        if squeeze:
            x = x[0]

        return x

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        """Gaussian Error Linear Unit activation."""
        return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

    def get_params(self) -> dict:
        """Get all parameters as dict."""
        return {
            "W_q": self.W_q.tolist(),
            "W_k": self.W_k.tolist(),
            "W_v": self.W_v.tolist(),
            "W_o": self.W_o.tolist(),
            "W_ff1": self.W_ff1.tolist(),
            "b_ff1": self.b_ff1.tolist(),
            "W_ff2": self.W_ff2.tolist(),
            "b_ff2": self.b_ff2.tolist(),
            "ln1_gamma": self.ln1_gamma.tolist(),
            "ln1_beta": self.ln1_beta.tolist(),
            "ln2_gamma": self.ln2_gamma.tolist(),
            "ln2_beta": self.ln2_beta.tolist(),
        }

    def set_params(self, params: dict):
        """Load parameters from dict."""
        self.W_q = np.array(params["W_q"], dtype=np.float32)
        self.W_k = np.array(params["W_k"], dtype=np.float32)
        self.W_v = np.array(params["W_v"], dtype=np.float32)
        self.W_o = np.array(params["W_o"], dtype=np.float32)
        self.W_ff1 = np.array(params["W_ff1"], dtype=np.float32)
        self.b_ff1 = np.array(params["b_ff1"], dtype=np.float32)
        self.W_ff2 = np.array(params["W_ff2"], dtype=np.float32)
        self.b_ff2 = np.array(params["b_ff2"], dtype=np.float32)
        self.ln1_gamma = np.array(params["ln1_gamma"], dtype=np.float32)
        self.ln1_beta = np.array(params["ln1_beta"], dtype=np.float32)
        self.ln2_gamma = np.array(params["ln2_gamma"], dtype=np.float32)
        self.ln2_beta = np.array(params["ln2_beta"], dtype=np.float32)


class MarketPredictor:
    """Transformer-based market prediction model.

    This model learns to predict market movements by distilling knowledge
    from an LLM oracle that analyzes historical scenarios.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        seq_len: int = 20,
        num_classes: int = 3,  # bearish, neutral, bullish
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.num_classes = num_classes

        # Input projection
        scale = math.sqrt(2.0 / input_dim)
        self.input_proj = np.random.randn(input_dim, hidden_dim).astype(np.float32) * scale
        self.input_bias = np.zeros(hidden_dim, dtype=np.float32)

        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(seq_len, hidden_dim)

        # Transformer blocks
        self.transformer_blocks = [
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ]

        # Output head
        self.output_proj = np.random.randn(hidden_dim, num_classes).astype(np.float32) * scale
        self.output_bias = np.zeros(num_classes, dtype=np.float32)

        # Confidence head (predicts uncertainty)
        self.confidence_proj = np.random.randn(hidden_dim, 1).astype(np.float32) * scale
        self.confidence_bias = np.zeros(1, dtype=np.float32)

        # Training state
        self._training = False
        self._epoch = 0
        self._best_loss = float("inf")

    def _create_positional_encoding(self, max_len: int, dim: int) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = np.zeros((max_len, dim), dtype=np.float32)
        pe[:, 0::2] = np.sin(pos * div_term)
        pe[:, 1::2] = np.cos(pos * div_term)

        return pe

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass through the model.

        Args:
            x: Input features of shape (batch, seq_len, input_dim) or (seq_len, input_dim)

        Returns:
            Tuple of (class_probs, confidence) where:
                - class_probs: Shape (batch, num_classes) with softmax probabilities
                - confidence: Shape (batch, 1) with confidence score (0-1)
        """
        # Handle 2D input
        squeeze = False
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
            squeeze = True

        batch_size, seq_len, _ = x.shape

        # Input projection
        x = x @ self.input_proj + self.input_bias

        # Add positional encoding
        x = x + self.pos_encoding[:seq_len]

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block.forward(x, training=self._training)

        # Use the last position for classification (like [CLS] token approach)
        x_last = x[:, -1, :]  # (batch, hidden_dim)

        # Classification output
        logits = x_last @ self.output_proj + self.output_bias
        probs = self._softmax(logits)

        # Confidence output (sigmoid to bound 0-1)
        conf_logit = x_last @ self.confidence_proj + self.confidence_bias
        confidence = self._sigmoid(conf_logit)

        if squeeze:
            probs = probs[0]
            confidence = confidence[0]

        return probs, confidence

    def predict(self, features_sequence: list[np.ndarray]) -> dict:
        """Make a prediction from a sequence of feature vectors.

        Args:
            features_sequence: List of feature arrays from FeatureExtractor.to_array()

        Returns:
            Dictionary with prediction details
        """
        if len(features_sequence) < self.seq_len:
            # Pad with zeros if sequence is too short
            padding = [np.zeros(self.input_dim, dtype=np.float32)] * (self.seq_len - len(features_sequence))
            features_sequence = padding + features_sequence

        # Take last seq_len features
        features_sequence = features_sequence[-self.seq_len:]

        # Stack into array
        x = np.stack(features_sequence, axis=0)

        # Forward pass
        probs, confidence = self.forward(x)

        # Interpret prediction
        class_names = ["bearish", "neutral", "bullish"]
        pred_class = int(np.argmax(probs))

        return {
            "prediction": class_names[pred_class],
            "class_index": pred_class,
            "probabilities": {
                "bearish": float(probs[0]),
                "neutral": float(probs[1]),
                "bullish": float(probs[2]),
            },
            "confidence": float(confidence[0]),
            "signal_strength": float(abs(probs[2] - probs[0])),  # Directional conviction
        }

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )

    def save(self, path: str):
        """Save model to file."""
        model_data = {
            "config": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "seq_len": self.seq_len,
                "num_classes": self.num_classes,
            },
            "input_proj": self.input_proj.tolist(),
            "input_bias": self.input_bias.tolist(),
            "output_proj": self.output_proj.tolist(),
            "output_bias": self.output_bias.tolist(),
            "confidence_proj": self.confidence_proj.tolist(),
            "confidence_bias": self.confidence_bias.tolist(),
            "transformer_blocks": [block.get_params() for block in self.transformer_blocks],
            "epoch": self._epoch,
            "best_loss": self._best_loss,
        }

        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MarketPredictor":
        """Load model from file."""
        with open(path, "r") as f:
            model_data = json.load(f)

        config = model_data["config"]
        model = cls(**config)

        model.input_proj = np.array(model_data["input_proj"], dtype=np.float32)
        model.input_bias = np.array(model_data["input_bias"], dtype=np.float32)
        model.output_proj = np.array(model_data["output_proj"], dtype=np.float32)
        model.output_bias = np.array(model_data["output_bias"], dtype=np.float32)
        model.confidence_proj = np.array(model_data["confidence_proj"], dtype=np.float32)
        model.confidence_bias = np.array(model_data["confidence_bias"], dtype=np.float32)

        for i, block_params in enumerate(model_data["transformer_blocks"]):
            model.transformer_blocks[i].set_params(block_params)

        model._epoch = model_data.get("epoch", 0)
        model._best_loss = model_data.get("best_loss", float("inf"))

        logger.info(f"Model loaded from {path}")
        return model
