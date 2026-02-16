"""
Cross-sectional alpha scoring module.

Purpose
-------
Convert per-symbol information (price, history, regime, rotation_score) into a
comparable scalar score that can be ranked across the universe.

Design Principles
-----------------
- Deterministic and stateless
- Cross-section comparable
- Penalizes volatility
- Regime-aware
- Encourages rotation of cheap tickers
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

EPSILON = 1e-8


@dataclass(frozen=True)
class ScoringConfig:
    lookback: int = 20
    vol_penalty: float = 1.0
    regime_multipliers: dict | None = None
    rotation_weight: float = 1.0  # influence of rotation score
    price_penalty_weight: float = 0.5  # penalize very cheap tickers


DEFAULT_REGIME_MULTIPLIERS = {
    "bull": 1.2,
    "neutral": 1.0,
    "bear": 0.7
}

# -------------------------------------------------------------------
# Core scoring logic
# -------------------------------------------------------------------

def compute_zscore(price: float, history: pd.Series) -> float:
    mean = history.mean()
    std = history.std()
    if std < EPSILON:
        return 0.0
    return (price - mean) / std


def compute_momentum(history: pd.Series) -> float:
    if len(history) < 2:
        return 0.0
    return history.iloc[-1] / history.iloc[0] - 1.0


def compute_volatility(history: pd.Series) -> float:
    returns = history.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    return returns.std()


def score_symbol(
    symbol: str,
    price: float,
    history: pd.Series,
    regime: str,
    rotation_score: float = 0.5,
    config: ScoringConfig = ScoringConfig()
) -> float:
    """
    Produce a single scalar score for a symbol.

    Higher score = more attractive.
    Penalizes cheap tickers and high-volatility assets.
    Encourages rotation of low-scoring symbols.
    """

    history = history.dropna()
    if len(history) < config.lookback:
        return float("-inf")

    # ----------------------------
    # Signal components
    # ----------------------------
    z = compute_zscore(price, history)               # mean reversion
    momentum = compute_momentum(history)            # recent trend
    vol = compute_volatility(history)               # volatility

    # ----------------------------
    # Base raw score
    # ----------------------------
    raw_score = (
        -z                  # mean reversion
        + 0.5 * momentum    # directional confirmation
    )

    # ----------------------------
    # Volatility penalty
    # ----------------------------
    adjusted_score = raw_score / (1.0 + config.vol_penalty * vol)

    # ----------------------------
    # Regime adjustment
    # ----------------------------
    multipliers = config.regime_multipliers or DEFAULT_REGIME_MULTIPLIERS
    regime_mult = multipliers.get(regime, 1.0)
    adjusted_score *= regime_mult

    # ----------------------------
    # Rotation score boost / cheap-ticker penalty
    # ----------------------------
    # Encourage rotation: higher rotation_score = keep longer, lower = rotate out
    adjusted_score *= (1.0 + config.rotation_weight * (rotation_score - 0.5))

    # Penalize very cheap tickers
    price_penalty = config.price_penalty_weight / max(price, EPSILON)
    final_score = adjusted_score - price_penalty

    return float(final_score)

