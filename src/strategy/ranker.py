"""
Cross-sectional ranking module.

Purpose
-------
Rank symbols by alpha score and select candidates for:
- entry (top-N)
- eviction (bottom-N)

This module is deliberately simple, deterministic,
and side-effect free.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class RankerConfig:
    top_n: int = 5
    bottom_n: int = 5
    min_score: float | None = None  # minimum threshold to be considered


# -------------------------------------------------------------------
# Core ranking logic
# -------------------------------------------------------------------

def rank_universe(
    scores: Dict[str, float],
    config: RankerConfig = RankerConfig()
) -> Dict[str, List[str]]:
    """
    Rank symbols by score and return top / bottom candidates.

    Parameters
    ----------
    scores : dict
        Mapping symbol -> score
    config : RankerConfig
        Ranking configuration

    Returns
    -------
    result : dict
        {
            "top": [symbols selected for entry/holding],
            "bottom": [symbols selected for eviction]
        }
    """
    if not scores:
        return {"top": [], "bottom": []}

    # Remove invalid scores
    clean_scores = {
        sym: score
        for sym, score in scores.items()
        if score is not None and score != float("-inf")
    }

    if not clean_scores:
        return {"top": [], "bottom": []}

    # Optional minimum score filter
    if config.min_score is not None:
        clean_scores = {
            sym: score
            for sym, score in clean_scores.items()
            if score >= config.min_score
        }

    if not clean_scores:
        return {"top": [], "bottom": []}

    # Sort descending (higher = better)
    ranked = sorted(clean_scores.items(), key=lambda x: x[1], reverse=True)

    # Select top-N and bottom-N
    top_symbols = [sym for sym, _ in ranked[: config.top_n]]
    bottom_symbols = [sym for sym, _ in ranked[-config.bottom_n:]]

    return {"top": top_symbols, "bottom": bottom_symbols}

