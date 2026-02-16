"""
Dynamic universe management module.

Purpose
-------
Control which symbols are eligible for trading over time.
Supports:
- periodic universe refresh
- liquidity and data-quality filtering
- graceful symbol entry/exit

This module is intentionally conservative and deterministic.
"""

from dataclasses import dataclass
from typing import List
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class UniverseConfig:
    min_price: float = 1.0
    min_avg_volume: float = 500_000
    refresh_frequency: int = 20     # days
    max_universe_size: int | None = None


# -------------------------------------------------------------------
# Core universe logic
# -------------------------------------------------------------------

class UniverseManager:
    """
    Maintains and refreshes the active trading universe.
    """

    def __init__(self, config: UniverseConfig = UniverseConfig()):
        self.config = config
        self._last_refresh_day: int | None = None
        self._current_universe: List[str] = []

    def should_refresh(self, day: int) -> bool:
        """
        Determine whether the universe should be refreshed.
        """
        if self._last_refresh_day is None:
            return True
        return (day - self._last_refresh_day) >= self.config.refresh_frequency

    def refresh_universe(
        self,
        day: int,
        universe_df: pd.DataFrame
    ) -> List[str]:
        """
        Refresh universe membership based on filters.

        Parameters
        ----------
        day : int
            Simulation day index
        universe_df : DataFrame
            Must contain columns:
            - symbol
            - price
            - avg_volume

        Returns
        -------
        list[str]
            Updated universe
        """

        # ----------------------------
        # Basic validation
        # ----------------------------
        required_cols = {"symbol", "price", "avg_volume"}
        if not required_cols.issubset(universe_df.columns):
            raise ValueError(
                f"Universe DataFrame must contain {required_cols}"
            )

        df = universe_df.copy()

        # ----------------------------
        # Apply filters
        # ----------------------------
        df = df[df["price"] >= self.config.min_price]
        df = df[df["avg_volume"] >= self.config.min_avg_volume]

        # ----------------------------
        # Sort by liquidity
        # ----------------------------
        df = df.sort_values(
            by="avg_volume",
            ascending=False
        )

        # ----------------------------
        # Cap universe size
        # ----------------------------
        if self.config.max_universe_size is not None:
            df = df.head(self.config.max_universe_size)

        # ----------------------------
        # Update internal state
        # ----------------------------
        self._current_universe = df["symbol"].tolist()
        self._last_refresh_day = day

        return self._current_universe

    def get_universe(self) -> List[str]:
        """
        Return the currently active universe.
        """
        return list(self._current_universe)

