"""
Capital allocation and rotation module.

Purpose
-------
Translate ranked symbols into portfolio actions:
- enter top-ranked symbols
- evict bottom-ranked symbols
- enforce position and capital constraints

This module is policy-driven and portfolio-aware.
"""

from dataclasses import dataclass
from typing import List, Dict


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass(frozen=True)
class AllocationConfig:
    max_positions: int = 5
    target_weight: float = 0.2        # equal-weight default
    min_trade_value: float = 50.0     # ignore tiny trades


# -------------------------------------------------------------------
# Core allocation logic
# -------------------------------------------------------------------

def allocate_capital(
    portfolio,
    top_symbols: List[str],
    bottom_symbols: List[str],
    prices: Dict[str, float],
    config: AllocationConfig = AllocationConfig(),
    risk_mgr=None
):
    """
    Apply capital allocation decisions to a portfolio.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object (must support evict, execute, cash)
    top_symbols : list[str]
        Symbols selected for entry / holding
    bottom_symbols : list[str]
        Symbols selected for eviction
    prices : dict
        symbol -> current price
    config : AllocationConfig
        Allocation policy
    risk_mgr : optional
        RiskManager for scaling position sizes
    """

    # ----------------------------
    # Step 1: Evict bottom-ranked symbols
    # ----------------------------
    for symbol in bottom_symbols:
        if symbol in portfolio.positions:
            portfolio.evict(symbol, price=prices.get(symbol))

    # ----------------------------
    # Step 2: Determine available slots
    # ----------------------------
    current_positions = list(portfolio.positions.keys())
    available_slots = max(0, config.max_positions - len(current_positions))

    if available_slots <= 0:
        # Already at max positions, may still rebalance existing positions later
        pass

    # ----------------------------
    # Step 3: Compute allocation per new position
    # ----------------------------
    equity = portfolio.total_equity(prices)
    alloc_per_position = equity * config.target_weight

    # ----------------------------
    # Step 4: Enter top-ranked symbols
    # ----------------------------
    for symbol in top_symbols:
        if symbol in portfolio.positions:
            continue  # already holding

        if available_slots <= 0:
            break  # max positions reached

        price = prices.get(symbol)
        if price is None or price <= 0:
            continue

        # Scale by risk manager if provided
        size_multiplier = 1.0
        if risk_mgr:
            hist_returns = portfolio.get_symbol_history(symbol)
            hist_vol = hist_returns.std() if hist_returns is not None else 0.0
            size_multiplier = risk_mgr.scale_position(1.0, hist_vol)

        # Compute number of shares to buy
        shares = int((alloc_per_position * size_multiplier) // price)
        trade_value = shares * price

        if shares <= 0 or trade_value < config.min_trade_value:
            continue

        # Execute trade
        portfolio.execute(
            symbol=symbol,
            price=price,
            signal="BUY",
            position_size=shares
        )

        available_slots -= 1

    # ----------------------------
    # Step 5: Optional rebalance existing positions to target weight
    # ----------------------------
    # This ensures equal-weight allocation and full capital usage
    for symbol in portfolio.positions.keys():
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        target_value = equity * config.target_weight
        current_value = portfolio.positions[symbol] * price
        delta_value = target_value - current_value

        if abs(delta_value) < config.min_trade_value:
            continue

        shares_delta = int(delta_value // price)
        if shares_delta == 0:
            continue

        signal = "BUY" if shares_delta > 0 else "SELL"
        portfolio.execute(
            symbol=symbol,
            price=price,
            signal=signal,
            position_size=abs(shares_delta)
        )

