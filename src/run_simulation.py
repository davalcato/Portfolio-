"""
Canonical simulation entry point with dynamic universe rotation.

Run with:
    python3 -m src.run_simulation
"""

import numpy as np
import pandas as pd
from collections import deque

from src.config import (
    INITIAL_CAPITAL,
    RANDOM_SEED,
    LOOKBACK,
    TRANSACTION_COST
)

from src.data_loader import load_universe_prices
from src.portfolio import Portfolio, compute_metrics
from src.scoring import score_symbol, ScoringConfig
from src.ranker import rank_universe, RankerConfig
from src.allocator import allocate_capital, AllocationConfig
from src.universe import load_universe, filter_universe
from src.regimes import classify_regime

# ----------------------------
# Simulation parameters
# ----------------------------

UNIVERSE_REFRESH_DAYS = 10       # refresh universe every N days
RECENTLY_HELD_MAXLEN = 50       # avoid immediate re-trades
RECENTLY_SOLD_MAXLEN = 50       # avoid re-entering sold tickers too soon

recently_held = deque(maxlen=RECENTLY_HELD_MAXLEN)
recently_sold = deque(maxlen=RECENTLY_SOLD_MAXLEN)

# ----------------------------
# Load universe & prices
# ----------------------------

df_universe = load_universe()
price_data = load_universe_prices()

# Initial active universe
active_universe = filter_universe(
    df_universe,
    recently_held=list(recently_held) + list(recently_sold)
)
print(f"Initial universe size: {len(active_universe)}")

# ----------------------------
# Initialize portfolio
# ----------------------------

portfolio = Portfolio(INITIAL_CAPITAL)
equity_curve = []
peak_equity = INITIAL_CAPITAL

# ----------------------------
# Precompute regimes
# ----------------------------

regimes_df = pd.DataFrame(index=price_data.index, columns=price_data.columns)
for symbol in price_data.columns:
    regimes_df[symbol] = classify_regime(price_data[symbol])

# ----------------------------
# Simulation loop
# ----------------------------

for day in range(LOOKBACK, len(price_data)):
    today_prices = price_data.iloc[day]
    history_window = price_data.iloc[day - LOOKBACK: day]

    # Refresh universe periodically
    if day % UNIVERSE_REFRESH_DAYS == 0:
        active_universe = filter_universe(
            df_universe,
            recently_held=list(recently_held),
            recently_sold=list(recently_sold)
        )
        print(f"\n=== DAY {day} ({price_data.index[day].date()}) ===")
        print(f"Refreshed universe size: {len(active_universe)}")

    # ----------------------------
    # Score symbols
    # ----------------------------
    scores = {}
    for symbol in active_universe:
        price = today_prices.get(symbol)
        hist = history_window[symbol].dropna()
        if price is None or len(hist) < LOOKBACK or price <= 0:
            continue

        today_regime = regimes_df[symbol].iloc[day]
        score = score_symbol(symbol, price, hist, today_regime)

        # Apply rotation_score from universe.csv
        if "rotation_score" in df_universe.columns:
            rotation = df_universe.loc[df_universe["ticker"] == symbol, "rotation_score"].values[0]
            score *= rotation  # higher rotation_score → more likely to stay

        scores[symbol] = score

    # ----------------------------
    # Rank universe
    # ----------------------------
    ranker_config = RankerConfig(top_n=5, bottom_n=5)
    top_symbols, bottom_symbols = rank_universe(scores, ranker_config)

    # ----------------------------
    # Allocate capital & execute trades
    # ----------------------------
    alloc_config = AllocationConfig(max_positions=5, target_weight=0.2)
    allocate_capital(
        portfolio,
        top_symbols,
        bottom_symbols,
        today_prices.to_dict(),
        alloc_config
    )

    # ----------------------------
    # Update recently held / sold
    # ----------------------------
    for sym in top_symbols:
        if sym not in recently_held:
            recently_held.append(sym)
    for sym in bottom_symbols:
        if sym not in recently_sold:
            recently_sold.append(sym)

    # ----------------------------
    # Track equity & drawdown
    # ----------------------------
    equity = portfolio.total_equity(today_prices.to_dict())
    equity_curve.append(equity)
    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity * 100

    # ----------------------------
    # Logging
    # ----------------------------
    print(f"\n=== FORECAST DAY {day} ({price_data.index[day].date()}) ===")
    for sym in top_symbols:
        print(f"{sym}: price={today_prices[sym]:.2f}, score={scores[sym]:.3f}, action=BUY")
    for sym in bottom_symbols:
        print(f"{sym}: price={today_prices[sym]:.2f}, score={scores[sym]:.3f}, action=SELL")
    print(f"Total Equity: {equity:.2f}")
    print(f"Drawdown: {drawdown:.2f}%")

# ----------------------------
# Final metrics
# ----------------------------

equity_series = pd.Series(equity_curve, index=price_data.index[LOOKBACK:])
metrics = compute_metrics(equity_series)

print("\n--- FINAL PORTFOLIO ---")
print(f"Cash: {portfolio.cash:.2f}")
print(f"Open Positions: {portfolio.positions}")
print("\n--- METRICS ---")
print(metrics)
print("\n--- TRADE LOG ---")
for trade in portfolio.trade_log:
    print(trade)

