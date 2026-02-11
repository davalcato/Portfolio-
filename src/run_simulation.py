"""
Canonical simulation entry point with dynamic Russell 3000 universe rotation,
ADV/liquidity filtering, parallelized price downloads, rotation score updates,
and true daily universe reshuffle.

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

np.random.seed(RANDOM_SEED)

# ----------------------------
# Simulation parameters
# ----------------------------
UNIVERSE_REFRESH_DAYS = 10       # full universe refresh every N days
RECENTLY_HELD_MAXLEN = 50
RECENTLY_SOLD_MAXLEN = 50
MIN_PRICE = 5
MIN_ADV = 1_000_000
BATCH_SIZE = 100
MAX_WORKERS = 5
UNIVERSE_SAMPLE_SIZE = 500       # daily random subset
TOP_N = 5
BOTTOM_N = 5
MAX_POSITIONS = 5
TARGET_WEIGHT = 0.2

recently_held = deque(maxlen=RECENTLY_HELD_MAXLEN)
recently_sold = deque(maxlen=RECENTLY_SOLD_MAXLEN)

# ----------------------------
# Load universe & prices
# ----------------------------
df_universe = load_universe()
price_data = load_universe_prices(
    batch_size=BATCH_SIZE,
    parallel=True,
    max_workers=MAX_WORKERS
)

# Initialize rotation scores if missing
if "rotation_score" not in df_universe.columns:
    df_universe["rotation_score"] = 1.0

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

    # ----------------------------
    # Refresh universe fully every N days
    # ----------------------------
    if day % UNIVERSE_REFRESH_DAYS == 0:
        # Keep only tickers with enough data
        df_universe = df_universe[df_universe["Ticker"].isin(price_data.columns)]
        print(f"\n=== DAY {day} ({price_data.index[day].date()}) ===")
        print(f"Universe refreshed: {len(df_universe)} tickers")

    # ----------------------------
    # Daily universe reshuffle
    # ----------------------------
    # Step 1: sample from entire universe
    if len(df_universe) > UNIVERSE_SAMPLE_SIZE:
        sampled_universe = np.random.choice(
            df_universe["Ticker"].values,
            size=UNIVERSE_SAMPLE_SIZE,
            replace=False
        ).tolist()
    else:
        sampled_universe = df_universe["Ticker"].tolist()

    # Step 2: apply liquidity and price filter
    filtered_universe = []
    for symbol in sampled_universe:
        if symbol not in price_data.columns:
            continue
        prices = price_data[symbol].dropna()
        if len(prices) < LOOKBACK:
            continue
        avg_volume = prices.tail(LOOKBACK).mean()
        last_price = prices.iloc[-1]
        if last_price >= MIN_PRICE and avg_volume >= MIN_ADV:
            filtered_universe.append(symbol)

    # Step 3: remove recently held/sold tickers
    active_universe = [
        s for s in filtered_universe
        if s not in recently_held and s not in recently_sold
    ]

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

        # Apply rotation score
        rotation = df_universe.loc[df_universe["Ticker"] == symbol, "rotation_score"].values
        if len(rotation) > 0:
            score *= rotation[0]

        scores[symbol] = score

    # ----------------------------
    # Rank universe
    # ----------------------------
    ranker_config = RankerConfig(top_n=TOP_N, bottom_n=BOTTOM_N)
    top_symbols, bottom_symbols = rank_universe(scores, ranker_config)

    # ----------------------------
    # Allocate capital & execute trades
    # ----------------------------
    alloc_config = AllocationConfig(max_positions=MAX_POSITIONS, target_weight=TARGET_WEIGHT)
    allocate_capital(
        portfolio,
        top_symbols,
        bottom_symbols,
        today_prices.to_dict(),
        alloc_config
    )

    # ----------------------------
    # Update recently held / sold and rotation scores
    # ----------------------------
    for sym in top_symbols:
        if sym not in recently_held:
            recently_held.append(sym)
        df_universe.loc[df_universe["Ticker"] == sym, "rotation_score"] *= 0.5

    for sym in bottom_symbols:
        if sym not in recently_sold:
            recently_sold.append(sym)

    # Gradually increase rotation scores to encourage exploration
    df_universe["rotation_score"] *= 1.01
    df_universe["rotation_score"] = df_universe["rotation_score"].clip(0.1, 5.0)

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

