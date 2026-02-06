"""
Canonical simulation entry point.

Run with:
    python3 -m src.run_simulation
"""

import numpy as np
import pandas as pd

from src.config import (
    INITIAL_CAPITAL,
    RANDOM_SEED,
    MAX_POSITION_PCT,
    BUY_ZSCORE,
    SELL_ZSCORE,
    LOOKBACK,
    TRANSACTION_COST
)

from src.data_loader import load_universe_prices
from src.portfolio import Portfolio, compute_metrics


def run_simulation():
    np.random.seed(RANDOM_SEED)

    # ----------------------------
    # Load price data
    # ----------------------------
    price_data = load_universe_prices()
    portfolio = Portfolio(INITIAL_CAPITAL)

    equity_curve = []
    peak_equity = INITIAL_CAPITAL

    # ----------------------------
    # Simulation loop
    # ----------------------------
    for day in range(LOOKBACK, len(price_data)):
        today_prices = price_data.iloc[day]
        history = price_data.iloc[day - LOOKBACK : day]

        print(f"\n=== DAY {day} ({price_data.index[day].date()}) ===")

        for symbol in price_data.columns:
            price = today_prices[symbol]
            hist = history[symbol].dropna()

            if len(hist) < LOOKBACK:
                continue

            mean = hist.mean()
            std = hist.std()

            if std == 0 or price <= 0:
                continue

            z = (price - mean) / std

            if z < BUY_ZSCORE:
                signal = "BUY"
            elif z > SELL_ZSCORE:
                signal = "SELL"
            else:
                signal = "HOLD"

            portfolio.execute(
                symbol=symbol,
                price=price,
                signal=signal,
                transaction_cost=TRANSACTION_COST
            )

            print(f"{symbol}: price={price:.2f}, z={z:.2f}, signal={signal}")

        equity = portfolio.total_equity(today_prices.to_dict())
        equity_curve.append(equity)

        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity * 100

        print(f"Total Equity: {equity:.2f}")
        print(f"Drawdown: {drawdown:.2f}%")

    # ----------------------------
    # Metrics
    # ----------------------------
    equity_series = pd.Series(equity_curve)
    metrics = compute_metrics(equity_series)

    print("\n--- Metrics ---")
    print(metrics)


if __name__ == "__main__":
    run_simulation()

