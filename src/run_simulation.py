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

# ---- NEW imports ----
from src.risk import RiskManager
from src.beliefs import ALL_BELIEFS
from src.regimes import classify_regime


def run_simulation():
    np.random.seed(RANDOM_SEED)

    # ----------------------------
    # Load price data
    # ----------------------------
    price_data = load_universe_prices()
    portfolio = Portfolio(INITIAL_CAPITAL)
    risk_mgr = RiskManager(max_drawdown=0.2, target_volatility=0.02)

    equity_curve = []
    peak_equity = INITIAL_CAPITAL

    # ----------------------------
    # Classify regimes
    # ----------------------------
    regimes_df = pd.DataFrame(index=price_data.index, columns=price_data.columns)
    for symbol in price_data.columns:
        regimes_df[symbol] = classify_regime(price_data[symbol])

    # ----------------------------
    # Simulation loop
    # ----------------------------
    for day in range(LOOKBACK, len(price_data)):
        today_prices = price_data.iloc[day]
        history = price_data.iloc[day - LOOKBACK: day]

        print(f"\n=== DAY {day} ({price_data.index[day].date()}) ===")

        for symbol in price_data.columns:
            price = today_prices[symbol]
            hist = history[symbol].dropna()

            if len(hist) < LOOKBACK or price <= 0:
                continue

            # Calculate z-score (existing logic)
            mean = hist.mean()
            std = hist.std()
            if std == 0:
                continue

            z = (price - mean) / std

            # ----------------------------
            # Check beliefs and regimes
            # ----------------------------
            today_regime = regimes_df[symbol].iloc[day]
            signal = "HOLD"  # default

            for belief in ALL_BELIEFS:
                if belief.name.lower() in symbol.lower():  # match signal to symbol convention
                    if today_regime in belief.regimes:
                        if z < BUY_ZSCORE:
                            signal = "BUY"
                        elif z > SELL_ZSCORE:
                            signal = "SELL"
                        # else HOLD
                    else:
                        signal = "HOLD"  # skip if regime mismatch

            # ----------------------------
            # Scale position by risk manager
            # ----------------------------
            hist_returns = hist.pct_change().dropna()
            hist_vol = hist_returns.std() if len(hist_returns) > 0 else 0
            pos_size = risk_mgr.scale_position(1.0, hist_vol)  # scale weight 1.0

            portfolio.execute(
                symbol=symbol,
                price=price,
                signal=signal,
                transaction_cost=TRANSACTION_COST,
                position_size=pos_size
            )

            print(f"{symbol}: price={price:.2f}, z={z:.2f}, regime={today_regime}, signal={signal}, pos_size={pos_size:.2f}")

        # Update equity curve
        equity = portfolio.total_equity(today_prices.to_dict())
        equity_curve.append(equity)

        # Track drawdown
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity * 100

        print(f"Total Equity: {equity:.2f}")
        print(f"Drawdown: {drawdown:.2f}%")

    # ----------------------------
    # Metrics
    # ----------------------------
    equity_series = pd.Series(equity_curve, index=price_data.index[LOOKBACK:])
    metrics = compute_metrics(equity_series)

    print("\n--- Metrics ---")
    print(metrics)


if __name__ == "__main__":
    run_simulation()

