"""
Stable Russell 3000 Simulation
Uses batch-based Yahoo downloads
"""

import numpy as np
import pandas as pd
from src.data_loader import load_universe, load_universe_prices


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

LOOKBACK_DAYS = 20
DAILY_SAMPLE_SIZE = 50
SIMULATION_DAYS = 120


def compute_momentum(price_df):
    return (price_df.iloc[-1] - price_df.iloc[-LOOKBACK_DAYS]) / price_df.iloc[-LOOKBACK_DAYS]


def sample_daily_universe(price_df, n=DAILY_SAMPLE_SIZE):
    tickers = price_df.columns.tolist()
    if len(tickers) <= n:
        return tickers
    return list(np.random.choice(tickers, n, replace=False))


def run_simulation():

    # 1️⃣ Load Universe
    df_universe = load_universe()
    tickers = df_universe["ticker"].tolist()

    # 2️⃣ Download Prices (batch-safe)
    price_data = load_universe_prices(
        tickers,
        period="1y",
        interval="1d",
        batch_size=100
    )

    print(f"Price data columns: {len(price_data.columns)}")

    cash = 500.0
    positions = {}
    trades = []

    for day in range(LOOKBACK_DAYS, min(SIMULATION_DAYS, len(price_data))):

        daily_slice = price_data.iloc[:day + 1]
        active_tickers = sample_daily_universe(daily_slice)

        momentum_scores = compute_momentum(daily_slice[active_tickers])

        print(f"\n=== DAY {day} ===")

        for ticker in active_tickers:
            price = daily_slice[ticker].iloc[-1]
            score = momentum_scores[ticker]

            if score > 0.05:
                signal = "BUY"
            elif score < -0.05:
                signal = "SELL"
            else:
                signal = "HOLD"

            print(f"{ticker}: price={price:.2f}, signal={signal}")

            if signal == "BUY" and cash > price:
                qty = int(cash // price)
                positions[ticker] = positions.get(ticker, 0) + qty
                cash -= qty * price
                trades.append((ticker, "BUY", qty, price))

            elif signal == "SELL" and ticker in positions:
                qty = positions[ticker]
                cash += qty * price
                trades.append((ticker, "SELL", qty, price))
                positions[ticker] = 0

        equity = cash + sum(
            daily_slice[t].iloc[-1] * positions.get(t, 0)
            for t in active_tickers
        )

        drawdown = 1 - (equity / 500.0)

        print(f"Total Equity: {equity:.2f}")
        print(f"Drawdown: {drawdown*100:.2f}%")

    print("\n--- FINAL PORTFOLIO ---")
    print("Cash:", cash)
    print("Positions:", positions)
    print("\n--- TRADES ---")
    for trade in trades:
        print(trade)


if __name__ == "__main__":
    run_simulation()

