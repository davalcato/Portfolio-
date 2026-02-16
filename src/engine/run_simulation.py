# src/engine/run_simulation.py

import os
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from src.data.data_loader import load_universe, load_universe_prices
from src.strategy.signals import generate_signals

# ✅ FIXED CONFIG IMPORT
from src.config.config import (
    INITIAL_CAPITAL,
    RANDOM_SEED,
    LOOKBACK,
    BUY_ZSCORE,
    SELL_ZSCORE,
    MAX_POSITION_PCT,
    TRANSACTION_COST,
)

# ---------------------------
# Set random seed
# ---------------------------
if RANDOM_SEED is not None:
    np.random.seed(RANDOM_SEED)

# ---------------------------
# Simulation parameters
# ---------------------------
START_DATE = "2023-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
ADV_THRESHOLD = 10_000_000  # Minimum Average Daily Volume for tradable tickers
DAILY_RESHUFFLE = True      # Whether to reshuffle universe daily

# ---------------------------
# Utility Functions
# ---------------------------
def compute_adv(price_df):
    """
    Compute approximate ADV (here using volume as proxy).
    Expects price_df to contain 'Volume' multi-level columns if downloaded from yfinance.
    """
    if "Volume" not in price_df.columns:
        return pd.Series(np.ones(price_df.shape[1]), index=price_df.columns)
    return price_df["Volume"].mean()

# ---------------------------
# Main Simulation Function
# ---------------------------
def run_simulation():
    print("Loading master universe...")
    df_universe = load_universe()
    print(f"Loaded universe size: {len(df_universe)} tickers")

    print("Downloading price data...")
    price_data = load_universe_prices(
        df_universe,
        start=START_DATE,
        end=END_DATE,
        parallel=True,
    )

    if price_data.empty:
        print("No price data available. Exiting.")
        return

    print(f"Price data loaded: {price_data.shape[1]} tickers, {price_data.shape[0]} days")

    # Filter by ADV
    print("Computing ADV and filtering...")
    adv_series = price_data.rolling(window=20).mean().mean()
    tradable_tickers = adv_series[adv_series > ADV_THRESHOLD].index.tolist()
    price_data = price_data[tradable_tickers]

    print(f"Tradable universe after ADV filter: {len(price_data.columns)} tickers")

    # Initialize portfolio
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []

    # Daily simulation loop
    print("Running simulation...")
    for current_day in tqdm(price_data.index, desc="Simulating Days"):

        if DAILY_RESHUFFLE:
            tickers_today = np.random.permutation(price_data.columns)
        else:
            tickers_today = price_data.columns

        prices_today = price_data.loc[current_day, tickers_today]

        # Generate signals (BUY, SELL, HOLD)
        signals = generate_signals(
            prices_today,
            lookback=LOOKBACK,
            buy_zscore=BUY_ZSCORE,
            sell_zscore=SELL_ZSCORE,
        )

        # Apply trades
        for ticker, signal in signals.items():
            price = prices_today[ticker]

            if signal == "BUY":
                max_position = int((cash * MAX_POSITION_PCT) // price)

                if max_position > 0:
                    positions[ticker] = positions.get(ticker, 0) + max_position
                    cash -= max_position * price * (1 + TRANSACTION_COST)
                    trades.append((ticker, "BUY", max_position, price))

            elif signal == "SELL" and positions.get(ticker, 0) > 0:
                qty = positions[ticker]
                cash += qty * price * (1 - TRANSACTION_COST)
                trades.append((ticker, "SELL", qty, price))
                positions[ticker] = 0

    # Portfolio summary
    total_equity = cash + sum(
        price_data.iloc[-1][t] * q
        for t, q in positions.items()
        if t in price_data.columns
    )

    print("\n--- FINAL PORTFOLIO ---")
    print(f"Cash: {cash:.2f}")
    print(f"Open Positions: {positions}")
    print(f"Total Equity: {total_equity:.2f}")

    print("\n--- TRADES ---")
    for trade in trades:
        print(trade)


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    run_simulation()

