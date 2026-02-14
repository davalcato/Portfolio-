import os
import pandas as pd
import yfinance as yf
import numpy as np
import time


# ---------------------------------
# Utility: Normalize Yahoo Tickers
# ---------------------------------
def normalize_ticker(ticker: str) -> str:
    """
    Convert class share formats to Yahoo-compatible format.
    Example:
        BRKB -> BRK-B
        BF.B -> BF-B
    """
    ticker = ticker.strip().upper()

    # Handle common class share cases
    if "." in ticker:
        ticker = ticker.replace(".", "-")

    if ticker.endswith("B") and len(ticker) > 4 and "-" not in ticker:
        # Special case for Berkshire
        if ticker == "BRKB":
            ticker = "BRK-B"

    return ticker


# ---------------------------------
# Load Universe CSV
# ---------------------------------
def load_universe(csv_path="data/universe.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Universe CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    if len(df.columns) == 1:
        df.columns = ["ticker"]

    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column")

    df["ticker"] = df["ticker"].astype(str).apply(normalize_ticker)

    df = df.drop_duplicates(subset="ticker").reset_index(drop=True)

    print(f"Loaded universe size: {len(df)} tickers")
    return df


# ---------------------------------
# Batch Download Prices
# ---------------------------------
def load_universe_prices(
    tickers,
    period="1y",
    interval="1d",
    batch_size=100,
    retry_attempts=2
):
    """
    Stable batch downloader to avoid Yahoo crumb errors.
    """

    tickers = [normalize_ticker(t) for t in tickers]
    all_prices = []

    print(f"Downloading {len(tickers)} tickers in batches of {batch_size}...")

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]

        for attempt in range(retry_attempts):
            try:
                data = yf.download(
                    batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )

                if data.empty:
                    raise ValueError("Empty batch download")

                if len(batch) == 1:
                    # Single ticker case
                    df = data[["Close"]].rename(columns={"Close": batch[0]})
                else:
                    df = pd.concat(
                        [data[t]["Close"] for t in batch if t in data],
                        axis=1
                    )
                    df.columns = [t for t in batch if t in data]

                all_prices.append(df)
                print(f"Batch {i//batch_size + 1} downloaded ({len(batch)} tickers)")
                break

            except Exception as e:
                print(f"Retry {attempt+1} failed for batch {i//batch_size + 1}: {e}")
                time.sleep(2)

        time.sleep(1)  # small delay to avoid rate limiting

    if not all_prices:
        raise ValueError("No price data could be downloaded.")

    price_df = pd.concat(all_prices, axis=1)

    # Remove columns with insufficient data
    min_obs = 50
    price_df = price_df.dropna(axis=1, thresh=min_obs)

    print(f"Final price universe size: {price_df.shape[1]} tickers")

    return price_df

