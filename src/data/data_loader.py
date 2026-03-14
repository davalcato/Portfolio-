# src/data/data_loader.py

import os
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------------------------------
# Load Universe (ROBUST CSV HANDLING)
# --------------------------------------------------
def load_universe(path="universe.csv"):
    """
    Load ticker universe from CSV safely.

    Supports:
    - CSV with header (Ticker / Symbol)
    - CSV with no header
    - CSV with multiple columns
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")

    df = pd.read_csv(path)

    possible_cols = ["Ticker", "ticker", "Symbol", "symbol"]

    ticker_col = None

    for col in possible_cols:
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        ticker_col = df.columns[0]

    tickers = (
        df[ticker_col]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
    )

    return pd.DataFrame({"Ticker": tickers})


# --------------------------------------------------
# Safe ticker downloader
# --------------------------------------------------
def download_ticker_data(ticker, start, end):

    try:
        data = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )

        # Skip bad downloads
        if data is None or data.empty:
            return None

        if "Close" not in data.columns:
            return None

        series = data["Close"]

        # Ensure we return a pandas Series
        if not isinstance(series, pd.Series):
            return None

        return ticker, series

    except Exception:
        return None


# --------------------------------------------------
# Load price data for entire universe
# --------------------------------------------------
def load_universe_prices(df_universe, start, end, parallel=True):

    tickers = df_universe["Ticker"].tolist()

    price_data = {}

    if parallel:

        with ThreadPoolExecutor(max_workers=10) as executor:

            futures = {
                executor.submit(download_ticker_data, ticker, start, end): ticker
                for ticker in tickers
            }

            for future in as_completed(futures):

                result = future.result()

                if result is None:
                    continue

                ticker, series = result

                price_data[ticker] = series

    else:

        for ticker in tickers:

            result = download_ticker_data(ticker, start, end)

            if result is None:
                continue

            ticker, series = result

            price_data[ticker] = series

    # --------------------------------------------------
    # Handle case where no tickers downloaded
    # --------------------------------------------------

    if not price_data:
        print("No valid price data downloaded.")
        return pd.DataFrame()

    print(f"Downloaded price series for {len(price_data)} tickers")

    # --------------------------------------------------
    # Correct way to build price matrix
    # --------------------------------------------------

    df = pd.concat(price_data, axis=1)

    df.columns = price_data.keys()

    return df.dropna(axis=1, how="all")
