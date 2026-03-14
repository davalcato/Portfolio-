# src/engine/run_simulation.py

import pandas as pd
from pathlib import Path
from datetime import datetime
from src.data.data_loader import load_universe_prices


def load_universe(universe_path: str) -> pd.DataFrame:
    """
    Load and normalize the universe CSV safely.
    Supports various column names such as ticker/Ticker/Symbol.
    """

    if not Path(universe_path).exists():
        raise FileNotFoundError(f"Universe file not found: {universe_path}")

    df = pd.read_csv(universe_path)

    if df.empty:
        raise ValueError("Universe CSV is empty")

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    possible_cols = ["Ticker", "ticker", "symbol", "Symbol"]

    ticker_col = None
    for col in possible_cols:
        if col in df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        raise ValueError(
            f"No ticker column found. Columns detected: {list(df.columns)}"
        )

    df = df.rename(columns={ticker_col: "Ticker"})

    # Clean tickers
    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.replace("$", "", regex=False)
        .str.replace(".", "-", regex=False)
    )

    df = df[df["Ticker"] != ""]
    df = df.dropna(subset=["Ticker"])

    return df


def clean_universe(df_universe: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates and invalid tickers.
    """

    df_universe = df_universe.drop_duplicates(subset=["Ticker"])

    df_universe = df_universe[df_universe["Ticker"].str.len() > 0]

    return df_universe.reset_index(drop=True)


def run_simulation():

    print("\nLoading universe...")

    universe_path = "src/data/universe.csv"

    df_universe = load_universe(universe_path)

    print(f"Initial universe size: {len(df_universe)}")

    df_universe = clean_universe(df_universe)

    print(f"Valid universe size: {len(df_universe)}")

    if df_universe.empty:
        print("Universe contains no valid tickers.")
        return

    start_date = "2015-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")

    print("\nDownloading prices...")

    price_data = load_universe_prices(
        df_universe,
        start=start_date,
        end=end_date,
        parallel=True,
    )

    if price_data is None or price_data.empty:
        print("No price data available")
        return

    print("\nPrice data downloaded successfully")
    print(f"Price matrix shape: {price_data.shape}")

    print("\nFirst rows of price data:")
    print(price_data.head())

    print("\nSimulation complete")


if __name__ == "__main__":
    run_simulation()
