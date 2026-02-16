import pandas as pd
from datetime import datetime

UNIVERSE_PATH = "universe.csv"

# -------------------------------------------------------------------
# Load universe
# -------------------------------------------------------------------
def load_universe(path=UNIVERSE_PATH, as_of: str = None) -> pd.DataFrame:
    """
    Load raw universe snapshot, optionally filtering by date.
    """
    df = pd.read_csv(path)

    # Ensure proper datetime column
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        if as_of:
            df = df[df["date"] <= pd.to_datetime(as_of)]

    # Fill missing columns for robustness
    for col in ["price", "adv", "delisted"]:
        if col not in df.columns:
            df[col] = 0 if col != "delisted" else False

    return df

# -------------------------------------------------------------------
# Filter universe
# -------------------------------------------------------------------
def filter_universe(
    df: pd.DataFrame,
    min_price: float = 5.0,
    min_adv: int = 1_000_000,
    exclude_delisted: bool = True
) -> list[str]:
    """
    Apply liquidity, price, and survivorship constraints.

    Parameters
    ----------
    df : DataFrame
        Raw universe snapshot
    min_price : float
        Minimum price threshold
    min_adv : int
        Minimum average daily volume
    exclude_delisted : bool
        Remove tickers flagged as delisted

    Returns
    -------
    list[str]
        Filtered tickers
    """
    filtered = df[
        (df["price"] >= min_price) &
        (df["adv"] >= min_adv)
    ]

    if exclude_delisted and "delisted" in df.columns:
        filtered = filtered[~filtered["delisted"]]

    # Remove duplicates and sort for determinism
    tickers = sorted(filtered["ticker"].dropna().unique().tolist())

    return tickers

# -------------------------------------------------------------------
# Optional: Dynamic universe refresh (future)
# -------------------------------------------------------------------
def refresh_universe(df: pd.DataFrame, add_new: list[str] = None, remove: list[str] = None) -> list[str]:
    """
    Update universe dynamically by adding/removing tickers.
    """
    tickers = set(df["ticker"].dropna().tolist())

    if add_new:
        tickers.update(add_new)

    if remove:
        tickers.difference_update(remove)

    return sorted(tickers)

