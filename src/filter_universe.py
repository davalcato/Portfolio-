import pandas as pd
from datetime import datetime

UNIVERSE_PATH = "universe.csv"


def load_universe(path=UNIVERSE_PATH, as_of: str = None) -> pd.DataFrame:
    """
    Load raw universe snapshot, optionally filtering by date.
    """
    df = pd.read_csv(path)
    if as_of and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] <= pd.to_datetime(as_of)]
    return df


def filter_universe(
    df: pd.DataFrame,
    min_price: float = 5.0,
    max_price: float = 500.0,
    min_adv: float = 10_000_000,
    max_adv: float = 500_000_000,
    exclude_delisted: bool = True,
    recently_held: list[str] | None = None,
    recently_sold: list[str] | None = None,
    min_rotation_score: float = 0.0
) -> list[str]:
    """
    Filter universe by liquidity, price, delisting, recent holding, and rotation score.

    Parameters
    ----------
    df : pd.DataFrame
        Universe dataframe with at least ['ticker', 'price', 'adv'] columns
    min_price, max_price : float
        Price filters
    min_adv, max_adv : float
        Average daily volume filters
    exclude_delisted : bool
        Remove tickers flagged as delisted
    recently_held : list[str] | None
        Exclude symbols that were recently held
    recently_sold : list[str] | None
        Exclude symbols that were recently sold
    min_rotation_score : float
        Optional minimum rotation_score to include

    Returns
    -------
    list[str]
        Filtered tickers
    """

    # Drop NaNs in critical columns
    df = df.dropna(subset=["ticker", "price", "adv"])

    # Price & volume filter
    filtered = df[
        (df["price"] >= min_price) &
        (df["price"] <= max_price) &
        (df["adv"] >= min_adv) &
        (df["adv"] <= max_adv)
    ]

    # Remove delisted
    if exclude_delisted and "delisted" in df.columns:
        filtered = filtered[~filtered["delisted"]]

    # Remove recently held or sold
    if recently_held:
        filtered = filtered[~filtered["ticker"].isin(recently_held)]
    if recently_sold:
        filtered = filtered[~filtered["ticker"].isin(recently_sold)]

    # Rotation score filter
    if "rotation_score" in filtered.columns:
        filtered = filtered[filtered["rotation_score"] >= min_rotation_score]

    return filtered["ticker"].astype(str).tolist()

