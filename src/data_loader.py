import pandas as pd
import yfinance as yf


def load_universe_prices():
    """
    Load adjusted close prices for the ticker universe.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with tickers as columns.
    """

    # Load universe
    universe = pd.read_csv("universe.csv")["Ticker"].dropna().tolist()

    # Download price data
    data = yf.download(
        universe,
        period="1y",
        auto_adjust=True,
        progress=False
    )

    # Handle single vs multi-ticker cases
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data.to_frame(name=universe[0])

    # Drop assets with insufficient data
    prices = prices.dropna(axis=1, thresh=int(0.8 * len(prices)))

    return prices

