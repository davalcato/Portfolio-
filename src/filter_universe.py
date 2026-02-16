import pandas as pd

def filter_universe(
    df_universe,
    price_data=None,
    recently_held=None,
    recently_sold=None,
    min_price=5,
    min_adv=1_000_000
):
    """
    Filters a universe of tickers based on:
    - recently held / sold tickers
    - minimum price
    - average daily volume (ADV)
    
    Parameters:
        df_universe (pd.DataFrame): Must contain "Ticker" column
        price_data (pd.DataFrame): Historical price data with columns = tickers and 'Volume'
        recently_held (list): tickers to exclude temporarily
        recently_sold (list): tickers to exclude temporarily
        min_price (float): minimum stock price to trade
        min_adv (float): minimum average daily volume (shares traded)
    
    Returns:
        list of tickers passing filters
    """
    tickers = df_universe["Ticker"].tolist()
    
    if recently_held:
        tickers = [t for t in tickers if t not in recently_held]
    if recently_sold:
        tickers = [t for t in tickers if t not in recently_sold]

    # Skip tickers not in price_data
    if price_data is not None:
        tickers = [t for t in tickers if t in price_data.columns]

        # Compute average price & ADV over available period
        adv_filtered = []
        for t in tickers:
            series = price_data[t].dropna()
            if len(series) == 0:
                continue
            # Average closing price
            avg_price = series.mean()
            # If volume data is available
            if hasattr(price_data, 'Volume') and t in price_data['Volume'].columns:
                adv = price_data['Volume'][t].dropna().mean()
            else:
                # Fallback: skip ADV filter if no volume
                adv = min_adv + 1
            if avg_price >= min_price and adv >= min_adv:
                adv_filtered.append(t)
        tickers = adv_filtered

    return tickers

