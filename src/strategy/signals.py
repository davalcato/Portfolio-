# src/signals.py

import pandas as pd
import numpy as np

def generate_signals(prices: pd.Series, lookback: int = 20, buy_zscore: float = -1.0, sell_zscore: float = 1.0):
    """
    Generate simple z-score based BUY, SELL, HOLD signals for a Series of prices.
    
    Parameters
    ----------
    prices : pd.Series
        Current day prices for tickers (index=ticker)
    lookback : int
        Lookback period for rolling mean/std
    buy_zscore : float
        Z-score threshold to BUY
    sell_zscore : float
        Z-score threshold to SELL
    
    Returns
    -------
    signals : dict
        Dictionary of ticker -> signal ('BUY', 'SELL', 'HOLD')
    """
    signals = dict()
    
    for ticker in prices.index:
        # For a single-ticker series, you would normally use historical price
        # Here we'll assume prices is a pd.Series of length lookback (mocking for now)
        # In real system, replace with historical series of that ticker
        price_history = prices[ticker:ticker+lookback]  # placeholder
        
        mean = price_history.mean()
        std = price_history.std()
        if std == 0:
            signals[ticker] = "HOLD"
            continue
        
        zscore = (prices[ticker] - mean) / std
        if zscore <= buy_zscore:
            signals[ticker] = "BUY"
        elif zscore >= sell_zscore:
            signals[ticker] = "SELL"
        else:
            signals[ticker] = "HOLD"
            
    return signals

