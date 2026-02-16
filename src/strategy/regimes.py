# src/regimes.py
import pandas as pd

def classify_regime(price_series, window=20, vol_thresh=0.02):
    """
    Simple regime classifier:
    - Trend: high volatility with directional bias
    - Range: low volatility, no directional bias
    """
    returns = price_series.pct_change().fillna(0)
    vol = returns.rolling(window).std()
    trend = returns.rolling(window).mean()

    regimes = []
    for i in range(len(price_series)):
        if vol[i] > vol_thresh and abs(trend[i]) > 0:
            regimes.append("high_vol_trend")
        elif vol[i] <= vol_thresh:
            regimes.append("low_vol_range")
        else:
            regimes.append("mid_vol_trend")
    return pd.Series(regimes, index=price_series.index)

