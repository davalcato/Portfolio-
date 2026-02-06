import numpy as np

def zscore_signal(series, buy_z, sell_z):
mean = series.mean()
std = series.std()

if std == 0:
return "HOLD"

z = (series.iloc[-1] - mean) / std

if z < buy_z:
return "BUY"
elif z > sell_z:
return "SELL"
return "HOLD"
