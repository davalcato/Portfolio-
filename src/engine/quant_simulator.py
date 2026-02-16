import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =========================
# CONFIG
# =========================
STARTING_CASH = 272.0
cash = STARTING_CASH
positions = {}
trade_log = []

HIST_DAYS = 60
FORECAST_DAYS = 60
MC_SIMULATIONS = 1000
TRANSACTION_COST = 0.001  # 10 bps

# =========================
# TICKER UNIVERSE
# =========================
RAW_TICKERS = [
    "BURU","CRBP","KITT","SRRK","RIO","LMND",
    "RKLB","OKLO","DRUG","SOXL","RGTI",
    "FJET","IBIO","RR","AYB.BE","CRWV","SLV","LB"
]

TICKERS = sorted(set(RAW_TICKERS))

print("\nLoaded tickers:")
print(TICKERS)

# =========================
# DOWNLOAD HISTORICAL DATA
# =========================
data = yf.download(
    TICKERS,
    period=f"{HIST_DAYS}d",
    group_by="ticker",
    auto_adjust=True,
    progress=False
)

# =========================
# MONTE CARLO FORECAST
# =========================
def monte_carlo_forecast(price_series, days, simulations):
    returns = price_series.pct_change().dropna()

    if len(returns) < 5:
        raise ValueError("Insufficient return history")

    mu = returns.mean()
    sigma = returns.std()

    rand = np.random.normal(mu, sigma, (days, simulations))
    paths = np.zeros_like(rand)
    paths[0] = price_series.iloc[-1] * (1 + rand[0])

    for i in range(1, days):
        paths[i] = paths[i - 1] * (1 + rand[i])

    return pd.DataFrame(paths)

# =========================
# SIGNAL FUNCTION
# =========================
def momentum_signal(series):
    if len(series) < 20:
        return "HOLD"

    fast = series.rolling(5).mean().iloc[-1]
    slow = series.rolling(20).mean().iloc[-1]

    if fast > slow * 1.01:
        return "BUY"
    elif fast < slow * 0.99:
        return "SELL"
    return "HOLD"

# =========================
# BUILD EXTENDED PRICE SERIES
# =========================
extended_prices = {}
dropped = {}

for t in TICKERS:
    try:
        if t not in data or data[t].empty:
            raise ValueError("No data returned")

        close = data[t]["Close"].dropna()

        if len(close) < 20:
            raise ValueError("Not enough price history")

        mc = monte_carlo_forecast(close, FORECAST_DAYS, MC_SIMULATIONS)
        forecast = mc.median(axis=1)

        future_dates = pd.bdate_range(
            start=close.index[-1] + pd.Timedelta(days=1),
            periods=FORECAST_DAYS
        )
        forecast.index = future_dates

        series = pd.concat([close, forecast])
        extended_prices[t] = series

    except Exception as e:
        dropped[t] = str(e)

# =========================
# REPORT DROPPED TICKERS
# =========================
print("\nDropped tickers:")
for k, v in dropped.items():
    print(f"{k}: {v}")

print("\nValid tickers:")
print(list(extended_prices.keys()))

# =========================
# SIMULATION LOOP
# =========================
dates = next(iter(extended_prices.values())).index
peak_equity = STARTING_CASH

for day, date in enumerate(dates):
    print(f"\n=== DAY {day + 1} ({date.date()}) ===")

    for t, series in extended_prices.items():
        price_series = series.iloc[:day + 1]
        price = price_series.iloc[-1]
        signal = momentum_signal(price_series)

        print(f"{t}: price={price:.2f}, signal={signal}")

        # BUY
        if signal == "BUY" and cash > price:
            qty = int(cash // price)
            if qty > 0:
                cost = qty * price * (1 + TRANSACTION_COST)
                cash -= cost
                positions[t] = positions.get(t, 0) + qty
                trade_log.append((date, t, "BUY", price, qty))

        # SELL
        elif signal == "SELL" and positions.get(t, 0) > 0:
            qty = positions[t]
            proceeds = qty * price * (1 - TRANSACTION_COST)
            cash += proceeds
            positions[t] = 0
            trade_log.append((date, t, "SELL", price, qty))

    equity = cash + sum(
        positions.get(t, 0) * extended_prices[t].iloc[day]
        for t in positions
    )

    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity * 100

    print(f"Total Equity: {equity:.2f}")
    print(f"Drawdown: {drawdown:.2f}%")

# =========================
# FINAL SUMMARY
# =========================
final_equity = cash + sum(
    positions.get(t, 0) * extended_prices[t].iloc[-1]
    for t in positions
)

print("\n--- FINAL PORTFOLIO ---")
print("Cash:", round(cash, 2))
print("Positions:", positions)
print("Total Equity:", round(final_equity, 2))

