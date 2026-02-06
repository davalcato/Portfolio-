import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# =========================
# CONFIG
# =========================
STARTING_CASH = 272.0
HIST_DAYS = 120
FORECAST_DAYS = 60
MAX_POSITION_PCT = 0.25      # 25% of equity per position
TRANSACTION_COST = 0.001    # 10 bps

# =========================
# UNIVERSE
# =========================
try:
    TICKERS = (
        pd.read_csv("universe.csv")["Ticker"]
        .dropna()
        .astype(str)
        .tolist()
    )
except FileNotFoundError:
    TICKERS = [
        "BURU","CRBP","KITT","SRRK","RIO","LMND",
        "RKLB","OKLO","DRUG","SOXL","RGTI",
        "FJET","IBIO","RR","AYB.BE"
    ]

print("Universe:", TICKERS)

# =========================
# LOAD PRICE DATA
# =========================
prices = yf.download(
    TICKERS,
    period=f"{HIST_DAYS + FORECAST_DAYS}d",
    auto_adjust=True,
    progress=False
)["Close"]

prices = prices.dropna(how="all")

# =========================
# SIGNAL ENGINE (KEY FIX)
# =========================
def momentum_signal(series):
    """
    Nathan-style signal:
    - BUY only on bullish crossover
    - SELL only on bearish crossover
    - HOLD otherwise
    """
    if len(series) < 30:
        return "HOLD"

    fast_prev = series.rolling(10).mean().iloc[-2]
    slow_prev = series.rolling(30).mean().iloc[-2]

    fast_now = series.rolling(10).mean().iloc[-1]
    slow_now = series.rolling(30).mean().iloc[-1]

    if fast_prev <= slow_prev and fast_now > slow_now:
        return "BUY"

    if fast_prev >= slow_prev and fast_now < slow_now:
        return "SELL"

    return "HOLD"

# =========================
# PORTFOLIO ENGINE
# =========================
cash = STARTING_CASH
positions = {t: 0 for t in TICKERS}
equity_curve = []
peak_equity = STARTING_CASH
trade_log = []

# =========================
# SIMULATION LOOP
# =========================
for date in prices.index:
    daily_prices = prices.loc[:date]

    equity = cash + sum(
        positions[t] * daily_prices[t].iloc[-1]
        for t in TICKERS if not np.isnan(daily_prices[t].iloc[-1])
    )

    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity
    equity_curve.append(equity)

    print(f"\n=== {date.date()} ===")

    for t in TICKERS:
        if t not in daily_prices or np.isnan(daily_prices[t].iloc[-1]):
            continue

        price_series = daily_prices[t].dropna()
        price = price_series.iloc[-1]
        signal = momentum_signal(price_series)

        print(f"{t}: price={price:.2f}, signal={signal}")

        # ================= BUY =================
        if signal == "BUY" and positions[t] == 0:
            max_alloc = equity * MAX_POSITION_PCT
            qty = int(max_alloc // price)

            if qty > 0 and cash >= qty * price:
                cost = qty * price * (1 + TRANSACTION_COST)
                cash -= cost
                positions[t] += qty

                trade_log.append((date, t, "BUY", qty, price))

        # ================= SELL =================
        elif signal == "SELL" and positions[t] > 0:
            qty = positions[t]
            proceeds = qty * price * (1 - TRANSACTION_COST)
            cash += proceeds
            positions[t] = 0

            trade_log.append((date, t, "SELL", qty, price))

    print(f"Equity: {equity:.2f} | Drawdown: {drawdown:.2%}")

# =========================
# FINAL REPORT
# =========================
final_equity = equity_curve[-1]

print("\n====== FINAL PORTFOLIO ======")
print("Final Cash:", round(cash, 2))
print("Positions:", {k:v for k,v in positions.items() if v > 0})
print("Total Equity:", round(final_equity, 2))
print("Max Drawdown:", f"{max((peak_equity - e)/peak_equity for e in equity_curve):.2%}")
print("Total Trades:", len(trade_log))

