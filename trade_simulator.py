import pandas as pd
import numpy as np
import yfinance as yf

# =========================
# CONFIG
# =========================
STARTING_CASH = 272.0
cash = STARTING_CASH

HIST_DAYS = 90
FORECAST_DAYS = 60
MC_SIMULATIONS = 800

LOOKBACK = 30          # signal window
LOW_PCTL = 20          # buy zone
HIGH_PCTL = 80         # sell zone

TRANSACTION_COST = 0.001  # 10 bps

positions = {}
trade_log = []

# =========================
# LOAD TICKERS
# =========================
try:
    tickers = pd.read_csv("universe.csv")["Ticker"].dropna().tolist()
except FileNotFoundError:
    tickers = [
        "BURU","CRBP","KITT","SRRK","RIO","LMND","RKLB","OKLO",
        "DRUG","SOXL","RGTI","FJET","IBIO","RR","AYB.BE"
    ]

print("\nLoaded tickers:", tickers)

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download(
    tickers,
    period=f"{HIST_DAYS}d",
    auto_adjust=True,
    group_by="ticker",
    progress=False
)

# =========================
# MONTE CARLO (LOG-NORMAL, NO NEGATIVE PRICES)
# =========================
def monte_carlo_paths(series, days, sims):
    log_returns = np.log(series / series.shift(1)).dropna()
    mu = log_returns.mean()
    sigma = log_returns.std()

    shocks = np.random.normal(mu, sigma, (days, sims))

    paths = np.zeros((days, sims))
    paths[0] = series.iloc[-1] * np.exp(shocks[0])

    for i in range(1, days):
        paths[i] = paths[i - 1] * np.exp(shocks[i])

    return pd.DataFrame(paths)

# =========================
# SIGNAL ENGINE (PERCENTILE BASED)
# =========================
def signal_engine(price_series, position):
    if len(price_series) < LOOKBACK:
        return "HOLD"

    window = price_series.iloc[-LOOKBACK:]
    current_price = window.iloc[-1]

    p_low = np.percentile(window, LOW_PCTL)
    p_high = np.percentile(window, HIGH_PCTL)

    # BUY: statistically cheap
    if current_price <= p_low and position == 0:
        return "BUY"

    # SELL: statistically expensive
    if current_price >= p_high and position > 0:
        return "SELL"

    # Forced exit to avoid infinite holds
    if position > 0:
        return "SELL"

    return "HOLD"

# =========================
# BUILD PRICE STRUCTURES
# =========================
historical_prices = {}
forecast_prices = {}

for t in tickers:
    try:
        close = data[t]["Close"].dropna()
        historical_prices[t] = close
        positions[t] = 0

        mc = monte_carlo_paths(close, FORECAST_DAYS, MC_SIMULATIONS)
        forecast_prices[t] = mc.mean(axis=1)

    except Exception as e:
        print(f"Skipping {t}: {e}")

# =========================
# SIMULATION LOOP
# =========================
peak_equity = STARTING_CASH
total_days = HIST_DAYS + FORECAST_DAYS

for day in range(total_days):
    label = "HIST" if day < HIST_DAYS else "FORECAST"
    print(f"\n=== {label} DAY {day + 1} ===")

    for t in tickers:
        hist_series = historical_prices[t]

        # Clamp signal data to historical only
        signal_series = hist_series.iloc[:min(day + 1, HIST_DAYS)]

        # Price used for execution
        if day < HIST_DAYS:
            price = signal_series.iloc[-1]
        else:
            price = forecast_prices[t].iloc[day - HIST_DAYS]

        position = positions[t]
        signal = signal_engine(signal_series, position)

        print(f"{t}: price={price:.2f}, signal={signal}")

        # BUY
        if signal == "BUY" and cash > price:
            qty = int(cash // price)
            if qty > 0:
                cost = qty * price * (1 + TRANSACTION_COST)
                cash -= cost
                positions[t] += qty

                trade_log.append((t, "BUY", qty, price))

        # SELL
        elif signal == "SELL" and position > 0:
            proceeds = position * price * (1 - TRANSACTION_COST)
            cash += proceeds
            positions[t] = 0

            trade_log.append((t, "SELL", position, price))

    equity = cash + sum(
        positions[t] * (
            historical_prices[t].iloc[-1]
            if day < HIST_DAYS
            else forecast_prices[t].iloc[day - HIST_DAYS]
        )
        for t in tickers
    )

    peak_equity = max(peak_equity, equity)
    drawdown = (peak_equity - equity) / peak_equity

    print(f"Total Equity: {equity:.2f}")
    print(f"Drawdown: {drawdown:.2%}")

# =========================
# FINAL REPORT
# =========================
print("\n--- FINAL PORTFOLIO ---")
print("Cash:", round(cash, 2))
print("Open Positions:", {k: v for k, v in positions.items() if v > 0})

print("\n--- TRADES ---")
for trade in trade_log[-20:]:
    print(trade)

