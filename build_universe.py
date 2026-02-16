import pandas as pd
import yfinance as yf

# -------------------------------------------------
# OPTION 1: Pull IWV (Russell 3000 ETF) holdings
# -------------------------------------------------

url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"

df = pd.read_csv(url, skiprows=9)

tickers = df["Ticker"].dropna().unique()

universe_df = pd.DataFrame({"Ticker": tickers})

universe_df.to_csv("src/universe.csv", index=False)

print(f"Saved {len(universe_df)} tickers to src/universe.csv")

