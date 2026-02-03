# =========================
# src/portfolio.py
# Portfolio management and metrics module
# =========================

from typing import Dict, List
import pandas as pd
import numpy as np
from src.config import MAX_POSITION_PCT

# ----------------------------
# Portfolio Class
# ----------------------------
class Portfolio:
    """
    Portfolio class to track cash, positions, equity curve, and trades.
    """

    def __init__(self, capital: float):
        self.cash: float = capital
        self.positions: Dict[str, int] = {}
        self.equity_curve: List[float] = []
        self.trade_log: List[List] = []  # Each trade: [Ticker, Side, Qty, Price]

    def total_equity(self, current_prices: dict) -> float:
        """
        Compute total equity = cash + market value of positions
        """
        equity = self.cash
        for symbol, shares in self.positions.items():
            equity += shares * current_prices.get(symbol, 0)
        return equity

    def execute(self, symbol: str, price: float, signal: str, debug: bool = True) -> None:
        """
        Execute a trade based on the signal.

        Parameters:
        - symbol: ticker symbol
        - price: current price
        - signal: "BUY", "SELL", "HOLD"
        - debug: if True, prints trade info
        """
        if price <= 0:
            return

        # Determine maximum position allocation
        current_positions_value = sum(shares * price for shares in self.positions.values())
        max_position_value = MAX_POSITION_PCT * (self.cash + current_positions_value)

        # ----------------------------
        # BUY signal
        # ----------------------------
        if signal.upper() == "BUY":
            allocation = min(self.cash, max_position_value)
            shares_to_buy = int(allocation // price)
            if shares_to_buy > 0:
                self.cash -= shares_to_buy * price
                self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                self.trade_log.append([symbol, "BUY", shares_to_buy, price])
                if debug:
                    print(f"DEBUG: Bought {shares_to_buy} of {symbol} at {price:.2f}")

        # ----------------------------
        # SELL signal
        # ----------------------------
        elif signal.upper() == "SELL" and symbol in self.positions:
            shares_to_sell = self.positions.pop(symbol)
            self.cash += shares_to_sell * price
            self.trade_log.append([symbol, "SELL", shares_to_sell, price])
            if debug:
                print(f"DEBUG: Sold {shares_to_sell} of {symbol} at {price:.2f}")


# ----------------------------
# Compute Metrics
# ----------------------------
def compute_metrics(equity_curve: pd.Series) -> dict:
    """
    Compute portfolio metrics: Sharpe ratio and max drawdown.
    """
    returns = equity_curve.pct_change().dropna()
    sharpe = (np.sqrt(252) * returns.mean() / returns.std()) if returns.std() > 0 else 0
    drawdown = (equity_curve.cummax() - equity_curve) / equity_curve
    max_dd = drawdown.max() if len(drawdown) > 0 else 0

    return {
        "Sharpe": round(sharpe, 2),
        "MaxDrawdown": round(max_dd, 2)
    }

