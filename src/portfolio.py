import numpy as np
import pandas as pd


class Portfolio:
    """
    Manages portfolio state, execution, and equity tracking.
    """

    def __init__(self, initial_capital, max_position_pct=0.10, transaction_cost=0.001):
        self.cash = initial_capital
        self.positions = {}
        self.max_position_pct = max_position_pct
        self.transaction_cost = transaction_cost
        self.trade_log = []

    def total_equity(self, prices):
        equity = self.cash
        for symbol, shares in self.positions.items():
            equity += shares * prices.get(symbol, 0)
        return equity

    def execute(self, symbol, price, signal, transaction_cost=None):
        if price <= 0:
            return

        tc = transaction_cost if transaction_cost is not None else self.transaction_cost

        # Portfolio value approximation
        portfolio_value = self.cash + sum(
            shares * price for shares in self.positions.values()
        )

        max_position_value = portfolio_value * self.max_position_pct

        # BUY logic
        if signal == "BUY":
            allocation = min(self.cash, max_position_value)
            shares = int(allocation // price)

            if shares > 0:
                cost = shares * price * (1 + tc)
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + shares
                    self.trade_log.append((symbol, "BUY", shares, price))

        # SELL logic
        elif signal == "SELL" and symbol in self.positions:
            shares = self.positions.pop(symbol)
            proceeds = shares * price * (1 - tc)
            self.cash += proceeds
            self.trade_log.append((symbol, "SELL", shares, price))


def compute_metrics(equity_curve: pd.Series):
    """
    Compute standard performance metrics.
    """

    returns = equity_curve.pct_change().dropna()

    sharpe = (
        np.sqrt(252) * returns.mean() / returns.std()
        if returns.std() > 0
        else 0.0
    )

    drawdown = (equity_curve.cummax() - equity_curve) / equity_curve.cummax()
    max_dd = drawdown.max() if not drawdown.empty else 0.0

    return {
        "Sharpe": round(sharpe, 2),
        "MaxDrawdown": round(float(max_dd), 2),
    }

