import numpy as np
import pandas as pd


class Portfolio:
    """
    Manages portfolio state, execution, equity tracking, and rebalancing.
    """

    def __init__(self, initial_capital, max_position_pct=0.10, transaction_cost=0.001):
        self.cash = initial_capital
        self.positions: dict[str, int] = {}
        self.max_position_pct = max_position_pct
        self.transaction_cost = transaction_cost
        self.trade_log: list[tuple] = []

        # Optional history tracking per symbol for risk scaling
        self.price_history: dict[str, pd.Series] = {}

    # ----------------------------
    # Portfolio equity
    # ----------------------------
    def total_equity(self, prices: dict) -> float:
        equity = self.cash
        for symbol, shares in self.positions.items():
            equity += shares * prices.get(symbol, 0)
        return equity

    # ----------------------------
    # Execute trade
    # ----------------------------
    def execute(self, symbol: str, price: float, signal: str, position_size: int = None, transaction_cost=None):
        if price <= 0:
            return

        tc = transaction_cost if transaction_cost is not None else self.transaction_cost

        if signal == "BUY":
            shares_to_buy = position_size or 0
            cost = shares_to_buy * price * (1 + tc)
            if cost <= self.cash and shares_to_buy > 0:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares_to_buy
                self.trade_log.append((symbol, "BUY", shares_to_buy, price))

        elif signal == "SELL" and symbol in self.positions:
            shares_to_sell = self.positions.pop(symbol)
            proceeds = shares_to_sell * price * (1 - tc)
            self.cash += proceeds
            self.trade_log.append((symbol, "SELL", shares_to_sell, price))

    # ----------------------------
    # Evict a position
    # ----------------------------
    def evict(self, symbol: str, price: float = None):
        if symbol in self.positions and price is not None:
            shares = self.positions.pop(symbol)
            proceeds = shares * price * (1 - self.transaction_cost)
            self.cash += proceeds
            self.trade_log.append((symbol, "SELL", shares, price))

    # ----------------------------
    # Rebalance all positions to target weights
    # ----------------------------
    def rebalance(self, prices: dict, target_weights: dict[str, float]):
        equity = self.total_equity(prices)
        for symbol, target_weight in target_weights.items():
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue

            target_value = equity * target_weight
            current_value = self.positions.get(symbol, 0) * price
            delta_value = target_value - current_value
            if abs(delta_value) < 1e-6:
                continue

            shares_delta = int(delta_value // price)
            if shares_delta == 0:
                continue

            signal = "BUY" if shares_delta > 0 else "SELL"
            self.execute(symbol, price, signal, position_size=abs(shares_delta))

    # ----------------------------
    # Optional: access symbol return history for risk scaling
    # ----------------------------
    def get_symbol_history(self, symbol: str) -> pd.Series | None:
        return self.price_history.get(symbol)

# ----------------------------
# Metrics
# ----------------------------
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

