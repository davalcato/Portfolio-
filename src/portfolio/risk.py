# src/risk.py
import numpy as np

class RiskManager:
    """
    Handles risk allocation, position sizing, and drawdown constraints.
    """
    def __init__(self, max_drawdown=0.2, target_volatility=0.1):
        self.max_drawdown = max_drawdown
        self.target_volatility = target_volatility
        self.equity_peak = 1.0

    def check_drawdown(self, equity_curve):
        self.equity_peak = max(self.equity_peak, equity_curve[-1])
        drawdown = (self.equity_peak - equity_curve[-1]) / self.equity_peak
        return drawdown <= self.max_drawdown

    def scale_position(self, signal_weight, historical_vol):
        """
        Scale position based on target volatility.
        """
        if historical_vol == 0:
            return 0
        return signal_weight * (self.target_volatility / historical_vol)

