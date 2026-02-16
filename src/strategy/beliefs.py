# src/beliefs.py

class SignalBelief:
    """
    Encodes the assumptions, target regimes, and failure modes for a signal.
    """
    def __init__(self, name, description, regimes, failure_conditions):
        self.name = name
        self.description = description
        self.regimes = regimes  # list of regimes where signal works
        self.failure_conditions = failure_conditions  # list of conditions that break it

# Example: Momentum signal
MOMENTUM_20_60 = SignalBelief(
    name="Momentum_20_60",
    description="Buys assets with positive 20-day returns vs 60-day trend.",
    regimes=["high_vol_trend", "mid_vol_trend"],
    failure_conditions=["low_vol_mean_reversion", "liquidity_crunch"]
)

# Example: Mean reversion
MEAN_REVERSION_10 = SignalBelief(
    name="MeanReversion_10",
    description="Exploits 10-day mean-reverting assets.",
    regimes=["low_vol_range"],
    failure_conditions=["high_vol_trend"]
)

# Store all beliefs for easy reference
ALL_BELIEFS = [MOMENTUM_20_60, MEAN_REVERSION_10]

