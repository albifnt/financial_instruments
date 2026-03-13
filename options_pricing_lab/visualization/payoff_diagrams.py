"""Payoff diagram generators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def vanilla_payoff_grid(strike: float, option_type: str = "call", premium: float = 0.0, s_min: float = 0.5, s_max: float = 1.5):
    s = np.linspace(s_min * strike, s_max * strike, 200)
    if option_type == "call":
        payoff = np.maximum(s - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - s, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    pnl = payoff - premium
    return pd.DataFrame({"spot_at_expiry": s, "payoff": payoff, "pnl": pnl})


def straddle_payoff_grid(strike: float, premium_call: float, premium_put: float):
    s = np.linspace(0.5 * strike, 1.5 * strike, 200)
    call = np.maximum(s - strike, 0.0) - premium_call
    put = np.maximum(strike - s, 0.0) - premium_put
    return pd.DataFrame({"spot_at_expiry": s, "pnl": call + put})
