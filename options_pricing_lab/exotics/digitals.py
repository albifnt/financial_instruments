"""Digital (cash-or-nothing) option pricing."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.black_scholes import d1_d2


def cash_or_nothing(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    payout: float = 1.0,
    option_type: str = "call",
    dividend: float = 0.0,
) -> float:
    if maturity <= 0:
        if option_type == "call":
            return payout if spot > strike else 0.0
        return payout if spot < strike else 0.0
    _, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend=dividend)
    disc = np.exp(-rate * maturity)
    if option_type == "call":
        return float(payout * disc * norm.cdf(d2))
    if option_type == "put":
        return float(payout * disc * norm.cdf(-d2))
    raise ValueError("option_type must be 'call' or 'put'.")
