"""Asian option pricing (geometric closed form + arithmetic MC)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from core.monte_carlo import gbm_paths


def geometric_asian_call_bs(spot: float, strike: float, maturity: float, rate: float, vol: float) -> float:
    sigma_g = vol / np.sqrt(3.0)
    mu_g = 0.5 * (rate - 0.5 * vol * vol) + 0.5 * sigma_g * sigma_g
    if maturity <= 0:
        return max(spot - strike, 0.0)
    d1 = (np.log(spot / strike) + (mu_g + 0.5 * sigma_g * sigma_g) * maturity) / (sigma_g * np.sqrt(maturity))
    d2 = d1 - sigma_g * np.sqrt(maturity)
    return float(np.exp(-rate * maturity) * (spot * np.exp(mu_g * maturity) * norm.cdf(d1) - strike * norm.cdf(d2)))


def arithmetic_asian_mc(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    n_paths: int = 40000,
    n_steps: int = 252,
    seed: int = 42,
) -> float:
    paths = gbm_paths(
        spot=spot, maturity=maturity, rate=rate, vol=vol, n_paths=n_paths, n_steps=n_steps, seed=seed
    )
    avg = np.mean(paths[:, 1:], axis=1)
    if option_type == "call":
        payoff = np.maximum(avg - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - avg, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    return float(np.exp(-rate * maturity) * np.mean(payoff))
