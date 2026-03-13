"""Monte Carlo pricers for vanilla/path-dependent options."""

from __future__ import annotations

import numpy as np


def european_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    n_paths: int = 20000,
    seed: int = 42,
    dividend: float = 0.0,
) -> float:
    if maturity <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return float(intrinsic)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    drift = (rate - dividend - 0.5 * vol * vol) * maturity
    diffusion = vol * np.sqrt(maturity) * z
    st = spot * np.exp(drift + diffusion)
    if option_type == "call":
        payoff = np.maximum(st - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - st, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    return float(np.exp(-rate * maturity) * np.mean(payoff))


def gbm_paths(
    spot: float,
    maturity: float,
    rate: float,
    vol: float,
    n_paths: int = 5000,
    n_steps: int = 252,
    seed: int = 42,
    dividend: float = 0.0,
) -> np.ndarray:
    dt = maturity / n_steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, n_steps))
    increments = (rate - dividend - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z
    log_s = np.cumsum(increments, axis=1)
    log_s = np.concatenate([np.zeros((n_paths, 1)), log_s], axis=1)
    return spot * np.exp(log_s)
