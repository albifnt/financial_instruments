"""Lookback option pricing via Monte Carlo."""

from __future__ import annotations

import numpy as np

from core.monte_carlo import gbm_paths


def floating_strike_call_mc(
    spot: float,
    maturity: float,
    rate: float,
    vol: float,
    n_paths: int = 30000,
    n_steps: int = 252,
    seed: int = 42,
) -> float:
    paths = gbm_paths(
        spot=spot, maturity=maturity, rate=rate, vol=vol, n_paths=n_paths, n_steps=n_steps, seed=seed
    )
    min_path = np.min(paths, axis=1)
    payoff = np.maximum(paths[:, -1] - min_path, 0.0)
    return float(np.exp(-rate * maturity) * np.mean(payoff))


def floating_strike_put_mc(
    spot: float,
    maturity: float,
    rate: float,
    vol: float,
    n_paths: int = 30000,
    n_steps: int = 252,
    seed: int = 42,
) -> float:
    paths = gbm_paths(
        spot=spot, maturity=maturity, rate=rate, vol=vol, n_paths=n_paths, n_steps=n_steps, seed=seed
    )
    max_path = np.max(paths, axis=1)
    payoff = np.maximum(max_path - paths[:, -1], 0.0)
    return float(np.exp(-rate * maturity) * np.mean(payoff))
