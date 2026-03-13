"""Barrier option pricing via Monte Carlo."""

from __future__ import annotations

import numpy as np

from core.monte_carlo import gbm_paths


def up_and_out_call_mc(
    spot: float,
    strike: float,
    barrier: float,
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
    knocked = np.max(paths, axis=1) >= barrier
    terminal = paths[:, -1]
    payoff = np.where(knocked, 0.0, np.maximum(terminal - strike, 0.0))
    return float(np.exp(-rate * maturity) * np.mean(payoff))


def down_and_out_put_mc(
    spot: float,
    strike: float,
    barrier: float,
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
    knocked = np.min(paths, axis=1) <= barrier
    terminal = paths[:, -1]
    payoff = np.where(knocked, 0.0, np.maximum(strike - terminal, 0.0))
    return float(np.exp(-rate * maturity) * np.mean(payoff))
