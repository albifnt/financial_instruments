"""Compound option pricing via nested Monte Carlo."""

from __future__ import annotations

import numpy as np


def call_on_call_mc(
    spot: float,
    strike_compound: float,
    strike_underlying: float,
    t1: float,
    t2: float,
    rate: float,
    vol: float,
    n_outer: int = 20000,
    n_inner: int = 2000,
    seed: int = 42,
) -> float:
    """
    Price a call on a call:
    - At t1 holder can buy a call (expiring t2, strike_underlying) for strike_compound.
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1 for compound options.")
    rng = np.random.default_rng(seed)

    z1 = rng.standard_normal(n_outer)
    s_t1 = spot * np.exp((rate - 0.5 * vol * vol) * t1 + vol * np.sqrt(t1) * z1)

    # Inner valuation of underlying call at t1 for each outer path.
    z2 = rng.standard_normal((n_outer, n_inner))
    dt = t2 - t1
    s_t2 = s_t1[:, None] * np.exp((rate - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z2)
    underlying_payoff = np.maximum(s_t2 - strike_underlying, 0.0)
    underlying_value_t1 = np.exp(-rate * dt) * np.mean(underlying_payoff, axis=1)

    compound_payoff_t1 = np.maximum(underlying_value_t1 - strike_compound, 0.0)
    return float(np.exp(-rate * t1) * np.mean(compound_payoff_t1))
