"""Value at Risk estimates for option positions."""

from __future__ import annotations

import numpy as np

from core.black_scholes import greeks, price as bs_price


def parametric_var_delta_gamma(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    position: float = 1.0,
    horizon_days: int = 1,
    spot_daily_vol: float = 0.015,
    confidence: float = 0.99,
) -> float:
    g = greeks(spot, strike, maturity, rate, vol, option_type)
    z = 2.3263478740408408 if confidence >= 0.99 else 1.6448536269514722
    sigma_s = spot * spot_daily_vol * np.sqrt(horizon_days)
    pnl_std = abs(position) * np.sqrt((g["delta"] * sigma_s) ** 2 + 0.5 * (g["gamma"] * sigma_s * sigma_s) ** 2)
    return float(z * pnl_std)


def monte_carlo_var_full_reval(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    position: float = 1.0,
    horizon_days: int = 1,
    confidence: float = 0.99,
    n_paths: int = 30000,
    seed: int = 42,
) -> float:
    rng = np.random.default_rng(seed)
    dt = horizon_days / 252.0
    z = rng.standard_normal(n_paths)
    spot_new = spot * np.exp((rate - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z)
    t_new = max(maturity - dt, 1e-8)

    base = bs_price(spot, strike, maturity, rate, vol, option_type)
    reval = np.array([bs_price(s, strike, t_new, rate, vol, option_type) for s in spot_new])
    pnl = position * (reval - base)
    q = np.quantile(pnl, 1.0 - confidence)
    return float(-q)
