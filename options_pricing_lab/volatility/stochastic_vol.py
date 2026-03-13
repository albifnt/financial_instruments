"""Stochastic volatility utilities: Heston simulation + SABR approximation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def heston_paths(
    spot: float,
    v0: float,
    rate: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    maturity: float,
    n_paths: int = 5000,
    n_steps: int = 252,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    dt = maturity / n_steps
    rng = np.random.default_rng(seed)

    s = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))
    s[:, 0] = spot
    v[:, 0] = np.maximum(v0, 1e-8)

    for i in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w1 = z1
        w2 = rho * z1 + np.sqrt(max(1.0 - rho * rho, 1e-8)) * z2
        v_prev = np.maximum(v[:, i], 0.0)
        v_next = v_prev + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * w2
        v[:, i + 1] = np.maximum(v_next, 1e-10)
        s[:, i + 1] = s[:, i] * np.exp((rate - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * w1)

    return s, v


def sabr_implied_vol(
    fwd: float,
    strike: float,
    maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """
    Hagan 2002 SABR approximation.
    """
    if fwd <= 0 or strike <= 0:
        raise ValueError("Forward and strike must be positive.")
    if abs(fwd - strike) < 1e-12:
        fk_beta = fwd ** (1.0 - beta)
        term1 = alpha / fk_beta
        term2 = (
            ((1.0 - beta) ** 2 / 24.0) * (alpha * alpha / (fwd ** (2.0 - 2.0 * beta)))
            + 0.25 * rho * beta * nu * alpha / (fwd ** (1.0 - beta))
            + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
        ) * maturity
        return float(term1 * (1.0 + term2))

    z = (nu / alpha) * (fwd * strike) ** ((1.0 - beta) / 2.0) * np.log(fwd / strike)
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    fk_beta = (fwd * strike) ** ((1.0 - beta) / 2.0)
    log_fk = np.log(fwd / strike)
    numerator = alpha * (1.0 + (((1.0 - beta) ** 2) / 24.0 * log_fk * log_fk + ((1.0 - beta) ** 4) / 1920.0 * log_fk**4))
    denominator = fk_beta * (
        1.0
        + ((1.0 - beta) ** 2 / 24.0) * (log_fk**2)
        + ((1.0 - beta) ** 4 / 1920.0) * (log_fk**4)
    )
    corr_term = 1.0 + (
        ((1.0 - beta) ** 2 / 24.0) * (alpha * alpha / (fk_beta * fk_beta))
        + 0.25 * rho * beta * nu * alpha / fk_beta
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    ) * maturity
    return float((numerator / denominator) * (z / xz) * corr_term)


def sample_sabr_smile(
    fwd: float, maturity: float, alpha: float, beta: float, rho: float, nu: float, strikes: np.ndarray
) -> pd.DataFrame:
    rows = [(k, sabr_implied_vol(fwd, float(k), maturity, alpha, beta, rho, nu)) for k in strikes]
    return pd.DataFrame(rows, columns=["strike", "implied_vol"])
