"""Black-Scholes-Merton pricing for European options."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _validate_inputs(spot: float, strike: float, maturity: float, vol: float) -> None:
    if spot <= 0 or strike <= 0:
        raise ValueError("Spot and strike must be positive.")
    if maturity < 0:
        raise ValueError("Maturity cannot be negative.")
    if vol < 0:
        raise ValueError("Volatility cannot be negative.")


def d1_d2(spot: float, strike: float, maturity: float, rate: float, vol: float, dividend: float = 0.0) -> tuple[float, float]:
    _validate_inputs(spot, strike, maturity, vol)
    if maturity == 0 or vol == 0:
        return np.inf, np.inf
    sigt = vol * np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend + 0.5 * vol * vol) * maturity) / sigt
    d2 = d1 - sigt
    return float(d1), float(d2)


def price(spot: float, strike: float, maturity: float, rate: float, vol: float, option_type: str = "call", dividend: float = 0.0) -> float:
    _validate_inputs(spot, strike, maturity, vol)
    disc_r = np.exp(-rate * maturity)
    disc_q = np.exp(-dividend * maturity)

    if maturity == 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return float(intrinsic)
    if vol == 0:
        fwd_payoff = max(spot * disc_q - strike * disc_r, 0.0)
        if option_type == "call":
            return float(fwd_payoff)
        put_val = max(strike * disc_r - spot * disc_q, 0.0)
        return float(put_val)

    d1, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend=dividend)
    if option_type == "call":
        return float(spot * disc_q * norm.cdf(d1) - strike * disc_r * norm.cdf(d2))
    if option_type == "put":
        return float(strike * disc_r * norm.cdf(-d2) - spot * disc_q * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'.")


def greeks(spot: float, strike: float, maturity: float, rate: float, vol: float, option_type: str = "call", dividend: float = 0.0) -> dict:
    _validate_inputs(spot, strike, maturity, vol)
    if maturity == 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    d1, d2 = d1_d2(spot, strike, maturity, rate, vol, dividend=dividend)
    disc_r = np.exp(-rate * maturity)
    disc_q = np.exp(-dividend * maturity)
    pdf_d1 = norm.pdf(d1)
    sqrt_t = np.sqrt(maturity)

    if option_type == "call":
        delta = disc_q * norm.cdf(d1)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
            - rate * strike * disc_r * norm.cdf(d2)
            + dividend * spot * disc_q * norm.cdf(d1)
        )
        rho = strike * maturity * disc_r * norm.cdf(d2)
    elif option_type == "put":
        delta = disc_q * (norm.cdf(d1) - 1.0)
        theta = (
            -spot * disc_q * pdf_d1 * vol / (2.0 * sqrt_t)
            + rate * strike * disc_r * norm.cdf(-d2)
            - dividend * spot * disc_q * norm.cdf(-d1)
        )
        rho = -strike * maturity * disc_r * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    gamma = disc_q * pdf_d1 / (spot * vol * sqrt_t)
    vega = spot * disc_q * pdf_d1 * sqrt_t
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega / 100.0),
        "theta": float(theta / 365.0),
        "rho": float(rho / 100.0),
    }
