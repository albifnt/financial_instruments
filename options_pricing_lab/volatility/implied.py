"""Implied volatility inversion utilities."""

from __future__ import annotations

from scipy.optimize import brentq

from core.black_scholes import price as bs_price


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str = "call",
    dividend: float = 0.0,
    vol_lower: float = 1e-6,
    vol_upper: float = 5.0,
) -> float:
    if market_price <= 0:
        return 0.0

    def f(vol: float) -> float:
        return bs_price(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            vol=vol,
            option_type=option_type,
            dividend=dividend,
        ) - market_price

    return float(brentq(f, vol_lower, vol_upper))
