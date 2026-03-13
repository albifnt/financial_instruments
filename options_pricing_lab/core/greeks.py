"""Greeks calculations: analytical (BS) + finite-difference generic helper."""

from __future__ import annotations

from typing import Callable

from core import black_scholes


def analytical_bs(
    spot: float, strike: float, maturity: float, rate: float, vol: float, option_type: str = "call", dividend: float = 0.0
) -> dict:
    return black_scholes.greeks(
        spot=spot, strike=strike, maturity=maturity, rate=rate, vol=vol, option_type=option_type, dividend=dividend
    )


def finite_difference(
    pricer: Callable[..., float],
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    ds: float = 0.5,
    dv: float = 0.01,
    dr: float = 0.0005,
    dt: float = 1.0 / 365.0,
) -> dict:
    p = pricer(spot, strike, maturity, rate, vol, option_type)
    p_up = pricer(spot + ds, strike, maturity, rate, vol, option_type)
    p_dn = pricer(max(spot - ds, 1e-6), strike, maturity, rate, vol, option_type)
    delta = (p_up - p_dn) / (2.0 * ds)
    gamma = (p_up - 2.0 * p + p_dn) / (ds * ds)

    v_up = pricer(spot, strike, maturity, rate, vol + dv, option_type)
    v_dn = pricer(spot, strike, maturity, rate, max(vol - dv, 1e-6), option_type)
    vega = (v_up - v_dn) / (2.0 * dv) / 100.0

    r_up = pricer(spot, strike, maturity, rate + dr, vol, option_type)
    r_dn = pricer(spot, strike, maturity, rate - dr, vol, option_type)
    rho = (r_up - r_dn) / (2.0 * dr) / 100.0

    t_short = max(maturity - dt, 1e-9)
    p_short = pricer(spot, strike, t_short, rate, vol, option_type)
    theta = (p_short - p) / dt / 365.0

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }
