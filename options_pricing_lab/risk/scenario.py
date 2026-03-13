"""Scenario analysis for options."""

from __future__ import annotations

import pandas as pd

from core.black_scholes import price as bs_price


def scenario_reprice(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    spot_shift_pct: float = 0.0,
    vol_shift_pct: float = 0.0,
    rate_shift_bps: float = 0.0,
    time_decay_days: int = 0,
) -> dict:
    s_new = spot * (1.0 + spot_shift_pct / 100.0)
    v_new = max(vol * (1.0 + vol_shift_pct / 100.0), 1e-8)
    r_new = rate + rate_shift_bps / 10000.0
    t_new = max(maturity - time_decay_days / 365.0, 1e-8)

    base = bs_price(spot, strike, maturity, rate, vol, option_type)
    stressed = bs_price(s_new, strike, t_new, r_new, v_new, option_type)
    return {
        "base_price": float(base),
        "scenario_price": float(stressed),
        "pnl": float(stressed - base),
    }


def scenario_grid(
    spot: float, strike: float, maturity: float, rate: float, vol: float, option_type: str = "call"
) -> pd.DataFrame:
    spot_shifts = [-20, -10, -5, 0, 5, 10, 20]
    vol_shifts = [-30, -15, 0, 15, 30]
    rows = []
    base = bs_price(spot, strike, maturity, rate, vol, option_type)
    for ds in spot_shifts:
        for dv in vol_shifts:
            res = scenario_reprice(
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                vol=vol,
                option_type=option_type,
                spot_shift_pct=ds,
                vol_shift_pct=dv,
            )
            rows.append((ds, dv, res["scenario_price"], res["scenario_price"] - base))
    return pd.DataFrame(rows, columns=["spot_shift_pct", "vol_shift_pct", "price", "pnl"])
