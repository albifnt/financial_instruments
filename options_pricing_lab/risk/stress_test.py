"""Stress testing routines for option books."""

from __future__ import annotations

import pandas as pd

from risk.scenario import scenario_reprice


def run_stress_tests(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
) -> pd.DataFrame:
    tests = [
        ("Crash + Vol Spike", -25.0, 40.0, -50.0, 5),
        ("Rally + Vol Crush", 20.0, -25.0, 25.0, 5),
        ("Rates Shock Up", 0.0, 0.0, 200.0, 0),
        ("Rates Shock Down", 0.0, 0.0, -150.0, 0),
        ("Weekend Decay", 0.0, 0.0, 0.0, 3),
    ]
    rows = []
    for name, ds, dv, dr, dt in tests:
        out = scenario_reprice(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            vol=vol,
            option_type=option_type,
            spot_shift_pct=ds,
            vol_shift_pct=dv,
            rate_shift_bps=dr,
            time_decay_days=dt,
        )
        rows.append((name, out["base_price"], out["scenario_price"], out["pnl"]))
    return pd.DataFrame(rows, columns=["scenario", "base_price", "stressed_price", "pnl"])
