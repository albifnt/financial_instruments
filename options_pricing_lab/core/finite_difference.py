"""Finite difference pricing for European options (explicit scheme)."""

from __future__ import annotations

import numpy as np


def explicit_european(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    option_type: str = "call",
    s_max_mult: float = 3.0,
    n_s: int = 200,
    n_t: int = 1000,
) -> float:
    if maturity <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return float(intrinsic)
    s_max = s_max_mult * max(spot, strike)
    ds = s_max / n_s
    dt = maturity / n_t

    s = np.linspace(0.0, s_max, n_s + 1)
    v = np.maximum(s - strike, 0.0) if option_type == "call" else np.maximum(strike - s, 0.0)

    for n in range(n_t):
        t = maturity - n * dt
        new_v = v.copy()
        for i in range(1, n_s):
            a = 0.5 * dt * (vol * vol * i * i - rate * i)
            b = 1.0 - dt * (vol * vol * i * i + rate)
            c = 0.5 * dt * (vol * vol * i * i + rate * i)
            new_v[i] = a * v[i - 1] + b * v[i] + c * v[i + 1]

        if option_type == "call":
            new_v[0] = 0.0
            new_v[-1] = s_max - strike * np.exp(-rate * (t - dt))
        else:
            new_v[0] = strike * np.exp(-rate * (t - dt))
            new_v[-1] = 0.0
        v = new_v

    return float(np.interp(spot, s, v))
