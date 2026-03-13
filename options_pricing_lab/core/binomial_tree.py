"""CRR binomial tree pricing for European and American options."""

from __future__ import annotations

import numpy as np


def price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    vol: float,
    steps: int = 200,
    option_type: str = "call",
    american: bool = False,
    dividend: float = 0.0,
) -> float:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if maturity <= 0:
        intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        return float(intrinsic)

    dt = maturity / steps
    u = np.exp(vol * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-rate * dt)
    q = (np.exp((rate - dividend) * dt) - d) / (u - d)
    q = min(max(q, 0.0), 1.0)

    stock = np.array([spot * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    if option_type == "call":
        values = np.maximum(stock - strike, 0.0)
    elif option_type == "put":
        values = np.maximum(strike - stock, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    for i in range(steps - 1, -1, -1):
        values = disc * (q * values[1:] + (1.0 - q) * values[:-1])
        if american:
            stock_i = np.array([spot * (u ** j) * (d ** (i - j)) for j in range(i + 1)])
            if option_type == "call":
                exercise = np.maximum(stock_i - strike, 0.0)
            else:
                exercise = np.maximum(strike - stock_i, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])
