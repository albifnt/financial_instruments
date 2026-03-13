"""Greeks heatmap data builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.black_scholes import greeks


def greek_heatmap(
    strike: float,
    maturity: float,
    rate: float,
    option_type: str,
    spot_grid: np.ndarray,
    vol_grid: np.ndarray,
    greek_name: str = "delta",
) -> pd.DataFrame:
    rows = []
    for s in spot_grid:
        for v in vol_grid:
            g = greeks(spot=float(s), strike=strike, maturity=maturity, rate=rate, vol=float(v), option_type=option_type)
            rows.append((float(s), float(v), float(g[greek_name])))
    return pd.DataFrame(rows, columns=["spot", "vol", "value"])
