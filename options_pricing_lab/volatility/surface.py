"""Volatility surface construction and interpolation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator


def build_surface(points: pd.DataFrame) -> LinearNDInterpolator:
    """
    points columns: maturity, strike, implied_vol
    """
    required = {"maturity", "strike", "implied_vol"}
    if not required.issubset(points.columns):
        raise ValueError(f"Input DataFrame must contain {required}.")
    xyz = points[["maturity", "strike"]].to_numpy()
    vals = points["implied_vol"].to_numpy()
    return LinearNDInterpolator(xyz, vals)


def evaluate_surface(
    interpolator: LinearNDInterpolator, maturities: np.ndarray, strikes: np.ndarray
) -> pd.DataFrame:
    rows = []
    for t in maturities:
        for k in strikes:
            v = interpolator(t, k)
            vol = float(v) if np.isfinite(v) else np.nan
            rows.append((t, k, vol))
    return pd.DataFrame(rows, columns=["maturity", "strike", "implied_vol"])
