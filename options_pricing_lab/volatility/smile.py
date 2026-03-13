"""Volatility smile diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def smile_metrics(smile_df: pd.DataFrame, atm_strike: float) -> dict:
    """
    smile_df columns: strike, implied_vol
    """
    df = smile_df.sort_values("strike").copy()
    atm_idx = (df["strike"] - atm_strike).abs().idxmin()
    atm_vol = float(df.loc[atm_idx, "implied_vol"])

    # First derivative (skew) and second derivative (curvature) around ATM.
    k = df["strike"].to_numpy()
    v = df["implied_vol"].to_numpy()
    dv = np.gradient(v, k)
    d2v = np.gradient(dv, k)
    atm_pos = list(df.index).index(atm_idx)

    return {
        "atm_vol": atm_vol,
        "skew_atm": float(dv[atm_pos]),
        "curvature_atm": float(d2v[atm_pos]),
        "smirk_strength": float(v.max() - v.min()),
    }
