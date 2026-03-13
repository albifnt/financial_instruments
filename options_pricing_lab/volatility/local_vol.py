"""Local volatility (Dupire-style finite-difference approximation)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dupire_local_vol(
    surface_df: pd.DataFrame,
    rate: float,
) -> pd.DataFrame:
    """
    Approximate local vol using finite differences on implied-vol surface.
    surface_df columns: maturity, strike, implied_vol.
    """
    df = surface_df.dropna().copy().sort_values(["maturity", "strike"])
    maturities = np.sort(df["maturity"].unique())
    strikes = np.sort(df["strike"].unique())

    pivot = df.pivot(index="maturity", columns="strike", values="implied_vol").reindex(index=maturities, columns=strikes)
    iv = pivot.to_numpy()

    # Approximate local vol by combining term derivative and smile curvature in a robust educational formula.
    d_iv_dt = np.gradient(iv, maturities, axis=0)
    d_iv_dk = np.gradient(iv, strikes, axis=1)
    d2_iv_dk2 = np.gradient(d_iv_dk, strikes, axis=1)

    local = np.sqrt(np.maximum(iv * iv + 2.0 * maturities[:, None] * iv * d_iv_dt + 0.1 * d2_iv_dk2, 1e-8))
    rows = []
    for i, t in enumerate(maturities):
        for j, k in enumerate(strikes):
            rows.append((t, k, float(local[i, j]), float(rate)))
    return pd.DataFrame(rows, columns=["maturity", "strike", "local_vol", "rate"])
