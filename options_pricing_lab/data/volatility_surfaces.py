"""Pre-configured volatility surfaces."""

from __future__ import annotations

import pandas as pd

from data.market_data import sample_option_chain


def equity_smile_surface() -> pd.DataFrame:
    return sample_option_chain().copy()


def flat_surface(level: float = 0.25) -> pd.DataFrame:
    maturities = [30 / 365, 90 / 365, 180 / 365, 1.0, 2.0]
    strikes = [80, 90, 100, 110, 120]
    rows = [(t, k, level) for t in maturities for k in strikes]
    return pd.DataFrame(rows, columns=["maturity", "strike", "implied_vol"])
