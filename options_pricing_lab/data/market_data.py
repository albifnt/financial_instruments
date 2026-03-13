"""Sample market snapshots and option chains."""

from __future__ import annotations

import pandas as pd


SPOT_SNAPSHOT = {
    "AAPL": {"spot": 210.0, "rate": 0.045, "dividend": 0.005},
    "SPX": {"spot": 5200.0, "rate": 0.042, "dividend": 0.015},
    "EURUSD": {"spot": 1.09, "rate": 0.035, "dividend": 0.020},
}


def sample_option_chain() -> pd.DataFrame:
    rows = [
        (30 / 365, 90, 0.34),
        (30 / 365, 95, 0.31),
        (30 / 365, 100, 0.28),
        (30 / 365, 105, 0.27),
        (30 / 365, 110, 0.29),
        (90 / 365, 90, 0.33),
        (90 / 365, 95, 0.30),
        (90 / 365, 100, 0.27),
        (90 / 365, 105, 0.26),
        (90 / 365, 110, 0.28),
        (180 / 365, 90, 0.31),
        (180 / 365, 95, 0.29),
        (180 / 365, 100, 0.26),
        (180 / 365, 105, 0.255),
        (180 / 365, 110, 0.27),
        (365 / 365, 90, 0.30),
        (365 / 365, 95, 0.28),
        (365 / 365, 100, 0.25),
        (365 / 365, 105, 0.245),
        (365 / 365, 110, 0.26),
    ]
    return pd.DataFrame(rows, columns=["maturity", "strike", "implied_vol"])
