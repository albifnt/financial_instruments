"""Bond pricing functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from core.curve import YieldCurve


class BondPricer:
    @staticmethod
    def zero_coupon_price(
        face: float,
        maturity: float,
        curve: YieldCurve | None = None,
        model=None,
        r_t: float | None = None,
    ) -> float:
        if maturity <= 0:
            raise ValueError("Maturity must be positive.")
        if model is not None:
            return float(face * model.bond_price(0.0, maturity, r_t=r_t))
        if curve is None:
            raise ValueError("Provide either a curve or model for pricing.")
        return float(face * curve.get_discount_factor(maturity))

    @staticmethod
    def cashflow_table(face: float, coupon_rate_pct: float, maturity: float, frequency: int = 2) -> pd.DataFrame:
        cpn = coupon_rate_pct / 100.0
        times = np.arange(1.0 / frequency, maturity + 1e-9, 1.0 / frequency)
        cashflows = np.full_like(times, fill_value=face * cpn / frequency, dtype=float)
        cashflows[-1] += face
        return pd.DataFrame({"time_years": times, "cashflow": cashflows})

    @staticmethod
    def coupon_bond_price(
        face: float,
        coupon_rate_pct: float,
        maturity: float,
        curve: YieldCurve,
        frequency: int = 2,
    ) -> float:
        cf = BondPricer.cashflow_table(face, coupon_rate_pct, maturity, frequency=frequency)
        discounts = np.array([curve.get_discount_factor(t) for t in cf["time_years"]])
        return float(np.sum(cf["cashflow"].to_numpy() * discounts))

    @staticmethod
    def yield_to_maturity(price: float, face: float, coupon_rate_pct: float, maturity: float, frequency: int = 2) -> float:
        cf = BondPricer.cashflow_table(face, coupon_rate_pct, maturity, frequency=frequency)
        t = cf["time_years"].to_numpy()
        c = cf["cashflow"].to_numpy()

        def obj(y: float) -> float:
            disc = (1.0 + y / frequency) ** (frequency * t)
            return np.sum(c / disc) - price

        ytm = brentq(obj, -0.95, 1.0)
        return float(ytm)

    @staticmethod
    def duration_convexity(
        price: float,
        face: float,
        coupon_rate_pct: float,
        maturity: float,
        ytm: float,
        frequency: int = 2,
    ) -> tuple[float, float, float]:
        cf = BondPricer.cashflow_table(face, coupon_rate_pct, maturity, frequency=frequency)
        t = cf["time_years"].to_numpy()
        c = cf["cashflow"].to_numpy()
        disc = (1.0 + ytm / frequency) ** (frequency * t)
        pv = c / disc
        macaulay = np.sum(t * pv) / price
        modified = macaulay / (1.0 + ytm / frequency)
        convexity = np.sum(pv * t * (t + 1.0 / frequency)) / (price * (1.0 + ytm / frequency) ** 2)
        return float(macaulay), float(modified), float(convexity)
