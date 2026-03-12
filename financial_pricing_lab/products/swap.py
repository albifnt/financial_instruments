"""Vanilla interest rate swap pricing and risk measures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.curve import YieldCurve


class InterestRateSwap:
    def __init__(self, notional: float, fixed_rate_pct: float, maturity: float, pay_freq: int = 2):
        self.notional = float(notional)
        self.fixed_rate = float(fixed_rate_pct) / 100.0
        self.maturity = float(maturity)
        self.pay_freq = int(pay_freq)

    def payment_times(self) -> np.ndarray:
        return np.arange(1.0 / self.pay_freq, self.maturity + 1e-9, 1.0 / self.pay_freq)

    def annuity(self, curve: YieldCurve) -> float:
        times = self.payment_times()
        delta = 1.0 / self.pay_freq
        return float(np.sum([delta * curve.get_discount_factor(t) for t in times]))

    def par_rate(self, curve: YieldCurve) -> float:
        df_T = curve.get_discount_factor(self.maturity)
        ann = self.annuity(curve)
        return float((1.0 - df_T) / max(ann, 1e-12))

    def fixed_leg_pv(self, curve: YieldCurve) -> float:
        return float(self.notional * self.fixed_rate * self.annuity(curve))

    def floating_leg_pv(self, curve: YieldCurve) -> float:
        return float(self.notional * (1.0 - curve.get_discount_factor(self.maturity)))

    def npv(self, curve: YieldCurve, payer_fixed: bool = True) -> float:
        fixed = self.fixed_leg_pv(curve)
        floating = self.floating_leg_pv(curve)
        return float(floating - fixed) if payer_fixed else float(fixed - floating)

    def dv01(self, curve: YieldCurve, payer_fixed: bool = True) -> float:
        bumped = curve.bumped_parallel(1.0)
        return float(self.npv(bumped, payer_fixed=payer_fixed) - self.npv(curve, payer_fixed=payer_fixed))

    def key_rate_durations(self, curve: YieldCurve, payer_fixed: bool = True, bump_bps: float = 1.0) -> pd.DataFrame:
        base = self.npv(curve, payer_fixed=payer_fixed)
        nodes = curve.get_nodes()
        rows = []
        for t in nodes["time_years"].to_numpy():
            if t == 0:
                continue
            bumped = curve.bumped_key_rate(t, bump_bps)
            bumped_npv = self.npv(bumped, payer_fixed=payer_fixed)
            krd = -(bumped_npv - base) / (base if abs(base) > 1e-9 else self.notional) / (bump_bps / 10000.0)
            rows.append((t, krd))
        return pd.DataFrame(rows, columns=["time_years", "key_rate_duration"])
