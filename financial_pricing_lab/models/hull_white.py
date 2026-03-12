"""One-factor Hull-White model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.curve import YieldCurve


class HullWhiteModel:
    def __init__(self, a: float, sigma: float, r0: float):
        self.a = float(a)
        self.sigma = float(sigma)
        self.r0 = float(r0)
        self._theta_times = np.array([0.0])
        self._theta_vals = np.array([0.0])
        self._curve: YieldCurve | None = None

    def _f0(self, t: float, eps: float = 1e-4) -> float:
        if self._curve is None:
            return self.r0
        t1 = max(t - eps, 0.0)
        t2 = t + eps
        df1 = self._curve.get_discount_factor(t1)
        df2 = self._curve.get_discount_factor(t2)
        return (np.log(df1) - np.log(df2)) / max(t2 - t1, 1e-8)

    def calibrate_theta(self, curve: YieldCurve, horizon: float = 10.0, step: float = 0.05) -> pd.DataFrame:
        self._curve = curve
        times = np.arange(0.0, horizon + step, step)
        f0 = np.array([self._f0(t) for t in times])
        dt = step
        dfdt = np.gradient(f0, dt)
        theta = dfdt + self.a * f0 + (self.sigma * self.sigma / (2.0 * self.a)) * (
            1.0 - np.exp(-2.0 * self.a * times)
        )
        self._theta_times = times
        self._theta_vals = theta
        return pd.DataFrame({"time": times, "theta": theta, "f0": f0})

    def theta(self, t: float) -> float:
        return float(np.interp(t, self._theta_times, self._theta_vals))

    def B(self, tau: float) -> float:
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def bond_price(self, t: float, T: float, r_t: float | None = None) -> float:
        if T < t:
            raise ValueError("Maturity T must be >= t.")
        if self._curve is None:
            raise ValueError("Call calibrate_theta(curve, ...) before pricing Hull-White bonds.")
        if r_t is None:
            r_t = self.r0
        tau = T - t
        B = self.B(tau)
        p0T = self._curve.get_discount_factor(T)
        p0t = self._curve.get_discount_factor(t) if t > 0 else 1.0
        f0t = self._f0(t)
        vol_adj = (self.sigma * self.sigma / (4.0 * self.a)) * (1.0 - np.exp(-2.0 * self.a * t)) * B * B
        A = (p0T / p0t) * np.exp(B * f0t - vol_adj)
        return float(A * np.exp(-B * r_t))

    def simulate_paths(
        self, n_paths: int = 100, n_steps: int = 250, horizon: float = 5.0, seed: int = 42
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        times = np.linspace(0.0, horizon, n_steps + 1)
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0

        for i in range(n_steps):
            t = times[i]
            z = rng.standard_normal(n_paths)
            th = self.theta(t)
            dr = (th - self.a * rates[:, i]) * dt + self.sigma * np.sqrt(dt) * z
            rates[:, i + 1] = rates[:, i] + dr

        cols = [f"path_{i+1}" for i in range(n_paths)]
        df = pd.DataFrame(rates.T, columns=cols)
        df.insert(0, "time", times)
        return df
