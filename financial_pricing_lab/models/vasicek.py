"""Vasicek short-rate model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class VasicekModel:
    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.r0 = float(r0)

    def B(self, tau: float) -> float:
        return (1.0 - np.exp(-self.a * tau)) / self.a

    def A(self, tau: float) -> float:
        B_t = self.B(tau)
        term1 = (B_t - tau) * (self.a * self.a * self.b - 0.5 * self.sigma * self.sigma) / (
            self.a * self.a
        )
        term2 = (self.sigma * self.sigma * B_t * B_t) / (4.0 * self.a)
        return float(np.exp(term1 - term2))

    def bond_price(self, t: float, T: float, r_t: float | None = None) -> float:
        if T < t:
            raise ValueError("Maturity T must be >= t.")
        if r_t is None:
            r_t = self.r0
        tau = T - t
        return float(self.A(tau) * np.exp(-self.B(tau) * r_t))

    def expected_rate(self, t: float) -> float:
        return float(self.b + (self.r0 - self.b) * np.exp(-self.a * t))

    def simulate_paths(
        self, n_paths: int = 100, n_steps: int = 250, horizon: float = 5.0, seed: int = 42
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        times = np.linspace(0.0, horizon, n_steps + 1)
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = self.r0

        for i in range(n_steps):
            z = rng.standard_normal(n_paths)
            dr = self.a * (self.b - rates[:, i]) * dt + self.sigma * np.sqrt(dt) * z
            rates[:, i + 1] = rates[:, i] + dr

        cols = [f"path_{i+1}" for i in range(n_paths)]
        df = pd.DataFrame(rates.T, columns=cols)
        df.insert(0, "time", times)
        return df
