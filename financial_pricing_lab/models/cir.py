"""Cox-Ingersoll-Ross short-rate model."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CIRModel:
    def __init__(self, a: float, b: float, sigma: float, r0: float):
        self.a = float(a)
        self.b = float(b)
        self.sigma = float(sigma)
        self.r0 = float(r0)

    def feller_condition(self) -> bool:
        return 2.0 * self.a * self.b >= self.sigma * self.sigma

    def bond_price(self, t: float, T: float, r_t: float | None = None) -> float:
        if T < t:
            raise ValueError("Maturity T must be >= t.")
        if r_t is None:
            r_t = self.r0
        tau = T - t
        gamma = np.sqrt(self.a * self.a + 2.0 * self.sigma * self.sigma)
        exp_gamma_tau = np.exp(gamma * tau)
        denom = (gamma + self.a) * (exp_gamma_tau - 1.0) + 2.0 * gamma
        B = 2.0 * (exp_gamma_tau - 1.0) / denom
        A = ((2.0 * gamma * np.exp((self.a + gamma) * tau / 2.0)) / denom) ** (
            2.0 * self.a * self.b / (self.sigma * self.sigma)
        )
        return float(A * np.exp(-B * r_t))

    def simulate_paths(
        self, n_paths: int = 100, n_steps: int = 250, horizon: float = 5.0, seed: int = 42
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dt = horizon / n_steps
        times = np.linspace(0.0, horizon, n_steps + 1)
        rates = np.zeros((n_paths, n_steps + 1))
        rates[:, 0] = max(self.r0, 0.0)

        for i in range(n_steps):
            z = rng.standard_normal(n_paths)
            rt = np.maximum(rates[:, i], 0.0)
            dr = self.a * (self.b - rt) * dt + self.sigma * np.sqrt(rt) * np.sqrt(dt) * z
            rates[:, i + 1] = np.maximum(rt + dr, 0.0)

        cols = [f"path_{i+1}" for i in range(n_paths)]
        df = pd.DataFrame(rates.T, columns=cols)
        df.insert(0, "time", times)
        return df
