"""Yield curve bootstrapping and analytics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def tenor_to_years(tenor: str) -> float:
    tenor = tenor.strip().upper()
    if tenor.endswith("M"):
        return float(tenor[:-1]) / 12.0
    if tenor.endswith("Y"):
        return float(tenor[:-1])
    raise ValueError(f"Unsupported tenor: {tenor}")


@dataclass
class YieldCurve:
    """Simple educational curve object with deposit + swap bootstrapping."""

    currency: str = "USD"
    valuation_date: date | None = None
    deposits: List[Tuple[str, float]] = field(default_factory=list)
    swaps: List[Tuple[str, float]] = field(default_factory=list)
    discount_factors: Dict[float, float] = field(default_factory=lambda: {0.0: 1.0})

    def add_deposit(self, tenor: str, rate_pct: float) -> None:
        self.deposits.append((tenor.upper(), float(rate_pct)))

    def add_swap(self, tenor: str, rate_pct: float) -> None:
        self.swaps.append((tenor.upper(), float(rate_pct)))

    def clear_instruments(self) -> None:
        self.deposits.clear()
        self.swaps.clear()
        self.discount_factors = {0.0: 1.0}

    def bootstrap(self) -> None:
        """Bootstrap discount factors sequentially from deposits and swaps."""
        dfs: Dict[float, float] = {0.0: 1.0}

        dep_nodes = sorted((tenor_to_years(t), r / 100.0) for t, r in self.deposits)
        for t, r in dep_nodes:
            if t <= 0:
                raise ValueError("Deposit maturity must be positive.")
            dfs[t] = 1.0 / (1.0 + r * t)

        swap_nodes = sorted((tenor_to_years(t), r / 100.0) for t, r in self.swaps)
        known_times: List[float] = sorted(dfs.keys())

        # Irregular-node par swap approximation:
        # S * sum_{j<=i}(delta_j * DF_j) = 1 - DF_i
        # Solve for DF_i sequentially.
        for maturity, swap_rate in swap_nodes:
            if maturity <= max(known_times):
                raise ValueError(
                    "Swap maturities must be strictly increasing and after deposit nodes."
                )
            prev_times = [t for t in known_times if t > 0.0]
            times_for_leg = prev_times + [maturity]
            deltas = np.diff([0.0] + times_for_leg).tolist()

            fixed_sum_known = 0.0
            for t, delta in zip(prev_times, deltas[:-1]):
                fixed_sum_known += delta * dfs[t]

            last_delta = deltas[-1]
            numerator = 1.0 - swap_rate * fixed_sum_known
            denominator = 1.0 + swap_rate * last_delta
            if denominator <= 0.0 or numerator <= 0.0:
                raise ValueError(
                    "Invalid instrument set produced non-positive discount factor during bootstrapping."
                )
            dfs[maturity] = numerator / denominator
            known_times = sorted(dfs.keys())

        self.discount_factors = dict(sorted(dfs.items()))

    def get_nodes(self) -> pd.DataFrame:
        times = np.array(sorted(self.discount_factors.keys()))
        dfs = np.array([self.discount_factors[t] for t in times])
        zeros = np.where(times > 0, -np.log(dfs) / np.maximum(times, 1e-12), 0.0)
        return pd.DataFrame(
            {
                "time_years": times,
                "discount_factor": dfs,
                "zero_rate": zeros,
                "zero_rate_pct": zeros * 100.0,
            }
        )

    def get_discount_factor(self, t: float) -> float:
        if t < 0:
            raise ValueError("Time must be non-negative.")
        if t == 0:
            return 1.0
        times = np.array(sorted(self.discount_factors.keys()))
        dfs = np.array([self.discount_factors[x] for x in times])
        if t in self.discount_factors:
            return self.discount_factors[t]
        zero = np.where(times > 0, -np.log(dfs) / np.maximum(times, 1e-12), 0.0)
        z_t = np.interp(t, times, zero, left=zero[1] if len(zero) > 1 else zero[0], right=zero[-1])
        return float(np.exp(-z_t * t))

    def zero_rate(self, t: float) -> float:
        if t <= 0:
            return 0.0
        df = self.get_discount_factor(t)
        return float(-np.log(max(df, 1e-12)) / t)

    def forward_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1.")
        df1 = self.get_discount_factor(t1)
        df2 = self.get_discount_factor(t2)
        return float((df1 / df2 - 1.0) / (t2 - t1))

    def forward_curve(self, horizon: float = 10.0, step: float = 0.25) -> pd.DataFrame:
        grid = np.arange(step, horizon + step, step)
        rows = []
        for t in grid:
            t1 = max(0.0, t - step)
            rows.append((t, self.forward_rate(t1, t)))
        return pd.DataFrame(rows, columns=["time_years", "forward_rate"])

    def plot_discount_factors(self) -> go.Figure:
        nodes = self.get_nodes()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=nodes["time_years"],
                y=nodes["discount_factor"],
                mode="lines+markers",
                name="Discount Factor",
            )
        )
        fig.update_layout(
            title=f"{self.currency} Discount Factors",
            xaxis_title="Maturity (years)",
            yaxis_title="Discount Factor",
            template="plotly_white",
        )
        return fig

    def plot_zero_rates(self) -> go.Figure:
        nodes = self.get_nodes()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=nodes["time_years"],
                y=nodes["zero_rate_pct"],
                mode="lines+markers",
                name="Zero Rate (%)",
            )
        )
        fig.update_layout(
            title=f"{self.currency} Zero Curve",
            xaxis_title="Maturity (years)",
            yaxis_title="Zero Rate (%)",
            template="plotly_white",
        )
        return fig

    def plot_forward_rates(self, horizon: float = 10.0, step: float = 0.25) -> go.Figure:
        fwds = self.forward_curve(horizon=horizon, step=step)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fwds["time_years"],
                y=fwds["forward_rate"] * 100.0,
                mode="lines",
                name="Forward Rate (%)",
            )
        )
        fig.update_layout(
            title=f"{self.currency} Forward Curve",
            xaxis_title="Maturity (years)",
            yaxis_title="Forward Rate (%)",
            template="plotly_white",
        )
        return fig

    def export_csv(self, path: str) -> None:
        self.get_nodes().to_csv(path, index=False)

    def export_json(self, path: str) -> None:
        payload = {
            "currency": self.currency,
            "valuation_date": str(self.valuation_date) if self.valuation_date else None,
            "nodes": self.get_nodes().to_dict(orient="records"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def to_json_string(self) -> str:
        payload = {
            "currency": self.currency,
            "nodes": self.get_nodes().to_dict(orient="records"),
        }
        return json.dumps(payload, indent=2)

    def copy(self) -> "YieldCurve":
        c = YieldCurve(currency=self.currency, valuation_date=self.valuation_date)
        c.deposits = list(self.deposits)
        c.swaps = list(self.swaps)
        c.discount_factors = dict(self.discount_factors)
        return c

    @classmethod
    def from_nodes(
        cls, times: Iterable[float], dfs: Iterable[float], currency: str = "USD"
    ) -> "YieldCurve":
        curve = cls(currency=currency)
        curve.discount_factors = {0.0: 1.0}
        for t, df in zip(times, dfs):
            if t > 0:
                curve.discount_factors[float(t)] = float(df)
        curve.discount_factors = dict(sorted(curve.discount_factors.items()))
        return curve

    def bumped_parallel(self, bps: float) -> "YieldCurve":
        nodes = self.get_nodes()
        times = nodes["time_years"].to_numpy()
        zeros = nodes["zero_rate"].to_numpy()
        shifted_zeros = zeros + bps / 10000.0
        dfs = np.exp(-shifted_zeros * times)
        return YieldCurve.from_nodes(times, dfs, currency=self.currency)

    def bumped_key_rate(self, key_time: float, bps: float, bandwidth: float = 1.5) -> "YieldCurve":
        nodes = self.get_nodes()
        times = nodes["time_years"].to_numpy()
        zeros = nodes["zero_rate"].to_numpy()
        distances = np.abs(times - key_time)
        weights = np.clip(1.0 - distances / bandwidth, 0.0, 1.0)
        shifted_zeros = zeros + (bps / 10000.0) * weights
        dfs = np.exp(-shifted_zeros * times)
        return YieldCurve.from_nodes(times, dfs, currency=self.currency)
