"""Plotly helpers for 3D volatility surfaces."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def vol_surface_3d(surface_df: pd.DataFrame, title: str = "Implied Volatility Surface") -> go.Figure:
    pivot = surface_df.pivot(index="maturity", columns="strike", values="implied_vol")
    x = pivot.columns.to_numpy()
    y = pivot.index.to_numpy()
    z = pivot.to_numpy()
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Viridis")])
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Maturity",
            "zaxis_title": "Implied Vol",
        },
    )
    return fig


def local_vol_3d(local_df: pd.DataFrame, title: str = "Local Volatility Surface") -> go.Figure:
    pivot = local_df.pivot(index="maturity", columns="strike", values="local_vol")
    x = pivot.columns.to_numpy()
    y = pivot.index.to_numpy()
    z = pivot.to_numpy()
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, colorscale="Plasma")])
    fig.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Maturity",
            "zaxis_title": "Local Vol",
        },
    )
    return fig
