"""Options Pricing Laboratory - Streamlit dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import black_scholes, binomial_tree, finite_difference, greeks as greek_mod, monte_carlo
from data.market_data import SPOT_SNAPSHOT, sample_option_chain
from data.volatility_surfaces import equity_smile_surface
from exotics import asians, barriers, compounds, digitals, lookbacks
from risk import scenario, stress_test, var
from visualization import heatmaps, payoff_diagrams, surface_plots
from volatility import implied, local_vol, smile, stochastic_vol, surface


st.set_page_config(page_title="Options Pricing Lab", layout="wide")
st.title("Options Pricing Laboratory")
st.caption("Interactive sandbox for vanilla options, exotics, volatility models, and risk analytics.")


def tutorial(page: str) -> None:
    with st.expander("Quick Tutorial"):
        tips = {
            "Vanilla Pricing": "Set contract parameters, compare Black-Scholes price and Greeks, and inspect payoff profile.",
            "Numerical Methods": "Benchmark Binomial / Monte Carlo / Finite Difference against Black-Scholes.",
            "Exotics": "Price barriers, digitals, asians, lookbacks, and compound options with intuitive controls.",
            "Volatility Lab": "Build implied-vol surfaces, compute local vol, and explore SABR/Heston behavior.",
            "Risk Lab": "Run scenario shocks, stress tests, and VaR calculations.",
            "Visualization Studio": "Generate payoff charts, Greeks heatmaps, and 3D surfaces.",
        }
        st.write(tips.get(page, ""))


asset = st.sidebar.selectbox("Underlying preset", options=list(SPOT_SNAPSHOT.keys()))
preset = SPOT_SNAPSHOT[asset]

page = st.sidebar.radio(
    "Section",
    [
        "Vanilla Pricing",
        "Numerical Methods",
        "Exotics",
        "Volatility Lab",
        "Risk Lab",
        "Visualization Studio",
    ],
)
tutorial(page)

with st.sidebar:
    st.subheader("Common Inputs")
    spot = st.number_input("Spot", min_value=0.01, value=float(preset["spot"]), step=float(max(preset["spot"] * 0.01, 0.01)))
    strike = st.number_input("Strike", min_value=0.01, value=float(preset["spot"]), step=float(max(preset["spot"] * 0.01, 0.01)))
    maturity = st.slider("Maturity (years)", 0.01, 3.0, 0.5, 0.01)
    rate = st.slider("Risk-free rate", -0.05, 0.15, float(preset["rate"]), 0.001)
    dividend = st.slider("Dividend/Carry", 0.0, 0.08, float(preset["dividend"]), 0.001)
    vol = st.slider("Volatility", 0.01, 1.50, 0.25, 0.005)
    option_type = st.selectbox("Option type", ["call", "put"])


if page == "Vanilla Pricing":
    c1, c2 = st.columns([1, 1])
    with c1:
        price_bs = black_scholes.price(spot, strike, maturity, rate, vol, option_type=option_type, dividend=dividend)
        st.metric("Black-Scholes Price", f"{price_bs:.6f}")
        g = black_scholes.greeks(spot, strike, maturity, rate, vol, option_type=option_type, dividend=dividend)
        greek_df = pd.DataFrame({"Greek": list(g.keys()), "Value": list(g.values())})
        st.dataframe(greek_df, use_container_width=True)
        if st.toggle("Show Formula"):
            st.latex(r"d_1=\frac{\ln(S/K)+(r-q+\frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2=d_1-\sigma\sqrt{T}")
            st.latex(r"C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2), \quad P=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)")
    with c2:
        payoff_df = payoff_diagrams.vanilla_payoff_grid(strike=strike, option_type=option_type, premium=price_bs)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=payoff_df["spot_at_expiry"], y=payoff_df["payoff"], name="Payoff"))
        fig.add_trace(go.Scatter(x=payoff_df["spot_at_expiry"], y=payoff_df["pnl"], name="PnL"))
        fig.update_layout(title="Payoff Diagram", xaxis_title="Spot at Expiry", yaxis_title="Value", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)


elif page == "Numerical Methods":
    st.subheader("Numerical Pricing Comparison")
    steps = st.slider("Binomial steps", 25, 1000, 250, 25)
    n_paths = st.slider("Monte Carlo paths", 5000, 100000, 30000, 5000)
    n_t = st.slider("Finite difference time steps", 200, 4000, 1200, 100)

    bs = black_scholes.price(spot, strike, maturity, rate, vol, option_type=option_type, dividend=dividend)
    bt_eur = binomial_tree.price(
        spot, strike, maturity, rate, vol, steps=steps, option_type=option_type, american=False, dividend=dividend
    )
    bt_am = binomial_tree.price(
        spot, strike, maturity, rate, vol, steps=steps, option_type=option_type, american=True, dividend=dividend
    )
    mc = monte_carlo.european_price(
        spot, strike, maturity, rate, vol, option_type=option_type, n_paths=n_paths, dividend=dividend
    )
    fd = finite_difference.explicit_european(spot, strike, maturity, rate, vol, option_type=option_type, n_t=n_t)

    comp = pd.DataFrame(
        {
            "Method": ["Black-Scholes", "Binomial (European)", "Binomial (American)", "Monte Carlo", "Finite Difference"],
            "Price": [bs, bt_eur, bt_am, mc, fd],
        }
    )
    comp["Error vs BS"] = comp["Price"] - bs
    st.dataframe(comp, use_container_width=True)
    fig = px.bar(comp, x="Method", y="Error vs BS", title="Numerical Error vs Black-Scholes")
    st.plotly_chart(fig, use_container_width=True)


elif page == "Exotics":
    st.subheader("Exotic Options Playground")
    exotic = st.selectbox("Exotic product", ["Digital", "Barrier", "Asian", "Lookback", "Compound"])

    if exotic == "Digital":
        payout = st.number_input("Digital payout", min_value=0.01, value=1.0, step=0.5)
        digital_px = digitals.cash_or_nothing(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            vol=vol,
            payout=payout,
            option_type=option_type,
            dividend=dividend,
        )
        st.metric("Digital Option Price", f"{digital_px:.6f}")

    elif exotic == "Barrier":
        style = st.selectbox("Barrier style", ["Up-and-Out Call", "Down-and-Out Put"])
        barrier = st.number_input("Barrier level", min_value=0.01, value=float(spot * (1.2 if style.startswith("Up") else 0.8)))
        if style == "Up-and-Out Call":
            px_bar = barriers.up_and_out_call_mc(spot, strike, barrier, maturity, rate, vol)
        else:
            px_bar = barriers.down_and_out_put_mc(spot, strike, barrier, maturity, rate, vol)
        st.metric("Barrier Price (MC)", f"{px_bar:.6f}")
        st.caption("Monte Carlo barrier pricing with discrete monitoring at simulation steps.")

    elif exotic == "Asian":
        geo = asians.geometric_asian_call_bs(spot, strike, maturity, rate, vol)
        ari = asians.arithmetic_asian_mc(spot, strike, maturity, rate, vol, option_type=option_type)
        st.metric("Geometric Asian (closed-form call)", f"{geo:.6f}")
        st.metric(f"Arithmetic Asian ({option_type}, MC)", f"{ari:.6f}")

    elif exotic == "Lookback":
        lb_call = lookbacks.floating_strike_call_mc(spot, maturity, rate, vol)
        lb_put = lookbacks.floating_strike_put_mc(spot, maturity, rate, vol)
        c1, c2 = st.columns(2)
        c1.metric("Floating-Strike Lookback Call", f"{lb_call:.6f}")
        c2.metric("Floating-Strike Lookback Put", f"{lb_put:.6f}")

    else:
        strike_compound = st.number_input("Compound strike (paid at t1)", min_value=0.0, value=5.0, step=0.5)
        t1 = st.slider("Compound option expiry t1", 0.05, float(max(maturity - 0.05, 0.1)), min(0.25, float(max(maturity - 0.05, 0.1))), 0.01)
        t2 = st.slider("Underlying option expiry t2", float(t1 + 0.05), 3.0, float(max(maturity, t1 + 0.2)), 0.01)
        comp_px = compounds.call_on_call_mc(
            spot=spot,
            strike_compound=strike_compound,
            strike_underlying=strike,
            t1=t1,
            t2=t2,
            rate=rate,
            vol=vol,
        )
        st.metric("Call on Call (nested MC)", f"{comp_px:.6f}")


elif page == "Volatility Lab":
    st.subheader("Volatility Surface and Models")
    if "vol_points" not in st.session_state:
        st.session_state.vol_points = equity_smile_surface()

    choice = st.selectbox("Data source", ["Sample surface", "Manual editor"])
    if choice == "Sample surface":
        st.session_state.vol_points = sample_option_chain()
    points = st.data_editor(st.session_state.vol_points, num_rows="dynamic", use_container_width=True, key="vol_points_editor")
    st.session_state.vol_points = points

    interp = surface.build_surface(points)
    t_grid = np.linspace(float(points["maturity"].min()), float(points["maturity"].max()), 25)
    k_grid = np.linspace(float(points["strike"].min()), float(points["strike"].max()), 25)
    surf_df = surface.evaluate_surface(interp, t_grid, k_grid).dropna()
    st.plotly_chart(surface_plots.vol_surface_3d(surf_df), use_container_width=True)

    # Smile slice
    t_slice = st.slider("Smile maturity slice", float(points["maturity"].min()), float(points["maturity"].max()), float(points["maturity"].median()), 0.01)
    smile_df = points.iloc[(points["maturity"] - t_slice).abs().argsort()[:8]].sort_values("strike")[["strike", "implied_vol"]]
    metrics = smile.smile_metrics(smile_df, atm_strike=strike)
    fig_smile = px.line(smile_df, x="strike", y="implied_vol", markers=True, title=f"Smile near maturity {t_slice:.2f}")
    st.plotly_chart(fig_smile, use_container_width=True)
    st.json(metrics)

    # Local vol
    local_df = local_vol.dupire_local_vol(surf_df, rate=rate)
    st.plotly_chart(surface_plots.local_vol_3d(local_df), use_container_width=True)

    # SABR
    st.markdown("### SABR Smile Generator")
    fwd = st.number_input("Forward", min_value=0.0001, value=float(spot))
    alpha = st.slider("SABR alpha", 0.01, 1.0, 0.25, 0.01)
    beta = st.slider("SABR beta", 0.0, 1.0, 0.6, 0.05)
    rho = st.slider("SABR rho", -0.99, 0.99, -0.2, 0.01)
    nu = st.slider("SABR nu", 0.01, 2.0, 0.8, 0.01)
    ks = np.linspace(0.7 * fwd, 1.3 * fwd, 21)
    sabr_df = stochastic_vol.sample_sabr_smile(fwd=fwd, maturity=maturity, alpha=alpha, beta=beta, rho=rho, nu=nu, strikes=ks)
    st.plotly_chart(px.line(sabr_df, x="strike", y="implied_vol", title="SABR Implied Vol Smile"), use_container_width=True)

    with st.expander("Implied vol from market price"):
        mkt_price = st.number_input("Market option price", min_value=0.0001, value=max(black_scholes.price(spot, strike, maturity, rate, vol, option_type), 0.0001))
        iv = implied.implied_volatility(mkt_price, spot, strike, maturity, rate, option_type=option_type, dividend=dividend)
        st.write(f"Implied volatility: **{iv:.4%}**")


elif page == "Risk Lab":
    st.subheader("Risk Analytics")
    with st.expander("Scenario Analysis"):
        ds = st.slider("Spot shift (%)", -40, 40, 0, 1)
        dv = st.slider("Vol shift (%)", -50, 100, 0, 1)
        dr = st.slider("Rate shift (bps)", -300, 300, 0, 5)
        decay = st.slider("Time decay (days)", 0, 30, 0, 1)
        res = scenario.scenario_reprice(
            spot=spot,
            strike=strike,
            maturity=maturity,
            rate=rate,
            vol=vol,
            option_type=option_type,
            spot_shift_pct=float(ds),
            vol_shift_pct=float(dv),
            rate_shift_bps=float(dr),
            time_decay_days=int(decay),
        )
        st.json(res)
        grid_df = scenario.scenario_grid(spot, strike, maturity, rate, vol, option_type)
        st.plotly_chart(
            px.density_heatmap(grid_df, x="spot_shift_pct", y="vol_shift_pct", z="pnl", color_continuous_scale="RdBu"),
            use_container_width=True,
        )

    with st.expander("Stress Tests"):
        stress_df = stress_test.run_stress_tests(spot, strike, maturity, rate, vol, option_type=option_type)
        st.dataframe(stress_df, use_container_width=True)
        st.plotly_chart(px.bar(stress_df, x="scenario", y="pnl", title="Stress PnL"), use_container_width=True)

    with st.expander("Value at Risk"):
        position = st.number_input("Position size (contracts)", value=1.0, step=1.0)
        conf = st.selectbox("Confidence", [0.95, 0.99], index=1)
        horizon = st.slider("Horizon (days)", 1, 20, 1)
        param_var = var.parametric_var_delta_gamma(
            spot, strike, maturity, rate, vol, option_type=option_type, position=position, horizon_days=horizon, confidence=conf
        )
        mc_var = var.monte_carlo_var_full_reval(
            spot, strike, maturity, rate, vol, option_type=option_type, position=position, horizon_days=horizon, confidence=conf
        )
        c1, c2 = st.columns(2)
        c1.metric("Parametric VaR", f"{param_var:.6f}")
        c2.metric("Monte Carlo VaR", f"{mc_var:.6f}")


else:
    st.subheader("Visualization Studio")
    studio = st.selectbox("Visualization type", ["Payoff", "Greeks Heatmap", "Heston Paths"])

    if studio == "Payoff":
        premium = black_scholes.price(spot, strike, maturity, rate, vol, option_type=option_type, dividend=dividend)
        vanilla = payoff_diagrams.vanilla_payoff_grid(strike, option_type=option_type, premium=premium)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=vanilla["spot_at_expiry"], y=vanilla["payoff"], name="Payoff"))
        fig.add_trace(go.Scatter(x=vanilla["spot_at_expiry"], y=vanilla["pnl"], name="PnL"))
        fig.update_layout(title="Vanilla Option Payoff and PnL", xaxis_title="Spot at Expiry")
        st.plotly_chart(fig, use_container_width=True)
        straddle = payoff_diagrams.straddle_payoff_grid(strike, premium_call=premium, premium_put=premium)
        st.plotly_chart(px.line(straddle, x="spot_at_expiry", y="pnl", title="Long Straddle PnL"), use_container_width=True)

    elif studio == "Greeks Heatmap":
        greek_name = st.selectbox("Greek", ["delta", "gamma", "vega", "theta", "rho"])
        s_grid = np.linspace(0.6 * strike, 1.4 * strike, 35)
        v_grid = np.linspace(0.05, 1.0, 30)
        hm = heatmaps.greek_heatmap(
            strike=strike,
            maturity=maturity,
            rate=rate,
            option_type=option_type,
            spot_grid=s_grid,
            vol_grid=v_grid,
            greek_name=greek_name,
        )
        st.plotly_chart(px.density_heatmap(hm, x="spot", y="vol", z="value", color_continuous_scale="Viridis"), use_container_width=True)

    else:
        st.markdown("Heston stochastic volatility simulation")
        v0 = st.slider("Initial variance v0", 0.0001, 0.5, 0.04, 0.0001)
        kappa = st.slider("kappa", 0.1, 10.0, 2.0, 0.1)
        theta = st.slider("theta", 0.0001, 0.5, 0.04, 0.0001)
        xi = st.slider("xi", 0.01, 2.0, 0.6, 0.01)
        rho_h = st.slider("rho", -0.99, 0.99, -0.5, 0.01)
        s_paths, v_paths = stochastic_vol.heston_paths(
            spot=spot,
            v0=v0,
            rate=rate,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho_h,
            maturity=maturity,
            n_paths=100,
            n_steps=252,
        )
        t = np.linspace(0.0, maturity, s_paths.shape[1])
        fig_s = go.Figure()
        for i in range(min(20, s_paths.shape[0])):
            fig_s.add_trace(go.Scatter(x=t, y=s_paths[i], mode="lines", line={"width": 1}, showlegend=False))
        fig_s.update_layout(title="Heston Spot Paths", xaxis_title="Time", yaxis_title="Spot")
        st.plotly_chart(fig_s, use_container_width=True)
        fig_v = go.Figure()
        for i in range(min(20, v_paths.shape[0])):
            fig_v.add_trace(go.Scatter(x=t, y=v_paths[i], mode="lines", line={"width": 1}, showlegend=False))
        fig_v.update_layout(title="Heston Variance Paths", xaxis_title="Time", yaxis_title="Variance")
        st.plotly_chart(fig_v, use_container_width=True)

st.markdown("---")
st.caption("Educational use only. Models are simplified and not intended for production trading/risk decisions.")
