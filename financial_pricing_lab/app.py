"""Financial Pricing Lab - interactive derivatives learning platform."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.curve import YieldCurve
from data.sample_curves import SAMPLE_CURVES
from models.black import BlackModel
from models.cir import CIRModel
from models.hull_white import HullWhiteModel
from models.vasicek import VasicekModel
from products.bond import BondPricer
from products.swap import InterestRateSwap


st.set_page_config(page_title="Financial Pricing Lab", layout="wide")


def tutorial_block(page: str) -> None:
    with st.expander("Quick Tutorial"):
        tutorials = {
            "Curve Bootstrapping": """
1. Choose a sample curve or edit market quotes directly.
2. Click **Bootstrap Curve** to build discount factors from short deposits + longer swaps.
3. Inspect discount, zero, and forward curves to understand term structure shape.
4. Export the resulting curve for offline analysis.
""",
            "Interest Rate Models": """
1. Select a short-rate model and tune mean-reversion / volatility.
2. Compare bond prices implied by model dynamics.
3. Use Monte Carlo paths to build intuition for rate evolution.
4. Toggle formulas to link implementation with theory.
""",
            "Product Pricing": """
1. Select Bond or Swap and enter contract terms.
2. Price using the bootstrapped curve and inspect risk metrics.
3. Review cash flow schedules and sensitivity measures.
""",
            "Risk Analysis": """
1. Define shock scenarios (parallel, steepener, flattener).
2. Measure P&L impact for Bonds/Swaps under each scenario.
3. Inspect key rate duration concentration by maturity bucket.
""",
            "Model Comparison": """
1. Set common parameters for Vasicek, CIR, and Hull-White.
2. Compare model bond prices against market-implied prices.
3. Review MAE/RMSE diagnostics to gauge fit quality.
""",
        }
        st.markdown(tutorials.get(page, ""))


def sample_to_frames(sample_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = SAMPLE_CURVES[sample_name]
    dep_df = pd.DataFrame(sample["deposits"], columns=["Tenor", "Rate (%)"])
    swap_df = pd.DataFrame(sample["swaps"], columns=["Tenor", "Rate (%)"])
    return dep_df, swap_df


def bootstrap_from_frames(currency: str, dep_df: pd.DataFrame, swap_df: pd.DataFrame) -> YieldCurve:
    curve = YieldCurve(currency=currency)
    for _, row in dep_df.iterrows():
        curve.add_deposit(str(row["Tenor"]), float(row["Rate (%)"]))
    for _, row in swap_df.iterrows():
        curve.add_swap(str(row["Tenor"]), float(row["Rate (%)"]))
    curve.bootstrap()
    return curve


def ensure_curve() -> YieldCurve:
    if "curve" not in st.session_state:
        dep, sw = sample_to_frames("USD")
        st.session_state.curve = bootstrap_from_frames("USD", dep, sw)
    return st.session_state.curve


def plot_paths(paths: pd.DataFrame, title: str, max_paths: int = 20) -> go.Figure:
    fig = go.Figure()
    cols = [c for c in paths.columns if c.startswith("path_")][:max_paths]
    for c in cols:
        fig.add_trace(go.Scatter(x=paths["time"], y=paths[c] * 100.0, mode="lines", line={"width": 1}, name=c))
    fig.update_layout(title=title, xaxis_title="Time (years)", yaxis_title="Short rate (%)", template="plotly_white")
    return fig


def price_bond_under_scenario(curve: YieldCurve, face: float, coupon: float, maturity: float, freq: int) -> float:
    return BondPricer.coupon_bond_price(face=face, coupon_rate_pct=coupon, maturity=maturity, curve=curve, frequency=freq)


def price_swap_under_scenario(curve: YieldCurve, notional: float, fixed_rate: float, maturity: float, freq: int) -> float:
    swap = InterestRateSwap(notional=notional, fixed_rate_pct=fixed_rate, maturity=maturity, pay_freq=freq)
    return swap.npv(curve, payer_fixed=True)


st.title("Financial Pricing Lab")
st.caption("Interactive environment for yield curves, interest rate models, and derivatives pricing.")

page = st.sidebar.radio(
    "Navigate",
    [
        "Curve Bootstrapping",
        "Interest Rate Models",
        "Product Pricing",
        "Risk Analysis",
        "Model Comparison",
    ],
)

tutorial_block(page)


if page == "Curve Bootstrapping":
    st.subheader("Page 1: Curve Bootstrapping")
    st.write("Build discount factors from market deposits and swaps.")
    with st.expander("Academic references"):
        st.markdown(
            "- John Hull, *Options, Futures, and Other Derivatives*, curve construction chapters.\n"
            "- Brigo & Mercurio, *Interest Rate Models: Theory and Practice*."
        )

    sample_name = st.selectbox("Sample curve", options=list(SAMPLE_CURVES.keys()), help="Loads pre-configured market quotes.")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Load sample quotes"):
            dep_df, swap_df = sample_to_frames(sample_name)
            st.session_state.dep_df = dep_df
            st.session_state.swap_df = swap_df
    with c2:
        if st.button("Bootstrap curve"):
            try:
                dep_df = st.session_state.get("dep_df", sample_to_frames(sample_name)[0])
                swap_df = st.session_state.get("swap_df", sample_to_frames(sample_name)[1])
                curve = bootstrap_from_frames(sample_name, dep_df, swap_df)
                st.session_state.curve = curve
                st.success("Curve bootstrapped successfully.")
            except Exception as exc:
                st.error(f"Bootstrapping failed: {exc}")

    dep_default, sw_default = sample_to_frames(sample_name)
    st.caption("Deposit quotes: short tenors anchor the money-market segment of the curve.")
    dep_df = st.data_editor(
        st.session_state.get("dep_df", dep_default),
        num_rows="dynamic",
        use_container_width=True,
        key="dep_editor",
    )
    st.caption("Swap quotes: medium/long tenors shape the rest of the curve.")
    swap_df = st.data_editor(
        st.session_state.get("swap_df", sw_default),
        num_rows="dynamic",
        use_container_width=True,
        key="swap_editor",
    )
    st.session_state.dep_df = dep_df
    st.session_state.swap_df = swap_df

    curve = ensure_curve()
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(curve.plot_discount_factors(), use_container_width=True)
    with c2:
        st.plotly_chart(curve.plot_zero_rates(), use_container_width=True)
    st.plotly_chart(curve.plot_forward_rates(horizon=10.0, step=0.25), use_container_width=True)

    nodes = curve.get_nodes()
    st.dataframe(nodes, use_container_width=True)
    csv_bytes = nodes.to_csv(index=False).encode("utf-8")
    st.download_button("Export curve CSV", csv_bytes, file_name=f"{curve.currency.lower()}_curve.csv", mime="text/csv")
    st.download_button(
        "Export curve JSON",
        curve.to_json_string().encode("utf-8"),
        file_name=f"{curve.currency.lower()}_curve.json",
        mime="application/json",
    )

    if st.toggle("Show Formula: Bootstrap logic"):
        st.latex(r"S_i\sum_{j \le i}\Delta_j P(0,t_j) = 1 - P(0,t_i)")
        st.latex(r"P(0,t_i)=\frac{1-S_i\sum_{j<i}\Delta_jP(0,t_j)}{1+S_i\Delta_i}")


elif page == "Interest Rate Models":
    st.subheader("Page 2: Interest Rate Models")
    curve = ensure_curve()
    model_name = st.selectbox("Select model", ["Vasicek", "CIR", "Hull-White"])
    n_paths = st.slider("Monte Carlo paths", 20, 300, 80, help="More paths improve smoothness but increase runtime.")
    n_steps = st.slider("Time steps", 50, 500, 250)
    horizon = st.slider("Simulation horizon (years)", 1, 30, 10)

    a = st.slider("Mean reversion a", 0.01, 2.00, 0.20, 0.01)
    b = st.slider("Long-run level b", 0.0, 0.20, 0.04, 0.001)
    sigma = st.slider("Volatility sigma", 0.001, 0.20, 0.02, 0.001)
    r0 = st.slider("Initial short rate r0", 0.0, 0.20, 0.04, 0.001)

    maturities = np.linspace(0.5, 30.0, 60)
    fig_prices = go.Figure()

    if model_name == "Vasicek":
        model = VasicekModel(a=a, b=b, sigma=sigma, r0=r0)
        prices = [model.bond_price(0.0, t) for t in maturities]
        paths = model.simulate_paths(n_paths=n_paths, n_steps=n_steps, horizon=float(horizon))
        fig_prices.add_trace(go.Scatter(x=maturities, y=prices, mode="lines", name="Vasicek"))
        st.plotly_chart(plot_paths(paths, "Vasicek Short-Rate Paths"), use_container_width=True)
        if st.toggle("Show Formula: Vasicek"):
            st.latex(r"dr_t = a(b-r_t)dt + \sigma dW_t")
            st.latex(r"P(t,T)=A(t,T)e^{-B(t,T)r_t}")

    elif model_name == "CIR":
        model = CIRModel(a=a, b=b, sigma=sigma, r0=r0)
        prices = [model.bond_price(0.0, t) for t in maturities]
        vas_ref = VasicekModel(a=a, b=b, sigma=sigma, r0=r0)
        vas_prices = [vas_ref.bond_price(0.0, t) for t in maturities]
        paths = model.simulate_paths(n_paths=n_paths, n_steps=n_steps, horizon=float(horizon))
        fig_prices.add_trace(go.Scatter(x=maturities, y=prices, mode="lines", name="CIR"))
        fig_prices.add_trace(go.Scatter(x=maturities, y=vas_prices, mode="lines", name="Vasicek reference"))
        st.plotly_chart(plot_paths(paths, "CIR Short-Rate Paths"), use_container_width=True)
        st.info(f"Feller condition 2ab >= sigma^2: {'Satisfied' if model.feller_condition() else 'Not satisfied'}")
        if st.toggle("Show Formula: CIR"):
            st.latex(r"dr_t = a(b-r_t)dt + \sigma\sqrt{r_t}dW_t")

    else:
        hw = HullWhiteModel(a=a, sigma=sigma, r0=r0)
        theta_df = hw.calibrate_theta(curve, horizon=float(horizon), step=0.05)
        prices = [hw.bond_price(0.0, t) for t in maturities]
        paths = hw.simulate_paths(n_paths=n_paths, n_steps=n_steps, horizon=float(horizon))
        fig_prices.add_trace(go.Scatter(x=maturities, y=prices, mode="lines", name="Hull-White"))
        st.plotly_chart(plot_paths(paths, "Hull-White Short-Rate Paths"), use_container_width=True)
        fig_theta = go.Figure()
        fig_theta.add_trace(go.Scatter(x=theta_df["time"], y=theta_df["theta"], mode="lines", name="theta(t)"))
        fig_theta.update_layout(title="Calibrated theta(t)", xaxis_title="Time", yaxis_title="Theta", template="plotly_white")
        st.plotly_chart(fig_theta, use_container_width=True)
        with st.expander("Calibrate to swaption quote (educational approximation)"):
            market_price = st.number_input("Market swaption price", min_value=0.0, value=0.01, step=0.001)
            expiry = st.number_input("Expiry (years)", min_value=0.25, value=2.0, step=0.25)
            tenor = st.number_input("Underlying swap tenor (years)", min_value=1.0, value=5.0, step=0.5)
            swap_tmp = InterestRateSwap(notional=1.0, fixed_rate_pct=0.0, maturity=tenor, pay_freq=2)
            fwd = swap_tmp.par_rate(curve)
            ann = swap_tmp.annuity(curve)
            imp_vol = BlackModel.implied_vol(
                market_price=max(market_price, 1e-9), forward=fwd, strike=fwd, expiry=expiry, annuity=ann
            )
            st.write(f"Implied Black vol from market price: **{imp_vol:.4%}**")
            sigma_fit = imp_vol * a / max(1.0 - np.exp(-a * expiry), 1e-8)
            st.write(f"Approximate Hull-White sigma from target swaption vol: **{sigma_fit:.4%}**")
            st.caption("Use this implied vol as a target reference when adjusting Hull-White sigma.")
        if st.toggle("Show Formula: Hull-White"):
            st.latex(r"dr_t = (\theta(t)-ar_t)dt + \sigma dW_t")
            st.latex(r"\theta(t)=\frac{\partial f(0,t)}{\partial t}+af(0,t)+\frac{\sigma^2}{2a}(1-e^{-2at})")

    fig_prices.add_trace(
        go.Scatter(
            x=maturities,
            y=[curve.get_discount_factor(t) for t in maturities],
            mode="lines",
            name="Market curve DF",
            line={"dash": "dash"},
        )
    )
    fig_prices.update_layout(title="Bond Price Curve Comparison", xaxis_title="Maturity (years)", yaxis_title="Price")
    st.plotly_chart(fig_prices, use_container_width=True)


elif page == "Product Pricing":
    st.subheader("Page 3: Product Pricing")
    curve = ensure_curve()
    product = st.selectbox("Select product", ["Bond", "Swap"])

    if product == "Bond":
        face = st.number_input("Face value", min_value=100.0, value=100.0, step=100.0)
        coupon = st.number_input("Coupon rate (%)", min_value=0.0, value=4.0, step=0.1)
        maturity = st.number_input("Maturity (years)", min_value=0.5, value=5.0, step=0.5)
        freq = st.selectbox("Coupon frequency", [1, 2, 4], index=1)
        if st.button("Price Bond"):
            price = BondPricer.coupon_bond_price(face, coupon, maturity, curve, frequency=freq)
            ytm = BondPricer.yield_to_maturity(price, face, coupon, maturity, frequency=freq)
            mac_dur, mod_dur, conv = BondPricer.duration_convexity(
                price, face, coupon, maturity, ytm=ytm, frequency=freq
            )
            st.metric("Bond Price", f"{price:,.4f}")
            st.metric("Yield-to-Maturity", f"{ytm:.4%}")
            c1, c2 = st.columns(2)
            c1.metric("Modified Duration", f"{mod_dur:.4f}")
            c2.metric("Convexity", f"{conv:.4f}")
            cf = BondPricer.cashflow_table(face, coupon, maturity, frequency=freq)
            st.dataframe(cf, use_container_width=True)
        if st.toggle("Show Formula: Bond pricing"):
            st.latex(r"P=\sum_i CF_i \cdot P(0,t_i)")
            st.latex(r"D_{mod}=\frac{D_{Mac}}{1+y/m}")

    else:
        notional = st.number_input("Notional", min_value=100000.0, value=1000000.0, step=100000.0)
        fixed_rate = st.number_input("Fixed rate (%)", min_value=0.0, value=4.0, step=0.05)
        maturity = st.number_input("Maturity (years)", min_value=1.0, value=5.0, step=0.5)
        pay_freq = st.selectbox("Payment frequency", [1, 2, 4], index=1)
        payer_fixed = st.toggle("Payer fixed swap", value=True, help="If on, you pay fixed and receive floating.")
        if st.button("Price Swap"):
            swap = InterestRateSwap(notional, fixed_rate, maturity, pay_freq=pay_freq)
            par = swap.par_rate(curve)
            fixed = swap.fixed_leg_pv(curve)
            floating = swap.floating_leg_pv(curve)
            npv = swap.npv(curve, payer_fixed=payer_fixed)
            dv01 = swap.dv01(curve, payer_fixed=payer_fixed)
            st.metric("Swap NPV", f"{npv:,.2f}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Par Swap Rate", f"{par:.4%}")
            c2.metric("Fixed Leg PV", f"{fixed:,.2f}")
            c3.metric("Floating Leg PV", f"{floating:,.2f}")
            st.metric("DV01 (1bp parallel)", f"{dv01:,.2f}")
            krd = swap.key_rate_durations(curve, payer_fixed=payer_fixed)
            fig_krd = go.Figure()
            fig_krd.add_trace(go.Bar(x=krd["time_years"], y=krd["key_rate_duration"], name="KRD"))
            fig_krd.update_layout(title="Key Rate Durations", xaxis_title="Maturity (years)", yaxis_title="Duration")
            st.plotly_chart(fig_krd, use_container_width=True)
        if st.toggle("Show Formula: Swap pricing"):
            st.latex(r"S_{par}=\frac{1-P(0,T)}{\sum_i\Delta_iP(0,t_i)}")


elif page == "Risk Analysis":
    st.subheader("Page 4: Risk Analysis")
    curve = ensure_curve()
    instrument = st.selectbox("Instrument for scenario analysis", ["Bond", "Swap"])
    parallel_bps = st.slider("Parallel shift (bps)", -200, 200, 25)
    tilt_bps = st.slider("Steepener/Flattener magnitude (bps)", -200, 200, 20)

    if instrument == "Bond":
        face = st.number_input("Bond face", min_value=100.0, value=100.0, step=100.0)
        coupon = st.number_input("Bond coupon (%)", min_value=0.0, value=4.0, step=0.1)
        maturity = st.number_input("Bond maturity (years)", min_value=0.5, value=7.0, step=0.5)
        freq = st.selectbox("Bond coupon frequency", [1, 2, 4], index=1)
        base_price = price_bond_under_scenario(curve, face, coupon, maturity, freq)
    else:
        notional = st.number_input("Swap notional", min_value=100000.0, value=1000000.0, step=100000.0)
        fixed_rate = st.number_input("Swap fixed rate (%)", min_value=0.0, value=4.0, step=0.05)
        maturity = st.number_input("Swap maturity (years)", min_value=1.0, value=7.0, step=0.5)
        freq = st.selectbox("Swap payment frequency", [1, 2, 4], index=1)
        base_price = price_swap_under_scenario(curve, notional, fixed_rate, maturity, freq)

    nodes = curve.get_nodes()
    max_t = float(nodes["time_years"].max())

    parallel_curve = curve.bumped_parallel(parallel_bps)
    steep_curve = curve.copy()
    flat_curve = curve.copy()
    for t in nodes["time_years"].to_numpy():
        if t == 0:
            continue
        w = (t / max_t) if max_t > 0 else 0.0
        steep_curve = steep_curve.bumped_key_rate(t, tilt_bps * w, bandwidth=1.0)
        flat_curve = flat_curve.bumped_key_rate(t, -tilt_bps * w, bandwidth=1.0)

    if instrument == "Bond":
        px_parallel = price_bond_under_scenario(parallel_curve, face, coupon, maturity, freq)
        px_steep = price_bond_under_scenario(steep_curve, face, coupon, maturity, freq)
        px_flat = price_bond_under_scenario(flat_curve, face, coupon, maturity, freq)
    else:
        px_parallel = price_swap_under_scenario(parallel_curve, notional, fixed_rate, maturity, freq)
        px_steep = price_swap_under_scenario(steep_curve, notional, fixed_rate, maturity, freq)
        px_flat = price_swap_under_scenario(flat_curve, notional, fixed_rate, maturity, freq)

    pnl_df = pd.DataFrame(
        {
            "Scenario": ["Base", "Parallel", "Steepener", "Flattener"],
            "Value": [base_price, px_parallel, px_steep, px_flat],
        }
    )
    pnl_df["PnL vs Base"] = pnl_df["Value"] - base_price

    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Bar(x=pnl_df["Scenario"], y=pnl_df["PnL vs Base"], name="P&L"))
    fig_pnl.update_layout(title="Scenario P&L Impact", yaxis_title="P&L")
    st.plotly_chart(fig_pnl, use_container_width=True)
    st.dataframe(pnl_df, use_container_width=True)

    if instrument == "Swap":
        swap = InterestRateSwap(notional, fixed_rate, maturity, pay_freq=freq)
        krd = swap.key_rate_durations(curve)
        fig_krd = go.Figure()
        fig_krd.add_trace(go.Bar(x=krd["time_years"], y=krd["key_rate_duration"], name="KRD"))
        fig_krd.update_layout(title="Key Rate Duration Profile", xaxis_title="Maturity (years)", yaxis_title="Duration")
        st.plotly_chart(fig_krd, use_container_width=True)


else:
    st.subheader("Page 5: Model Comparison")
    curve = ensure_curve()

    a = st.slider("a (mean reversion)", 0.01, 2.0, 0.20, 0.01, key="cmp_a")
    b = st.slider("b (long-run mean)", 0.0, 0.20, 0.04, 0.001, key="cmp_b")
    sigma = st.slider("sigma (vol)", 0.001, 0.20, 0.02, 0.001, key="cmp_sigma")
    r0 = st.slider("r0 (initial short rate)", 0.0, 0.20, 0.04, 0.001, key="cmp_r0")

    vas = VasicekModel(a=a, b=b, sigma=sigma, r0=r0)
    cir = CIRModel(a=a, b=b, sigma=sigma, r0=r0)
    hw = HullWhiteModel(a=a, sigma=sigma, r0=r0)
    hw.calibrate_theta(curve, horizon=30.0, step=0.05)

    maturities = np.linspace(0.5, 30.0, 60)
    market = np.array([curve.get_discount_factor(t) for t in maturities])
    vas_px = np.array([vas.bond_price(0.0, t) for t in maturities])
    cir_px = np.array([cir.bond_price(0.0, t) for t in maturities])
    hw_px = np.array([hw.bond_price(0.0, t) for t in maturities])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=maturities, y=market, name="Market", mode="lines", line={"width": 4}))
    fig.add_trace(go.Scatter(x=maturities, y=vas_px, name="Vasicek", mode="lines"))
    fig.add_trace(go.Scatter(x=maturities, y=cir_px, name="CIR", mode="lines"))
    fig.add_trace(go.Scatter(x=maturities, y=hw_px, name="Hull-White", mode="lines"))
    fig.update_layout(title="Bond Price Curves by Model", xaxis_title="Maturity (years)", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    metrics = pd.DataFrame(
        {
            "Model": ["Vasicek", "CIR", "Hull-White"],
            "MAE": [
                np.mean(np.abs(vas_px - market)),
                np.mean(np.abs(cir_px - market)),
                np.mean(np.abs(hw_px - market)),
            ],
            "RMSE": [
                np.sqrt(np.mean((vas_px - market) ** 2)),
                np.sqrt(np.mean((cir_px - market) ** 2)),
                np.sqrt(np.mean((hw_px - market) ** 2)),
            ],
        }
    )
    st.dataframe(metrics, use_container_width=True)

    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(x=metrics["Model"], y=metrics["MAE"], name="MAE"))
    fig_err.add_trace(go.Bar(x=metrics["Model"], y=metrics["RMSE"], name="RMSE"))
    fig_err.update_layout(title="Model Error Metrics", barmode="group", yaxis_title="Error")
    st.plotly_chart(fig_err, use_container_width=True)

    params = pd.DataFrame(
        {"Parameter": ["a", "b", "sigma", "r0"], "Value": [a, b, sigma, r0]}
    )
    st.write("Side-by-side parameter panel")
    st.dataframe(params, use_container_width=True)
