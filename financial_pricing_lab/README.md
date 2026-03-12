# Financial Pricing Lab

Interactive educational platform for interest-rate derivatives pricing built with Python + Streamlit.

## Features

- Yield curve bootstrapping from deposits and swaps
- Zero, forward, and discount factor analytics with interactive Plotly charts
- Interest-rate models:
  - Vasicek
  - CIR (with Feller condition check)
  - Hull-White (theta calibration to input curve)
  - Black model for swaptions/caps/floors/bond options
- Product pricing:
  - Zero-coupon and coupon bonds
  - Vanilla interest-rate swaps
- Risk analytics:
  - Duration and convexity
  - DV01 and key-rate durations
  - Parallel, steepener, and flattener scenarios
- Model comparison dashboard with MAE/RMSE diagnostics

## Project Structure

```text
financial_pricing_lab/
├── app.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── curve.py
│   └── daycount.py
├── models/
│   ├── __init__.py
│   ├── vasicek.py
│   ├── cir.py
│   ├── hull_white.py
│   └── black.py
├── products/
│   ├── __init__.py
│   ├── bond.py
│   └── swap.py
└── data/
    └── sample_curves.py
```

## Installation

From the project root:

```bash
cd financial_pricing_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## App Pages

1. **Curve Bootstrapping**
   - Edit market quotes with tables
   - Bootstrap and inspect discount/zero/forward curves
   - Export curve to CSV/JSON
2. **Interest Rate Models**
   - Tune model parameters with sliders
   - Simulate Monte Carlo short-rate paths
   - Compare model bond prices to market curve
3. **Product Pricing**
   - Price bonds and swaps
   - Inspect cashflows and risk measures
4. **Risk Analysis**
   - Run scenario shocks
   - Visualize P&L impact and key-rate profiles
5. **Model Comparison**
   - Side-by-side Vasicek/CIR/Hull-White comparisons
   - Error diagnostics vs market-implied discount factors

## Educational Notes

- Use the **Quick Tutorial** expander on each page.
- Use **Show Formula** toggles to view LaTeX equations.
- Suggested references:
  - Hull, *Options, Futures, and Other Derivatives*
  - Brigo & Mercurio, *Interest Rate Models: Theory and Practice*

## Important Caveats

This application is for learning and prototyping. It uses simplified curve bootstrapping and model calibration assumptions and is **not** production-ready for trading/risk reporting without further validation.
