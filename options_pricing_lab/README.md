# Options Pricing Laboratory

Interactive educational platform for options pricing and volatility analytics built with Python + Streamlit.

## Highlights

- Vanilla option pricing with:
  - Black-Scholes closed-form
  - Binomial tree (European/American)
  - Monte Carlo
  - Finite differences
- Greeks analytics:
  - Analytical Greeks
  - Finite-difference Greeks
- Exotics:
  - Barrier options
  - Digital options
  - Asian options
  - Lookback options
  - Compound options
- Volatility analytics:
  - Implied volatility inversion
  - Volatility surface interpolation
  - Smile metrics
  - Local volatility approximation
  - Heston path simulation
  - SABR smile approximation
- Risk tools:
  - Scenario analysis
  - Stress testing
  - Parametric and Monte Carlo VaR
- Visualization suite:
  - 3D vol surfaces
  - Payoff diagrams
  - Greeks heatmaps

## Structure

```text
options_pricing_lab/
├── app.py
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── black_scholes.py
│   ├── binomial_tree.py
│   ├── monte_carlo.py
│   ├── finite_difference.py
│   └── greeks.py
├── exotics/
│   ├── __init__.py
│   ├── barriers.py
│   ├── digitals.py
│   ├── asians.py
│   ├── lookbacks.py
│   └── compounds.py
├── volatility/
│   ├── __init__.py
│   ├── surface.py
│   ├── implied.py
│   ├── local_vol.py
│   ├── stochastic_vol.py
│   └── smile.py
├── risk/
│   ├── __init__.py
│   ├── scenario.py
│   ├── stress_test.py
│   └── var.py
├── visualization/
│   ├── __init__.py
│   ├── surface_plots.py
│   ├── payoff_diagrams.py
│   └── heatmaps.py
└── data/
    ├── __init__.py
    ├── market_data.py
    └── volatility_surfaces.py
```

## Installation

```bash
cd options_pricing_lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Independence from `financial_pricing_lab`

This dashboard is fully separate in its own folder (`options_pricing_lab`) and does not modify or import from `financial_pricing_lab`.

## Disclaimer

Educational/prototyping code only. Numerical methods and calibration routines are simplified and should be validated before professional use.
