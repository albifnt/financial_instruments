"""Black model formulas for rates options."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


class BlackModel:
    @staticmethod
    def black_price(
        forward: float,
        strike: float,
        vol: float,
        expiry: float,
        annuity: float = 1.0,
        option_type: str = "call",
    ) -> float:
        if expiry <= 0.0:
            intrinsic = max(forward - strike, 0.0) if option_type == "call" else max(strike - forward, 0.0)
            return annuity * intrinsic
        if vol <= 0.0:
            intrinsic = max(forward - strike, 0.0) if option_type == "call" else max(strike - forward, 0.0)
            return annuity * intrinsic
        if forward <= 0.0 or strike <= 0.0:
            raise ValueError("Forward and strike must be positive for Black formula.")

        st = vol * np.sqrt(expiry)
        d1 = (np.log(forward / strike) + 0.5 * vol * vol * expiry) / st
        d2 = d1 - st
        if option_type == "call":
            return float(annuity * (forward * norm.cdf(d1) - strike * norm.cdf(d2)))
        if option_type == "put":
            return float(annuity * (strike * norm.cdf(-d2) - forward * norm.cdf(-d1)))
        raise ValueError("option_type must be 'call' or 'put'.")

    @staticmethod
    def swaption_price(forward_swap_rate: float, strike: float, vol: float, expiry: float, annuity: float) -> float:
        return BlackModel.black_price(forward_swap_rate, strike, vol, expiry, annuity=annuity, option_type="call")

    @staticmethod
    def implied_vol(market_price: float, forward: float, strike: float, expiry: float, annuity: float = 1.0) -> float:
        if market_price <= 0:
            return 0.0

        def objective(v: float) -> float:
            return BlackModel.black_price(forward, strike, v, expiry, annuity=annuity) - market_price

        return float(brentq(objective, 1e-6, 5.0))

    @staticmethod
    def caplet_floorlet_price(
        forward_rate: float,
        strike: float,
        vol: float,
        expiry: float,
        accrual: float,
        discount_factor: float,
        option_type: str = "caplet",
    ) -> float:
        opt = "call" if option_type == "caplet" else "put"
        return discount_factor * accrual * BlackModel.black_price(
            forward_rate, strike, vol, expiry, annuity=1.0, option_type=opt
        )

    @staticmethod
    def bond_option_price(
        bond_forward: float,
        strike: float,
        vol: float,
        expiry: float,
        discount_to_expiry: float,
        option_type: str = "call",
    ) -> float:
        return discount_to_expiry * BlackModel.black_price(
            bond_forward, strike, vol, expiry, annuity=1.0, option_type=option_type
        )
