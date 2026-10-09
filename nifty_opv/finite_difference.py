"""
Finite-difference Greeks for Black-Scholes validation.

This module provides an independent numerical oracle for the analytical
Greeks in :mod:`nifty_opv.black_scholes`. It uses central differences on
the pricing function, which means any error in the analytical formulas
that is not also present in the price function will be caught.

Sign conventions
----------------
- delta : dV/dS
- gamma : d^2V/dS^2
- vega  : dV/dsigma                (per 1.00 change in sigma)
- rho   : dV/dr                    (per 1.00 change in r)
- theta : dV/dt_calendar           (per year; matches bs_theta)

"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nifty_opv.black_scholes import bs_price

__all__ = [
    "fd_delta",
    "fd_gamma",
    "fd_vega",
    "fd_rho",
    "fd_theta",
    "fd_all_greeks",
]

PricerFn = Callable[[float, float, float, float, float, float, str], float]


def _default_pricer(
    S: float, K: float, T: float, r: float, sigma: float, q: float, option_type: str
) -> float:
    return float(bs_price(S, K, T, r, sigma, q, option_type))


def fd_delta(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    h: float = 1e-4,
    pricer: PricerFn = _default_pricer,
) -> float:
    """
    Central-difference Delta: [P(S+h) - P(S-h)] / (2h).

    Parameters
    ----------
    h
        Absolute bump size in spot units. For S ~ 21000, h=1e-4 * S = 2.1
        would be too large; prefer relative bumps at large S. The default
        here is an absolute bump. Callers should scale h if needed.
    """
    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")
    up = pricer(S + h, K, T, r, sigma, q, option_type)
    dn = pricer(S - h, K, T, r, sigma, q, option_type)
    return (up - dn) / (2.0 * h)


def fd_gamma(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    h: float = 1e-3,
    pricer: PricerFn = _default_pricer,
) -> float:
    """
    Central-difference Gamma: [P(S+h) - 2 P(S) + P(S-h)] / h^2.

    Gamma is identical for calls and puts in the BS model, but we accept
    the argument for uniformity with the analytical API.
    """
    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")
    up = pricer(S + h, K, T, r, sigma, q, option_type)
    mid = pricer(S, K, T, r, sigma, q, option_type)
    dn = pricer(S - h, K, T, r, sigma, q, option_type)
    return (up - 2.0 * mid + dn) / (h * h)


def fd_vega(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    h: float = 1e-4,
    pricer: PricerFn = _default_pricer,
) -> float:
    """
    Central-difference Vega: [P(sigma+h) - P(sigma-h)] / (2h).

    Returns sensitivity per 1.00 change in sigma. Divide by 100 for per 1%.
    """
    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")
    up = pricer(S, K, T, r, sigma + h, q, option_type)
    dn = pricer(S, K, T, r, sigma - h, q, option_type)
    return (up - dn) / (2.0 * h)


def fd_rho(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    h: float = 1e-4,
    pricer: PricerFn = _default_pricer,
) -> float:
    """
    Central-difference Rho: [P(r+h) - P(r-h)] / (2h).

    Returns sensitivity per 1.00 change in r. Divide by 100 for per 1%.
    """
    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")
    up = pricer(S, K, T, r + h, sigma, q, option_type)
    dn = pricer(S, K, T, r - h, sigma, q, option_type)
    return (up - dn) / (2.0 * h)


def fd_theta(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    h: float = 1e-5,
    pricer: PricerFn = _default_pricer,
) -> float:
    """
    Central-difference Theta: dV/dt_calendar = -dV/dT.

    Bumps time to expiry T, then flips the sign so the convention
    matches :func:`nifty_opv.black_scholes.bs_theta` (calendar time
    advancing).
    """
    if h <= 0.0:
        raise ValueError(f"h must be > 0, got {h}")
    if T - h <= 0.0:
        # Near expiry, a symmetric bump would use negative T. Fall back to
        # a one-sided forward difference.
        up = pricer(S, K, T + h, r, sigma, q, option_type)
        mid = pricer(S, K, T, r, sigma, q, option_type)
        return -(up - mid) / h
    up = pricer(S, K, T + h, r, sigma, q, option_type)
    dn = pricer(S, K, T - h, r, sigma, q, option_type)
    return -(up - dn) / (2.0 * h)


def fd_all_greeks(
    S: float, K: float, T: float, r: float, sigma: float,
    q: float = 0.0, option_type: str = "call",
    h_spot: float = 1e-4,
    h_gamma: float = 1e-3,
    h_vol: float = 1e-4,
    h_rate: float = 1e-4,
    h_time: float = 1e-5,
    pricer: PricerFn = _default_pricer,
) -> dict[str, float]:
    """
    Compute all finite-difference Greeks in one call. Returns a dict with
    keys delta, gamma, vega, rho, theta.
    """
    return {
        "delta": fd_delta(S, K, T, r, sigma, q, option_type, h_spot, pricer),
        "gamma": fd_gamma(S, K, T, r, sigma, q, option_type, h_gamma, pricer),
        "vega":  fd_vega (S, K, T, r, sigma, q, option_type, h_vol, pricer),
        "rho":   fd_rho  (S, K, T, r, sigma, q, option_type, h_rate, pricer),
        "theta": fd_theta(S, K, T, r, sigma, q, option_type, h_time, pricer),
    }