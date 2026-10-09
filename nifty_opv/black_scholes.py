"""
European Black-Scholes-Merton pricing and analytical Greeks

Formulas
--------
Call:  C = S e^{-qT} N(d1) - K e^{-rT} N(d2)
Put:   P = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

Conventions
-----------
- "T" is in years.
- "r" and "q" are continuously compounded annual rates.
- "sigma" is annualized volatility, e.g. 0.20 for 20%.

Greeks
------
- Delta  : dV/dS
- Gamma  : d^2V/dS^2
- Vega   : dV/dsigma     (per 1.00 change in sigma; divide by 100 for per 1%)
- Theta  : dV/dt, calendar time advancing (per year; divide by 365 for per day)
- Rho    : dV/dr         (per 1.00 change in r; divide by 100 for per 1%)

Edge cases considered:
----------
- T <= 0     : returns intrinsic value; Greeks return their expiry limits.
- sigma <= 0 : returns discounted forward intrinsic; Greeks return limits.
- S <= 0 or K <= 0 : raises ValueError.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "bs_theta",
    "bs_rho",
    "bs_all_greeks",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard normal PDF, vectorized."""
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Standard normal CDF, vectorized, using the error function.

    Uses ``scipy.special.ndtr`` semantics via erf, which is accurate to
    machine precision over the whole real line.
    """
    from scipy.special import ndtr  # local import keeps top-level cheap
    return ndtr(x)


def _validate_option_type(option_type: str) -> str:
    ot = option_type.strip().lower()
    if ot in {"c", "call", "ce"}:
        return "call"
    if ot in {"p", "put", "pe"}:
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _as_float_array(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def _intrinsic(S: NDArray[np.float64],
               K: NDArray[np.float64],
               option_type: str) -> NDArray[np.float64]:
    """Undiscounted intrinsic at expiry."""
    if option_type == "call":
        return np.maximum(S - K, 0.0)
    return np.maximum(K - S, 0.0)


def _d1_d2(S: NDArray[np.float64],
           K: NDArray[np.float64],
           T: NDArray[np.float64],
           r: NDArray[np.float64],
           sigma: NDArray[np.float64],
           q: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute d1 and d2. Assumes T > 0 and sigma > 0."""
    vol_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    return d1, d2


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def bs_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> NDArray[np.float64]:
    """
    European Black-Scholes-Merton price with continuous dividend yield.

    Parameters
    ----------
    S, K, T, r, sigma, q : array_like
        Spot, strike, time to expiry (years), risk-free rate, volatility,
        continuous dividend yield. Scalars or arrays; broadcast together.
    option_type : {"call", "put"}
        Case-insensitive.

    Returns
    -------
    ndarray
        Option price, broadcast to the input shape. Returns a 0-d array if
        all inputs are scalar.

    Raises
    ------
    ValueError
        If ``S <= 0`` or ``K <= 0``, or if ``option_type`` is invalid.
    """
    ot = _validate_option_type(option_type)

    S_ = _as_float_array(S)
    K_ = _as_float_array(K)
    T_ = _as_float_array(T)
    r_ = _as_float_array(r)
    sigma_ = _as_float_array(sigma)
    q_ = _as_float_array(q)

    if np.any(S_ <= 0.0):
        raise ValueError("Spot S must be strictly positive")
    if np.any(K_ <= 0.0):
        raise ValueError("Strike K must be strictly positive")

    # Broadcast everything to a common shape.
    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(S_, K_, T_, r_, sigma_, q_)

    # Deliberate degenerate handling.
    T_pos = np.maximum(T_, 0.0)
    sigma_pos = np.maximum(sigma_, 0.0)

    # Mask: normal BS regime.
    normal = (T_pos > 0.0) & (sigma_pos > 0.0)

    # Pre-allocate output.
    out = np.zeros_like(S_, dtype=np.float64)

    # --- Degenerate case: T == 0 OR sigma == 0 -> deterministic limit ---
    # With T == 0: price = intrinsic.
    # With sigma == 0 (T > 0): forward is S e^{(r-q)T}, price is the
    # discounted payoff of that deterministic forward.
    deg = ~normal
    if np.any(deg):
        fwd = S_ * np.exp((r_ - q_) * T_pos)
        if ot == "call":
            pay = np.maximum(fwd - K_, 0.0)
        else:
            pay = np.maximum(K_ - fwd, 0.0)
        disc = np.exp(-r_ * T_pos)
        # When T == 0, disc == 1, fwd == S, so this collapses to intrinsic.
        out = np.where(deg, disc * pay, out)

    # --- Normal case ---
    if np.any(normal):
        d1, d2 = _d1_d2(S_, K_, T_pos, r_, sigma_pos, q_)
        disc_r = np.exp(-r_ * T_pos)
        disc_q = np.exp(-q_ * T_pos)
        if ot == "call":
            val = S_ * disc_q * _norm_cdf(d1) - K_ * disc_r * _norm_cdf(d2)
        else:
            val = K_ * disc_r * _norm_cdf(-d2) - S_ * disc_q * _norm_cdf(-d1)
        out = np.where(normal, val, out)

    return out


# ---------------------------------------------------------------------------
# Analytical Greeks
# ---------------------------------------------------------------------------

def bs_delta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> NDArray[np.float64]:
    """
    Delta = dV/dS.

    Call: e^{-qT} N(d1)
    Put : -e^{-qT} N(-d1)

    At expiry (T == 0):
        Call -> 1 if S > K, 0 if S < K (undefined at S == K, returns 0.5)
        Put  -> -1 if S < K, 0 if S > K (returns -0.5 at S == K)
    """
    ot = _validate_option_type(option_type)

    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(
        _as_float_array(S), _as_float_array(K), _as_float_array(T),
        _as_float_array(r), _as_float_array(sigma), _as_float_array(q),
    )
    if np.any(S_ <= 0.0) or np.any(K_ <= 0.0):
        raise ValueError("S and K must be strictly positive")

    T_pos = np.maximum(T_, 0.0)
    sigma_pos = np.maximum(sigma_, 0.0)
    normal = (T_pos > 0.0) & (sigma_pos > 0.0)

    out = np.zeros_like(S_, dtype=np.float64)

    if np.any(normal):
        d1, _ = _d1_d2(S_, K_, T_pos, r_, sigma_pos, q_)
        disc_q = np.exp(-q_ * T_pos)
        if ot == "call":
            val = disc_q * _norm_cdf(d1)
        else:
            val = -disc_q * _norm_cdf(-d1)
        out = np.where(normal, val, out)

    # Degenerate: T == 0 or sigma == 0 -> step function on forward vs strike.
    deg = ~normal
    if np.any(deg):
        fwd = S_ * np.exp((r_ - q_) * T_pos)
        if ot == "call":
            step = np.where(fwd > K_, 1.0, np.where(fwd < K_, 0.0, 0.5))
            val_deg = np.exp(-q_ * T_pos) * step
        else:
            step = np.where(fwd < K_, -1.0, np.where(fwd > K_, 0.0, -0.5))
            val_deg = np.exp(-q_ * T_pos) * step
        out = np.where(deg, val_deg, out)

    return out


def bs_gamma(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """
    Gamma = d^2V/dS^2, identical for call and put.

    Gamma = e^{-qT} phi(d1) / (S sigma sqrt(T))

    At expiry (T == 0 or sigma == 0) Gamma is zero almost everywhere;
    it is a Dirac mass at S == K. We return 0 for the degenerate case.
    """
    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(
        _as_float_array(S), _as_float_array(K), _as_float_array(T),
        _as_float_array(r), _as_float_array(sigma), _as_float_array(q),
    )
    if np.any(S_ <= 0.0) or np.any(K_ <= 0.0):
        raise ValueError("S and K must be strictly positive")

    T_pos = np.maximum(T_, 0.0)
    sigma_pos = np.maximum(sigma_, 0.0)
    normal = (T_pos > 0.0) & (sigma_pos > 0.0)

    out = np.zeros_like(S_, dtype=np.float64)
    if np.any(normal):
        d1, _ = _d1_d2(S_, K_, T_pos, r_, sigma_pos, q_)
        disc_q = np.exp(-q_ * T_pos)
        val = disc_q * _norm_pdf(d1) / (S_ * sigma_pos * np.sqrt(T_pos))
        out = np.where(normal, val, out)
    return out


def bs_vega(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
) -> NDArray[np.float64]:
    """
    Vega = dV/dsigma, identical for call and put.

    Vega = S e^{-qT} phi(d1) sqrt(T)

    Units: per 1.00 change in sigma. Divide by 100 for per 1% vol move.
    """
    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(
        _as_float_array(S), _as_float_array(K), _as_float_array(T),
        _as_float_array(r), _as_float_array(sigma), _as_float_array(q),
    )
    if np.any(S_ <= 0.0) or np.any(K_ <= 0.0):
        raise ValueError("S and K must be strictly positive")

    T_pos = np.maximum(T_, 0.0)
    sigma_pos = np.maximum(sigma_, 0.0)
    normal = (T_pos > 0.0) & (sigma_pos > 0.0)

    out = np.zeros_like(S_, dtype=np.float64)
    if np.any(normal):
        d1, _ = _d1_d2(S_, K_, T_pos, r_, sigma_pos, q_)
        disc_q = np.exp(-q_ * T_pos)
        val = S_ * disc_q * _norm_pdf(d1) * np.sqrt(T_pos)
        out = np.where(normal, val, out)
    return out


def bs_theta(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> NDArray[np.float64]:
    """
    Theta = dV/dt with calendar time advancing (per year).

    Call:
        Theta = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                - r K e^{-rT} N(d2)
                + q S e^{-qT} N(d1)

    Put:
        Theta = -S e^{-qT} phi(d1) sigma / (2 sqrt(T))
                + r K e^{-rT} N(-d2)
                - q S e^{-qT} N(-d1)

    Sign convention: this is the change in option value as calendar time
    advances. It is typically negative for long vanilla options. To get
    decay per calendar day, divide by 365.
    """
    ot = _validate_option_type(option_type)

    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(
        _as_float_array(S), _as_float_array(K), _as_float_array(T),
        _as_float_array(r), _as_float_array(sigma), _as_float_array(q),
    )
    if np.any(S_ <= 0.0) or np.any(K_ <= 0.0):
        raise ValueError("S and K must be strictly positive")

    T_pos = np.maximum(T_, 0.0)
    sigma_pos = np.maximum(sigma_, 0.0)
    normal = (T_pos > 0.0) & (sigma_pos > 0.0)

    out = np.zeros_like(S_, dtype=np.float64)
    if np.any(normal):
        d1, d2 = _d1_d2(S_, K_, T_pos, r_, sigma_pos, q_)
        disc_r = np.exp(-r_ * T_pos)
        disc_q = np.exp(-q_ * T_pos)
        common = -S_ * disc_q * _norm_pdf(d1) * sigma_pos / (2.0 * np.sqrt(T_pos))
        if ot == "call":
            val = common - r_ * K_ * disc_r * _norm_cdf(d2) \
                  + q_ * S_ * disc_q * _norm_cdf(d1)
        else:
            val = common + r_ * K_ * disc_r * _norm_cdf(-d2) \
                  - q_ * S_ * disc_q * _norm_cdf(-d1)
        out = np.where(normal, val, out)
    return out


def bs_rho(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> NDArray[np.float64]:
    """
    Rho = dV/dr, per 1.00 change in continuously compounded rate.

    Call: K T e^{-rT} N(d2)
    Put : -K T e^{-rT} N(-d2)

    Divide by 100 for per 1% rate move.
    """
    ot = _validate_option_type(option_type)

    S_, K_, T_, r_, sigma_, q_ = np.broadcast_arrays(
        _as_float_array(S), _as_float_array(K), _as_float_array(T),
        _as_float_array(r), _as_float_array(sigma), _as_float_array(q),
    )
    if np.any(S_ <= 0.0) or np.any(K_ <= 0.0):
        raise ValueError("S and K must be strictly positive")

    T_pos = np.maximum(T_, 0.0)
    sigma_pos = np.maximum(sigma_, 0.0)
    normal = (T_pos > 0.0) & (sigma_pos > 0.0)

    out = np.zeros_like(S_, dtype=np.float64)
    if np.any(normal):
        _, d2 = _d1_d2(S_, K_, T_pos, r_, sigma_pos, q_)
        disc_r = np.exp(-r_ * T_pos)
        if ot == "call":
            val = K_ * T_pos * disc_r * _norm_cdf(d2)
        else:
            val = -K_ * T_pos * disc_r * _norm_cdf(-d2)
        out = np.where(normal, val, out)
    return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def bs_all_greeks(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    q: ArrayLike = 0.0,
    option_type: str = "call",
) -> dict[str, NDArray[np.float64]]:
    """
    Return all Greeks plus price in a dict.

    Keys: ``price``, ``delta``, ``gamma``, ``vega``, ``theta``, ``rho``.
    """
    return {
        "price": bs_price(S, K, T, r, sigma, q, option_type),
        "delta": bs_delta(S, K, T, r, sigma, q, option_type),
        "gamma": bs_gamma(S, K, T, r, sigma, q),
        "vega": bs_vega(S, K, T, r, sigma, q),
        "theta": bs_theta(S, K, T, r, sigma, q, option_type),
        "rho": bs_rho(S, K, T, r, sigma, q, option_type),
    }