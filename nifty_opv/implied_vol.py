"""
Implied volatility inversion using Brent's method.


- Never return a bare float. Always return an :class:`IVResult`.
- Validate no-arbitrage bounds before calling the solver.
- Expand the search bracket when needed.
- Reprice with the returned IV to verify the round trip.
- Distinguish "solver failed" from "input price is infeasible".

Typical use
-----------
>>> res = implied_volatility(price=120.5, S=21000, K=21000, T=0.05,
...                          r=0.053, option_type="call", q=0.012)
>>> res.status
'success'
>>> round(res.iv, 4)  # doctest: +SKIP
0.1432
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from nifty_opv.black_scholes import bs_price

__all__ = ["IVResult", "implied_volatility"]

IVStatus = Literal[
    "success",
    "below_intrinsic",
    "above_upper_bound",
    "no_bracket",
    "invalid_input",
    "solver_error",
]


@dataclass(frozen=True)
class IVResult:
    """
    Structured result of an implied volatility inversion.

    Attributes
    ----------
    iv
        Implied volatility as a decimal (e.g. 0.15 for 15%). ``None`` if
        the inversion failed.
    status
        One of "success", "below_intrinsic", "above_upper_bound",
        "no_bracket", "invalid_input", "solver_error".
    repricing_error
        |bs_price(iv) - market_price| when iv is available, else
        None.
    failure_reason
        Text explanation when status != "success".
    """

    iv: float | None
    status: IVStatus
    repricing_error: float | None
    failure_reason: str | None


def _no_arbitrage_bounds(
    S: float, K: float, T: float, r: float, q: float, option_type: str
) -> tuple[float, float]:
    """
    Return (lower, upper) bounds on the option price under Black-Scholes no-arbitrage for the given inputs.
    Both bounds are discounted to present value at time T.
    """
    disc_r = float(np.exp(-r * T))
    disc_q = float(np.exp(-q * T))
    forward = S * disc_q / disc_r  # F = S e^{(r-q)T}

    if option_type == "call":
        # Lower: max(S e^{-qT} - K e^{-rT}, 0)
        # Upper: S e^{-qT}
        lower = max(S * disc_q - K * disc_r, 0.0)
        upper = S * disc_q
    else:
        # Lower: max(K e^{-rT} - S e^{-qT}, 0)
        # Upper: K e^{-rT}
        lower = max(K * disc_r - S * disc_q, 0.0)
        upper = K * disc_r

    # "forward" is unused above; kept for clarity of the derivation.
    _ = forward
    return lower, upper


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    lo: float = 1e-6,
    hi: float = 5.0,
    tol: float = 1e-8,
    max_bracket_expansions: int = 12,
    bracket_multiplier: float = 2.0,
    repricing_tol: float = 1e-6,
) -> IVResult:
    """
    Invert Black-Scholes for implied volatility using Brent's method.

    Parameters
    ----------
    market_price
        Observed option price (same units as S and K).
    S, K, T, r, q
        Spot, strike, time to expiry in years, risk-free rate,
        continuous dividend yield.
    option_type
        call or put
    lo, hi
        Initial volatility bracket. Must satisfy 0 < lo < hi.
    tol
        Brent tolerance.
    max_bracket_expansions
        Number of times to widen hi when the root is not bracketed.
    bracket_multiplier
        Each expansion multiplies the failing bound by this factor.
    repricing_tol
        Tolerance for the round-trip check.

    Returns
    -------
    IVResult
        Structured result. Never raises for ordinary input failures
    """
    # Input validation
    ot = option_type.strip().lower()
    if ot in {"c", "call", "ce"}:
        ot = "call"
    elif ot in {"p", "put", "pe"}:
        ot = "put"
    else:
        return IVResult(None, "invalid_input", None,
                        f"option_type must be 'call' or 'put', got {option_type!r}")

    if not np.isfinite(market_price) or market_price <= 0.0:
        return IVResult(None, "invalid_input", None,
                        f"market_price must be finite and > 0, got {market_price}")
    if not np.isfinite(S) or S <= 0.0:
        return IVResult(None, "invalid_input", None, f"S must be > 0, got {S}")
    if not np.isfinite(K) or K <= 0.0:
        return IVResult(None, "invalid_input", None, f"K must be > 0, got {K}")
    if not np.isfinite(T) or T <= 0.0:
        return IVResult(None, "invalid_input", None, f"T must be > 0, got {T}")
    if not np.isfinite(r):
        return IVResult(None, "invalid_input", None, f"r must be finite, got {r}")
    if not np.isfinite(q):
        return IVResult(None, "invalid_input", None, f"q must be finite, got {q}")
    if not (0.0 < lo < hi):
        return IVResult(None, "invalid_input", None,
                        f"bracket must satisfy 0 < lo < hi, got lo={lo}, hi={hi}")

    # No-arbitrage bound check
    lower, upper = _no_arbitrage_bounds(S, K, T, r, q, ot)
    # Small slack to avoid rejecting prices that sit exactly on the bound due to floating-point round-off.
    slack = 1e-10 * max(1.0, abs(upper))
    if market_price < lower - slack:
        return IVResult(None, "below_intrinsic", None,
                        f"price {market_price:.6g} below lower bound {lower:.6g}")
    if market_price > upper + slack:
        return IVResult(None, "above_upper_bound", None,
                        f"price {market_price:.6g} above upper bound {upper:.6g}")

    # Define objective
    def objective(sigma: float) -> float:
        return float(bs_price(S, K, T, r, sigma, q, ot)) - market_price

    # Bracket the root
    f_lo = objective(lo)
    f_hi = objective(hi)

    # If we are on the boundaries of the feasible region, the target price may be achievable only in the limit sigma -> 0 or sigma -> +inf.
    if f_lo > 0.0:
        # Even at extremely low vol, our model price exceeds the market price.
        return IVResult(None, "no_bracket", None,
                        f"BS price at sigma={lo} is already above market price")
    if f_hi < 0.0:
        # Even at very high vol, our model price is below the market price.
        # Try expanding hi.
        hi_cur = hi
        f_hi_cur = f_hi
        expanded = 0
        while f_hi_cur < 0.0 and expanded < max_bracket_expansions:
            hi_cur *= bracket_multiplier
            f_hi_cur = objective(hi_cur)
            expanded += 1
        if f_hi_cur < 0.0:
            return IVResult(None, "no_bracket", None,
                            f"could not bracket root after {expanded} expansions "
                            f"(hi={hi_cur:.3g}, f={f_hi_cur:.3g})")
        hi = hi_cur
        f_hi = f_hi_cur

    # Solve
    try:
        iv = float(brentq(objective, lo, hi, xtol=tol, rtol=1e-12, maxiter=200))
    except (ValueError, RuntimeError) as exc:
        return IVResult(None, "solver_error", None, f"brentq failed: {exc}")

    if not np.isfinite(iv) or iv <= 0.0:
        return IVResult(None, "solver_error", None, f"brentq returned invalid iv={iv}")

    # Round-trip repricing check
    model_price = float(bs_price(S, K, T, r, iv, q, ot))
    repricing_error = abs(model_price - market_price)

        # Relative tolerance: 1e-6 of the option's own price, with an absolute floor of 1e-8 so near-zero prices do not produce spurious failures.
    effective_tol = max(repricing_tol * max(abs(market_price), 1.0), 1e-8)
    if repricing_error > effective_tol:
        return IVResult(iv, "solver_error", repricing_error,
                        f"repricing error {repricing_error:.3e} exceeds "
                        f"effective tol {effective_tol:.3e}")

    return IVResult(iv, "success", repricing_error, None)


def implied_volatility_array(
    market_prices: np.ndarray,
    S: float,
    strikes: np.ndarray,
    T: float,
    r: float,
    option_types: np.ndarray,
    q: float = 0.0,
    **kwargs: float,
) -> list[IVResult]:
    """
    Vectorized convenience wrapper. Loops over observations and returns a
    list of :class:`IVResult`.

    Parameters
    ----------
    market_prices, strikes, option_types
        Same-length arrays.
    S, T, r, q
        Scalars shared across all observations.
    **kwargs
        Forwarded to :func:implied_volatility (lo, hi, tol, ...).
    """
    n = len(market_prices)
    if not (len(strikes) == n and len(option_types) == n):
        raise ValueError("market_prices, strikes, option_types must have equal length")

    results: list[IVResult] = []
    for i in range(n):
        results.append(
            implied_volatility(
                market_price=float(market_prices[i]),
                S=float(S),
                K=float(strikes[i]),
                T=float(T),
                r=float(r),
                option_type=str(option_types[i]),
                q=float(q),
                **kwargs,
            )
        )
    return results