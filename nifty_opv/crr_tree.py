"""
Cox-Ross-Rubinstein (CRR) binomial tree for European (and optionally
American) vanilla options on a continuous-dividend underlying.

Parameters
----------
u     = exp( sigma * sqrt(dt) )       up factor per step
d     = 1 / u                         down factor per step
p     = [ exp((r - q) dt) - d ] / (u - d)   risk-neutral up probability
dt    = T / N                         step size in years

Terminal spots at step N:
    S_j = S * u^j * d^(N-j),  j = 0, ..., N

Backward induction (European):
    V_j = exp(-r dt) * [ p V_{j+1} + (1 - p) V_j ]  for j = 0, ..., i-1

American:
    V_j = max( V_j, payoff( S_j ) )

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["crr_price", "crr_price_array", "crr_convergence"]


def _validate_option_type(option_type: str) -> str:
    ot = option_type.strip().lower()
    if ot in {"c", "call", "ce"}:
        return "call"
    if ot in {"p", "put", "pe"}:
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _payoff(spot: NDArray[np.float64], strike: float, option_type: str) -> NDArray[np.float64]:
    """European/American payoff at exercise for the terminal (or any) spot."""
    if option_type == "call":
        return np.maximum(spot - strike, 0.0)
    return np.maximum(strike - spot, 0.0)


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps: int = 500,
    american: bool = False,
) -> float:
    """
    Price a vanilla option with the CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q
        Spot, strike, time to expiry in years, risk-free rate, volatility,
        continuous dividend yield.
    option_type
        call or put.
    n_steps
        Number of tree steps. Must be >= 1.
    american
        If True, allow early exercise. Default False (European).

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        If any of the inputs are invalid, or if the risk-neutralprobability leaves (0, 1).
    """
    ot = _validate_option_type(option_type)

    if not np.isfinite(S) or S <= 0.0:
        raise ValueError(f"S must be finite and > 0, got {S}")
    if not np.isfinite(K) or K <= 0.0:
        raise ValueError(f"K must be finite and > 0, got {K}")
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError(f"T must be finite and > 0, got {T}")
    if not np.isfinite(r):
        raise ValueError(f"r must be finite, got {r}")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"sigma must be finite and > 0, got {sigma}")
    if not np.isfinite(q):
        raise ValueError(f"q must be finite, got {q}")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError(f"n_steps must be an integer >= 1, got {n_steps!r}")

    n = int(n_steps)
    dt = T / n

    u = float(np.exp(sigma * np.sqrt(dt)))
    d = 1.0 / u
    p = (float(np.exp((r - q) * dt)) - d) / (u - d)

    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Risk-neutral probability p={p:.6g} outside (0, 1). "
            f"Inputs: T={T}, n={n}, sigma={sigma}, r={r}, q={q}. "
            f"Reduce dt by increasing n_steps, or check sigma."
        )

    disc = float(np.exp(-r * dt))
    one_minus_p = 1.0 - p

    # Build via logs to avoid overflow for large n and large sigma.
    j = np.arange(n + 1, dtype=np.float64)
    log_u = np.log(u)
    log_d = np.log(d)
    log_S = np.log(S)
    log_spots = log_S + j * log_u + (n - j) * log_d
    spots: NDArray[np.float64] = np.exp(log_spots)

    # Terminal payoffs
    values: NDArray[np.float64] = _payoff(spots, K, ot)

    # Backward induction
    # At step i (from n-1 down to 0), the active slice is values[:i+1].
    # Using a single array and slicing in place is O(n^2) time but O(n) memory.
    for i in range(n - 1, -1, -1):
        # values[:i+1] <- disc * ( p * values[1:i+2] + (1-p) * values[:i+1] )
        cont = disc * (p * values[1 : i + 2] + one_minus_p * values[: i + 1])

        if american:
            # Spot at step i, node j: S * u^j * d^(i-j)
            j = np.arange(i + 1, dtype=np.float64)
            log_spots_i = log_S + j * log_u + (i - j) * log_d
            spots_i = np.exp(log_spots_i)
            exercise = _payoff(spots_i, K, ot)
            values[: i + 1] = np.maximum(cont, exercise)
        else:
            values[: i + 1] = cont

    return float(values[0])


def crr_price_array(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps_list: ArrayLike = (10, 50, 100, 500, 1000),
    american: bool = False,
) -> list[tuple[int, float]]:
    """
    Price the same option at multiple step counts.

    Returns
    -------
    list of (n_steps, price)
    """
    out: list[tuple[int, float]] = []
    for n in n_steps_list:
        out.append((
            int(n),
            crr_price(S, K, T, r, sigma, q, option_type, n_steps=int(n), american=american),
        ))
    return out


def crr_convergence(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: str = "call",
    n_steps_list: ArrayLike = (10, 25, 50, 100, 250, 500, 1000, 2000),
    american: bool = False,
) -> list[tuple[int, float]]:
    """
    Alias for :func:`crr_price_array`, kept for readability at call sites.
    """
    return crr_price_array(
        S, K, T, r, sigma, q, option_type,
        n_steps_list=n_steps_list, american=american,
    )