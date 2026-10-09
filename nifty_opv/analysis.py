"""
Empirical analysis layer for the NIFTY option pricing validator.

Input:  a cleaned DataFrame from :mod:`nifty_opv.datapipe`.
Output: DataFrames of results consumed by the plotting and CLI layers.

- time-to-expiry and moneyness calculations
- implied volatility for every observation
- ATM volatility estimation per (trade_date, expiry)
- three pricing experiments
- error aggregation
- put-call parity diagnostic
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from nifty_opv.black_scholes import bs_price
from nifty_opv.config import Config
from nifty_opv.crr_tree import crr_price
from nifty_opv.implied_vol import implied_volatility

logger = logging.getLogger(__name__)

__all__ = [
    "add_time_to_expiry",
    "add_moneyness",
    "assign_moneyness_bucket",
    "compute_iv_chain",
    "estimate_atm_vol",
    "experiment_iv_repricing",
    "experiment_constant_atm_vol",
    "experiment_crr",
    "summarize_errors",
    "parity_diagnostic",
]



# Derived columns


def add_time_to_expiry(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Add a time_to_expiry column (in years) using the config day-count.

    - calendar_days_365: T = calendar_days / 365
    - trading_days_252 : T = business_days / 252
      Business days exclude weekends but not NSE holidays. This is an
      approximation; document it when you report numbers.
    """
    work = df.copy()
    if work.empty:
        work["time_to_expiry"] = pd.Series(dtype=float)
        return work

    if cfg.market.day_count == "calendar_days_365":
        days = (work["expiry_date"] - work["trade_date"]).dt.days.astype(float)
        work["time_to_expiry"] = days / 365.0
    elif cfg.market.day_count == "trading_days_252":
        td = work["trade_date"].values.astype("datetime64[D]")
        ed = work["expiry_date"].values.astype("datetime64[D]")
        bdays = np.busday_count(td, ed).astype(float)
        work["time_to_expiry"] = bdays / 252.0
    else:
        raise ValueError(f"Unknown day_count: {cfg.market.day_count!r}")

    return work


def add_moneyness(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Add forward, k_over_s, and log_moneyness columns.

    forward       = S * exp((r - q) * T)
    k_over_s      = K / S
    log_moneyness = ln(K / F)
    """
    work = df.copy()
    if work.empty:
        work["forward"] = pd.Series(dtype=float)
        work["k_over_s"] = pd.Series(dtype=float)
        work["log_moneyness"] = pd.Series(dtype=float)
        return work

    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    work["forward"] = work["underlying_spot"] * np.exp((r - q) * work["time_to_expiry"])
    work["k_over_s"] = work["strike"] / work["underlying_spot"]
    work["log_moneyness"] = np.log(work["strike"] / work["forward"])
    return work


def assign_moneyness_bucket(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Assign each row to a moneyness bucket based on log_moneyness.

    Labels come from cfg.moneyness_buckets.labels. Values outside the
    outer edges are labeled outside_bucket_range.
    """
    work = df.copy()
    if work.empty:
        work["moneyness_bucket"] = pd.Series(dtype=object)
        return work

    edges = np.asarray(cfg.moneyness_buckets.edges, dtype=float)
    labels = list(cfg.moneyness_buckets.labels)

    # pd.cut requires monotonically increasing edges and len(labels) = len(edges) - 1.
    work["moneyness_bucket"] = pd.cut(
        work["log_moneyness"],
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=True,
    )
    # Anything outside the bucket range gets a named bucket.
    work["moneyness_bucket"] = work["moneyness_bucket"].astype(object)
    outside = work["moneyness_bucket"].isna() & work["log_moneyness"].notna()
    work.loc[outside, "moneyness_bucket"] = "outside_bucket_range"
    return work



# Implied volatility across a chain


def compute_iv_chain(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Run the IV solver over every row and attach diagnostics.

    Adds columns:
        iv                    : implied vol (NaN if failed)
        iv_status             : status string from IVResult
        iv_repricing_error    : |model - market| after inversion
        iv_failure_reason     : reason string if status != "success"

    Uses the price column specified by ``cfg.models.black_scholes.price_field``.
    """
    work = df.copy()
    if work.empty:
        for col in ("iv", "iv_status", "iv_repricing_error", "iv_failure_reason"):
            work[col] = pd.Series(dtype=float if col != "iv_status" and col != "iv_failure_reason" else object)
        return work

    price_field = cfg.models.black_scholes.price_field
    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    iv_cfg = cfg.implied_vol

    ivs: list[float] = []
    statuses: list[str] = []
    errors: list[float] = []
    reasons: list[str | None] = []

    for row in work.itertuples(index=False):
        px = float(getattr(row, price_field))
        res = implied_volatility(
            market_price=px,
            S=float(row.underlying_spot),
            K=float(row.strike),
            T=float(row.time_to_expiry),
            r=r,
            option_type=str(row.option_type),
            q=q,
            lo=iv_cfg.lo,
            hi=iv_cfg.hi,
            tol=iv_cfg.tol,
            max_bracket_expansions=iv_cfg.max_bracket_expansions,
            bracket_multiplier=iv_cfg.bracket_multiplier,
            repricing_tol=iv_cfg.repricing_tol,
        )
        ivs.append(res.iv if res.iv is not None else np.nan)
        statuses.append(res.status)
        errors.append(res.repricing_error if res.repricing_error is not None else np.nan)
        reasons.append(res.failure_reason)

    work["iv"] = ivs
    work["iv_status"] = statuses
    work["iv_repricing_error"] = errors
    work["iv_failure_reason"] = reasons
    return work



# ATM volatility estimation


def estimate_atm_vol(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Estimate one ATM implied volatility per (trade_date, expiry).

    Rule (from config):
      - ``closest_to_forward``: pick the strike whose |log_moneyness| is
        smallest; take the median IV across calls and puts at that strike.
      - ``closest_to_spot``: pick the strike whose |K/S - 1| is smallest.

    Returns a DataFrame with columns: trade_date, expiry_date, atm_iv,
    atm_strike, atm_rule, n_obs.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "trade_date", "expiry_date", "atm_iv", "atm_strike", "atm_rule", "n_obs"
        ])

    rule = cfg.experiments.constant_atm_vol.atm_rule

    # Only use rows with a valid IV.
    valid = df[df["iv"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=[
            "trade_date", "expiry_date", "atm_iv", "atm_strike", "atm_rule", "n_obs"
        ])

    if rule == "closest_to_forward":
        valid["_abs_m"] = valid["log_moneyness"].abs()
    elif rule == "closest_to_spot":
        valid["_abs_m"] = (valid["k_over_s"] - 1.0).abs()
    else:
        raise ValueError(f"Unknown atm_rule: {rule!r}")

    out_rows: list[dict] = []
    for (td, ed), grp in valid.groupby(["trade_date", "expiry_date"], sort=True):
        g = grp.sort_values("_abs_m")
        if g.empty:
            continue
        atm_strike = float(g.iloc[0]["strike"])
        atm_rows = g[g["strike"] == atm_strike]
        atm_iv = float(np.nanmedian(atm_rows["iv"].values))
        out_rows.append({
            "trade_date": td,
            "expiry_date": ed,
            "atm_iv": atm_iv,
            "atm_strike": atm_strike,
            "atm_rule": rule,
            "n_obs": int(len(atm_rows)),
        })

    return pd.DataFrame(out_rows)



# Experiments


def _market_price_column(cfg: Config) -> str:
    return cfg.models.black_scholes.price_field


def experiment_iv_repricing(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Experiment 1: reprice each option with its own implied volatility.

    This is a numerical consistency test (inversion round-trip). It is
    NOT independent evidence that Black-Scholes explains the market. A
    small error here only proves the IV solver converged.
    """
    work = df.copy()
    if work.empty:
        return work.assign(bs_price_own_iv=pd.Series(dtype=float),
                           error_own_iv=pd.Series(dtype=float))

    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    price_col = _market_price_column(cfg)

    prices: list[float] = []
    for row in work.itertuples(index=False):
        iv = float(row.iv)
        if not np.isfinite(iv):
            prices.append(np.nan)
            continue
        prices.append(float(bs_price(
            S=float(row.underlying_spot),
            K=float(row.strike),
            T=float(row.time_to_expiry),
            r=r,
            sigma=iv,
            q=q,
            option_type=str(row.option_type),
        )))

    work["bs_price_own_iv"] = prices
    work["error_own_iv"] = work["bs_price_own_iv"] - work[price_col]
    return work


def experiment_constant_atm_vol(df: pd.DataFrame, cfg: Config,
                                atm_table: pd.DataFrame) -> pd.DataFrame:
    """
    Experiment 2: reprice each option with its group's constant ATM volatility.

    Parameters
    ----------
    df
        Frame with ``trade_date``, ``expiry_date``, and the usual columns.
    cfg
        Config.
    atm_table
        Output of :func:`estimate_atm_vol`.
    """
    work = df.copy()
    if work.empty:
        return work.assign(atm_iv=pd.Series(dtype=float),
                           bs_price_atm_iv=pd.Series(dtype=float),
                           error_atm_iv=pd.Series(dtype=float))

    # Drop any stale ATM columns from a previous merge so the new merge
    # does not produce suffixed columns (atm_iv_x, atm_iv_y, ...).
    stale = [c for c in ("atm_iv", "atm_strike", "atm_rule", "n_obs")
             if c in work.columns]
    if stale:
        work = work.drop(columns=stale)

    work = work.merge(atm_table, on=["trade_date", "expiry_date"], how="left")

    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    price_col = _market_price_column(cfg)

    prices: list[float] = []
    for row in work.itertuples(index=False):
        sigma = float(row.atm_iv) if np.isfinite(row.atm_iv) else np.nan
        if not np.isfinite(sigma):
            prices.append(np.nan)
            continue
        prices.append(float(bs_price(
            S=float(row.underlying_spot),
            K=float(row.strike),
            T=float(row.time_to_expiry),
            r=r,
            sigma=sigma,
            q=q,
            option_type=str(row.option_type),
        )))

    work["bs_price_atm_iv"] = prices
    work["error_atm_iv"] = work["bs_price_atm_iv"] - work[price_col]
    return work


def experiment_crr(df: pd.DataFrame, cfg: Config,
                   atm_table: pd.DataFrame,
                   n_steps: int | None = None) -> pd.DataFrame:
    """
    Experiment 3: reprice each option with CRR using the constant ATM vol.

    Parameters
    ----------
    df
        Frame with the usual columns.
    cfg
        Config.
    atm_table
        Output of :func:`estimate_atm_vol`.
    n_steps
        Tree steps. Defaults to ``cfg.experiments.crr_tree.steps``.

    Notes
    -----
    This is the slowest experiment. With n_steps=500 and ~1000 rows per
    expiry, expect a few seconds per expiry.
    """
    work = df.copy()
    if work.empty:
        return work.assign(crr_price_atm_iv=pd.Series(dtype=float),
                           crr_minus_bs=pd.Series(dtype=float),
                           crr_error_atm_iv=pd.Series(dtype=float))

    n = int(n_steps if n_steps is not None else cfg.experiments.crr_tree.steps)

    # Drop any stale ATM columns from a previous merge so the new merge
    # does not produce suffixed columns (atm_iv_x, atm_iv_y, ...).
    stale = [c for c in ("atm_iv", "atm_strike", "atm_rule", "n_obs")
             if c in work.columns]
    if stale:
        work = work.drop(columns=stale)

    work = work.merge(atm_table, on=["trade_date", "expiry_date"], how="left")

    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    price_col = _market_price_column(cfg)

    crr_prices: list[float] = []
    for row in work.itertuples(index=False):
        sigma = float(row.atm_iv) if np.isfinite(row.atm_iv) else np.nan
        if not np.isfinite(sigma):
            crr_prices.append(np.nan)
            continue
        crr_prices.append(crr_price(
            S=float(row.underlying_spot),
            K=float(row.strike),
            T=float(row.time_to_expiry),
            r=r,
            sigma=sigma,
            q=q,
            option_type=str(row.option_type),
            n_steps=n,
            american=False,
        ))

    work["crr_price_atm_iv"] = crr_prices
    work["crr_error_atm_iv"] = work["crr_price_atm_iv"] - work[price_col]

    if "bs_price_atm_iv" in work.columns:
        work["crr_minus_bs"] = work["crr_price_atm_iv"] - work["bs_price_atm_iv"]
    else:
        work["crr_minus_bs"] = np.nan

    return work



# Error aggregation


def _error_stats(group: pd.DataFrame, error_col: str, price_col: str) -> pd.Series:
    err = group[error_col].dropna()
    price = group[price_col].dropna()
    n = int(err.shape[0])
    if n == 0:
        return pd.Series({
            "n": 0,
            "mean_abs_error": np.nan,
            "median_abs_error": np.nan,
            "mean_signed_error": np.nan,
            "rmse": np.nan,
            "mean_rel_error": np.nan,
        })
    abs_err = err.abs()
    rel_err = (abs_err / price.reindex(err.index).replace(0.0, np.nan)).dropna()
    return pd.Series({
        "n": n,
        "mean_abs_error": float(abs_err.mean()),
        "median_abs_error": float(abs_err.median()),
        "mean_signed_error": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "mean_rel_error": float(rel_err.mean()) if not rel_err.empty else np.nan,
    })


def summarize_errors(df: pd.DataFrame, cfg: Config,
                     error_col: str = "error_atm_iv",
                     price_col: str | None = None) -> pd.DataFrame:
    """
    Aggregate pricing errors overall and by grouping variables.

    Returns a long DataFrame with columns:
        grouping, group_value, n, mean_abs_error, median_abs_error,
        mean_signed_error, rmse, mean_rel_error
    """
    price_col = price_col or _market_price_column(cfg)
    if df.empty or error_col not in df.columns:
        return pd.DataFrame(columns=[
            "grouping", "group_value", "n", "mean_abs_error",
            "median_abs_error", "mean_signed_error", "rmse", "mean_rel_error",
        ])

    rows: list[dict] = []

    # Overall
    overall = _error_stats(df, error_col, price_col).to_dict()
    rows.append({"grouping": "overall", "group_value": "all", **overall})

    # By groupings that may or may not be present.
    for grouping in ("option_type", "moneyness_bucket", "expiry_date", "trade_date"):
        if grouping not in df.columns:
            continue
        for val, grp in df.groupby(grouping, sort=True, dropna=False):
            stats = _error_stats(grp, error_col, price_col).to_dict()
            rows.append({"grouping": grouping, "group_value": str(val), **stats})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Put-call parity diagnostic
# ---------------------------------------------------------------------------

def parity_diagnostic(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Compute put-call parity residuals on matched call/put pairs.

    Model identity (European, continuous carry):
        C - P = S e^{-qT} - K e^{-rT}

    Residual:
        residual = C_market - P_market - (S e^{-qT} - K e^{-rT})

    Returns one row per matched pair with trade_date, expiry_date, strike,
    moneyness, residual, and absolute residual.

    Notes
    -----
    This is a consistency diagnostic, not an executable arbitrage test.
    Daily Bhavcopy prices are not executable bid/ask quotes, and the
    calculation ignores transaction costs, financing, and short-sale
    constraints. A nonzero residual is not evidence of a profitable trade.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "trade_date", "expiry_date", "strike", "log_moneyness",
            "call_price", "put_price", "forward_parity", "residual",
            "abs_residual",
        ])

    price_col = _market_price_column(cfg)
    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield

    calls = df[df["option_type"] == "CE"].copy()
    puts = df[df["option_type"] == "PE"].copy()

    key = ["trade_date", "expiry_date", "strike"]
    calls_small = calls[key + [price_col, "underlying_spot", "time_to_expiry",
                               "log_moneyness"]].rename(
        columns={price_col: "call_price"}
    )
    puts_small = puts[key + [price_col]].rename(columns={price_col: "put_price"})

    merged = calls_small.merge(puts_small, on=key, how="inner")
    if merged.empty:
        return pd.DataFrame(columns=[
            "trade_date", "expiry_date", "strike", "log_moneyness",
            "call_price", "put_price", "forward_parity", "residual",
            "abs_residual",
        ])

    S = merged["underlying_spot"].astype(float)
    K = merged["strike"].astype(float)
    T = merged["time_to_expiry"].astype(float)
    merged["forward_parity"] = S * np.exp(-q * T) - K * np.exp(-r * T)
    merged["residual"] = merged["call_price"] - merged["put_price"] - merged["forward_parity"]
    merged["abs_residual"] = merged["residual"].abs()

    return merged[[
        "trade_date", "expiry_date", "strike", "log_moneyness",
        "call_price", "put_price", "forward_parity", "residual",
        "abs_residual",
    ]].reset_index(drop=True)