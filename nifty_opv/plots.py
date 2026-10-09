"""
Every function takes a pandas DataFrame and a :class:`nifty_opv.config.Config`, and writes a PNG to
"cfg.paths.reports_figures". Functions return the output path so the CLI
can log what it produced.

Nothing in this module computes prices, volatilities, or errors. If you find
yourself wanting to add math here, add it to ``analysis.py`` instead and
pass the result in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from nifty_opv.config import Config

logger = logging.getLogger(__name__)

__all__ = [
    "setup_style",
    "plot_iv_vs_strike",
    "plot_iv_vs_log_moneyness",
    "plot_atm_term_structure",
    "plot_error_by_bucket",
    "plot_error_vs_log_moneyness",
    "plot_crr_convergence",
    "plot_greeks_validation",
    "plot_parity_residuals",
]



# Style


def setup_style(cfg: Config) -> None:
    """Apply the matplotlib/seaborn style from the config."""
    try:
        plt.style.use(cfg.plots.style)
    except OSError:
        # Older matplotlib may not ship the versioned name.
        plt.style.use("seaborn-whitegrid")
    sns.set_palette(cfg.plots.palette)


def _save(fig: plt.Figure, cfg: Config, name: str) -> Path:
    """Save a figure to reports/figures/<name>.<fmt> and close it."""
    out_dir = cfg.paths.reports_figures
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.{cfg.plots.save_format}"
    fig.savefig(path, dpi=cfg.plots.dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", path)
    return path



# Volatility smile / skew


def plot_iv_vs_strike(df: pd.DataFrame, cfg: Config,
                      trade_date: pd.Timestamp | None = None,
                      filename: str = "iv_vs_strike") -> Path:
    """
    Plot implied volatility against strike, one line per expiry, calls and
    puts on the same axes.

    Parameters
    ----------
    df
        Clean frame with iv, strike, expiry_date, option_type.
    trade_date
        If given, restrict to that trade date. Otherwise plot all dates
        pooled (rarely useful).
    """
    setup_style(cfg)
    work = df.copy()
    if trade_date is not None:
        work = work[work["trade_date"] == pd.Timestamp(trade_date)]
    work = work[work["iv"].notna()]

    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if work.empty:
        ax.text(0.5, 0.5, "No valid IV observations",
                ha="center", va="center", transform=ax.transAxes)
    else:
        expiries = sorted(work["expiry_date"].unique())
        for exp in expiries:
            sub = work[work["expiry_date"] == exp]
            for ot, marker in (("CE", "o"), ("PE", "x")):
                s = sub[sub["option_type"] == ot].sort_values("strike")
                if s.empty:
                    continue
                ax.plot(
                    s["strike"], s["iv"],
                    marker=marker, linestyle="-",
                    label=f"{pd.Timestamp(exp).date()} {ot}",
                    alpha=0.8,
                )
    ax.set_xlabel("Strike (index points)")
    ax.set_ylabel("Implied volatility (annualized)")
    ax.set_title("Implied volatility vs strike")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, cfg, filename)


def plot_iv_vs_log_moneyness(df: pd.DataFrame, cfg: Config,
                             trade_date: pd.Timestamp | None = None,
                             filename: str = "iv_vs_log_moneyness") -> Path:
    """
    Plot implied volatility against log-moneyness ln(K/F). This is the
    canonical smile/skew view. If the smile is skewed rather than symmetric
    it will be obvious here.
    """
    setup_style(cfg)
    work = df.copy()
    if trade_date is not None:
        work = work[work["trade_date"] == pd.Timestamp(trade_date)]
    work = work[work["iv"].notna() & work["log_moneyness"].notna()]

    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if work.empty:
        ax.text(0.5, 0.5, "No valid IV observations",
                ha="center", va="center", transform=ax.transAxes)
    else:
        for (exp, ot), sub in work.groupby(["expiry_date", "option_type"]):
            sub = sub.sort_values("log_moneyness")
            marker = "o" if ot == "CE" else "x"
            ax.plot(
                sub["log_moneyness"], sub["iv"],
                marker=marker, linestyle="-",
                label=f"{pd.Timestamp(exp).date()} {ot}",
                alpha=0.8,
            )
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Log-moneyness  ln(K / F)")
    ax.set_ylabel("Implied volatility (annualized)")
    ax.set_title("Volatility smile / skew in log-moneyness")
    ax.legend(fontsize=8, loc="best")
    return _save(fig, cfg, filename)


def plot_atm_term_structure(atm_table: pd.DataFrame, cfg: Config,
                            filename: str = "atm_term_structure") -> Path:
    """
    Plot ATM implied volatility against time to expiry, one line per
    trade date.
    """
    setup_style(cfg)
    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if atm_table.empty:
        ax.text(0.5, 0.5, "No ATM IV available",
                ha="center", va="center", transform=ax.transAxes)
    else:
        atm_table = atm_table.copy()
        atm_table["tte_days"] = (
            pd.to_datetime(atm_table["expiry_date"]) - pd.to_datetime(atm_table["trade_date"])
        ).dt.days
        for td, sub in atm_table.groupby("trade_date"):
            sub = sub.sort_values("tte_days")
            ax.plot(sub["tte_days"], sub["atm_iv"],
                    marker="o", label=str(pd.Timestamp(td).date()))
    ax.set_xlabel("Calendar days to expiry")
    ax.set_ylabel("ATM implied volatility (annualized)")
    ax.set_title("ATM implied volatility term structure")
    if not atm_table.empty:
        ax.legend(fontsize=8, loc="best")
    return _save(fig, cfg, filename)



# Pricing errors


def plot_error_by_bucket(df: pd.DataFrame, cfg: Config,
                         error_col: str = "error_atm_iv",
                         filename: str = "error_by_bucket") -> Path:
    """
    Bar chart of mean absolute error per moneyness bucket, split by
    option type. Uses ``log_moneyness`` buckets assigned by analysis.py.
    """
    setup_style(cfg)
    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if df.empty or error_col not in df.columns or "moneyness_bucket" not in df.columns:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, cfg, filename)

    work = df[df[error_col].notna()].copy()
    work["abs_err"] = work[error_col].abs()

    bucket_order = list(cfg.moneyness_buckets.labels) + ["outside_bucket_range"]
    present = [b for b in bucket_order if b in work["moneyness_bucket"].unique()]

    grouped = (work.groupby(["moneyness_bucket", "option_type"])["abs_err"]
                   .mean().reset_index())

    x = np.arange(len(present))
    width = 0.4
    for i, ot in enumerate(("CE", "PE")):
        vals = []
        for b in present:
            row = grouped[(grouped["moneyness_bucket"] == b) & (grouped["option_type"] == ot)]
            vals.append(float(row["abs_err"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width, label=ot)

    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean absolute error (index points)")
    ax.set_title(f"Pricing error by moneyness bucket  ({error_col})")
    ax.legend()
    return _save(fig, cfg, filename)


def plot_error_vs_log_moneyness(df: pd.DataFrame, cfg: Config,
                                error_col: str = "error_atm_iv",
                                filename: str = "error_vs_log_moneyness") -> Path:
    """
    Scatter of signed pricing error against log-moneyness, coloured by
    option type. A horizontal line at zero is drawn.
    """
    setup_style(cfg)
    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if df.empty or error_col not in df.columns:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, cfg, filename)

    work = df[df[error_col].notna() & df["log_moneyness"].notna()]
    if work.empty:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, cfg, filename)

    for ot, color in (("CE", "tab:blue"), ("PE", "tab:orange")):
        s = work[work["option_type"] == ot]
        if s.empty:
            continue
        ax.scatter(s["log_moneyness"], s[error_col],
                   s=18, alpha=0.6, label=ot, color=color)

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Log-moneyness  ln(K / F)")
    ax.set_ylabel("Signed error (index points)")
    ax.set_title(f"Signed pricing error vs log-moneyness  ({error_col})")
    ax.legend()
    return _save(fig, cfg, filename)



# Numerical convergence


def plot_crr_convergence(convergence: pd.DataFrame, cfg: Config,
                         bs_reference: float,
                         filename: str = "crr_convergence") -> Path:
    """
    Plot |CRR - BS| against step count on a log-log axis.

    Parameters
    ----------
    convergence
        DataFrame with columns n_steps and price.
    bs_reference
        Black-Scholes price under identical inputs. The absolute difference
        is plotted.
    """
    setup_style(cfg)
    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if convergence.empty:
        ax.text(0.5, 0.5, "No convergence data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, cfg, filename)

    conv = convergence.copy()
    conv["abs_diff"] = (conv["price"] - bs_reference).abs()
    conv["parity"] = np.where(conv["n_steps"] % 2 == 0, "even", "odd")

    for parity, sub in conv.groupby("parity"):
        sub = sub.sort_values("n_steps")
        ax.loglog(sub["n_steps"], sub["abs_diff"],
                  marker="o", label=f"{parity} steps")

    ax.set_xlabel("Number of CRR steps")
    ax.set_ylabel("|CRR - Black-Scholes|  (index points)")
    ax.set_title("CRR convergence to Black-Scholes")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    return _save(fig, cfg, filename)


def plot_greeks_validation(greeks_df: pd.DataFrame, cfg: Config,
                           greek: str = "delta",
                           filename: str | None = None) -> Path:
    """
    Scatter of analytical vs finite-difference Greek with a y=x line.

    Parameters
    ----------
    greeks_df
        DataFrame with columns ``analytical_<greek>`` and ``fd_<greek>``.
    greek
        One of delta, gamma, vega, rho, theta.
    """
    setup_style(cfg)
    an_col = f"analytical_{greek}"
    fd_col = f"fd_{greek}"
    fig, ax = plt.subplots(figsize=tuple(cfg.plots.figsize))
    if greeks_df.empty or an_col not in greeks_df.columns or fd_col not in greeks_df.columns:
        ax.text(0.5, 0.5, "No data",
                ha="center", va="center", transform=ax.transAxes)
        return _save(fig, cfg, filename or f"greeks_validation_{greek}")

    x = greeks_df[an_col].astype(float)
    y = greeks_df[fd_col].astype(float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    ax.scatter(x, y, s=18, alpha=0.6)
    if len(x) > 0:
        lo = min(float(x.min()), float(y.min()))
        hi = max(float(x.max()), float(y.max()))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=0.8, label="y = x")
    ax.set_xlabel(f"Analytical {greek}")
    ax.set_ylabel(f"Finite-difference {greek}")
    ax.set_title(f"Analytical vs finite-difference {greek}")
    ax.legend()
    return _save(fig, cfg, filename or f"greeks_validation_{greek}")



# Put-call parity


def plot_parity_residuals(parity_df: pd.DataFrame, cfg: Config,
                          filename: str = "parity_residuals") -> Path:
    """
    Two-panel figure: histogram of residuals and residuals vs strike.
    """
    setup_style(cfg)
    fig, axes = plt.subplots(1, 2, figsize=(cfg.plots.figsize[0] * 1.6,
                                            cfg.plots.figsize[1]))
    ax_hist, ax_scatter = axes

    if parity_df.empty or "residual" not in parity_df.columns:
        for ax in axes:
            ax.text(0.5, 0.5, "No parity data",
                    ha="center", va="center", transform=ax.transAxes)
        return _save(fig, cfg, filename)

    residual = parity_df["residual"].dropna()
    ax_hist.hist(residual, bins=30, color="steelblue", alpha=0.8)
    ax_hist.axvline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_hist.set_xlabel("Parity residual C - P - (S e^{-qT} - K e^{-rT})")
    ax_hist.set_ylabel("Count")
    ax_hist.set_title("Parity residual distribution")

    ax_scatter.scatter(parity_df["strike"], parity_df["residual"],
                       s=15, alpha=0.6, color="tab:red")
    ax_scatter.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax_scatter.set_xlabel("Strike (index points)")
    ax_scatter.set_ylabel("Residual")
    ax_scatter.set_title("Parity residual vs strike")

    fig.suptitle("Put-call parity diagnostic (consistency check, not arbitrage)")
    return _save(fig, cfg, filename)