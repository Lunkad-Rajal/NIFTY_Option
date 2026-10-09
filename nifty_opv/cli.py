"""
Runs the full pipeline end-to-end from a single command:

    python -m nifty_opv.cli --config config.yaml --synthetic

Inputs
------
- --config : path to the YAML config (default: config.yaml).
- --synthetic : generate a synthetic NIFTY option chain instead of reading ``data/raw/``. Useful for smoke-testing the pipeline without NSE files.
- --no-plots : skip figure generation.
- ``--no-crr`` : skip the CRR experiment.
- --crr-steps : override the CRR step count.

Outputs
-------
- reports/tables/data_quality.json     : row counts and exclusion reasons
- reports/tables/iv_chain.csv          : per-observation IV
- reports/tables/atm_vol.csv           : ATM IV per (date, expiry)
- reports/tables/pricing_errors_summary.csv
- reports/tables/parity_residuals.csv  : matched call/put pairs
- reports/tables/parity_summary.csv    : aggregated parity stats
- reports/tables/greeks_validation.csv : analytical vs FD Greeks
- reports/figures/*.png                : all plots
- reports/summary.json                 : machine-readable summary

Exit codes
----------
0  success
1  configuration or input error
2  no data to process
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_opv.analysis import (
    add_moneyness,
    add_time_to_expiry,
    assign_moneyness_bucket,
    compute_iv_chain,
    estimate_atm_vol,
    experiment_constant_atm_vol,
    experiment_crr,
    experiment_iv_repricing,
    parity_diagnostic,
    summarize_errors,
)
from nifty_opv.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vega,
)
from nifty_opv.config import Config, load_config, setup_logging
from nifty_opv.crr_tree import crr_price
from nifty_opv.datapipe import DataQualityReport, load_all_raw
from nifty_opv.finite_difference import (
    fd_delta,
    fd_gamma,
    fd_rho,
    fd_theta,
    fd_vega,
)
from nifty_opv.plots import (
    plot_atm_term_structure,
    plot_crr_convergence,
    plot_error_by_bucket,
    plot_error_vs_log_moneyness,
    plot_greeks_validation,
    plot_iv_vs_log_moneyness,
    plot_iv_vs_strike,
    plot_parity_residuals,
)

logger = logging.getLogger("nifty_opv.cli")



# Synthetic data generator


def generate_synthetic_chain(cfg: Config, seed: int | None = None) -> pd.DataFrame:
    """
    Build a synthetic NIFTY option chain that mimics the shape of a real
    Bhavcopy file: multiple trade dates, multiple expiries, a skewed smile,
    and a mixture of liquid and illiquid strikes.
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.project.random_seed)

    # Two weeks of weekdays, plus three expiries.
    trade_dates = pd.bdate_range("2024-01-08", periods=10)
    expiries = pd.to_datetime(["2024-01-25", "2024-02-29", "2024-03-28"])

    spot = 21000.0
    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield

    # Strikes around spot with a 250-point step.
    strikes = np.arange(19500, 22501, 250.0)

    rows: list[dict] = []
    for td in trade_dates:
        # Small daily spot moves so the panel is not a single state.
        S_t = spot * (1.0 + 0.002 * rng.standard_normal())

        for exp in expiries:
            T = (exp - td).days / 365.0
            if T <= 0:
                continue

            # Skewed smile: high IV for OTM puts, low IV around ATM,
            # moderate rise for OTM calls.
            for K in strikes:
                m = np.log(K / (S_t * np.exp((r - q) * T)))
                iv = 0.14 + 0.40 * (m - 0.01) ** 2 + 0.20 * max(0.0, -m)

                # OTM calls far from spot: less liquid.
                vol_level = int(np.clip(
                    5000.0 * np.exp(-3.0 * abs(m)) + 50.0 * rng.random(),
                    0.0, 20000.0,
                ))
                # Add some zero-volume rows deliberately.
                if abs(m) > 0.10 and rng.random() < 0.25:
                    vol_level = 0

                for ot in ("CE", "PE"):
                    px = float(bs_price(S_t, K, T, r, iv, q, ot))
                    rows.append({
                        "trade_date": td.normalize(),
                        "expiry_date": exp.normalize(),
                        "strike": float(K),
                        "option_type": ot,
                        "settlement_price": px,
                        "close": px,
                        "volume": vol_level,
                        "open_interest": int(max(vol_level, 100)),
                        "underlying_spot": S_t,
                        "instrument_type": "OPTIDX",
                        "underlying_symbol": "NIFTY",
                        "source_file": "synthetic",
                    })

    df = pd.DataFrame(rows)
    logger.info("Synthetic chain: %d rows, %d trade dates, %d expiries",
                len(df), df["trade_date"].nunique(), df["expiry_date"].nunique())
    return df



# Greeks validation table


def _build_greeks_table(cfg: Config) -> pd.DataFrame:
    """
    Produce a small grid of options and compare analytical Greeks to
    finite-difference Greeks. Returns a DataFrame suitable for CSV and
    for the validation scatter plot.
    """
    r = cfg.market.risk_free_rate
    q = cfg.market.dividend_yield
    S = 21000.0

    strikes = np.linspace(19500, 22500, 13)
    ttes = np.array([7, 14, 30, 60]) / 365.0
    sigmas = np.array([0.10, 0.15, 0.25])
    option_types = ["CE", "PE"]

    rows: list[dict] = []
    for K in strikes:
        for T in ttes:
            for sigma in sigmas:
                for ot in option_types:
                    a_delta = float(bs_delta(S, K, T, r, sigma, q, ot))
                    a_gamma = float(bs_gamma(S, K, T, r, sigma, q))
                    a_vega  = float(bs_vega (S, K, T, r, sigma, q))
                    a_rho   = float(bs_rho  (S, K, T, r, sigma, q, ot))
                    a_theta = float(bs_theta(S, K, T, r, sigma, q, ot))

                    # Absolute spot bump scaled to spot size.
                    h_spot = max(1e-3, 1e-5 * S)
                    f_delta = fd_delta(S, K, T, r, sigma, q, ot, h=h_spot)
                    f_gamma = fd_gamma(S, K, T, r, sigma, q, ot, h=1.0)
                    f_vega  = fd_vega (S, K, T, r, sigma, q, ot, h=1e-4)
                    f_rho   = fd_rho  (S, K, T, r, sigma, q, ot, h=1e-4)
                    f_theta = fd_theta(S, K, T, r, sigma, q, ot, h=1e-5)

                    rows.append({
                        "strike": K, "T": T, "sigma": sigma, "option_type": ot,
                        "analytical_delta": a_delta, "fd_delta": f_delta,
                        "analytical_gamma": a_gamma, "fd_gamma": f_gamma,
                        "analytical_vega":  a_vega,  "fd_vega":  f_vega,
                        "analytical_rho":   a_rho,   "fd_rho":   f_rho,
                        "analytical_theta": a_theta, "fd_theta": f_theta,
                    })

    df = pd.DataFrame(rows)
    for g in ("delta", "gamma", "vega", "rho", "theta"):
        df[f"abs_diff_{g}"] = (df[f"analytical_{g}"] - df[f"fd_{g}"]).abs()
    return df



# Output helpers


def _write_json(obj: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
    logger.info("Wrote %s", path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))


def _parity_summary(parity_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate parity residuals overall and by expiry."""
    if parity_df.empty:
        return pd.DataFrame(columns=[
            "grouping", "group_value", "n_pairs",
            "mean_residual", "median_residual",
            "mean_abs_residual", "max_abs_residual",
        ])

    rows: list[dict] = []

    def _stats(g: pd.DataFrame, grouping: str, value: str) -> dict:
        r = g["residual"].dropna()
        ar = g["abs_residual"].dropna()
        return {
            "grouping": grouping,
            "group_value": value,
            "n_pairs": int(len(r)),
            "mean_residual": float(r.mean()) if len(r) else np.nan,
            "median_residual": float(r.median()) if len(r) else np.nan,
            "mean_abs_residual": float(ar.mean()) if len(ar) else np.nan,
            "max_abs_residual": float(ar.max()) if len(ar) else np.nan,
        }

    rows.append(_stats(parity_df, "overall", "all"))
    for exp, g in parity_df.groupby("expiry_date", sort=True):
        rows.append(_stats(g, "expiry_date", str(pd.Timestamp(exp).date())))
    return pd.DataFrame(rows)



# Pipeline


def run_pipeline(cfg: Config, use_synthetic: bool,
                 do_plots: bool, do_crr: bool,
                 crr_steps_override: int | None) -> int:
    """
    Execute the full pipeline. Returns the process exit code.
    """
    logger.info("=== NIFTY Option Pricing Validator ===")
    logger.info("Config       : %s", cfg.project_root / "config.yaml")
    logger.info("Raw data dir : %s", cfg.paths.data_raw)
    logger.info("Synthetic    : %s", use_synthetic)
    logger.info("Plots        : %s", do_plots)
    logger.info("CRR          : %s", do_crr)

    # Data acquisition
    quality = DataQualityReport()
    if use_synthetic:
        df = generate_synthetic_chain(cfg)
        quality.input_rows = len(df)
        quality.output_rows = len(df)
        quality.source_files = ["synthetic"]
        quality.notes.append("Synthetic data used. Results are not empirical.")
    else:
        df, quality = load_all_raw(cfg, quality)

    if df.empty:
        logger.error(
            "No usable data. Place NSE Bhavcopy files in %s, or run with "
            "--synthetic to smoke-test the pipeline.",
            cfg.paths.data_raw,
        )
        _write_json(quality.to_dict(),
                    cfg.paths.reports_tables / "data_quality.json")
        return 2

    logger.info("Rows after cleaning: %d", len(df))

    # Derived columns
    df = add_time_to_expiry(df, cfg)
    df = add_moneyness(df, cfg)
    df = assign_moneyness_bucket(df, cfg)

    # Implied volatility per row
    df = compute_iv_chain(df, cfg)
    n_success = int((df["iv_status"] == "success").sum())
    logger.info("IV inverted for %d / %d rows", n_success, len(df))

    # ATM volatility per (date, expiry)
    atm = estimate_atm_vol(df, cfg)
    logger.info("ATM table: %d (date, expiry) groups", len(atm))

    # Experiments
    df = experiment_iv_repricing(df, cfg)
    df = experiment_constant_atm_vol(df, cfg, atm)
    if do_crr:
        df = experiment_crr(df, cfg, atm, n_steps=crr_steps_override)
    else:
        df["crr_price_atm_iv"] = np.nan
        df["crr_error_atm_iv"] = np.nan
        df["crr_minus_bs"] = np.nan

    # Error aggregation
    errors_own_iv = summarize_errors(df, cfg, error_col="error_own_iv")
    errors_own_iv["error_kind"] = "iv_repricing"
    errors_atm = summarize_errors(df, cfg, error_col="error_atm_iv")
    errors_atm["error_kind"] = "constant_atm_vol"
    if do_crr:
        errors_crr = summarize_errors(df, cfg, error_col="crr_error_atm_iv")
        errors_crr["error_kind"] = "crr"
        errors_all = pd.concat([errors_own_iv, errors_atm, errors_crr],
                               ignore_index=True)
    else:
        errors_all = pd.concat([errors_own_iv, errors_atm], ignore_index=True)

    # Parity diagnostic
    parity_df = parity_diagnostic(df, cfg)
    parity_summary = _parity_summary(parity_df)
    logger.info("Parity pairs matched: %d", len(parity_df))

    # Greeks validation
    greeks_df = _build_greeks_table(cfg)

    # Write tables
    tables = cfg.paths.reports_tables
    _write_json(quality.to_dict(), tables / "data_quality.json")
    _write_csv(df, tables / "iv_chain.csv")
    _write_csv(atm, tables / "atm_vol.csv")
    _write_csv(errors_all, tables / "pricing_errors_summary.csv")
    _write_csv(parity_df, tables / "parity_residuals.csv")
    _write_csv(parity_summary, tables / "parity_summary.csv")
    _write_csv(greeks_df, tables / "greeks_validation.csv")

    # Figures
    if do_plots:
        # Pick one representative trade date for the smile plots.
        td_pick = pd.Timestamp(df["trade_date"].max())
        plot_iv_vs_strike(df, cfg, trade_date=td_pick)
        plot_iv_vs_log_moneyness(df, cfg, trade_date=td_pick)
        plot_atm_term_structure(atm, cfg)
        plot_error_by_bucket(df, cfg, error_col="error_atm_iv")
        plot_error_vs_log_moneyness(df, cfg, error_col="error_atm_iv")
        plot_parity_residuals(parity_df, cfg)

        # Greeks validation: one figure per Greek.
        for g in ("delta", "gamma", "vega", "rho", "theta"):
            plot_greeks_validation(greeks_df, cfg, greek=g)

        # CRR convergence: pick the ATM strike of the near expiry on the last trade date.
        if not atm.empty:
            row = atm.sort_values(["trade_date", "expiry_date"]).iloc[-1]
            S = float(df["underlying_spot"].iloc[-1])
            K = float(row["atm_strike"])
            T = float((pd.Timestamp(row["expiry_date"]) -
                       pd.Timestamp(row["trade_date"])).days / 365.0)
            sigma = float(row["atm_iv"])
            bs_ref = float(bs_price(S, K, T, cfg.market.risk_free_rate,
                                    sigma, cfg.market.dividend_yield, "CE"))
            steps = cfg.models.crr.convergence_steps
            conv = pd.DataFrame([
                {"n_steps": int(n),
                 "price": crr_price(S, K, T, cfg.market.risk_free_rate,
                                    sigma, cfg.market.dividend_yield,
                                    "CE", n_steps=int(n))}
                for n in steps
            ])
            plot_crr_convergence(conv, cfg, bs_ref)

    # Summary JSON
    summary = {
        "project": cfg.project.name,
        "version": cfg.project.version,
        "synthetic_data": bool(use_synthetic),
        "n_rows_after_cleaning": int(len(df)),
        "n_trade_dates": int(df["trade_date"].nunique()),
        "n_expiries": int(df["expiry_date"].nunique()),
        "n_strikes": int(df["strike"].nunique()),
        "iv_success_rate": float((df["iv_status"] == "success").mean()),
        "data_quality": quality.to_dict(),
        "market_inputs": {
            "risk_free_rate": cfg.market.risk_free_rate,
            "dividend_yield": cfg.market.dividend_yield,
            "day_count": cfg.market.day_count,
        },
        "pricing_errors": {
            "iv_repricing_mean_abs": float(
                errors_all.query("error_kind == 'iv_repricing' and grouping == 'overall'")
                ["mean_abs_error"].iloc[0]) if not errors_all.empty else None,
            "constant_atm_mean_abs": float(
                errors_all.query("error_kind == 'constant_atm_vol' and grouping == 'overall'")
                ["mean_abs_error"].iloc[0]) if not errors_all.empty else None,
            "crr_mean_abs": float(
                errors_all.query("error_kind == 'crr' and grouping == 'overall'")
                ["mean_abs_error"].iloc[0])
                if (do_crr and not errors_all.empty and
                    (errors_all["error_kind"] == "crr").any()) else None,
        },
        "parity": {
            "n_pairs": int(len(parity_df)),
            "mean_residual": float(parity_df["residual"].mean())
                if not parity_df.empty else None,
            "mean_abs_residual": float(parity_df["abs_residual"].mean())
                if not parity_df.empty else None,
        },
        "greeks_validation": {
            g: float(greeks_df[f"abs_diff_{g}"].max())
            for g in ("delta", "gamma", "vega", "rho", "theta")
        },
    }
    _write_json(summary, cfg.paths.reports_tables.parent / "summary.json")

    logger.info("Pipeline finished successfully.")
    return 0



# Argument parsing and entry point


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="nifty_opv.cli",
        description="NIFTY option pricing validator",
    )
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML config file (default: config.yaml)")
    p.add_argument("--synthetic", action="store_true",
                   help="Generate synthetic data instead of reading data/raw/")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip figure generation")
    p.add_argument("--no-crr", action="store_true",
                   help="Skip the CRR experiment (slowest step)")
    p.add_argument("--crr-steps", type=int, default=None,
                   help="Override CRR step count")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, KeyError, ValueError, TypeError) as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    setup_logging(cfg)

    # Allow a temporary strike-range override when the synthetic chain is
    # wider than the config range. Useful for smoke tests.
    if args.synthetic:
        cfg = replace(
            cfg,
            data=replace(cfg.data, min_strike=0.0, max_strike=1e9,
                         date_start="2000-01-01", date_end="2100-01-01"),
        )

    try:
        return run_pipeline(
            cfg=cfg,
            use_synthetic=args.synthetic,
            do_plots=not args.no_plots,
            do_crr=not args.no_crr,
            crr_steps_override=args.crr_steps,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())