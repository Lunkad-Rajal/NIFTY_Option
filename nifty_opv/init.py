"""
NIFTY Option Pricing Validator.

An empirical study comparing Black-Scholes and Cox-Ross-Rubinstein
(CRR) binomial tree model prices against traded NIFTY index option prices from
NSE Bhavcopy data.

Modules
-------
config              : loads config.yaml into a typed settings object
black_scholes       : European BS price and analytical Greeks
implied_vol         : implied volatility inversion (Brent)
crr_tree            : Cox-Ross-Rubinstein binomial tree
finite_difference   : finite-difference Greeks for validation
data_pipeline       : NSE download, ingestion, cleaning
analysis            : smile, pricing errors, parity, diagnostics
plots               : all figures
cli                 : end-to-end pipeline entry point
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = ["__version__", "__author__"]
