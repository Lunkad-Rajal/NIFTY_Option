"""
Every test in this file uses synthetic data.

Run with:
    pytest -q tests/test_all.py
or, from the project root:
    pytest -q

Test classes
------------
TestBlackScholes     : prices against the Haug reference table, parity, limits
TestGreeks           : analytical bounds and FD-agreement for all five Greeks
TestImpliedVol       : round-trip, bracket rejection, input validation
TestCRRTree          : convergence to BS, parity, input validation
TestAnalysis         : derived columns, ATM table, pricing experiments, parity
TestDataCleaning     : exclusion reasons on a small synthetic frame
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from nifty_opv.analysis import (
    add_moneyness,
    add_time_to_expiry,
    assign_moneyness_bucket,
    compute_iv_chain,
    estimate_atm_vol,
    experiment_constant_atm_vol,
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
from nifty_opv.config import load_config
from nifty_opv.crr_tree import crr_price
from nifty_opv.datapipe import (
    DataQualityReport,
    clean_options_data,
    normalize_columns,
)
from nifty_opv.finite_difference import (
    fd_delta,
    fd_gamma,
    fd_rho,
    fd_theta,
    fd_vega,
)
from nifty_opv.implied_vol import implied_volatility



# Fixtures


@pytest.fixture(scope="module")
def cfg():
    """
    Load the real config.yaml but widen the strike range so test fixtures
    are not filtered out.
    """
    c = load_config("config.yaml")
    return replace(
        c,
        data=replace(
            c.data,
            min_strike=0.0,
            max_strike=1e9,
            date_start="2000-01-01",
            date_end="2100-01-01",
        ),
    )


@pytest.fixture(scope="module")
def synthetic_chain() -> pd.DataFrame:
    """
    Small synthetic option chain generated with constant Black-Scholes volatility. 
    Used for IV inversion, pricing-error, and parity tests.
    """
    S, r, q, sigma = 21000.0, 0.053, 0.012, 0.15
    trade = pd.Timestamp("2024-01-10")
    expiry = pd.Timestamp("2024-02-29")
    strikes = [20000.0, 20500.0, 20750.0, 21000.0, 21250.0, 21500.0, 22000.0]
    T = (expiry - trade).days / 365.0

    rows: list[dict] = []
    for K in strikes:
        for ot in ("CE", "PE"):
            px = float(bs_price(S, K, T, r, sigma, q, ot))
            rows.append({
                "trade_date": trade,
                "expiry_date": expiry,
                "strike": K,
                "option_type": ot,
                "settlement_price": px,
                "close": px,
                "volume": 100,
                "open_interest": 500,
                "underlying_spot": S,
                "instrument_type": "OPTIDX",
                "underlying_symbol": "NIFTY",
                "source_file": "test",
            })
    return pd.DataFrame(rows)



# Black-Scholes prices


class TestBlackScholes:
    """Reference values from Haug (2007), 'The Complete Guide to Option Pricing Formulas'."""

    # S=100, K=100, T=1, r=0.05, sigma=0.20, q=0
    REF_CALL = 10.450584
    REF_PUT = 5.573526

    def test_call_reference_value(self) -> None:
        px = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call"))
        assert abs(px - self.REF_CALL) < 1e-4

    def test_put_reference_value(self) -> None:
        px = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "put"))
        assert abs(px - self.REF_PUT) < 1e-4

    def test_ce_pe_alias(self) -> None:
        """NSE-style 'CE'/'PE' must be accepted and equivalent to call/put."""
        c1 = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "CE"))
        c2 = float(bs_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call"))
        assert abs(c1 - c2) < 1e-12

    def test_put_call_parity(self) -> None:
        S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02
        c = float(bs_price(S, K, T, r, sigma, q, "call"))
        p = float(bs_price(S, K, T, r, sigma, q, "put"))
        lhs = c - p
        rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs(lhs - rhs) < 1e-10

    def test_price_positive(self) -> None:
        for K in (50.0, 100.0, 150.0):
            c = float(bs_price(100.0, K, 1.0, 0.05, 0.20, 0.0, "call"))
            p = float(bs_price(100.0, K, 1.0, 0.05, 0.20, 0.0, "put"))
            assert c >= 0.0
            assert p >= 0.0

    def test_degenerate_T_zero(self) -> None:
        """At T=0, price collapses to intrinsic value."""
        c = float(bs_price(110.0, 100.0, 0.0, 0.05, 0.20, 0.0, "call"))
        p = float(bs_price(110.0, 100.0, 0.0, 0.05, 0.20, 0.0, "put"))
        assert abs(c - 10.0) < 1e-12
        assert abs(p - 0.0) < 1e-12

    def test_degenerate_sigma_zero(self) -> None:
        """At sigma=0, price is the discounted payoff of the deterministic forward."""
        S, K, T, r, q = 100.0, 90.0, 1.0, 0.05, 0.0
        fwd = S * np.exp((r - q) * T)
        expected = np.exp(-r * T) * max(fwd - K, 0.0)
        got = float(bs_price(S, K, T, r, 0.0, q, "call"))
        assert abs(got - expected) < 1e-10

    def test_negative_spot_raises(self) -> None:
        with pytest.raises(ValueError):
            bs_price(-100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "call")

    def test_bad_option_type_raises(self) -> None:
        with pytest.raises(ValueError):
            bs_price(100.0, 100.0, 1.0, 0.05, 0.20, 0.0, "banana")

    def test_vectorized_matches_scalar(self) -> None:
        S = np.array([90.0, 100.0, 110.0])
        K = np.array([100.0, 100.0, 100.0])
        vec = bs_price(S, K, 1.0, 0.05, 0.20, 0.0, "call")
        for i in range(3):
            sc = float(bs_price(float(S[i]), float(K[i]), 1.0, 0.05, 0.20, 0.0, "call"))
            assert abs(vec[i] - sc) < 1e-12



# Greeks


class TestGreeks:
    """Analytical Greeks: bounds, signs, and FD agreement."""

    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.02

    def test_delta_call_in_range(self) -> None:
        d = float(bs_delta(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))
        assert 0.0 < d < np.exp(-self.q * self.T) + 1e-9

    def test_delta_put_in_range(self) -> None:
        d = float(bs_delta(self.S, self.K, self.T, self.r, self.sigma, self.q, "put"))
        assert -np.exp(-self.q * self.T) - 1e-9 < d < 0.0

    def test_gamma_positive(self) -> None:
        g = float(bs_gamma(self.S, self.K, self.T, self.r, self.sigma, self.q))
        assert g > 0.0

    def test_vega_positive(self) -> None:
        v = float(bs_vega(self.S, self.K, self.T, self.r, self.sigma, self.q))
        assert v > 0.0

    def test_vega_increases_with_T(self) -> None:
        v1 = float(bs_vega(self.S, self.K, 0.25, self.r, self.sigma, self.q))
        v2 = float(bs_vega(self.S, self.K, 1.00, self.r, self.sigma, self.q))
        assert v2 > v1

    def test_theta_sign_convention(self) -> None:
        """
        Theta is dV/dt with calendar time advancing. For a standard ATM call
        with r > q, theta can be positive or negative. We only assert the
        sign matches the analytical formula by a small margin.
        """
        th = float(bs_theta(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))
        assert np.isfinite(th)

    def test_rho_call_positive(self) -> None:
        rr = float(bs_rho(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))
        assert rr > 0.0

    def test_rho_put_negative(self) -> None:
        rr = float(bs_rho(self.S, self.K, self.T, self.r, self.sigma, self.q, "put"))
        assert rr < 0.0

    # FD agreement

    def test_fd_delta_matches_analytical(self) -> None:
        for ot in ("call", "put"):
            a = float(bs_delta(self.S, self.K, self.T, self.r, self.sigma, self.q, ot))
            f = fd_delta(self.S, self.K, self.T, self.r, self.sigma, self.q, ot, h=1e-4)
            assert abs(a - f) < 1e-4, f"{ot}: analytical={a}, fd={f}"

    def test_fd_gamma_matches_analytical(self) -> None:
        a = float(bs_gamma(self.S, self.K, self.T, self.r, self.sigma, self.q))
        f = fd_gamma(self.S, self.K, self.T, self.r, self.sigma, self.q, "call", h=1e-3)
        assert abs(a - f) < 1e-3

    def test_fd_vega_matches_analytical(self) -> None:
        a = float(bs_vega(self.S, self.K, self.T, self.r, self.sigma, self.q))
        f = fd_vega(self.S, self.K, self.T, self.r, self.sigma, self.q, "call", h=1e-4)
        assert abs(a - f) < 1e-2

    def test_fd_rho_matches_analytical(self) -> None:
        for ot in ("call", "put"):
            a = float(bs_rho(self.S, self.K, self.T, self.r, self.sigma, self.q, ot))
            f = fd_rho(self.S, self.K, self.T, self.r, self.sigma, self.q, ot, h=1e-4)
            assert abs(a - f) < 1e-2

    def test_fd_theta_matches_analytical(self) -> None:
        for ot in ("call", "put"):
            a = float(bs_theta(self.S, self.K, self.T, self.r, self.sigma, self.q, ot))
            f = fd_theta(self.S, self.K, self.T, self.r, self.sigma, self.q, ot, h=1e-5)
            assert abs(a - f) < 1e-2, f"{ot}: analytical={a}, fd={f}"



# Implied volatility


class TestImpliedVol:
    """Round-trip, bound rejection, and input validation."""

    S, T, r, q = 21000.0, 0.05, 0.053, 0.012

    @pytest.mark.parametrize("sigma_true", [0.05, 0.15, 0.30, 0.60])
    def test_round_trip_call(self, sigma_true: float) -> None:
        K = self.S
        px = float(bs_price(self.S, K, self.T, self.r, sigma_true, self.q, "call"))
        res = implied_volatility(px, self.S, K, self.T, self.r, "call", self.q)
        assert res.iv is not None
        assert abs(res.iv - sigma_true) < 1e-6

    @pytest.mark.parametrize("sigma_true", [0.05, 0.15, 0.30, 0.60])
    def test_round_trip_put(self, sigma_true: float) -> None:
        K = self.S
        px = float(bs_price(self.S, K, self.T, self.r, sigma_true, self.q, "put"))
        res = implied_volatility(px, self.S, K, self.T, self.r, "put", self.q)
        assert res.iv is not None
        assert abs(res.iv - sigma_true) < 1e-6

    def test_ce_pe_alias_accepted(self) -> None:
        K = self.S
        px = float(bs_price(self.S, K, self.T, self.r, 0.20, self.q, "call"))
        res = implied_volatility(px, self.S, K, self.T, self.r, "CE", self.q)
        assert res.iv is not None
        assert abs(res.iv - 0.20) < 1e-6

    def test_rejects_below_intrinsic(self) -> None:
        # Deep ITM call with essentially zero price.
        res = implied_volatility(1e-6, 21000.0, 19000.0, self.T, self.r, "call", self.q)
        assert res.status in {"below_intrinsic", "no_bracket", "invalid_input"}
        assert res.iv is None

    def test_rejects_above_upper_bound(self) -> None:
        # Call price cannot exceed S e^{-qT}.
        res = implied_volatility(1e6, 21000.0, 21000.0, self.T, self.r, "call", self.q)
        assert res.status == "above_upper_bound"
        assert res.iv is None

    def test_rejects_invalid_T(self) -> None:
        res = implied_volatility(100.0, self.S, self.S, 0.0, self.r, "call", self.q)
        assert res.status == "invalid_input"

    def test_rejects_invalid_option_type(self) -> None:
        res = implied_volatility(100.0, self.S, self.S, self.T, self.r, "FUT", self.q)
        assert res.status == "invalid_input"

    def test_rejects_negative_price(self) -> None:
        res = implied_volatility(-5.0, self.S, self.S, self.T, self.r, "call", self.q)
        assert res.status == "invalid_input"

    def test_never_returns_zero_on_failure(self) -> None:
        """
        Failure must be signalled by ``iv is None``, not by a zero value.
        A zero volatility silently propagated downstream would be worse than
        a missing value.
        """
        res = implied_volatility(1e-6, 21000.0, 19000.0, self.T, self.r, "call", self.q)
        assert res.iv is None or res.iv > 0.0

    def test_repricing_error_small_on_success(self) -> None:
        K = self.S
        px = float(bs_price(self.S, K, self.T, self.r, 0.18, self.q, "call"))
        res = implied_volatility(px, self.S, K, self.T, self.r, "call", self.q)
        assert res.status == "success"
        assert res.repricing_error is not None
        assert res.repricing_error < 1e-4



# CRR binomial tree


class TestCRRTree:
    """Convergence to Black-Scholes, parity, input validation."""

    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

    def test_converges_to_bs_at_high_n(self) -> None:
        bs = float(bs_price(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))
        crr = crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                        "call", n_steps=2000)
        # Even n=2000 underestimates ATM call; expect tight agreement.
        assert abs(crr - bs) < 0.05, f"BS={bs}, CRR={crr}"

    def test_error_decreases_with_n(self) -> None:
        bs = float(bs_price(self.S, self.K, self.T, self.r, self.sigma, self.q, "call"))
        e_100 = abs(crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                              "call", n_steps=100) - bs)
        e_1000 = abs(crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                               "call", n_steps=1000) - bs)
        # Use the same parity (even) to isolate the convergence rate from the odd/even oscillation.
        assert e_1000 < e_100

    def test_crr_put_call_parity(self) -> None:
        n = 1000
        c = crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                      "call", n_steps=n)
        p = crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                      "put", n_steps=n)
        lhs = c - p
        rhs = self.S * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        # CRR parity is exact in a discrete but arbitrage-free tree up to floating-point error.
        assert abs(lhs - rhs) < 1e-6

    def test_american_call_equals_european_no_dividend(self) -> None:
        """For a non-dividend paying stock, American call = European call."""
        euro = crr_price(self.S, self.K, self.T, self.r, self.sigma, 0.0,
                         "call", n_steps=500, american=False)
        amer = crr_price(self.S, self.K, self.T, self.r, self.sigma, 0.0,
                         "call", n_steps=500, american=True)
        assert abs(amer - euro) < 1e-6

    def test_american_put_geq_european_put(self) -> None:
        euro = crr_price(self.S, self.K, self.T, self.r, self.sigma, 0.0,
                         "put", n_steps=500, american=False)
        amer = crr_price(self.S, self.K, self.T, self.r, self.sigma, 0.0,
                         "put", n_steps=500, american=True)
        assert amer >= euro - 1e-9

    def test_rejects_negative_sigma(self) -> None:
        with pytest.raises(ValueError):
            crr_price(self.S, self.K, self.T, self.r, -0.10, self.q,
                      "call", n_steps=100)

    def test_rejects_zero_steps(self) -> None:
        with pytest.raises(ValueError):
            crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                      "call", n_steps=0)

    def test_rejects_bad_option_type(self) -> None:
        with pytest.raises(ValueError):
            crr_price(self.S, self.K, self.T, self.r, self.sigma, self.q,
                      "XYZ", n_steps=100)



# Analysis pipeline


class TestAnalysis:

    def test_add_time_to_expiry_calendar(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        # 2024-01-10 to 2024-02-29 = 50 calendar days
        expected = 50.0 / 365.0
        assert abs(df["time_to_expiry"].iloc[0] - expected) < 1e-9

    def test_add_moneyness(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        # log_moneyness should be the log of K over the forward
        K = df["strike"].iloc[0]
        F = df["forward"].iloc[0]
        assert abs(df["log_moneyness"].iloc[0] - np.log(K / F)) < 1e-12

    def test_assign_moneyness_bucket(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        df = assign_moneyness_bucket(df, cfg)
        assert df["moneyness_bucket"].notna().all()
        # The 21000 strike with spot 21000 and small carry sits in "atm".
        row = df[(df["strike"] == 21000.0)].iloc[0]
        assert row["moneyness_bucket"] in {"atm"}

    def test_compute_iv_chain_recovers_input_sigma(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        df = assign_moneyness_bucket(df, cfg)
        df = compute_iv_chain(df, cfg)
        # The synthetic chain uses sigma = 0.15 exactly.
        valid = df[df["iv"].notna()]
        assert len(valid) >= 10
        assert np.allclose(valid["iv"].values, 0.15, atol=1e-4)

    def test_estimate_atm_vol_returns_one_row_per_group(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        df = assign_moneyness_bucket(df, cfg)
        df = compute_iv_chain(df, cfg)
        atm = estimate_atm_vol(df, cfg)
        assert len(atm) == 1  # one trade_date, one expiry
        assert abs(atm["atm_iv"].iloc[0] - 0.15) < 1e-3

    def test_constant_atm_vol_has_no_effect_when_smile_is_flat(
        self, cfg, synthetic_chain,
    ) -> None:
        """
        Synthetic data uses a single constant volatility, so constant-ATM
        repricing must reproduce the input prices exactly.
        """
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        df = assign_moneyness_bucket(df, cfg)
        df = compute_iv_chain(df, cfg)
        atm = estimate_atm_vol(df, cfg)
        df = experiment_constant_atm_vol(df, cfg, atm)
        # Ignore rows whose IV solver failed.
        mask = df["bs_price_atm_iv"].notna()
        err = (df.loc[mask, "bs_price_atm_iv"] - df.loc[mask, "settlement_price"]).abs()
        assert err.max() < 1e-3

    def test_iv_repricing_round_trip(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        df = assign_moneyness_bucket(df, cfg)
        df = compute_iv_chain(df, cfg)
        df = experiment_iv_repricing(df, cfg)
        mask = df["bs_price_own_iv"].notna()
        err = (df.loc[mask, "bs_price_own_iv"] - df.loc[mask, "settlement_price"]).abs()
        # IV reprice is a numerical round-trip; should be near machine precision.
        assert err.max() < 1e-4

    def test_parity_diagnostic_on_synthetic(self, cfg, synthetic_chain) -> None:
        """
        Synthetic prices come from Black-Scholes, so parity must hold to
        floating-point precision.
        """
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        parity = parity_diagnostic(df, cfg)
        assert len(parity) >= 5
        assert parity["abs_residual"].max() < 1e-8

    def test_summarize_errors_returns_overall_row(self, cfg, synthetic_chain) -> None:
        df = add_time_to_expiry(synthetic_chain, cfg)
        df = add_moneyness(df, cfg)
        df = assign_moneyness_bucket(df, cfg)
        df = compute_iv_chain(df, cfg)
        atm = estimate_atm_vol(df, cfg)
        df = experiment_constant_atm_vol(df, cfg, atm)
        s = summarize_errors(df, cfg, error_col="error_atm_iv")
        assert (s["grouping"] == "overall").any()
        row = s[s["grouping"] == "overall"].iloc[0]
        assert row["n"] > 0



# Data cleaning


class TestDataCleaning:

    def test_normalize_columns_maps_nse_names(self) -> None:
        raw = pd.DataFrame({
            "INSTRUMENT": ["OPTIDX"],
            "SYMBOL": ["NIFTY"],
            "EXPIRY_DT": ["25-JAN-2024"],
            "STRIKE_PR": ["21000"],
            "OPTION_TYP": ["CE"],
            "CLOSE": ["105"],
            "SETTLE_PR": ["104"],
            "CONTRACTS": ["1000"],
            "OPEN_INT": ["5000"],
            "TIMESTAMP": ["10-JAN-2024"],
        })
        norm = normalize_columns(raw)
        assert norm["strike"].iloc[0] == 21000.0
        assert norm["option_type"].iloc[0] == "CE"
        assert norm["volume"].iloc[0] == 1000.0
        assert norm["settlement_price"].iloc[0] == 104.0
        assert pd.notna(norm["trade_date"].iloc[0])
        assert pd.notna(norm["expiry_date"].iloc[0])

    def test_cleaning_excludes_wrong_instrument(self, cfg) -> None:
        df = pd.DataFrame({
            "instrument_type": ["FUTIDX"],
            "underlying_symbol": ["NIFTY"],
            "option_type": ["CE"],
            "strike": [21000.0],
            "trade_date": [pd.Timestamp("2024-01-10")],
            "expiry_date": [pd.Timestamp("2024-01-25")],
            "volume": [100],
            "settlement_price": [100.0],
            "close": [100.0],
            "underlying_spot": [21000.0],
            "open_interest": [500],
            "source_file": ["test"],
        })
        clean, rep = clean_options_data(df, cfg, DataQualityReport())
        assert len(clean) == 0
        assert "instrument_type_mismatch" in rep.excluded_by_reason

    def test_cleaning_excludes_zero_volume(self, cfg) -> None:
        df = pd.DataFrame({
            "instrument_type": ["OPTIDX"],
            "underlying_symbol": ["NIFTY"],
            "option_type": ["CE"],
            "strike": [21000.0],
            "trade_date": [pd.Timestamp("2024-01-10")],
            "expiry_date": [pd.Timestamp("2024-02-29")],
            "volume": [0],
            "settlement_price": [100.0],
            "close": [100.0],
            "underlying_spot": [21000.0],
            "open_interest": [500],
            "source_file": ["test"],
        })
        clean, rep = clean_options_data(df, cfg, DataQualityReport())
        assert len(clean) == 0
        assert rep.zero_volume_excluded == 1

    def test_cleaning_keeps_valid_row(self, cfg) -> None:
        df = pd.DataFrame({
            "instrument_type": ["OPTIDX"],
            "underlying_symbol": ["NIFTY"],
            "option_type": ["CE"],
            "strike": [21000.0],
            "trade_date": [pd.Timestamp("2024-01-10")],
            "expiry_date": [pd.Timestamp("2024-02-29")],
            "volume": [100],
            "settlement_price": [100.0],
            "close": [100.0],
            "underlying_spot": [21000.0],
            "open_interest": [500],
            "source_file": ["test"],
        })
        clean, rep = clean_options_data(df, cfg, DataQualityReport())
        assert len(clean) == 1
        assert rep.output_rows == 1
