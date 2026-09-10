import sys
import os
import pytest
import numpy as np
import pandas as pd

# Add src/ directory to the Python path for deterministic test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from markowitz import (
    min_variance_weights,
    efficient_frontier_closed_form,
    solve_constrained_qp,
)
from metrics import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    portfolio_cumulative_returns,
    max_drawdown,
    summary_stats,
)


@pytest.fixture
def portfolio_data():

    mu = np.array([0.05, 0.10, 0.15, 0.20])
    
    var_diag = 0.04
    cov_offdiag = 0.01
    sigma = np.full((4, 4), cov_offdiag)
    np.fill_diagonal(sigma, var_diag)
    
    return mu, sigma


# Real-world check: Fully-invested portfolios must allocate 100% of available capital without leverage or cash drag.
def test_min_variance_weights_sum_to_one(portfolio_data):
    _, sigma = portfolio_data
    weights = min_variance_weights(sigma)
    assert np.isclose(np.sum(weights), 1.0, atol=1e-6), "Min variance weights must sum to 1.0"


# Real-world check: On the efficient frontier, higher expected return requires taking strictly non-decreasing volatility.
def test_closed_form_frontier_variance_increases(portfolio_data):
    mu, sigma = portfolio_data
    target_returns, frontier_vols, _ = efficient_frontier_closed_form(mu, sigma, num_points=30)
    
    # Locate global minimum variance portfolio index on sampled curve
    min_vol_idx = np.argmin(frontier_vols)
    
    # Extract upper efficient branch (returns above GMV return)
    efficient_vols = frontier_vols[min_vol_idx:]
    vol_diffs = np.diff(efficient_vols)
    
    # Volatility along upper branch must be non-decreasing (allowing minor floating-point tolerances)
    assert np.all(vol_diffs >= -1e-6), "Volatility must increase as target return increases along upper frontier"


# Real-world check: Portfolio risk controls must strictly prevent short-selling and over-concentration in single stocks.
def test_constrained_qp_respects_bounds(portfolio_data):
    mu, sigma = portfolio_data
    w_max = 0.3
    target_return = 0.125
    
    weights = solve_constrained_qp(mu, sigma, target_return, w_max=w_max)
    
    assert weights is not None, "QP solver failed to return valid weights"
    assert np.all(weights >= -1e-6), f"Weights violate long-only constraint: {weights}"
    assert np.all(weights <= w_max + 1e-6), f"Weights exceed single asset cap of {w_max}: {weights}"
    assert np.isclose(np.sum(weights), 1.0, atol=1e-5), f"Weights sum to {np.sum(weights)}, expected 1.0"


# Real-world check: Risk managers enforce sector limits to prevent hidden structural correlation exposure across industries.
def test_constrained_qp_respects_sector_cap(portfolio_data):
    mu, sigma = portfolio_data
    sector_mapper = {"Tech": [0, 1]}
    sector_cap = 0.40
    target_return = 0.125
    
    weights = solve_constrained_qp(
        mu, sigma, target_return, w_max=0.3, sector_mapper=sector_mapper, sector_cap=sector_cap
    )
    
    assert weights is not None, "QP solver failed under sector constraints"
    tech_exposure = np.sum(weights[[0, 1]])
    assert tech_exposure <= sector_cap + 1e-5, f"Tech sector allocation {tech_exposure} exceeds cap {sector_cap}"


# Real-world check: When business constraints are non-binding, numerical QP optimizers must match exact closed-form math.
def test_closed_form_and_constrained_agree_at_min_variance(portfolio_data):
    mu, sigma = portfolio_data
    
    # Unconstrained global min-variance weights
    cf_weights = min_variance_weights(sigma)
    cf_vol = np.sqrt(cf_weights @ sigma @ cf_weights)
    gmv_return = float(cf_weights @ mu)
    
    # Constrained QP solved at identical target return
    qp_weights = solve_constrained_qp(mu, sigma, target_return=gmv_return, w_max=0.3)
    qp_vol = np.sqrt(qp_weights @ sigma @ qp_weights)
    
    assert np.isclose(qp_vol, cf_vol, atol=1e-4), f"QP vol ({qp_vol}) mismatched closed-form vol ({cf_vol})"


# Real-world check: Analytics functions must compute non-degenerate risk, valid wealth compounding, and non-positive drawdowns.
def test_metrics_functions(portfolio_data):
    mu, sigma = portfolio_data
    np.random.seed(42)
    
    # Generate 50 days of synthetic log return data
    fake_log_returns = np.random.normal(0.0005, 0.01, size=(50, 4))
    dates = pd.date_range(start="2026-01-01", periods=50, freq="B")
    returns_df = pd.DataFrame(fake_log_returns, index=dates, columns=["A1", "A2", "A3", "A4"])
    
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    ret = annualized_return(weights, mu)
    vol = annualized_volatility(weights, sigma)
    s_ratio = sharpe_ratio(weights, mu, sigma, risk_free_rate=0.01)
    cum_returns = portfolio_cumulative_returns(weights, returns_df)
    mdd = max_drawdown(cum_returns)
    stats = summary_stats(weights, mu, sigma, returns_df, risk_free_rate=0.01)
    
    assert vol > 0, "Annualized volatility must be strictly positive"
    assert np.isclose(ret, 0.125), f"Expected annualized return should be 0.125, got {ret}"
    assert s_ratio > 0, "Sharpe ratio should be positive when return > risk-free rate"
    assert np.isclose(cum_returns.iloc[0], 1.0 + np.expm1(returns_df.iloc[0].values) @ weights), "Day 1 compounding error"
    assert mdd <= 0.0, f"Max drawdown must be <= 0.0, got {mdd}"
    assert stats["max_drawdown"] == mdd, "Summary stats dictionary contains inconsistent drawdown metric"