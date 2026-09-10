# Portfolio Optimization & Sensitivity Analysis Pipeline

A Mean-Variance Optimization (MVO) pipeline implementing Markowitz portfolio theory, convex quadratic programming with institutional-style constraints, and Monte Carlo estimation error sensitivity diagnostics — validated with a unit test suite.

---

## Overview

This project bridges theoretical quantitative finance and practical portfolio implementation. While modern portfolio theory provides closed-form mathematical solutions for unconstrained asset allocation, real-world execution requires handling strict business rules, position caps, and sector limits.

Furthermore, classical MVO is known as an **"error maximizer"**—where small noise in expected asset returns ($\mu$) causes massive, volatile swings in portfolio weights due to matrix inversion ($\Sigma^{-1}$). This pipeline quantifies that sensitivity using Monte Carlo simulations.

---

## Features & Architecture

* **Data Ingestion & Caching:** Automated downloading and local disk caching of 5-year daily adjusted close prices via `yfinance`.
* **Parameter Estimation:** Computation of annualized expected return vectors ($\mu$) and sample covariance matrices ($\Sigma$).
* **Closed-Form Frontier (Benchmark):** Matrix algebra solver using Lagrange multipliers for unconstrained efficiency.
* **Constrained QP Solver:** Convex optimization powered by `cvxpy` enforcing:
  * Budget constraint: $\sum w_i = 1.0$ (fully invested)
  * Long-only constraint: $w_i \ge 0$ (no short selling)
  * Asset caps: $w_i \le 0.30$ (maximum 30% per individual asset)
  * Sector diversification caps: $\sum w_{\text{sector}} \le 0.40$ (maximum 40% in any sector)
* **Sensitivity Analysis Engine:** 100-run Monte Carlo perturbation test adding $\pm 1\%$ Gaussian noise to $\mu$ to measure weight variance.
* **Performance Metrics:** Annualized return, volatility, Sharpe ratio, cumulative returns, and max drawdown (`src/metrics.py`).
* **Test Suite:** `pytest` coverage for the closed-form solution, QP bound/sector-cap constraints, closed-form/QP agreement at the min-variance point, and the metrics functions (6/6 passing).

---

## Roadmap

* **CVaR-based tail-risk optimization (in progress):** extending the constrained QP to a linear program minimizing Conditional Value-at-Risk (Rockafellar–Uryasev formulation) as an alternative to variance-based risk.

---

## Project Structure

```text
linear_programming_project/
├── data/
│   └── price_data.csv          # Cached historical asset prices
├── plots/
│   ├── efficient_frontier.png  # Closed-form vs Constrained frontier
│   └── sensitivity_analysis.png# Weight instability box plots
├── src/
│   ├── data_loader.py          # Price download, caching, log returns
│   ├── markowitz.py            # Closed-form + constrained QP, sensitivity analysis
│   └── metrics.py              # Return, volatility, Sharpe, drawdown
├── tests/
│   └── test_markowitz.py       # Unit tests for solvers and metrics
├── .gitignore
├── README.md
└── requirements.txt