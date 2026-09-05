# Portfolio Optimizer Project — Full Roadmap

**Goal:** Build a portfolio optimization engine implementing Markowitz QP (baseline),
CVaR-based LP (headline technique), and Hierarchical Risk Parity via Kruskal's MST
(graph algorithm), with backtested comparison — for a Bloomberg/Google internship CV.

**Your starting point:** Strong math (2nd year, but fast learner), solid C++, basic
algorithms knowledge, ~zero Python, zero optimization theory. This roadmap assumes that
and builds up in order — nothing is used before it's explained.

---

## Phase 0 — Setup and mental model (Day 0–1)

### 0.1 Python, but fast
You don't need a Python course. You need the ~20% of Python that maps to what you
already know in C++:
- Dynamic typing, no compilation step, indentation = blocks
- Lists (`[]`) ≈ `std::vector`, dicts (`{}`) ≈ `std::unordered_map`
- NumPy arrays are what you'll actually compute with — think of them as `std::vector`
  with vectorized math built in (no manual loops for elementwise ops)
- List comprehensions `[x**2 for x in range(10)]` ≈ a one-line transform loop

**Action:** Spend 2–3 hours on a "Python for people who already code" tutorial (not a
full course). Then learn NumPy basics (arrays, slicing, matrix multiply `@`) — maybe
2 more hours. That's genuinely enough to start; you'll pick up the rest by reading code
and asking me when something's unclear.

### 0.2 Tools
- Python 3.11+, `venv` for environment isolation
- Libraries: `numpy`, `pandas`, `scipy`, `cvxpy`, `matplotlib`, `yfinance` (free market
  data), `networkx` (for MST/graph part — or you implement Kruskal's yourself, better
  for the CV)
- Jupyter notebooks for exploration, then convert final code to proper `.py` modules
  with tests — CV projects should look like software, not just notebooks

### 0.3 Repo structure (set this up now, fill in as you go)
```
portfolio-optimizer/
├── data/               # cached price data
├── src/
│   ├── data_loader.py
│   ├── markowitz.py    # QP
│   ├── cvar_lp.py       # LP
│   ├── hrp.py           # Kruskal's MST + hierarchical allocation
│   ├── backtest.py
│   └── metrics.py
├── notebooks/           # exploration only, not the deliverable
├── tests/
├── README.md
└── requirements.txt
```

---

## Phase 1 — Math foundations (Day 2–5)

This is the part your strong math background makes fast. Do this **before** writing
any optimization code — you should be able to derive what the code does on paper first.

### 1.1 What "optimization" means formally
- Decision variables, objective function, constraints
- Feasible region, optimal solution
- Convexity: why convex problems are "easy" (any local min is global) — this is the
  single most important concept for everything that follows

### 1.2 Linear Programming (LP)
- Standard form: minimize `c^T x` subject to `Ax ≤ b`, `x ≥ 0`
- Geometric intuition: feasible region is a polytope, optimum is at a vertex
- **Duality** (since your math is strong, don't skip this): every LP has a dual problem;
  strong duality, complementary slackness. You won't need this to *code* the project,
  but understanding it will make you dangerous in an interview when they ask "why LP
  and not QP" — you'll be able to talk about shadow prices / sensitivity.
- Simplex method — understand the idea (pivot along vertices), don't implement it
  (you'll use `scipy`/`cvxpy` solvers — implementing your own Simplex is a fun but
  separate side project, not necessary here)

**Resource:** Boyd & Vandenberghe, *Convex Optimization*, Chapter 1–2 (free PDF online).
Given your math level, you can read the relevant sections directly rather than needing
a hand-holding course.

### 1.3 Quadratic Programming (QP)
- Standard form: minimize `(1/2) x^T Q x + c^T x` subject to linear constraints
- Why it's harder than LP: optimum can be interior to the feasible region, not just at
  a vertex (no simplex-style combinatorial structure)
- KKT conditions (generalize Lagrange multipliers to inequality constraints) — this
  is the theoretical bridge between "here's a QP" and "here's why cvxpy's answer is
  actually optimal." Worth understanding since your math background makes this
  tractable, and it's a great interview topic.

### 1.4 Finance-specific math
- Expected return, variance, covariance of a portfolio: if `w` = weight vector,
  `μ` = expected returns vector, `Σ` = covariance matrix:
  - Portfolio return = `w^T μ`
  - Portfolio variance = `w^T Σ w`  ← this quadratic form is *why* Markowitz is QP
- Efficient frontier: the set of portfolios that minimize variance for each level of
  target return
- Sharpe ratio: `(return - risk_free_rate) / volatility` — the standard way to compare
  portfolios
- **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)**:
  - VaR(α) = the loss threshold not exceeded with probability α (e.g. "95% VaR" = the
    loss you won't exceed 95% of the time)
  - CVaR(α) = the *expected* loss *given* that you're in the worst (1−α) tail —
    this is what makes CVaR a better tail-risk measure than VaR (VaR ignores how bad
    the tail actually is; CVaR doesn't)
  - The Rockafellar-Uryasev (2000) trick: CVaR minimization can be written as an LP by
    introducing an auxiliary variable for the VaR threshold and penalizing shortfalls
    linearly. **This is the core LP formulation you'll implement** — I'll walk through
    the derivation with you step by step when we get to Phase 3, on paper first.

**Milestone check:** before moving to Phase 2, you should be able to explain to me,
without notes: why portfolio variance minimization is a QP, and roughly why CVaR
minimization can be turned into an LP. If either is fuzzy, that's fine — flag it and
we'll work through the derivation together before touching code.

---

## Phase 2 — Markowitz QP implementation (Day 6–9)

### 2.1 Data pipeline (`data_loader.py`)
- Pull historical daily prices for ~15–30 stocks (a mix of sectors) via `yfinance`
- Compute daily/monthly log returns
- Compute the sample mean vector `μ` and covariance matrix `Σ`
- **Learning point:** discuss estimation error in `Σ` — with N assets you need roughly
  N² independent data points to estimate covariance reliably; this is exactly why HRP
  (Phase 4) is a real improvement, not just an algorithmic flex. Plant this seed now,
  pays off later.

### 2.2 Solve unconstrained Markowitz by hand first (paper + NumPy, no solver)
For the special case of only an equality constraint (`w^T 1 = 1`, no inequality
constraints like "no short selling"), the solution has a **closed form** via Lagrange
multipliers — you can derive and code this directly with `numpy.linalg.solve`, no
solver library needed. Do this first:
- Confirms you understand the math (you're solving your own derivation, not calling
  a black box)
- Gives you a sanity check for step 2.3

### 2.3 General constrained QP via `cvxpy`
Real portfolios need constraints a closed form can't handle: no short-selling
(`w ≥ 0`), max position size, sector caps. Implement:
```python
import cvxpy as cp
w = cp.Variable(n)
risk = cp.quad_form(w, Sigma)
constraints = [cp.sum(w) == 1, w >= 0]
prob = cp.Problem(cp.Minimize(risk), constraints)
prob.solve()
```
- Sweep target returns to trace the full **efficient frontier**, plot it
- Mark the max-Sharpe portfolio on the plot

### 2.4 Deliverable for this phase
A script that takes tickers + date range → outputs efficient frontier plot +
max-Sharpe portfolio weights. This is your baseline for everything after.

---

## Phase 3 — CVaR LP implementation (Day 10–14)

### 3.1 Derive the LP formulation on paper (with me) before coding
We'll work through Rockafellar-Uryasev together: introducing auxiliary variables
`z_i` for shortfall in each historical scenario, and the threshold variable for VaR,
turning a non-smooth tail-risk objective into a clean LP. I want you to be able to
write this formulation from memory afterward, not just paste code.

### 3.2 Scenario generation
CVaR-LP is *scenario-based*, not covariance-based — a fundamentally different
paradigm from Markowitz (worth stating explicitly on your CV/interview):
- Use historical daily returns directly as scenarios (historical simulation), or
- Bootstrap/simulate scenarios (more advanced, optional stretch goal)

### 3.3 Implement in `cvxpy` or `scipy.optimize.linprog`
Doing it in both is a good exercise: `cvxpy` is higher-level (matches the math
directly), `linprog` forces you to hand-build the `A`, `b`, `c` matrices — closer to
what you'd do in C++ and better for really nailing the LP standard form.

### 3.4 Deliverable
A function: tickers + date range + confidence level α → optimal CVaR-minimizing
portfolio weights, plus a plot of the loss distribution with the CVaR threshold
marked.

---

## Phase 4 — Kruskal's MST → Hierarchical Risk Parity (Day 15–19)

This is your "real algorithm" section — implement Kruskal's yourself (don't use
`networkx`), since that's the part that demonstrates algorithms skill, not just
library calls.

### 4.1 Build the correlation-distance graph
- Distance metric: `d(i,j) = sqrt(0.5 * (1 - correlation(i,j)))` — turns correlation
  into a proper distance (satisfies triangle inequality)
- Complete graph on N assets, N(N-1)/2 edges

### 4.2 Implement Kruskal's MST
- Sort edges by weight, Union-Find (disjoint set) with path compression + union by
  rank — this is a genuinely good data structure to have coded from scratch on a CV,
  and it's C++-flavored territory you'll be comfortable in
- Build the MST

### 4.3 Hierarchical clustering + recursive bisection (López de Prado's HRP)
- Convert MST/linkage into a dendrogram (tree)
- Quasi-diagonalize the covariance matrix by the tree ordering
- Recursively split the tree, allocating weight inversely proportional to cluster
  variance at each split — no matrix inversion anywhere (this is the numerical
  stability advantage over Markowitz worth calling out)

### 4.4 Deliverable
HRP weights for the same asset universe, comparable to Phase 2 and 3 outputs.

---

## Phase 5 — Backtesting and comparison (Day 20–24)

This phase is what turns three separate implementations into one project with a
conclusion — the most CV-valuable part.

### 5.1 Backtest engine (`backtest.py`)
- Rolling window: estimate weights on data up to time t, hold for a rebalance
  period, roll forward
- Apply to all three strategies (Markowitz max-Sharpe, CVaR-LP, HRP) + a naive
  equal-weight benchmark

### 5.2 Metrics (`metrics.py`)
- Annualized return, volatility, Sharpe ratio, max drawdown, CVaR realized
- **Stress test on known crisis periods**: 2008 financial crisis, March 2020 COVID
  crash — this is where CVaR-LP and HRP should visibly outperform naive Markowitz on
  drawdown, and that comparison is your headline result

### 5.3 Deliverable
A comparison table/plot: cumulative returns of all four strategies over the full
backtest period + a specific drawdown comparison during the crisis windows.

---

## Phase 6 — Polish for CV (Day 25–28)

### 6.1 README
- Clear problem statement, math summary (your derivations, briefly), architecture
  diagram, results with numbers (e.g. "CVaR-LP portfolio had 23% smaller max drawdown
  than mean-variance during the 2020 crash, at a Sharpe ratio cost of only 0.05")
- Reproducibility: `requirements.txt`, clear run instructions

### 6.2 Tests
- Unit tests for Kruskal's/Union-Find correctness, and sanity checks on solver
  outputs (weights sum to 1, no negative weights when constrained, etc.) — shows
  software engineering maturity, not just math

### 6.3 CV bullet, drafted now so you're writing toward it
> "Built a portfolio optimization engine comparing Markowitz QP, CVaR-based linear
> programming, and Hierarchical Risk Parity (custom Kruskal's MST implementation);
> backtested across 2008/2020 crisis periods, showing CVaR-LP reduced max drawdown by
> X% versus mean-variance at comparable Sharpe ratio."

Fill in X once you have real numbers — a specific, honest number beats a vague claim
every time in front of a Bloomberg or Google interviewer.

---

## How we'll work through this together

Suggested rhythm: for each phase, we do the math/derivation conversation first (on
paper, in words — I'll walk you through it and check your understanding), *then* move
to code, and I'll explain what each block does rather than just handing you a finished
file, since you said you want to fully understand it, not just have it work.

Want to start with **Phase 1.2–1.4** (the LP/QP/finance math) now, or do you want to
get Phase 0 environment setup out of the way first?
