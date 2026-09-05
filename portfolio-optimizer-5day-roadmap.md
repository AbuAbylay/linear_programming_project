# Portfolio Optimizer — 5-Day Roadmap

**Scope for this version:** Markowitz mean-variance (QP) baseline + CVaR-based
portfolio optimization (LP), with a backtested comparison. HRP/Kruskal's MST is
dropped from this version — noted as future work in the README, not implemented.

**Why this scope:** the QP-vs-LP comparison with a real backtested result is the
entire value of the project for a CV. Two techniques done properly and understood
deeply beats three done shallowly.

---

## Day 1 — Tooling + Markowitz math

### Morning (~3h): Python/NumPy speed-run
- Python syntax essentials mapped to C++ concepts (see roadmap Phase 0.1 from before)
- NumPy: arrays, broadcasting, `@` for matrix multiply, `np.linalg.solve`,
  `np.cov`, slicing
- Set up repo skeleton, venv, `requirements.txt` (`numpy`, `pandas`, `scipy`,
  `cvxpy`, `matplotlib`, `yfinance`)

### Afternoon (~3–4h): Markowitz math, on paper first
- Portfolio return `w^T μ`, portfolio variance `w^T Σ w` — why the quadratic form
  makes this a QP, not LP
- Convexity: why `Σ` being positive semi-definite guarantees a global minimum
- Derive the **closed-form unconstrained solution** via Lagrange multipliers
  (only equality constraint `w^T 1 = 1`, no short-selling restriction yet):
  the classic `w* = (Σ⁻¹ μ) / (1^T Σ⁻¹ μ)`-style result — work through the
  Lagrangian, take the gradient, solve. I'll walk this through with you.
- Sharpe ratio, efficient frontier concept

**End of day checkpoint:** you can write the Lagrangian for the unconstrained
problem from memory and explain why it's convex.

---

## Day 2 — Markowitz implementation

### Morning (~2–3h): Data pipeline
- `data_loader.py`: pull ~15–20 tickers (pick a mix — some tech, some
  defensive/utilities, maybe one index ETF — so the frontier looks interesting)
  via `yfinance`, 3–5 years of daily data
- Compute log returns, annualized `μ` and `Σ`
- Cache to disk (`data/`) so you're not re-hitting the API every run

### Afternoon (~3–4h): Two implementations
1. **Closed-form** version using your Day 1 derivation directly in NumPy
   (`np.linalg.solve`) — no solver library. This is your proof you understand
   the math, not just the API.
2. **Constrained QP via `cvxpy`** — add realistic constraints (`w ≥ 0`, no
   short-selling; optionally a max single-position cap like `w ≤ 0.3`)
   ```python
   w = cp.Variable(n)
   risk = cp.quad_form(w, Sigma)
   prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])
   prob.solve()
   ```
3. Sweep target returns → plot the efficient frontier, mark max-Sharpe point

**Deliverable:** `markowitz.py` + efficient frontier plot. This is your baseline,
fully working.

---

## Day 3 — CVaR/LP theory + scenario setup

### Morning (~3h): VaR/CVaR theory, on paper
- VaR(α): the loss not exceeded with probability α
- CVaR(α): expected loss *given* you're in the worst (1−α) tail — why this is a
  better tail-risk measure than VaR (VaR says nothing about how bad the tail is)
- **Rockafellar–Uryasev formulation** — the key derivation, I'll walk you through
  it step by step:
  - Introduce auxiliary variable `ζ` (candidate VaR threshold)
  - Introduce auxiliary variables `z_i ≥ 0` for shortfall in each historical
    scenario `i`: `z_i ≥ -(w^T r_i) - ζ`
  - Objective becomes: minimize `ζ + (1/(1-α)N) * Σ z_i` — this is **linear** in
    `w`, `ζ`, `z` — that's the whole trick, a non-smooth tail objective becomes
    a clean LP with extra variables and constraints
- Write out the full LP in standard form (`c^T x`, `Ax ≤ b`) on paper for your
  own asset universe before coding it

**End of day checkpoint:** you can explain to me, without notes, why adding the
`z_i` auxiliary variables linearizes the problem.

### Afternoon (~1–2h): Scenario setup
- Use historical daily/weekly returns directly as scenarios (historical
  simulation — simplest, most defensible choice for a 5-day project; note in
  README that bootstrapped/simulated scenarios are a possible extension)

---

## Day 4 — CVaR-LP implementation

### Morning (~3h): Implement in `cvxpy`
```python
w = cp.Variable(n)
zeta = cp.Variable()
z = cp.Variable(N, nonneg=True)  # N = number of historical scenarios
losses = -returns_matrix @ w      # shape (N,)
constraints = [
    z >= losses - zeta,
    cp.sum(w) == 1,
    w >= 0,
]
cvar = zeta + (1 / ((1 - alpha) * N)) * cp.sum(z)
prob = cp.Problem(cp.Minimize(cvar), constraints)
prob.solve()
```
- Confirm this matches your Day 3 paper derivation term-for-term — this
  matching is the whole point, don't skip it

### Afternoon (~2–3h): Also implement via `scipy.optimize.linprog`
- Hand-build the `A_ub`, `b_ub`, `c` matrices yourself instead of using
  `cvxpy`'s modeling layer — this is closer to what you'd do in C++, forces you
  to fully internalize the standard LP form, and gives you two independent
  implementations to cross-check against each other
- Plot the realized loss distribution with the CVaR threshold marked

**Deliverable:** `cvar_lp.py`, working, cross-validated two ways, plus a loss
distribution plot.

---

## Day 5 — Backtest, comparison, polish

### Morning (~3h): Backtest
- Simple approach given the time budget: **single train/test split** rather
  than a full rolling-window backtest (state this explicitly and honestly in
  the README as a scoping choice, not a limitation you're hiding)
- Train (estimate `μ`, `Σ`, scenarios) on data up to some cutoff date, hold the
  resulting weights fixed, evaluate on the out-of-sample period after
- Run three strategies on the same split: Markowitz max-Sharpe, CVaR-LP,
  equal-weight benchmark
- Metrics: cumulative return, volatility, Sharpe ratio, max drawdown, realized
  CVaR

**Stretch goal if time allows:** pick your out-of-sample window to include a
volatile period (e.g. a slice of 2022, or COVID-era data if your date range
allows) so the CVaR-LP vs Markowitz drawdown difference actually shows up. If
you don't have time to hunt for this, a flat comparison table with honest
numbers is still a legitimate result — don't manufacture a dramatic story.

### Afternoon (~2–3h): README + polish
- Problem statement, the two math formulations (briefly, in your own words),
  architecture, how to run it, results table/plot
- Explicitly note HRP/Kruskal's MST as a scoped-out extension — shows you knew
  the tradeoff, didn't just run out of time unknowingly
- A few basic tests: weights sum to 1, no negative weights, `cvxpy` and
  `linprog` CVaR results agree within tolerance
- Draft your CV bullet with real numbers from your actual results, e.g.:
  > "Built a portfolio optimizer implementing Markowitz mean-variance (QP) and
  > CVaR-based tail-risk optimization (LP, Rockafellar–Uryasev formulation);
  > out-of-sample backtest showed CVaR-LP reduced max drawdown by X% at a
  > Sharpe ratio cost of Y."

---

## Time-pressure fallbacks, decided now so you don't have to decide mid-panic

- **Behind after Day 2:** skip the closed-form derivation implementation, keep
  only the `cvxpy` QP version — you still have the paper derivation from Day 1
  to talk about in interviews even if it's not in the code.
- **Behind after Day 3:** skip the `scipy.optimize.linprog` manual version,
  keep only `cvxpy` for CVaR-LP.
- **Behind on Day 5:** cut the backtest metrics down to just cumulative return
  + max drawdown (drop Sharpe/volatility detail) — a smaller honest result beats
  a rushed broken one.
- **Never cut:** the paper derivations (Day 1, Day 3 mornings). These are cheap
  in time and are what let you actually defend the project in an interview —
  code without understanding is the one failure mode that actively hurts you.

---

## Suggested working rhythm with me

Same as before, compressed: for each math block, we talk it through and I check
your understanding before you write code for it. Given the timeline, I'll be
more direct about giving you working code snippets faster once the concept is
solid — but I won't skip the "do you actually get why this works" step, since
that's the part that makes this a real project instead of a copy-pasted one.

Ready to start Day 1 math (Markowitz Lagrangian derivation) now, or do you want
the environment/repo setup out of the way first?
