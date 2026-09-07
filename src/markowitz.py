import os
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from data_loader import load_returns

def compute_parameters(returns: pd.DataFrame, trading_days : int = 252):
    mu = returns.mean() * trading_days
    sigma = returns.cov() * trading_days

    return mu, sigma

def min_variance_weights(sigma: np.ndarray) -> np.ndarray:
    n = len(sigma)
    ones = np.ones(n)

    sigma_inv_ones = np.linalg.solve(sigma, ones) # Sigma^-1 * 1

    C = ones @ sigma_inv_ones

    weights = sigma_inv_ones / C

    return weights

def efficient_frontier_closed_form(
        mu: np.ndarray, 
        sigma: np.ndarray, 
        num_points: int = 50
        ) -> tuple[list[float], list[float], list[np.ndarray]]:
    n = len(sigma)
    ones = np.ones(n)

    sig_inv_1 = np.linalg.solve(sigma, ones)
    sig_inv_mu = np.linalg.solve(sigma, mu)

    A = mu @ sig_inv_mu
    B = mu @ sig_inv_1
    C = ones @ sig_inv_1

    min_ret = np.min(mu)
    max_ret = np.max(mu) * 0.95
    target_returns = np.linspace(min_ret, max_ret, num_points)

    frontier_vols = []
    frontier_weights = []

    for r_target in target_returns:
        matrix = np.array([[A, B], [B, C]])
        rhs = np.array([r_target, 1.0])

        lam, gamma = np.linalg.solve(matrix, rhs)
        w_star = lam * sig_inv_mu + gamma * sig_inv_1

        port_variance = w_star @ sigma @ w_star
        port_volatility = np.sqrt(port_variance)

        frontier_vols.append(port_volatility)
        frontier_weights.append(w_star)

    return list(target_returns), frontier_vols, frontier_weights

def solve_constrained_qp(
        mu: np.ndarray, 
        sigma: np.ndarray, 
        target_return: float, 
        w_max: float = 0.3, 
        sector_mapper: dict[str, list[int]] | None = None, 
        sector_cap: float = 0.40
        ) -> np.ndarray:
    """
    minimize w^T * Sigma * w
    constraints: 
        sum(w) = 1
        w >= 0
        w <= w_max
        mu^T * w = target_return
    """
    n = len(mu)
    w = cp.Variable(n)
    risk = cp.quad_form(w, sigma)

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= w_max,
        mu @ w == target_return
    ]

    if sector_mapper:
        for sector_name, asset_indices in sector_mapper.items():
            constraints.append(cp.sum(w[asset_indices]) <= sector_cap)

    
    prob = cp.Problem(cp.Minimize(risk), constraints)
    prob.solve()

    return w.value

def efficient_frontier_constrained(
        mu: np.ndarray, 
        sigma: np.ndarray, 
        num_points: int = 50, 
        w_max: float = 0.3, 
        sector_mapper: dict[str, list[int]] | None = None, 
        sector_cap: float = 0.40
    ):
    min_ret = np.min(mu)
    max_ret = np.max(mu) * 0.95
    target_returns = np.linspace(min_ret, max_ret, num_points)
    
    frontier_vols = []
    frontier_weights = []
    valid_returns = []

    for r_target in target_returns:
        w_val = solve_constrained_qp(mu, sigma, r_target, w_max, sector_mapper, sector_cap)
        
        if w_val is not None:
            port_variance = w_val @ sigma @ w_val
            port_volatility = np.sqrt(port_variance)
            
            frontier_vols.append(port_volatility)
            frontier_weights.append(w_val)
            valid_returns.append(r_target)
            
    return valid_returns, frontier_vols, frontier_weights

def plot_efficient_frontier(
        mu: np.ndarray, 
        sigma: np.ndarray, 
        risk_free_rate: float = 0.0, 
        sector_mapper: dict[str, list[int]] | None = None,
        sector_cap: float = 0.40,
        save_path: str = "plots/efficient_frontier.png"
    ):
    """
    Computes both the closed-form and cvxpy-constrained efficient frontiers,
    finds the max-Sharpe portfolio, and plots them together.
    """
    print("Computing closed-form efficient frontier...")
    cf_returns, cf_vols, cf_weights = efficient_frontier_closed_form(mu, sigma, num_points=60)
    
    print("Computing cvxpy constrained efficient frontier (w >= 0, w <= 0.3)...")
    con_returns, con_vols, con_weights = efficient_frontier_constrained(
        mu, 
        sigma, 
        num_points=60, 
        w_max=0.3,
        sector_mapper=sector_mapper,
        sector_cap=sector_cap
    )
    
    # Find Max-Sharpe portfolio along the constrained frontier
    sharpe_ratios = [(r - risk_free_rate) / v for r, v in zip(con_returns, con_vols)]
    max_sharpe_idx = np.argmax(sharpe_ratios)
    
    best_return = con_returns[max_sharpe_idx]
    best_vol = con_vols[max_sharpe_idx]
    best_sharpe = sharpe_ratios[max_sharpe_idx]
    best_weights = con_weights[max_sharpe_idx]
    
    # Plotting setup
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(cf_vols, cf_returns, label="Closed-Form Frontier (Unconstrained)", color="blue", linestyle="--", linewidth=2)
    plt.plot(con_vols, con_returns, label=r"Constrained Frontier ($0 \leq w \leq 0.3$, No Shorting)", color="darkgreen", linewidth=2.5)
    
    # Plot Max-Sharpe Point
    plt.scatter([best_vol], [best_return], color="red", marker="*", s=200, zorder=5, 
                label=f"Max-Sharpe Portfolio (Sharpe: {best_sharpe:.2f})")
    
    # Chart styling
    plt.title("Markowitz Efficient Frontier: Closed-Form vs. Constrained QP", fontsize=14, fontweight="bold")
    plt.xlabel(r"Annualized Volatility ($\sigma$)", fontsize=12)
    plt.ylabel(r"Annualized Expected Return ($E[R]$)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved efficient frontier plot to {save_path}!")
    plt.close()
    
    return best_weights

def analyze_sensitivity(
    mu: np.ndarray, 
    sigma: np.ndarray, 
    tickers: list[str], 
    sector_mapper: dict[str, list[int]] | None = None,
    sector_cap: float = 0.40,
    num_simulations: int = 100, 
    noise_std: float = 0.01,
    save_path: str = "plots/sensitivity_analysis.png"
):
    print(f"\nRunning sensitivity analysis across {num_simulations} simulations (Noise std = {noise_std})...")
    
    n_assets = len(mu)
    successful_weights = []
    
    for i in range(num_simulations):
        noise = np.random.normal(0, noise_std, size=mu.shape)
        mu_perturbed = mu + noise
        
        returns, vols, weights = efficient_frontier_constrained(
            mu_perturbed, 
            sigma, 
            num_points=40,
            sector_mapper=sector_mapper,
            sector_cap=sector_cap
        )
        
        if len(returns) > 0 and len(vols) > 0:
            sharpes = [r / v if v > 0 else -np.inf for r, v in zip(returns, vols)]
            best_idx = np.argmax(sharpes)
            successful_weights.append(weights[best_idx])

    if not successful_weights:
        print("Error: No valid portfolios were solved during sensitivity analysis.")
        return None

    weight_matrix = np.array(successful_weights)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    plt.boxplot(
        [weight_matrix[:, col] for col in range(n_assets)],
        tick_labels=tickers,
        patch_artist=True,
        boxprops=dict(facecolor='lightskyblue', color='blue'),
        medianprops=dict(color='red', linewidth=2)
    )
    
    plt.title(r"Sensitivity Analysis: Allocation Swings under $\mu$ Perturbation ($\pm 1\%$ Noise)", fontsize=14, fontweight="bold")
    plt.xlabel("Assets", fontsize=12)
    plt.ylabel("Portfolio Weight Allocation ($w$)", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved sensitivity plot to {save_path}!")
    plt.close()
    
    return weight_matrix

if __name__ == "__main__":
    returns_df = load_returns()
    mu, sigma = compute_parameters(returns_df)
    sigma_np = sigma.to_numpy()
    mu_np = mu.to_numpy()
    tickers = list(returns_df.columns)
    
    # Sector dictionary mapping
    sector_dict = {
        "Tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
        "Finance": ["JPM", "BAC", "GS", "MS"],
        "Healthcare": ["JNJ", "PFE", "UNH"],
        "Consumer": ["PG", "KO", "PEP"]
    }
    
    sector_mapper = {}
    for sector, sector_tickers in sector_dict.items():
        indices = [tickers.index(t) for t in sector_tickers if t in tickers]
        if indices:
            sector_mapper[sector] = indices

    # 1. Plot Frontier
    optimal_weights = plot_efficient_frontier(
        mu_np, 
        sigma_np, 
        sector_mapper=sector_mapper, 
        sector_cap=0.40
    )
    
    print("\n--- Optimal Max-Sharpe Portfolio Allocation ---")
    for ticker, weight in zip(tickers, optimal_weights):
        if weight > 1e-4:
            print(f"{ticker}: {weight * 100:.2f}%")

    # 2. Sensitivity Analysis
    analyze_sensitivity(
        mu_np, 
        sigma_np, 
        tickers, 
        sector_mapper=sector_mapper, 
        sector_cap=0.40,
        num_simulations=100, 
        noise_std=0.01
    )