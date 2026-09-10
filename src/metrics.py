import numpy as np
import pandas as pd


def annualized_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(np.dot(weights, mu))


def annualized_volatility(weights: np.ndarray, sigma: np.ndarray) -> float:
    variance = weights @ sigma @ weights
    return float(np.sqrt(variance))


def sharpe_ratio(
    weights: np.ndarray, 
    mu: np.ndarray, 
    sigma: np.ndarray, 
    risk_free_rate: float = 0.0
) -> float:
    ret = annualized_return(weights, mu)
    vol = annualized_volatility(weights, sigma)
    if vol == 0:
        return 0.0
    return (ret - risk_free_rate) / vol


def portfolio_cumulative_returns(
    weights: np.ndarray, 
    returns_df: pd.DataFrame
) -> pd.Series:
    
    simple_asset_returns = np.expm1(returns_df)
    
    portfolio_daily_simple = simple_asset_returns.values @ weights
    
    cum_returns = pd.Series(
        (1 + portfolio_daily_simple).cumprod(), 
        index=returns_df.index
    )
    return cum_returns


def max_drawdown(cumulative_returns: pd.Series) -> float:
    
    running_max = cumulative_returns.cummax()
    
    drawdowns = (cumulative_returns - running_max) / running_max
    return float(drawdowns.min())


def summary_stats(
    weights: np.ndarray, 
    mu: np.ndarray, 
    sigma: np.ndarray, 
    returns_df: pd.DataFrame, 
    risk_free_rate: float = 0.0
) -> dict[str, float]:

    cum_ret = portfolio_cumulative_returns(weights, returns_df)
    
    return {
        "annualized_return": annualized_return(weights, mu),
        "annualized_volatility": annualized_volatility(weights, sigma),
        "sharpe_ratio": sharpe_ratio(weights, mu, sigma, risk_free_rate),
        "max_drawdown": max_drawdown(cum_ret)
    }