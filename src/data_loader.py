import os
import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "AMZN",
    "JNJ", "PG", "KO", "XLU",
    "SPY",
    "AGG", "TLT",
    "JPM", "XOM", "VNQ"
]

def load_prices(tickers: list[str] = TICKERS, start: str = "2021-01-01", end: str = "2026-01-01", save_path: str = "data/price_data.csv") -> pd.DataFrame:
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")

    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close']
    else:
        prices = data[['Adj Close']]

    prices = prices.dropna(how='all').ffill().dropna()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    prices.to_csv(save_path)
    print(f"Successfully cached raw prices to {save_path}!")
    
    return prices

def load_returns(tickers: list[str] = TICKERS, start: str = "2021-01-01", end: str = "2026-01-01") -> pd.DataFrame:
    prices = load_prices(tickers=tickers, start=start, end=end)
    
    log_returns = np.log(prices / prices.shift(1))
    
    log_returns = log_returns.dropna()
    
    return log_returns

if __name__ == "__main__":
    returns_df = load_returns()
    print("\nFirst 5 rows of daily log returns:")
    print(returns_df.head())