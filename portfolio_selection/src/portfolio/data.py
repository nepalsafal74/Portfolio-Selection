"""Market data: a reproducible synthetic price generator and a CSV loader."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def synthetic_prices(n_assets: int = 30, n_days: int = 2520, seed: int = 7) -> pd.DataFrame:
    """Generate a one-factor price panel with realistic-looking correlation.

    Daily return of asset i:
        r_it = mu_i / 252 + sigma_i / sqrt(252) * (rho_i * z_mt + sqrt(1 - rho_i^2) * e_it)
    where z_mt is a common market shock, so assets are correlated through the market factor.
    Returns a DataFrame of prices indexed by business day, one column per asset.
    """
    rng = np.random.default_rng(seed)
    names = [f"A{i:03d}" for i in range(n_assets)]

    mu = rng.uniform(-0.02, 0.22, n_assets)    # annual expected return
    vol = rng.uniform(0.10, 0.45, n_assets)    # annual volatility
    rho = rng.uniform(0.30, 0.70, n_assets)    # loading on the market factor
    p0 = rng.uniform(40.0, 300.0, n_assets)    # starting share prices

    z_m = rng.standard_normal(n_days)
    eps = rng.standard_normal((n_days, n_assets))
    z = rho * z_m[:, None] + np.sqrt(1.0 - rho**2) * eps
    daily = mu / TRADING_DAYS + vol / np.sqrt(TRADING_DAYS) * z

    prices = p0 * np.cumprod(1.0 + daily, axis=0)
    index = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame(prices, index=index, columns=names)


def load_price_csv(path: str) -> pd.DataFrame:
    """Load a wide CSV: first column = date, remaining columns = one price series per asset."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index().ffill().dropna(axis=1, how="all").dropna()
    return df
