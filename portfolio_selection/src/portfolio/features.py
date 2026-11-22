"""Turn a price window into per-asset statistics and integer knapsack items."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TRADING_DAYS
from .models import Item

MIN_VOL = 1e-8  # below this an asset has effectively no price movement


def asset_table(prices: pd.DataFrame, rf_annual: float = 0.02) -> pd.DataFrame:
    """Annualised statistics computed from a price window.

    Columns: expected_return, risk (annualised volatility), sharpe, price (latest).
    Only data inside the window is used, so there is no look-ahead when the window
    ends on the rebalance date. Assets with zero or missing volatility keep their
    row here (Sharpe becomes NaN or infinite) and are removed later by discretize().
    """
    rets = prices.pct_change().dropna()
    mu = rets.mean() * TRADING_DAYS
    vol = rets.std(ddof=1) * np.sqrt(TRADING_DAYS)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (mu - rf_annual) / vol
    return pd.DataFrame({
        "expected_return": mu,
        "risk": vol,
        "sharpe": sharpe,
        "price": prices.iloc[-1],
    })


def is_usable(row: pd.Series) -> bool:
    """An asset can be used only with a positive, finite price, volatility and Sharpe."""
    vals = [row["price"], row["risk"], row["sharpe"], row["expected_return"]]
    return all(np.isfinite(v) for v in vals) and row["price"] > 0 and row["risk"] > MIN_VOL


def dropped_assets(table: pd.DataFrame) -> list:
    """Names of assets that cannot be used, with a short reason for the user."""
    out = []
    for name, row in table.iterrows():
        if not is_usable(row):
            if not np.isfinite(row["price"]) or row["price"] <= 0:
                reason = "missing or invalid price"
            else:
                reason = "no price movement (zero volatility)"
            out.append((str(name), reason))
    return out


def discretize(table: pd.DataFrame, lot_size: float = 10.0, risk_scale: int = 100) -> list:
    """Convert the statistics table into integer 0/1 knapsack items.

    Costs and risks are rounded UP, so any solution feasible in units is feasible in
    dollars and in volatility (the discretised model is conservative). Both strategies
    receive exactly these items, so the comparison is fair. Unusable assets are skipped
    (see dropped_assets for the list and reasons).
    """
    items = []
    for name, row in table.iterrows():
        if not is_usable(row):
            continue
        items.append(Item(
            name=str(name),
            cost=max(1, int(np.ceil(row["price"] / lot_size))),
            risk=max(1, int(np.ceil(row["risk"] * risk_scale))),
            value=float(row["expected_return"] * row["price"]),
            sharpe=float(row["sharpe"]),
        ))
    return items
