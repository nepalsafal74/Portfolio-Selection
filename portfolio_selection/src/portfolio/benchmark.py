"""Runtime, quality and memory experiments for Greedy vs. DP."""
from __future__ import annotations

import time

import pandas as pd

from .data import synthetic_prices
from .features import asset_table, discretize
from .greedy import greedy_select
from .knapsack import knapsack_2d, knapsack_table_bytes


def make_items(n_assets: int, seed: int = 0, lot_size: float = 10.0,
               risk_scale: int = 100, lookback: int = 252):
    prices = synthetic_prices(n_assets=n_assets, n_days=lookback + 1, seed=seed)
    return discretize(asset_table(prices), lot_size, risk_scale)


def best_time(fn, repeats: int = 3):
    """Return (best wall-clock seconds over repeats, result of the last call)."""
    best, out = float("inf"), None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


def scale_with_n(sizes, budget_units=500, risk_units=150, seed=0, repeats=3) -> pd.DataFrame:
    rows = []
    for n in sizes:
        items = make_items(n, seed=seed)
        t_g, g = best_time(lambda: greedy_select(items, budget_units, risk_units), repeats)
        t_k, k = best_time(lambda: knapsack_2d(items, budget_units, risk_units), repeats)
        gap = (k.value - g.value) / k.value * 100 if k.value > 0 else 0.0
        rows.append({
            "n": n,
            "greedy_ms": t_g * 1e3,
            "dp_ms": t_k * 1e3,
            "greedy_value": g.value,
            "dp_value": k.value,
            "greedy_gap_pct": gap,
            "greedy_count": len(g.items),
            "dp_count": len(k.items),
            "dp_table_mb": knapsack_table_bytes(n, budget_units, risk_units) / 1e6,
        })
    return pd.DataFrame(rows)


def scale_with_budget(budgets, n=100, risk_units=150, seed=0, repeats=3) -> pd.DataFrame:
    items = make_items(n, seed=seed)
    rows = []
    for B in budgets:
        t_k, _ = best_time(lambda: knapsack_2d(items, B, risk_units), repeats)
        rows.append({"B": B, "dp_ms": t_k * 1e3, "work_units": n * B * (risk_units + 1)})
    return pd.DataFrame(rows)
