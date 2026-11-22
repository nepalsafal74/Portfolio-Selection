"""Application layer shared by the command line (main.py) and the web app (app.py).

Everything the user sees is computed here, so the interface contains no algorithm logic.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import summarize, walk_forward
from .data import load_price_csv, synthetic_prices
from .features import asset_table, discretize, dropped_assets
from .greedy import greedy_select
from .knapsack import fill_table, knapsack_2d
from .models import Item, is_feasible

DEFAULT_LOT = 10.0
DEFAULT_RISK_SCALE = 100


# ---------------------------------------------------------------- data

def load_universe(n_assets: int = 12, n_days: int = 253, seed: int = 3,
                  csv_source=None) -> tuple[pd.DataFrame, str]:
    """Return (prices, label). Label says where the data came from; synthetic data is named.

    csv_source may be a file path or the bytes of an uploaded CSV file.
    """
    if csv_source is not None:
        if isinstance(csv_source, (bytes, bytearray)):
            csv_source = io.BytesIO(csv_source)
        prices = load_price_csv(csv_source)
        return prices, f"CSV file ({prices.shape[1]} assets, {prices.shape[0]} rows)"
    prices = synthetic_prices(n_assets=n_assets, n_days=n_days, seed=seed)
    return prices, f"SYNTHETIC test data ({n_assets} assets, seed {seed}); not real market prices"


def analyse(prices: pd.DataFrame, rf_annual: float = 0.02, lot_size: float = DEFAULT_LOT,
            risk_scale: int = DEFAULT_RISK_SCALE):
    """Return (table, items, dropped) for the latest window of prices."""
    table = asset_table(prices, rf_annual)
    items = discretize(table, lot_size, risk_scale)
    dropped = dropped_assets(table)
    return table, items, dropped


def budget_to_units(budget_dollars: float, lot_size: float = DEFAULT_LOT) -> int:
    return int(budget_dollars // lot_size)


# ---------------------------------------------------------------- algorithms

@dataclass
class Comparison:
    greedy: object
    knapsack: object
    greedy_ms: float
    knapsack_ms: float
    budget_units: int
    risk_units: int
    gap_pct: float                     # how far greedy is below the optimum
    both_feasible: bool
    notes: list = field(default_factory=list)


def compare(items, budget_units: int, risk_units: int, repeats: int = 3) -> Comparison:
    """Run both algorithms on the same items and time them (best of `repeats`)."""
    def best(fn):
        t_best, out = float("inf"), None
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = fn()
            t_best = min(t_best, time.perf_counter() - t0)
        return t_best * 1e3, out

    g_ms, g = best(lambda: greedy_select(items, budget_units, risk_units))
    k_ms, k = best(lambda: knapsack_2d(items, budget_units, risk_units))
    gap = (k.value - g.value) / k.value * 100 if k.value > 0 else 0.0
    notes = []
    if g.value >= k.value - 1e-9:
        notes.append("Greedy matched the optimum on this input.")
    else:
        notes.append(f"Greedy missed the optimum by ${k.value - g.value:,.2f}.")
    return Comparison(
        greedy=g, knapsack=k, greedy_ms=g_ms, knapsack_ms=k_ms,
        budget_units=budget_units, risk_units=risk_units, gap_pct=gap,
        both_feasible=is_feasible(g, budget_units, risk_units) and is_feasible(k, budget_units, risk_units),
        notes=notes,
    )


def dp_stages(items, budget_units: int, risk_units: int) -> list:
    """DP table after each item is considered: stages[k] covers items[:k].

    Used by the table explorer. Cost grows with n, so keep n small for interactive use.
    """
    stages = []
    for k in range(len(items) + 1):
        dp, _ = fill_table(items[:k], budget_units, risk_units)
        stages.append(dp)
    return stages


def allocation(selection, table: pd.DataFrame, budget_dollars: float) -> pd.DataFrame:
    """Dollar view of a selection: each chosen asset is bought as one whole share."""
    rows = []
    spent = 0.0
    for it in selection.items:
        price = float(table.loc[it.name, "price"])
        spent += price
        rows.append({"asset": it.name, "share price": price, "shares": 1,
                     "cost ($)": price, "expected gain ($)": it.value})
    df = pd.DataFrame(rows, columns=["asset", "share price", "shares", "cost ($)", "expected gain ($)"])
    if not df.empty:
        df["weight (% of spend)"] = df["cost ($)"] / spent * 100
    df.attrs["spent"] = spent
    df.attrs["cash_left"] = budget_dollars - spent
    return df


# ---------------------------------------------------------------- backtest & benchmark

def run_backtest(prices: pd.DataFrame, budget_dollars: float = 5000.0, risk_ceiling: int = 150,
                 lookback: int = 252, rebalance: int = 63):
    results = walk_forward(prices, budget_dollars=budget_dollars, lot_size=DEFAULT_LOT,
                           risk_ceiling=risk_ceiling, risk_scale=DEFAULT_RISK_SCALE,
                           lookback=lookback, rebalance=rebalance)
    return results, summarize(results)
