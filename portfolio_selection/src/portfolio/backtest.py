"""Walk-forward backtest that compares Greedy and Knapsack on the same rebalance dates."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import asset_table, discretize
from .greedy import greedy_select
from .knapsack import knapsack_2d
from .models import is_feasible


def walk_forward(
    prices: pd.DataFrame,
    budget_dollars: float = 5000.0,
    lot_size: float = 10.0,
    risk_ceiling: int = 150,     # in risk units: 150 = sum of annual vols of 1.50
    risk_scale: int = 100,
    lookback: int = 252,
    rebalance: int = 63,
    rf: float = 0.02,
) -> pd.DataFrame:
    """Rebalance every `rebalance` trading days using only the trailing `lookback` window.

    At each rebalance date t the strategy sees prices[t - lookback : t + 1] only. It buys one
    share of each selected asset at prices[t] and holds until the next rebalance date.
    """
    B = int(budget_dollars // lot_size)
    R = int(risk_ceiling)
    rows = []
    prev_sets = {"greedy": set(), "knapsack": set()}

    for t in range(lookback, len(prices) - 1, rebalance):
        window = prices.iloc[t - lookback : t + 1]
        items = discretize(asset_table(window, rf), lot_size, risk_scale)

        t0 = time.perf_counter()
        g = greedy_select(items, B, R)
        t_g = time.perf_counter() - t0
        t0 = time.perf_counter()
        k = knapsack_2d(items, B, R)
        t_k = time.perf_counter() - t0

        end = min(t + rebalance, len(prices) - 1)
        p_start, p_end = prices.iloc[t], prices.iloc[end]

        for label, sel, rt in (("greedy", g, t_g), ("knapsack", k, t_k)):
            names = sel.names
            if names:
                invested = float(p_start[names].sum())
                pnl = float((p_end[names] - p_start[names]).sum())
                period_ret = pnl / invested
            else:
                invested, pnl, period_ret = 0.0, 0.0, 0.0

            new_set = set(names)
            turnover = (len(new_set ^ prev_sets[label]) / max(len(new_set | prev_sets[label]), 1))
            prev_sets[label] = new_set

            rows.append({
                "date": prices.index[t],
                "strategy": label,
                "n_selected": len(names),
                "invested_dollars": invested,
                "est_value": sel.value,
                "realized_pnl": pnl,
                "period_return": period_ret,
                "turnover": turnover,
                "feasible": is_feasible(sel, B, R),
                "runtime_ms": rt * 1e3,
            })
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame, periods_per_year: float = 252 / 63) -> pd.DataFrame:
    """Per-strategy summary statistics."""
    out = []
    for label, grp in results.groupby("strategy"):
        r = grp["period_return"].to_numpy()
        wealth = np.prod(1.0 + r)
        years = len(r) / periods_per_year
        cagr = wealth ** (1.0 / years) - 1.0 if years > 0 else np.nan
        sd = r.std(ddof=1) * np.sqrt(periods_per_year) if len(r) > 1 else np.nan
        sharpe = (r.mean() * periods_per_year) / sd if sd and sd > 0 else np.nan
        out.append({
            "strategy": label,
            "periods": len(r),
            "cumulative_return": wealth - 1.0,
            "cagr": cagr,
            "ann_volatility": sd,
            "ann_sharpe_like": sharpe,
            "mean_est_value": grp["est_value"].mean(),
            "mean_turnover": grp["turnover"].iloc[1:].mean() if len(grp) > 1 else np.nan,
            "mean_runtime_ms": grp["runtime_ms"].mean(),
            "all_feasible": bool(grp["feasible"].all()),
        })
    return pd.DataFrame(out)
