#!/usr/bin/env python3
"""Command-line entry point.

  python main.py demo        # small worked example with brute-force check
  python main.py benchmark   # runtime / quality / memory experiments -> results/
  python main.py backtest    # walk-forward comparison -> results/
  python main.py all         # everything above
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import pandas as pd  # noqa: E402

from portfolio.brute import brute_force  # noqa: E402
from portfolio.data import synthetic_prices, load_price_csv  # noqa: E402
from portfolio.features import asset_table, discretize  # noqa: E402
from portfolio.greedy import greedy_select  # noqa: E402
from portfolio.knapsack import knapsack_2d  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(RESULTS, "figures")

BUDGET_DOLLARS = 5000.0
LOT = 10.0
RISK_CEILING = 150
RISK_SCALE = 100


def cmd_demo(args):
    prices = synthetic_prices(n_assets=12, n_days=253, seed=3)
    table = asset_table(prices)
    items = discretize(table, LOT, RISK_SCALE)
    B, R = int(BUDGET_DOLLARS // LOT), RISK_CEILING

    print("Candidate universe (12 assets, 1 year of history)")
    show = table.copy()
    show["cost_units"] = [it.cost for it in items]
    show["risk_units"] = [it.risk for it in items]
    print(show.round(3).to_string())
    print(f"\nBudget = {B} units (${BUDGET_DOLLARS:,.0f}), risk ceiling = {R} units\n")

    g = greedy_select(items, B, R)
    k = knapsack_2d(items, B, R)
    bf = brute_force(items, B, R)

    print(f"Greedy   : value ${g.value:8.2f}  cost {g.cost:3d}/{B}  risk {g.risk:3d}/{R}  picks {g.names}")
    print(f"Knapsack : value ${k.value:8.2f}  cost {k.cost:3d}/{B}  risk {k.risk:3d}/{R}  picks {k.names}")
    print(f"Brute    : value ${bf.value:8.2f}  (exhaustive check of all 2^{len(items)} subsets)")
    ok = abs(k.value - bf.value) < 1e-6 and abs(k.dp_optimum - k.value) < 1e-6
    print(f"DP matches brute force and its traceback: {ok}")


def cmd_benchmark(args):
    from portfolio.benchmark import scale_with_budget, scale_with_n
    from portfolio.plots import plot_budget_scaling, plot_quality, plot_runtime_vs_n

    os.makedirs(FIGS, exist_ok=True)
    sizes = [10, 20, 40, 80, 160, 320]
    df_n = scale_with_n(sizes, budget_units=int(BUDGET_DOLLARS // LOT), risk_units=RISK_CEILING)
    df_n.to_csv(os.path.join(RESULTS, "benchmark_n.csv"), index=False)
    print(df_n.round(3).to_string(index=False))

    df_b = scale_with_budget([100, 200, 400, 800, 1600], n=100, risk_units=RISK_CEILING)
    df_b.to_csv(os.path.join(RESULTS, "benchmark_budget.csv"), index=False)
    print()
    print(df_b.round(3).to_string(index=False))

    plot_runtime_vs_n(df_n, os.path.join(FIGS, "runtime_vs_n.png"))
    plot_quality(df_n, os.path.join(FIGS, "quality_vs_n.png"))
    plot_budget_scaling(df_b, os.path.join(FIGS, "dp_runtime_vs_budget.png"))
    print("\nFigures written to results/figures/")


def cmd_backtest(args):
    from portfolio.backtest import summarize, walk_forward
    from portfolio.plots import plot_equity

    os.makedirs(FIGS, exist_ok=True)
    if args.csv:
        prices = load_price_csv(args.csv)
        print(f"Loaded {prices.shape[1]} assets, {prices.shape[0]} rows from {args.csv}")
    else:
        prices = synthetic_prices(n_assets=args.assets, n_days=args.days, seed=args.seed)
        print(f"Synthetic universe: {args.assets} assets, {args.days} business days, seed {args.seed}")

    res = walk_forward(prices, budget_dollars=BUDGET_DOLLARS, lot_size=LOT,
                       risk_ceiling=RISK_CEILING, risk_scale=RISK_SCALE)
    res.to_csv(os.path.join(RESULTS, "backtest_rebalances.csv"), index=False)
    summ = summarize(res)
    summ.to_csv(os.path.join(RESULTS, "backtest_summary.csv"), index=False)
    print(summ.round(4).to_string(index=False))
    plot_equity(res, os.path.join(FIGS, "equity_curves.png"))
    print("\nBacktest written to results/")


def cmd_sweep(args):
    """Repeat the backtest over many synthetic universes and test the paired difference."""
    import numpy as np
    from scipy import stats  # listed in requirements.txt
    from portfolio.backtest import summarize, walk_forward

    rows = []
    for seed in range(1, args.seeds + 1):
        prices = synthetic_prices(n_assets=args.assets, n_days=args.days, seed=seed)
        res = walk_forward(prices, budget_dollars=BUDGET_DOLLARS, lot_size=LOT,
                           risk_ceiling=RISK_CEILING, risk_scale=RISK_SCALE)
        s = summarize(res).set_index("strategy")
        rows.append({"seed": seed,
                     "greedy_cum": s.loc["greedy", "cumulative_return"],
                     "knapsack_cum": s.loc["knapsack", "cumulative_return"],
                     "greedy_sharpe": s.loc["greedy", "ann_sharpe_like"],
                     "knapsack_sharpe": s.loc["knapsack", "ann_sharpe_like"],
                     "greedy_est": s.loc["greedy", "mean_est_value"],
                     "knapsack_est": s.loc["knapsack", "mean_est_value"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "seed_sweep.csv"), index=False)

    diff = df["knapsack_cum"] - df["greedy_cum"]
    t, pval = stats.ttest_rel(df["knapsack_cum"], df["greedy_cum"])
    print(df.round(3).to_string(index=False))
    print(f"\nMean cumulative return: greedy {df.greedy_cum.mean():.3f}, knapsack {df.knapsack_cum.mean():.3f}")
    print(f"Paired difference (knapsack - greedy): mean {diff.mean():.3f}, "
          f"seeds where knapsack wins {int((diff > 0).sum())}/{len(diff)}, "
          f"paired t = {t:.3f}, p = {pval:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    sub.add_parser("benchmark").set_defaults(fn=cmd_benchmark)
    bt = sub.add_parser("backtest")
    bt.add_argument("--csv", help="optional wide price CSV (date column + one column per asset)")
    bt.add_argument("--assets", type=int, default=30)
    bt.add_argument("--days", type=int, default=2520)
    bt.add_argument("--seed", type=int, default=7)
    bt.set_defaults(fn=cmd_backtest)
    sw = sub.add_parser("sweep")
    sw.add_argument("--seeds", type=int, default=20)
    sw.add_argument("--assets", type=int, default=30)
    sw.add_argument("--days", type=int, default=2520)
    sw.set_defaults(fn=cmd_sweep)
    sub.add_parser("all").set_defaults(fn=None)
    args = p.parse_args(argv)

    if args.cmd == "all":
        args.csv, args.assets, args.days, args.seed = None, 30, 2520, 7
        for fn in (cmd_demo, cmd_benchmark, cmd_backtest):
            fn(args)
            print("-" * 72)
    else:
        args.fn(args)


if __name__ == "__main__":
    main()
