"""Figures for the report. Uses the non-interactive Agg backend so it runs headless."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_runtime_vs_n(df, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(df["n"], df["greedy_ms"], "o-", label="Greedy (sort + scan)")
    ax.loglog(df["n"], df["dp_ms"], "s-", label="0/1 Knapsack DP")
    ref = df["dp_ms"].iloc[0] * (df["n"] / df["n"].iloc[0])
    ax.loglog(df["n"], ref, "k--", alpha=0.5, label="O(n) reference")
    ax.set_xlabel("number of candidate assets n")
    ax.set_ylabel("runtime (ms, best of 3)")
    ax.set_title("Runtime vs. universe size (B = 500, R = 150)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_quality(df, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x = np.arange(len(df))
    w = 0.38
    ax.bar(x - w / 2, df["greedy_value"], w, label="Greedy")
    ax.bar(x + w / 2, df["dp_value"], w, label="Knapsack (optimal)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in df["n"]])
    ax.set_xlabel("number of candidate assets n")
    ax.set_ylabel("expected one-year gain ($)")
    ax.set_title("Objective achieved by each strategy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_budget_scaling(df, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(df["B"], df["dp_ms"], "s-", label="measured DP time")
    slope = df["dp_ms"].iloc[-1] / df["B"].iloc[-1]
    ax.plot(df["B"], slope * df["B"], "k--", alpha=0.5, label="linear in B")
    ax.set_xlabel("budget capacity B (units)")
    ax.set_ylabel("DP runtime (ms)")
    ax.set_title("DP runtime grows linearly with budget (n = 100, R = 150)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_equity(results, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, grp in results.groupby("strategy"):
        wealth = (1.0 + grp["period_return"]).cumprod()
        ax.plot(grp["date"], wealth, label=label.capitalize())
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_ylabel("growth of $1")
    ax.set_title("Walk-forward equity curves (quarterly rebalancing)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
