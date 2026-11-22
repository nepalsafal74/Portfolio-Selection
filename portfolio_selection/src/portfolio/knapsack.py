"""Exact 0/1 knapsack with two capacity dimensions (budget x risk).

DP state dp[b, r] = maximum value using at most b budget units and at most r risk units.
A boolean table keep[i, b, r] records whether item i was taken at that state, so the
chosen set can be recovered by backtracking from dp[B, R].

Time  : O(n * B * R)
Memory: O(n * B * R) bytes for the boolean traceback table, plus O(B * R) floats for dp.
"""
from __future__ import annotations

import numpy as np

from .models import build_selection


def fill_table(items, budget_units: int, risk_units: int):
    """Fill the DP table. Returns (dp, keep).

    dp has shape (B+1, R+1); keep has shape (n, B+1, R+1).
    """
    B, R = int(budget_units), int(risk_units)
    if B < 0 or R < 0:
        raise ValueError("capacities must be non-negative")
    if any(it.cost <= 0 or it.risk <= 0 for it in items):
        raise ValueError("every item needs a positive integer cost and risk")

    n = len(items)
    dp = np.zeros((B + 1, R + 1))
    keep = np.zeros((n, B + 1, R + 1), dtype=bool)

    for i, it in enumerate(items):
        c, r = it.cost, it.risk
        if c > B or r > R:
            continue  # cannot fit even alone
        # `candidate` is a fresh array, so reading dp before the in-place update is safe.
        candidate = dp[: B + 1 - c, : R + 1 - r] + it.value
        target = dp[c:, r:]                       # view into dp
        better = candidate > target
        keep[i, c:, r:] = better
        np.copyto(target, candidate, where=better)
    return dp, keep


def knapsack_2d(items, budget_units: int, risk_units: int):
    B, R = int(budget_units), int(risk_units)
    dp, keep = fill_table(items, B, R)

    # Backtrack from the full-capacity state.
    chosen = []
    b, rr = B, R
    for i in range(len(items) - 1, -1, -1):
        if keep[i, b, rr]:
            chosen.append(items[i])
            b -= items[i].cost
            rr -= items[i].risk
    chosen.reverse()

    sel = build_selection(chosen)
    sel.dp_optimum = float(dp[B, R])  # must equal sel.value (checked in tests)
    return sel


def knapsack_table_bytes(n: int, budget_units: int, risk_units: int) -> int:
    """Bytes used by the boolean traceback table (one byte per boolean in NumPy)."""
    return int(n) * (int(budget_units) + 1) * (int(risk_units) + 1)
