"""Exhaustive search, used only as a correctness oracle on small instances."""
from __future__ import annotations

from .models import build_selection


def brute_force(items, budget_units: int, risk_units: int):
    """Try all 2^n subsets. Practical only for small n."""
    n = len(items)
    if n > 16:
        raise ValueError("brute_force is only intended for n <= 16")
    best_value, best_subset = -1.0, ()
    for mask in range(1 << n):
        cost = risk = 0
        value = 0.0
        subset = []
        for i in range(n):
            if mask >> i & 1:
                cost += items[i].cost
                risk += items[i].risk
                value += items[i].value
                subset.append(items[i])
        if cost <= budget_units and risk <= risk_units and value > best_value:
            best_value, best_subset = value, tuple(subset)
    return build_selection(best_subset)
