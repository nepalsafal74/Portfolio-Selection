"""Greedy selection: rank candidates once, then add each one that still fits.

Two entry points:
  greedy_select  - batch ranking, O(n log n)
  StreamingTopK  - online top-k maintenance with a min-heap, O(log k) per update

Assets with non-positive expected dollar gain are never selected. This matches the
knapsack objective, so both algorithms solve the same problem and can be compared fairly.
"""
from __future__ import annotations

import heapq

from .models import build_selection

RANK_KEYS = ("sharpe", "density")


def greedy_select(items, budget_units: int, risk_units: int, key: str = "sharpe"):
    """Greedy 0/1 selection.

    key="sharpe"  : rank by risk-adjusted return (the rule specified in the project brief)
    key="density" : rank by value per budget unit (used only for comparison in the report)
    """
    if key == "sharpe":
        order = sorted(items, key=lambda it: it.sharpe, reverse=True)
    elif key == "density":
        order = sorted(items, key=lambda it: it.value / it.cost, reverse=True)
    else:
        raise ValueError(f"unknown key {key!r}; choose from {RANK_KEYS}")

    chosen, cost_used, risk_used = [], 0, 0
    for it in order:
        if it.value <= 0:          # never buy an asset that loses expected money
            continue
        if cost_used + it.cost <= budget_units and risk_used + it.risk <= risk_units:
            chosen.append(it)
            cost_used += it.cost
            risk_used += it.risk
    return build_selection(chosen)


class StreamingTopK:
    """Keep the k best candidates seen so far in a live signal stream."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self._heap: list = []  # min-heap of (sharpe, name); root is the weakest kept item

    def update(self, name: str, sharpe: float) -> None:
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (sharpe, name))
        elif sharpe > self._heap[0][0]:
            heapq.heapreplace(self._heap, (sharpe, name))

    def top(self) -> list:
        """Current top-k as (sharpe, name) pairs, best first."""
        return sorted(self._heap, reverse=True)
