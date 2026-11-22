"""Shared data structures used by both selection strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Item:
    """One candidate security, discretised to integer capacity units.

    cost   : ceil(price / lot_size)          -> budget units
    risk   : ceil(annual_vol * risk_scale)   -> risk units
    value  : expected one-year dollar gain of buying one share
    sharpe : (annual_return - rf) / annual_vol, used by the greedy ranking
    """

    name: str
    cost: int
    risk: int
    value: float
    sharpe: float


@dataclass
class Selection:
    """Result of a selection strategy."""

    items: list = field(default_factory=list)
    value: float = 0.0
    cost: int = 0
    risk: int = 0
    dp_optimum: Optional[float] = None  # DP table entry dp[B, R], knapsack only

    @property
    def names(self) -> list:
        return [it.name for it in self.items]


def build_selection(items) -> Selection:
    """Create a Selection and compute its totals from the chosen items."""
    items = list(items)
    return Selection(
        items=items,
        value=float(sum(it.value for it in items)),
        cost=int(sum(it.cost for it in items)),
        risk=int(sum(it.risk for it in items)),
    )


def is_feasible(sel: Selection, budget_units: int, risk_units: int) -> bool:
    """True when the selection respects both the budget and the risk ceiling."""
    return sel.cost <= budget_units and sel.risk <= risk_units
