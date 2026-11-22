import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio.greedy import StreamingTopK, greedy_select  # noqa: E402
from portfolio.models import Item, is_feasible  # noqa: E402


def item(name, cost, risk, value, sharpe):
    return Item(name=name, cost=cost, risk=risk, value=value, sharpe=sharpe)


class TestGreedy(unittest.TestCase):
    def test_counterexample_from_brief(self):
        """Greedy by Sharpe takes A (value 10) and blocks B+C (value 12)."""
        items = [
            item("A", cost=10, risk=1, value=10.0, sharpe=1.0),
            item("B", cost=5, risk=1, value=6.0, sharpe=0.9),
            item("C", cost=5, risk=1, value=6.0, sharpe=0.9),
        ]
        sel = greedy_select(items, budget_units=10, risk_units=10)
        self.assertEqual(sel.names, ["A"])
        self.assertAlmostEqual(sel.value, 10.0)

    def test_respects_both_constraints(self):
        rng = random.Random(0)
        for _ in range(200):
            items = [item(f"i{k}", rng.randint(1, 20), rng.randint(1, 20),
                          rng.uniform(0, 10), rng.uniform(-1, 2)) for k in range(15)]
            B, R = rng.randint(5, 80), rng.randint(5, 80)
            sel = greedy_select(items, B, R)
            self.assertTrue(is_feasible(sel, B, R))

    def test_order_is_by_sharpe(self):
        items = [item("low", 1, 1, 1.0, 0.1), item("high", 1, 1, 1.0, 2.0)]
        sel = greedy_select(items, budget_units=1, risk_units=1)
        self.assertEqual(sel.names, ["high"])

    def test_density_key(self):
        items = [item("big", 10, 1, 10.0, 1.0), item("small", 2, 1, 5.0, 0.5)]
        self.assertEqual(greedy_select(items, 10, 10, key="density").names, ["small"])
        self.assertRaises(ValueError, greedy_select, items, 10, 10, "bogus")

    def test_streaming_topk_matches_sort(self):
        rng = random.Random(1)
        values = [(rng.uniform(-1, 3), f"s{i}") for i in range(500)]
        tk = StreamingTopK(k=7)
        for s, name in values:
            tk.update(name, s)
        expected = sorted(values, reverse=True)[:7]
        self.assertEqual(tk.top(), expected)


if __name__ == "__main__":
    unittest.main()
