import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio.brute import brute_force  # noqa: E402
from portfolio.greedy import greedy_select  # noqa: E402
from portfolio.knapsack import knapsack_2d  # noqa: E402
from portfolio.models import Item, is_feasible  # noqa: E402


def item(name, cost, risk, value, sharpe=0.0):
    return Item(name=name, cost=cost, risk=risk, value=value, sharpe=sharpe)


class TestKnapsack(unittest.TestCase):
    def test_counterexample_from_brief(self):
        items = [
            item("A", cost=10, risk=1, value=10.0, sharpe=1.0),
            item("B", cost=5, risk=1, value=6.0, sharpe=0.9),
            item("C", cost=5, risk=1, value=6.0, sharpe=0.9),
        ]
        sel = knapsack_2d(items, budget_units=10, risk_units=10)
        self.assertEqual(sorted(sel.names), ["B", "C"])
        self.assertAlmostEqual(sel.value, 12.0)
        self.assertAlmostEqual(sel.dp_optimum, 12.0)

    def test_risk_constraint_binds(self):
        """Budget is ample; the risk ceiling alone must force the choice."""
        items = [
            item("X", cost=1, risk=6, value=7.0),
            item("Y", cost=1, risk=5, value=5.0),
            item("Z", cost=1, risk=5, value=5.0),
        ]
        sel = knapsack_2d(items, budget_units=100, risk_units=10)
        self.assertEqual(sorted(sel.names), ["Y", "Z"])
        self.assertAlmostEqual(sel.value, 10.0)

    def test_matches_brute_force_on_random_instances(self):
        rng = random.Random(42)
        for trial in range(300):
            n = rng.randint(1, 10)
            items = [item(f"i{k}", rng.randint(1, 12), rng.randint(1, 12), rng.uniform(0, 20))
                     for k in range(n)]
            B, R = rng.randint(1, 40), rng.randint(1, 40)
            dp = knapsack_2d(items, B, R)
            bf = brute_force(items, B, R)
            self.assertAlmostEqual(dp.value, bf.value, places=9, msg=f"trial {trial}")
            self.assertAlmostEqual(dp.dp_optimum, bf.value, places=9, msg=f"trial {trial}")
            self.assertTrue(is_feasible(dp, B, R), msg=f"trial {trial}")

    def test_dp_never_worse_than_greedy(self):
        rng = random.Random(7)
        for _ in range(200):
            n = rng.randint(1, 12)
            items = [item(f"i{k}", rng.randint(1, 15), rng.randint(1, 15),
                          rng.uniform(0, 20), rng.uniform(-1, 3)) for k in range(n)]
            B, R = rng.randint(1, 60), rng.randint(1, 60)
            self.assertGreaterEqual(knapsack_2d(items, B, R).value + 1e-9,
                                    greedy_select(items, B, R).value)

    def test_edge_cases(self):
        self.assertEqual(knapsack_2d([], 10, 10).items, [])
        big = [item("huge", cost=50, risk=50, value=100.0)]
        self.assertEqual(knapsack_2d(big, 10, 10).items, [])
        self.assertRaises(ValueError, knapsack_2d, [item("bad", 0, 1, 1.0)], 5, 5)
        self.assertRaises(ValueError, knapsack_2d, [], -1, 5)

    def test_traceback_value_equals_table_value(self):
        rng = np.random.default_rng(3)
        items = [item(f"i{k}", int(rng.integers(1, 30)), int(rng.integers(1, 30)),
                      float(rng.uniform(0, 50))) for k in range(40)]
        sel = knapsack_2d(items, 200, 150)
        self.assertAlmostEqual(sel.value, sel.dp_optimum, places=6)
        self.assertLessEqual(sel.cost, 200)
        self.assertLessEqual(sel.risk, 150)


if __name__ == "__main__":
    unittest.main()
