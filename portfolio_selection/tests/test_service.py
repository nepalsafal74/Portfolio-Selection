import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio import service  # noqa: E402
from portfolio.features import dropped_assets  # noqa: E402
from portfolio.models import Item  # noqa: E402


class TestService(unittest.TestCase):
    def setUp(self):
        self.prices, _ = service.load_universe(n_assets=10, n_days=253, seed=5)
        self.table, self.items, _ = service.analyse(self.prices)

    def test_compare_is_feasible_and_dp_not_worse(self):
        c = service.compare(self.items, 500, 150, repeats=1)
        self.assertTrue(c.both_feasible)
        self.assertGreaterEqual(c.knapsack.value + 1e-9, c.greedy.value)
        self.assertAlmostEqual(c.knapsack.dp_optimum, c.knapsack.value, places=6)

    def test_last_dp_stage_equals_full_table(self):
        stages = service.dp_stages(self.items, 200, 80)
        self.assertEqual(len(stages), len(self.items) + 1)
        self.assertAlmostEqual(stages[-1][200, 80], service.knapsack_2d(self.items, 200, 80).value)

    def test_allocation_totals(self):
        c = service.compare(self.items, 500, 150, repeats=1)
        alloc = service.allocation(c.knapsack, self.table, 5000.0)
        self.assertAlmostEqual(alloc.attrs["spent"], alloc["cost ($)"].sum())
        self.assertAlmostEqual(alloc.attrs["cash_left"], 5000.0 - alloc.attrs["spent"])
        if len(alloc):
            self.assertAlmostEqual(alloc["weight (% of spend)"].sum(), 100.0)

    def test_zero_volatility_asset_is_dropped_not_crashing(self):
        prices = self.prices.copy()
        prices["FLAT"] = 50.0
        table, items, dropped = service.analyse(prices)
        self.assertIn("FLAT", [name for name, _ in dropped])
        self.assertNotIn("FLAT", [it.name for it in items])
        service.compare(items, 500, 150, repeats=1)  # must not raise

    def test_dropped_reason_for_bad_price(self):
        from portfolio.features import asset_table
        prices = self.prices.iloc[:, :3].copy()
        prices["BAD"] = np.nan
        reasons = dict(dropped_assets(asset_table(prices)))
        self.assertEqual(reasons.get("BAD"), "missing or invalid price")


class TestUploadedCsv(unittest.TestCase):
    def test_csv_bytes_are_parsed(self):
        csv = b"date,AAA,BBB\n2024-01-02,10,20\n2024-01-03,11,19\n2024-01-04,12,21\n"
        prices, label = service.load_universe(csv_source=csv)
        self.assertEqual(list(prices.columns), ["AAA", "BBB"])
        self.assertEqual(len(prices), 3)
        self.assertIn("CSV file", label)

    def test_synthetic_label_is_honest(self):
        _, label = service.load_universe(n_assets=3, n_days=50)
        self.assertIn("SYNTHETIC", label)


class TestNegativeValueGreedy(unittest.TestCase):
    def test_greedy_never_buys_losing_asset(self):
        from portfolio.greedy import greedy_select
        items = [Item("neg", 1, 1, -5.0, -0.5), Item("ok", 1, 1, 1.0, 0.1)]
        self.assertEqual(greedy_select(items, 10, 10).names, ["ok"])


if __name__ == "__main__":
    unittest.main()
