import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from portfolio.data import TRADING_DAYS, synthetic_prices  # noqa: E402
from portfolio.features import asset_table, discretize  # noqa: E402


class TestFeatures(unittest.TestCase):
    def test_sharpe_formula_matches_definition(self):
        prices = synthetic_prices(n_assets=5, n_days=300, seed=1)
        t = asset_table(prices, rf_annual=0.02)
        rets = prices.pct_change().dropna()
        mu = rets.mean().to_numpy() * TRADING_DAYS
        vol = rets.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS)
        np.testing.assert_allclose(t["sharpe"].to_numpy(), (mu - 0.02) / vol)
        np.testing.assert_allclose(t["risk"].to_numpy(), vol)

    def test_no_look_ahead(self):
        """Changing prices after the window must not change the table."""
        prices = synthetic_prices(n_assets=4, n_days=400, seed=2)
        window = prices.iloc[:300]
        future = prices.copy()
        future.iloc[300:] *= 10.0
        pd.testing.assert_frame_equal(asset_table(window), asset_table(prices.iloc[:300]))
        pd.testing.assert_frame_equal(asset_table(window), asset_table(future.iloc[:300]))

    def test_discretize_rounds_up(self):
        t = pd.DataFrame({
            "expected_return": [0.10],
            "risk": [0.2534],
            "sharpe": [0.5],
            "price": [101.0],
        }, index=["X"])
        (it,) = discretize(t, lot_size=10.0, risk_scale=100)
        self.assertEqual(it.cost, 11)   # ceil(101 / 10)
        self.assertEqual(it.risk, 26)   # ceil(25.34)
        self.assertAlmostEqual(it.value, 0.10 * 101.0)


if __name__ == "__main__":
    unittest.main()
