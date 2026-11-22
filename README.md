# Portfolio-Selection# 
Greedy vs. 0/1 Knapsack: Portfolio Selection Lab

**CSCE 4110 - Algorithms, Semester Project**
Author: Safal Nepal (individual project)
Demonstration video: https://drive.google.com/file/d/1BF6-K0QWCe5O88BAiZ7uGUGgNJR9TixS/view?usp=sharing
Report: `report/report.pdf`

## Project description

Investors and automated trading systems often must choose a few assets under a budget
limit and a risk limit. This project implements and compares two algorithms for that choice:

* **Greedy ranking** sorts assets by Sharpe ratio and adds each one that still fits. It runs
  in O(n log n) and is suitable for real-time filtering.
* **0/1 Knapsack dynamic programming** solves the same problem exactly over budget and risk
  at the same time. It runs in O(n x B x R).

The project includes a command-line tool, a web dashboard with interactive charts, a walk-forward
backtest, benchmarks, and automated tests. The dashboard lets you see the DP table fill up
stage by stage, which is the algorithm the course focuses on.

## Repository layout

```
app.py                    Streamlit web dashboard (run this for the interactive view)
main.py                   Command-line tool: demo | benchmark | backtest | sweep | all
requirements.txt
src/portfolio/
    models.py             Item and Selection data classes, feasibility check
    data.py               synthetic price generator, CSV loader
    features.py           asset statistics, validity checks, discretisation to integer items
    greedy.py             greedy selection and StreamingTopK (min-heap)
    knapsack.py           exact 2D 0/1 knapsack DP with traceback
    brute.py              exhaustive search (correctness oracle for small n)
    backtest.py           walk-forward rebalancing and summary statistics
    benchmark.py          runtime, quality and budget-scaling experiments
    plots.py              static figures for the report
    service.py            shared application layer used by app.py and main.py
tests/                    unit tests (unittest, no extra test dependency)
results/                  generated CSV files and figures from the full experiments
report/                   project report (Markdown source and PDF)
docs/demo_script.md       outline for the demonstration video
CONTRIBUTIONS.md          contribution statement by module and task
```

## Setup (Windows PowerShell)

Requires Python 3.10 or newer.

```powershell
cd portfolio_selection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux use `python3 -m venv .venv` and `source .venv/bin/activate`.

## Running the project

Web dashboard (interactive, recommended):

```powershell
streamlit run app.py
```

Your browser opens at http://localhost:8501. The dashboard has six tabs:

1. **Overview:** asset statistics, Sharpe ratios, risk vs. return, and correlations.
2. **Algorithm comparison:** greedy vs. optimal results on your budget and risk limits, with a
   hypothetical purchase table.
3. **DP table explorer:** the DP table filling stage by stage, with a heatmap.
4. **Backtest:** quarterly rebalancing of both methods on past data.
5. **Benchmarks:** saved runtime results and a quick timing run on your computer.
6. **Custom knapsack:** enter your own items and capacities, then compare the two methods.

The sidebar controls the data source (synthetic test data or your own CSV upload), budget,
risk ceiling, lookback window, and risk-free rate.

Command line:

```powershell
python main.py demo                  # 12-asset example, DP checked against brute force
python main.py benchmark             # runtime and quality experiments -> results/
python main.py backtest              # walk-forward backtest on synthetic data
python main.py sweep --seeds 20      # paired comparison over 20 synthetic universes
python -m unittest discover -s tests -t . -v   # run the 22 tests
```

### CSV format for your own prices

A wide CSV: the first column is the date, and each other column is one asset's daily price.
Missing values are forward-filled.

```
date,AAPL,MSFT,SPY
2020-01-02,75.09,158.78,324.87
2020-01-03,74.36,156.51,322.41
```

## Example output

`python main.py demo` (abridged):

```
Greedy   : value $  819.73  cost 151/500  risk 142/150  picks ['A002', 'A006', 'A008', 'A005', 'A000']
Knapsack : value $  883.37  cost 182/500  risk 146/150  picks ['A002', 'A005', 'A006', 'A007', 'A010']
Brute    : value $  883.37  (exhaustive check of all 2^12 subsets)
DP matches brute force and its traceback: True
```

Using the algorithms directly in Python:

```python
from portfolio.service import load_universe, analyse, compare, budget_to_units

prices, label = load_universe(n_assets=12, n_days=253, seed=3)
table, items, dropped = analyse(prices)
result = compare(items, budget_units=budget_to_units(5000), risk_units=150)
print(result.greedy.names, result.knapsack.names)
```

## Modelling assumptions

* Each asset is a 0/1 decision: buy one whole share or nothing.
* The budget is split into $10 units, and each asset's cost is rounded up.
* Risk is the sum of annualised volatilities, in units of 0.01. This additive measure ignores
  correlation (see Limitations).
* The DP maximises expected one-year dollar gain, price x expected return. Greedy ranks by
  Sharpe ratio, but it never buys an asset whose expected gain is negative.
* Assets with missing prices or zero volatility are excluded, and the dashboard lists them.

## Limitations

* The risk model is additive. A covariance-based constraint would need a mixed-integer solver,
  which is listed as future work.
* Synthetic data is generated by a one-factor model with known parameters. The dashboard labels
  it clearly. Real-market results are only as good as the CSV you provide.
* Sharpe and return estimates from a single year are noisy, and backtests describe past behavior
  only.
* Transaction costs are not modeled in the backtest.

## Academic integrity

The implementation, tests, and report are the author's own work. Libraries used: NumPy, pandas,
SciPy, matplotlib, Streamlit, and Plotly. Background references are listed in the report.
