"""Web dashboard for the Greedy vs. 0/1 Knapsack portfolio project.

Run with:  streamlit run app.py
All algorithm logic lives in src/portfolio; this file only lays out inputs and charts.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from portfolio import service  # noqa: E402
from portfolio.benchmark import scale_with_n  # noqa: E402
from portfolio.models import Item  # noqa: E402

st.set_page_config(page_title="Portfolio & Algorithm Lab", page_icon="📊", layout="wide")

BENCH_CSV = os.path.join(ROOT, "results", "benchmark_n.csv")


# ------------------------------------------------------------ cached helpers

@st.cache_data(show_spinner="Loading prices...")
def cached_universe(n_assets, n_days, seed, csv_bytes):
    return service.load_universe(n_assets=n_assets, n_days=n_days, seed=seed, csv_source=csv_bytes)


@st.cache_data(show_spinner=False)
def cached_analysis(prices_key, prices_df, lookback, rf, lot, risk_scale):
    window = prices_df.iloc[-(lookback + 1):]
    return service.analyse(window, rf, lot, risk_scale)


@st.cache_data(show_spinner="Filling DP table stages...")
def cached_stages(items_key, items, B, R):
    return service.dp_stages(list(items), B, R)


@st.cache_data(show_spinner="Running backtest...")
def cached_backtest(prices_key, prices_df, budget, risk_ceiling, lookback, rebalance):
    return service.run_backtest(prices_df, budget, risk_ceiling, lookback, rebalance)


def money(x):
    return f"${x:,.2f}"


def pct(x):
    return f"{x:.1f}%"


# ------------------------------------------------------------ sidebar

st.sidebar.title("Settings")
source = st.sidebar.radio("Price data", ["Synthetic test data", "Upload CSV"], horizontal=False)
csv_bytes = None
if source == "Upload CSV":
    up = st.sidebar.file_uploader("Wide CSV: date column + one price column per asset", type=["csv"])
    if up is not None:
        csv_bytes = up.getvalue()
    else:
        st.sidebar.info("Upload a CSV, or switch to synthetic data.")

n_assets = st.sidebar.slider("Synthetic universe size", 5, 30, 12, disabled=csv_bytes is not None)
seed = st.sidebar.number_input("Random seed", min_value=1, max_value=10_000, value=3, disabled=csv_bytes is not None)
lookback = st.sidebar.slider("Lookback window (trading days)", 60, 504, 252)
budget = st.sidebar.number_input("Budget ($)", min_value=100.0, max_value=1_000_000.0, value=5000.0, step=500.0)
risk_ceiling = st.sidebar.slider("Risk ceiling (units; 100 = 1.00 total volatility)", 20, 400, 150)
rf = st.sidebar.number_input("Risk-free rate (annual)", min_value=0.0, max_value=0.2, value=0.02, step=0.005, format="%.3f")
st.sidebar.caption("Lot size is $10 per budget unit. Risk unit = 0.01 of annual volatility.")

LOT = service.DEFAULT_LOT
RISK_SCALE = service.DEFAULT_RISK_SCALE
B_UNITS = service.budget_to_units(budget, LOT)

# Synthetic universes use 10 years of daily data; uploaded CSVs are used as they are.
prices, label = cached_universe(n_assets, 2520, seed, csv_bytes)
prices_key = (label, prices.shape, str(prices.index[-1]))
if prices.shape[0] < lookback + 2:
    st.error(f"Only {prices.shape[0]} rows of data; the lookback needs at least {lookback + 2}. Lower the lookback.")
    st.stop()

table, items, dropped = cached_analysis(prices_key, prices, lookback, rf, LOT, RISK_SCALE)
items_key = tuple((it.name, it.cost, it.risk, round(it.value, 6), round(it.sharpe, 6)) for it in items)

# ------------------------------------------------------------ header

st.title("Portfolio & Algorithm Lab")
st.caption("Greedy ranking vs. exact 0/1 Knapsack dynamic programming on the same asset data.")
if csv_bytes is None:
    st.warning("Data is SYNTHETIC test data generated for this demo, not real market prices.")

if len(items) < 1:
    st.error("No usable assets in this window. Check the data or lengthen the lookback.")
    st.stop()

tab_overview, tab_compare, tab_dp, tab_backtest, tab_bench, tab_custom = st.tabs(
    ["Overview", "Algorithm comparison", "DP table explorer", "Backtest", "Benchmarks", "Custom knapsack"])

# ------------------------------------------------------------ overview

with tab_overview:
    latest = prices.index[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets loaded", prices.shape[1])
    c2.metric("Usable for optimization", len(items))
    c3.metric("Latest price date", str(pd.Timestamp(latest).date()))
    c4.metric("Window (trading days)", lookback)

    if dropped:
        with st.expander(f"{len(dropped)} asset(s) excluded"):
            st.dataframe(pd.DataFrame(dropped, columns=["asset", "reason"]), hide_index=True, width="stretch")

    st.subheader("Asset statistics (lookback window)")
    stats = table.rename(columns={"expected_return": "expected return", "risk": "volatility",
                                  "sharpe": "Sharpe", "price": "latest price"})
    st.dataframe(
        stats.style.format({"expected return": "{:.2%}", "volatility": "{:.2%}", "Sharpe": "{:.2f}",
                            "latest price": "${:,.2f}"}),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        fig = px.bar(stats.sort_values("Sharpe"), x="Sharpe", y=stats.sort_values("Sharpe").index,
                     orientation="h", title="Sharpe ratio by asset", color="Sharpe",
                     color_continuous_scale="RdYlGn")
        fig.update_layout(height=420, yaxis_title="", coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.scatter(stats, x="volatility", y="expected return", text=stats.index,
                         title="Risk vs. return (each point is one asset)")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, xaxis_tickformat=".0%", yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")

    window = prices.iloc[-(lookback + 1):]
    usable_cols = [it.name for it in items]
    corr = window[usable_cols].pct_change().dropna().corr()
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto",
                    title="Correlation of daily returns")
    fig.update_layout(height=460)
    st.plotly_chart(fig, width="stretch")
    st.caption("Correlation shows how assets move together. Greedy and the DP below use each asset's own "
               "risk and do not account for correlation (see Limitations in the README).")

# ------------------------------------------------------------ comparison

with tab_compare:
    cmp_ = service.compare(items, B_UNITS, risk_ceiling, repeats=3)
    g, k = cmp_.greedy, cmp_.knapsack

    st.subheader("Result on the current budget and risk limits")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Greedy expected gain", money(g.value))
    m2.metric("Knapsack (optimal) gain", money(k.value), delta=money(k.value - g.value))
    m3.metric("Greedy gap to optimum", pct(cmp_.gap_pct))
    m4.metric("Runtime greedy / DP", f"{cmp_.greedy_ms:.2f} / {cmp_.knapsack_ms:.2f} ms")
    st.info(" ".join(cmp_.notes) + " Both answers respect the budget and risk limits."
            if cmp_.both_feasible else "Warning: a solution exceeded a limit.")

    st.subheader("Budget and risk used")
    usage = pd.DataFrame({
        "strategy": ["Greedy", "Knapsack"] * 2,
        "resource": ["Budget used", "Budget used", "Risk used", "Risk used"],
        "fraction": [g.cost / B_UNITS, k.cost / B_UNITS, g.risk / risk_ceiling, k.risk / risk_ceiling],
    })
    fig = px.bar(usage, x="fraction", y="resource", color="strategy", barmode="group", orientation="h",
                 text=usage["fraction"].map(lambda v: f"{v:.0%}"))
    fig.update_layout(height=280, xaxis=dict(range=[0, 1.05], tickformat=".0%"), yaxis_title="")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**Greedy picks** (ranked by Sharpe ratio)")
        st.dataframe(pd.DataFrame({"asset": g.names, "expected gain ($)": [it.value for it in g.items]}),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("**Knapsack picks** (exact optimum)")
        st.dataframe(pd.DataFrame({"asset": k.names, "expected gain ($)": [it.value for it in k.items]}),
                     hide_index=True, width="stretch")

    st.subheader("Hypothetical purchase with the optimal selection")
    alloc = service.allocation(k, table, budget)
    if alloc.empty:
        st.warning("No asset fits the budget and risk limits. Increase the budget or risk ceiling.")
    else:
        st.dataframe(
            alloc.style.format({"share price": "${:,.2f}", "cost ($)": "${:,.2f}",
                                "expected gain ($)": "${:,.2f}", "weight (% of spend)": "{:.1f}%"}),
            hide_index=True, width="stretch")
        st.write(f"Spent {money(alloc.attrs['spent'])} of {money(budget)}; cash left {money(alloc.attrs['cash_left'])}.")
    st.caption("Each selected asset is bought as one whole share. Expected gains are estimates from past data, "
               "not forecasts. This is an educational simulation, not investment advice.")

# ------------------------------------------------------------ DP explorer

with tab_dp:
    st.subheader("How the DP table fills up")
    if len(items) > 15:
        st.info("The table explorer is limited to 15 assets so it stays responsive. Lower the universe size.")
    else:
        st.write("Each stage adds one asset. The heatmap shows the best total gain achievable for each "
                 "budget (horizontal) and risk (vertical) limit, using only the assets considered so far.")
        stage = st.slider("Stage (number of assets considered)", 0, len(items), len(items))
        stages = cached_stages(items_key, tuple(items), B_UNITS, risk_ceiling)
        dp = stages[stage]
        if stage == 0:
            st.write("Stage 0: no assets considered yet, so every cell is 0.")
        else:
            it = items[stage - 1]
            st.write(f"Considering **{it.name}**: cost {it.cost} units, risk {it.risk} units, "
                     f"expected gain {money(it.value)}.")
        fig = go.Figure(go.Heatmap(z=dp.T, x=np.arange(dp.shape[0]), y=np.arange(dp.shape[1]),
                                   colorscale="Viridis", colorbar=dict(title="gain ($)"),
                                   hovertemplate="budget %{x} units, risk %{y} units<br>gain $%{z:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[B_UNITS], y=[risk_ceiling], mode="markers",
                                 marker=dict(symbol="x", size=14, color="red"), name="your limits"))
        fig.update_layout(height=460, xaxis_title="budget units used at most",
                          yaxis_title="risk units used at most", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Red X marks your limits ({B_UNITS} budget units, {risk_ceiling} risk units). "
                   f"Its value at stage {len(items)} equals the knapsack optimum: {money(stages[-1][B_UNITS, risk_ceiling])}.")
        with st.expander("Recurrence used"):
            st.latex(r"dp_i[b][r] = \max\bigl(dp_{i-1}[b][r],\; dp_{i-1}[b-c_i][r-k_i] + v_i\bigr)")
            st.write("Each cell chooses between skipping the asset or taking it. The table is built from "
                     "the first asset to the last, and the chosen set is recovered by walking back from "
                     "the bottom-right cell.")

# ------------------------------------------------------------ backtest

with tab_backtest:
    st.subheader("Walk-forward backtest (quarterly rebalancing)")
    st.write("Each quarter, both methods pick assets using only the past year of data, then hold them for "
             "a quarter. Results are reported for the period after each decision.")
    need = lookback + 63 + 2
    if prices.shape[0] < need:
        st.info(f"Backtest needs at least {need} rows of data. Use a longer universe or lower the lookback.")
    elif st.button("Run backtest", type="primary"):
        results, summ = cached_backtest(prices_key, prices, budget, risk_ceiling, lookback, 63)
        eq = results.assign(wealth=lambda d: d.groupby("strategy")["period_return"]
                            .transform(lambda r: (1 + r).cumprod()))
        fig = px.line(eq, x="date", y="wealth", color="strategy", title="Growth of $1")
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
        fig.update_layout(height=420, yaxis_title="growth of $1")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(summ.style.format({
            "cumulative_return": "{:.1%}", "cagr": "{:.1%}", "ann_volatility": "{:.1%}",
            "ann_sharpe_like": "{:.2f}", "mean_est_value": "${:,.0f}", "mean_turnover": "{:.2f}",
            "mean_runtime_ms": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("Past performance does not predict future results. The backtest uses no data from after "
                   "each decision date. Transaction costs are not modeled in this version.")
    else:
        st.write("Press **Run backtest** to simulate the strategies on this data.")

# ------------------------------------------------------------ benchmarks

with tab_bench:
    st.subheader("Runtime vs. number of assets")
    st.write("Greedy grows roughly with n log n; the DP grows with n × budget × risk. "
             "Saved results come from the project's full benchmark run.")
    if os.path.exists(BENCH_CSV):
        bench = pd.read_csv(BENCH_CSV)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bench["n"], y=bench["greedy_ms"], mode="lines+markers", name="Greedy"))
        fig.add_trace(go.Scatter(x=bench["n"], y=bench["dp_ms"], mode="lines+markers", name="Knapsack DP"))
        fig.update_layout(height=420, xaxis_type="log", yaxis_type="log",
                          xaxis_title="assets (n)", yaxis_title="runtime (ms)")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(bench.style.format({"greedy_ms": "{:.3f}", "dp_ms": "{:.2f}", "greedy_value": "${:,.0f}",
                                         "dp_value": "${:,.0f}", "greedy_gap_pct": "{:.1f}%",
                                         "dp_table_mb": "{:.2f}"}), hide_index=True, width="stretch")
    else:
        st.info("No saved benchmark found. Run `python main.py benchmark` first.")

    if st.button("Run quick benchmark on this machine"):
        with st.spinner("Timing both algorithms..."):
            live = scale_with_n([10, 20, 40, 80], budget_units=B_UNITS, risk_units=risk_ceiling, repeats=1)
        st.dataframe(live.style.format({"greedy_ms": "{:.3f}", "dp_ms": "{:.2f}"}), hide_index=True,
                     width="stretch")
        st.caption("Measured on this computer; results vary between machines.")

# ------------------------------------------------------------ custom knapsack

with tab_custom:
    st.subheader("Knapsack decision support")
    st.write("Enter your own items. Costs and risks are whole units; value is the expected gain in dollars; "
             "score is used only to rank items for the greedy method (higher is better).")
    default = pd.DataFrame({
        "name": ["A", "B", "C", "D", "E"],
        "cost": [10, 5, 5, 3, 4],
        "risk": [1, 1, 1, 2, 3],
        "value": [10.0, 6.0, 6.0, 4.0, 5.0],
        "score": [1.0, 0.9, 0.9, 0.6, 0.5],
    })
    edited = st.data_editor(default, num_rows="dynamic", width="stretch", key="custom_items")
    c1, c2 = st.columns(2)
    cap_b = c1.number_input("Budget capacity (units)", min_value=1, max_value=500, value=10)
    cap_r = c2.number_input("Risk capacity (units)", min_value=1, max_value=400, value=5)

    try:
        rows = edited.dropna(subset=["name", "cost", "risk", "value", "score"])
        custom = [Item(str(r["name"]), int(r["cost"]), int(r["risk"]), float(r["value"]), float(r["score"]))
                  for _, r in rows.iterrows()]
        if any(c.cost <= 0 or c.risk <= 0 for c in custom):
            raise ValueError("cost and risk must be positive whole numbers")
        if len(custom) > 15:
            st.info("Please use at most 15 items for the table view.")
        elif custom:
            cc = service.compare(custom, int(cap_b), int(cap_r), repeats=1)
            r1, r2 = st.columns(2)
            r1.metric("Greedy value", money(cc.greedy.value), help="Ranks by score, then adds what fits")
            r2.metric("Optimal (DP) value", money(cc.knapsack.value), delta=money(cc.knapsack.value - cc.greedy.value))
            st.write("**Optimal selection:** " + (", ".join(cc.knapsack.names) or "nothing fits"))
            st.write("**Greedy selection:** " + (", ".join(cc.greedy.names) or "nothing fits"))
            custom_key = tuple((c.name, c.cost, c.risk, round(c.value, 6), round(c.sharpe, 6)) for c in custom)
            stages = cached_stages(custom_key, tuple(custom), int(cap_b), int(cap_r))
            dp = stages[-1]
            fig = go.Figure(go.Heatmap(z=dp.T, colorscale="Viridis", colorbar=dict(title="value")))
            fig.update_layout(height=340, xaxis_title="budget units used at most",
                              yaxis_title="risk units used at most")
            st.plotly_chart(fig, width="stretch")
    except (ValueError, TypeError) as exc:
        st.error(f"Check your items: {exc}")
