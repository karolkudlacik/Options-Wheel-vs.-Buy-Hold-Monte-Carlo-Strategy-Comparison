"""
Wheel Strategy vs Buy & Hold — Monte Carlo (Streamlit app)
==========================================================
Thin UI layer. All financial/mathematical logic lives in ``simulation.py``;
this file only collects inputs, orchestrates the cached run, and displays
the results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import simulation as sim
from simulation import INITIAL_CAPITAL

st.set_page_config(page_title="Wheel Strategy Monte Carlo", layout="wide")

# Scenario presets: (mu, sigma are overwritten for non-Custom; days is fixed per scenario)
PRESETS = {
    "Bull Market":     dict(mu=0.15,  sigma=0.12, days=1022),
    "Bear Market":     dict(mu=-0.20, sigma=0.35, days=322),
    "Sideways Market": dict(mu=0.03,  sigma=0.18, days=272),
    "Custom":          dict(mu=0.15,  sigma=0.12, days=504),  # 2-year default horizon
}


# ─────────────────────────────────────────────────────────────
# Cached orchestration — runs the full Monte Carlo for ONE scenario.
# Every parameter is an argument so the cache invalidates correctly.
# Determinism is enforced inside sim.run_monte_carlo (np.random.seed(42)).
# ─────────────────────────────────────────────────────────────
@st.cache_data
def run_simulation(mu, sigma, vrp, target_delta, days, num_paths,
                   skew_slope, smile_curvature, seed=42):
    prices = sim.run_monte_carlo(100, mu, sigma, days, num_paths)

    wheel_results, bh_results = [], []
    wheel_histories, bh_histories = [], []

    for i in range(num_paths):
        iv = sim.calculate_iv(prices[:, i], vrp)
        h_w, wv = sim.simulate_wheel(prices[:, i], iv_path=iv, target_delta=target_delta,
                                     skew_slope=skew_slope,
                                     smile_curvature=smile_curvature)
        h_b, bv = sim.simulate_bh(prices[:, i])
        wheel_results.append(wv);   wheel_histories.append(h_w)
        bh_results.append(bv);      bh_histories.append(h_b)

    return (np.array(wheel_results), np.array(bh_results),
            wheel_histories, bh_histories)


# ─────────────────────────────────────────────────────────────
# Sidebar controls
# (the preset selector comes first so it can drive the μ/σ sliders)
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Simulation Parameters")

scenario = st.sidebar.selectbox(
    "Scenario Preset",
    ["Bull Market", "Bear Market", "Sideways Market", "Custom"],
)

is_custom = scenario == "Custom"
preset = PRESETS[scenario]

mu = st.sidebar.slider(
    "Annual Drift (μ)", -0.40, 0.40,
    value=float(preset["mu"]), step=0.01, disabled=not is_custom,
)
sigma = st.sidebar.slider(
    "Annual Volatility (σ)", 0.05, 0.60,
    value=float(preset["sigma"]), step=0.01, disabled=not is_custom,
)
vrp = st.sidebar.slider(
    "Volatility Risk Premium", 0.00, 0.10,
    value=0.02, step=0.005,
)
target_delta = st.sidebar.slider(
    "Option Delta", 0.10, 0.50,
    value=0.30, step=0.05,
)

st.sidebar.subheader("Volatility Skew")
use_skew = st.sidebar.checkbox(
    "Enable volatility skew", value=True,
    help="When off, all strikes are priced at a single flat volatility "
         "(the original Black-Scholes assumption).",
)
skew_slope = st.sidebar.slider(
    "Skew Slope", 0.00, 1.00,
    value=0.40, step=0.05, disabled=not use_skew,
    help="Linear term of σ(K) = σ_ATM − slope·ln(K/S) + curvature·ln(K/S)². "
         "A positive slope makes OTM puts richer than equidistant OTM calls — "
         "the structural skew of equity index options.",
)
smile_curvature = st.sidebar.slider(
    "Smile Curvature", 0.00, 3.00,
    value=0.50, step=0.25, disabled=not use_skew,
    help="Quadratic term lifting both wings of the smile.",
)
if not use_skew:
    skew_slope, smile_curvature = 0.0, 0.0

days = preset["days"]

num_paths = st.sidebar.number_input(
    "Number of Paths", min_value=1000, max_value=10000,
    value=2000, step=500,
)

run = st.sidebar.button("Run Simulation", type="primary", width="stretch")


# ─────────────────────────────────────────────────────────────
# Run only on click; persist results across reruns
# ─────────────────────────────────────────────────────────────
st.title("Wheel Strategy vs Buy & Hold — Monte Carlo")


# ─────────────────────────────────────────────────────────────
# Section 0 — Implied volatility smile (pricing layer)
# A pure function of the sidebar parameters, so it renders live,
# before (and independently of) any simulation run.
# ─────────────────────────────────────────────────────────────
st.subheader("Implied Volatility Smile")

sigma_atm = sigma + vrp   # representative ATM vol: scenario vol + risk premium
moneyness = np.linspace(0.80, 1.20, 81)
iv_curve  = np.array([
    sim.skewed_vol(sigma_atm, 1.0, m, skew_slope, smile_curvature)
    for m in moneyness
])

fig_smile = go.Figure()
fig_smile.add_trace(go.Scatter(
    x=moneyness, y=iv_curve * 100, mode="lines", name="Skew-adjusted IV",
    line=dict(color="#2563EB", width=2.5),
))
fig_smile.add_trace(go.Scatter(
    x=moneyness, y=np.full_like(moneyness, sigma_atm * 100), mode="lines",
    name="Flat vol (Black-Scholes)",
    line=dict(color="gray", width=1.5, dash="dash"),
))
fig_smile.add_vline(
    x=1.0, line_dash="dot", line_color="black",
    annotation_text="ATM", annotation_position="top",
)
fig_smile.update_layout(
    xaxis_title="Moneyness (K / S)",
    yaxis_title="Implied Volatility (%)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
    height=350,
)
st.plotly_chart(fig_smile, width="stretch")
st.caption(
    "Static skew in log-moneyness: σ(K) = σ_ATM − slope·ln(K/S) + curvature·ln(K/S)². "
    "OTM puts (K < S) trade above ATM vol, matching the structural skew of equity index "
    "options. Parameters are stylized, not calibrated to market data, and the skew affects "
    "only the option-pricing layer — price paths are still generated at a single flat "
    "volatility. The curve reflects the current sidebar settings (ATM vol = scenario σ + VRP)."
)

if run:
    with st.spinner(f"Running {num_paths:,} paths over {days} trading days…"):
        st.session_state["results"] = run_simulation(
            mu, sigma, vrp, target_delta, days, num_paths,
            skew_slope, smile_curvature,
        )
        st.session_state["years"] = days / 252

if "results" not in st.session_state:
    st.info("Configure parameters in the sidebar and click **Run Simulation** to begin.")
    st.stop()

wheel_results, bh_results, wheel_histories, bh_histories = st.session_state["results"]
years = st.session_state["years"]

m_wheel = sim.compute_metrics(wheel_results, wheel_histories, years)
m_bh    = sim.compute_metrics(bh_results, bh_histories, years)
p_beat  = float((wheel_results > bh_results).mean())


# ─────────────────────────────────────────────────────────────
# Section 1 — Metrics
# ─────────────────────────────────────────────────────────────
st.subheader("Performance Metrics")

c1, c2, c3 = st.columns(3)
c1.metric("CAGR (Wheel)",     f"{m_wheel['cagr']:.2%}")
c2.metric("Sharpe (Wheel)",   f"{m_wheel['sharpe']:.3f}")
c3.metric("P(Wheel > B&H)",   f"{p_beat:.2%}")

money = "{:,.0f}".format
pct   = "{:.2%}".format
ratio = "{:.3f}".format

table = pd.DataFrame(
    [
        ["Mean Final Value ($)",   money(m_wheel["mean"]),    money(m_bh["mean"])],
        ["Median Final Value ($)", money(m_wheel["median"]),  money(m_bh["median"])],
        ["CAGR",                   pct(m_wheel["cagr"]),      pct(m_bh["cagr"])],
        ["Sharpe Ratio",           ratio(m_wheel["sharpe"]),  ratio(m_bh["sharpe"])],
        ["Sortino Ratio",          ratio(m_wheel["sortino"]), ratio(m_bh["sortino"])],
        ["Avg Max Drawdown",       pct(m_wheel["mdd"]),       pct(m_bh["mdd"])],
        ["CVaR 95%",               pct(m_wheel["cvar_95"]),   pct(m_bh["cvar_95"])],
        ["P(loss)",                pct(m_wheel["p_loss"]),    pct(m_bh["p_loss"])],
        ["P(Wheel > B&H)",         pct(p_beat),               "—"],
    ],
    columns=["Metric", "Wheel", "Buy & Hold"],
)
st.dataframe(table, hide_index=True, width="stretch")


# Shared x-axis window: from the joint minimum to the 99th percentile
# (clips long right tails, e.g. Buy & Hold in a bull market).
x_min = float(min(wheel_results.min(), bh_results.min()))
x_max = float(np.percentile(np.concatenate([wheel_results, bh_results]), 99))
bin_size = (x_max - x_min) / 60   # 60 bins across the clipped range

WHEEL_COLOR = "#2563EB"
BH_COLOR    = "#93C5FD"


# ─────────────────────────────────────────────────────────────
# Section 2 — Distribution of final portfolio values
# ─────────────────────────────────────────────────────────────
st.subheader("Distribution of Final Portfolio Values")

fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(
    x=bh_results, name="Buy & Hold", marker_color=BH_COLOR, opacity=0.7,
    xbins=dict(start=x_min, end=x_max, size=bin_size),
))
fig_hist.add_trace(go.Histogram(
    x=wheel_results, name="Wheel", marker_color=WHEEL_COLOR, opacity=0.7,
    xbins=dict(start=x_min, end=x_max, size=bin_size),
))
fig_hist.add_vline(
    x=INITIAL_CAPITAL, line_dash="dash", line_color="black",
    annotation_text="Initial Capital", annotation_position="top",
)
fig_hist.update_layout(
    barmode="overlay",
    xaxis_title="Final Portfolio Value ($)",
    yaxis_title="Number of Paths",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
)
fig_hist.update_xaxes(range=[x_min, x_max])
st.plotly_chart(fig_hist, width="stretch")


# ─────────────────────────────────────────────────────────────
# Section 3 — Empirical CDFs
# ─────────────────────────────────────────────────────────────
st.subheader("Empirical Cumulative Distribution")

fig_cdf = go.Figure()
for values, label, color in [
    (bh_results, "Buy & Hold", BH_COLOR),
    (wheel_results, "Wheel", WHEEL_COLOR),
]:
    sorted_fv = np.sort(values)
    cdf = np.arange(1, len(values) + 1) / len(values)
    fig_cdf.add_trace(go.Scatter(
        x=sorted_fv, y=cdf, mode="lines", name=label,
        line=dict(color=color, width=2),
    ))
fig_cdf.add_vline(
    x=INITIAL_CAPITAL, line_dash="dash", line_color="black",
    annotation_text="Initial Capital", annotation_position="top",
)
fig_cdf.update_layout(
    xaxis_title="Final Portfolio Value ($)",
    yaxis_title="Cumulative Probability",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
)
fig_cdf.update_xaxes(range=[x_min, x_max])
st.plotly_chart(fig_cdf, width="stretch")


# ─────────────────────────────────────────────────────────────
# Section 4 — Portfolio value over time
# Median path plus the 10th–90th percentile band for each strategy,
# computed across all Monte Carlo paths at every trading day.
# ─────────────────────────────────────────────────────────────
st.subheader("Portfolio Value Over Time")

wheel_matrix = np.vstack(wheel_histories)   # shape: (paths, days)
bh_matrix    = np.vstack(bh_histories)
t_axis       = np.arange(wheel_matrix.shape[1]) + sim.START_DAY

fig_time = go.Figure()
for matrix, label, line_color, band_color in [
    (bh_matrix,    "Buy & Hold", BH_COLOR,    "rgba(147, 197, 253, 0.25)"),
    (wheel_matrix, "Wheel",      WHEEL_COLOR, "rgba(37, 99, 235, 0.15)"),
]:
    p10    = np.percentile(matrix, 10, axis=0)
    p90    = np.percentile(matrix, 90, axis=0)
    median = np.median(matrix, axis=0)

    fig_time.add_trace(go.Scatter(
        x=t_axis, y=p90, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig_time.add_trace(go.Scatter(
        x=t_axis, y=p10, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor=band_color, name=f"{label} 10–90%",
    ))
    fig_time.add_trace(go.Scatter(
        x=t_axis, y=median, mode="lines", name=f"{label} median",
        line=dict(color=line_color, width=2.5),
    ))

fig_time.add_hline(
    y=INITIAL_CAPITAL, line_dash="dash", line_color="black",
    annotation_text="Initial Capital", annotation_position="bottom right",
)
fig_time.update_layout(
    xaxis_title="Trading Day",
    yaxis_title="Portfolio Value ($)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(t=40, b=10),
)
st.plotly_chart(fig_time, width="stretch")
