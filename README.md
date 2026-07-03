# The Options Wheel vs. Buy & Hold — A Monte Carlo Study Across Market Regimes

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-vectorised-013243?logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Black--Scholes-8CAAE6?logo=scipy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Paths](https://img.shields.io/badge/Monte%20Carlo-10%2C000%20paths%20%C3%97%203%20regimes-555)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wheel-strategy-montecarlo.streamlit.app/)

A simulation study that asks a simple question: **does systematically selling option
premium (the "Wheel") actually beat just holding the asset?** Both strategies are run on
the *same* 10,000 simulated price paths in each of three market regimes — a bull, a bear,
and a sideways market — so the comparison is paired path-by-path rather than a comparison
of two separate experiments.

The short answer is nuanced, and that nuance is the point. The Wheel does **not** maximise
expected wealth. What it does, across every regime tested, is reshape the *distribution* of
outcomes — shallower drawdowns, a thinner left tail, and a higher risk-adjusted return —
at the cost of giving up the right tail in a strong bull market. It is best read as a tool
for **managing volatility**, not for chasing return.

> This project was presented at the **14th Kraków Conference on Financial Mathematics**
> (*XIV Krakowska Konferencja Matematyki Finansowej*), 9 May 2026.

**[▶ Live app](https://wheel-strategy-montecarlo.streamlit.app/)** · [Conference paper — PDF (Polish)](Options_strategy_wheel_vs_Buy_paper_from_kkmf.pdf) · [Research notebook — Jupyter (Polish)](PL_wheel_strategy_risk_analysis_mc.ipynb)

---

## Table of contents

- [Motivation](#motivation)
- [What the Wheel is](#what-the-wheel-is)
- [Methodology](#methodology)
- [Performance and risk metrics](#performance-and-risk-metrics)
- [Results](#results)
- [Key findings](#key-findings)
- [Assumptions and limitations](#assumptions-and-limitations)
- [Repository structure](#repository-structure)
- [Interactive app](#interactive-app)
- [Reproducing the study](#reproducing-the-study)
- [Reading the results responsibly](#reading-the-results-responsibly)

---

## Motivation

Selling options is often marketed as a way to "generate income" from a portfolio, with the
implicit promise of higher returns and lower risk at the same time. That framing is too
good to be true, and the interesting question is what the trade-off *actually* looks like.

Selling a put or a call hands away the tails of the return distribution in exchange for a
premium received up front. A premium seller is, in effect, short volatility: the position
profits when realised moves are small and loses when they are large. There is a well-known
structural reason a seller can be compensated for this — the **volatility risk premium
(VRP)**, the empirical tendency for options to be priced at an implied volatility slightly
above the volatility that subsequently realises. Buyers pay up for insurance; sellers
collect the spread.

This study isolates that mechanic in a controlled, simulated environment and measures its
consequences. **Buy & Hold** is the benchmark — buy the asset on day one, hold to the end.
The **Wheel** is the systematic premium-selling strategy described below. The same
simulated price paths feed both, and the question is not only *which earns more on average*
but *how the whole distribution of outcomes differs*, regime by regime.

---

## What the Wheel is

The Wheel is a cyclical, rules-based strategy that alternates between selling cash-secured
puts and selling covered calls. One full turn of the wheel:

1. **Sell a cash-secured put.** Collect the premium. Hold enough cash to buy the shares if
   assigned.
2. **Get assigned (if the put expires in-the-money).** Buy 100 shares per contract at the
   strike — i.e. acquire the asset at a discount to the price when the put was sold, net of
   premium.
3. **Sell a covered call against the shares.** Collect more premium, which keeps lowering
   the effective cost basis.
4. **Get called away (if the call expires in-the-money).** Sell the shares at the strike,
   booking the gain, and return to step 1.

If a sold option expires out-of-the-money, no shares change hands and the same leg is simply
re-sold — premium keeps accruing while the position waits. The decision rules used here are
deliberately mechanical, so that the simulation tests the *strategy* and not a trader's
discretion:

- **Strike selection by delta.** Each option is written at an absolute delta of
  **|Δ| = 0.30** — roughly a 30%-ish chance of finishing in-the-money under the model. This
  fixes how aggressively the strategy reaches for premium versus how often it gets assigned.
- **Tenor.** Every option is **21 observation steps** to expiry, then rolled.
- **Cost-basis protection on the call leg.** When selling the covered call, the strike is
  set to $K = \max(K_{\text{assigned}}, K_\Delta)$ — never below the price at which the shares were
  acquired. This stops the strategy from being forced to sell its stock at a loss just to
  collect a call premium.

---

## Methodology

### Market model: geometric Brownian motion

Each price path is a geometric Brownian motion (GBM). The asset follows

$$
dS_t = \mu S_t \, dt + \sigma S_t \, dW_t
$$

which is simulated with the exact log-Euler discretisation (no discretisation bias):

$$
S_{t+\Delta t} = S_t \cdot \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right)\Delta t + \sigma\sqrt{\Delta t} \cdot Z\right], \quad Z \sim \mathcal{N}(0,1)
$$

with $\Delta t = 1/252$ (252 trading days per year). Paths are generated in a fully vectorised
way: a matrix of standard-normal shocks is drawn once and turned into prices with a single
cumulative product, so a 10,000-path scenario is one NumPy operation rather than a Python
loop.

![Simulated price paths for the three regimes](path_simulations.png)

*A sample of GBM paths in each regime, with the median and the 10th–90th percentile band.
The bull market drifts up with a fanning cone of outcomes; the bear market drifts down under
high volatility; the sideways market stays anchored near its starting level.*

### The three regimes

The three scenarios are **stylised** — they are not calibrated to one specific ETF, but
chosen to span qualitatively different environments and stress the strategy in each.

| Regime          | Drift μ | Volatility σ | Horizon (active trading days) |
| --------------- | :-----: | :----------: | :---------------------------: |
| **Bull**        |  +15%   |     12%      |        ≈ 1000 (≈ 4.0 yr)      |
| **Bear**        |  −20%   |     35%      |        ≈ 300  (≈ 1.2 yr)      |
| **Sideways**    |   +3%   |     18%      |        ≈ 250  (≈ 1.0 yr)      |

### Monte Carlo design

- **10,000 paths per regime.**
- **Starting capital: \$100,000.** Buy & Hold deploys it all on the first active day; the
  Wheel uses it as collateral for cash-secured puts.
- **Risk-free rate r = 4%.** Idle cash in the Wheel earns this rate, compounded daily, while
  positions are open — so the comparison is not unfairly tilted by leaving the Wheel's cash
  inert.
- **A 22-step warm-up.** Both strategies start trading on day 22. The warm-up exists so the
  trailing-volatility estimator (below) is populated with history before the first option is
  written.
- **Paired comparison.** Every path is fed to *both* strategies. This is what makes
  `P(Wheel > Buy & Hold)` meaningful: it is the fraction of identical-market scenarios in
  which the Wheel's terminal wealth beats Buy & Hold's, not a comparison of two unrelated
  distributions.
- **Reproducible.** A fixed random seed makes every figure and number in this README
  exactly reproducible.

### Option pricing and the analytical strike

Options are priced with the **Black–Scholes** model for European puts and calls. Rather than
searching numerically for the strike that yields a 0.30-delta option, the code **inverts the
delta in closed form**. Because a call's delta is $N(d_1)$, a target delta pins down $d_1$
directly, and the strike follows analytically:

$$
d_1 = N^{-1}(\Delta_{\text{call}})
$$

$$
K = S \cdot \exp\left[-d_1 \cdot \sigma\sqrt{T} + \left(r + \tfrac{1}{2}\sigma^2\right)T\right]
$$

This gives the exact target-delta strike in one step, with no root-finding — a small but
clean efficiency that keeps the per-path inner loop fast. (When the volatility skew below is
enabled, the strike and its volatility depend on each other, so this closed form is applied
inside a short fixed-point iteration — see the next-but-one subsection.)

### Implied volatility and the volatility risk premium

The premium a seller collects depends entirely on the implied volatility used to price the
option. Here, implied volatility is modelled as the asset's **trailing 21-day realised
volatility plus a constant 2% (200 bps) volatility risk premium**:

$$
\text{IV}_t = \sigma_{\text{realised},\,21\text{d}}\ (\text{annualised}) + 2\%
$$

The 2% spread is the structural edge the Wheel is designed to harvest: it represents the
market reality that option sellers are, on average, paid slightly more than the volatility
that ultimately materialises. Modelling it as a constant additive spread is a deliberate
simplification (see [limitations](#assumptions-and-limitations)).

### Volatility skew (the smile)

Black–Scholes assumes one flat volatility across all strikes. Listed equity index options do
not trade that way: out-of-the-money puts are systematically priced at a **higher** implied
volatility than equidistant out-of-the-money calls — the volatility *skew* (or "smirk") —
driven by the leverage effect (falling prices coincide with rising volatility) and by
structural institutional demand for downside protection.

The app reproduces this with a deliberately simple **static parametrisation in
log-moneyness**, applied at the option-pricing step:

$$
\sigma(K) = \sigma_{\text{ATM}} - b \ln(K/S) + c \ln(K/S)^2
$$

A positive slope $b$ makes lower strikes (OTM puts) richer and higher strikes (OTM calls)
cheaper; a positive curvature $c$ lifts both wings into a smile. Because the target-delta
strike and the volatility *at* that strike now depend on each other, the two are solved
jointly with a short fixed-point iteration around the closed-form strike above (convergence
is geometric; five iterations leave a delta residual below ~10⁻⁶ even for steep skews).

Two honest boundaries of this approach:

- **It is a pricing-layer approximation only.** Price paths are still generated at a single
  flat volatility; the skew changes what the strategy is *paid*, not how the market *moves*.
  A full stochastic-volatility treatment (e.g. Heston), where the smile emerges from the
  dynamics itself, is out of scope by design.
- **The parameters are stylised, not calibrated.** The default slope and curvature are set
  to typical equity-index magnitudes (a few volatility points across the 25-delta wings),
  not fitted to any specific day's option chain.

For the Wheel the first-order effect is intuitive: the cash-secured puts it sells (struck
below spot) collect a somewhat richer premium, while the covered calls (struck above spot)
collect a somewhat poorer one. The sidebar toggle **Enable volatility skew** switches the
mechanism off entirely, restoring the flat-vol baseline used in the notebook.

---

## Performance and risk metrics

All metrics are computed across the cross-section of 10,000 Monte Carlo terminal outcomes.
Two of them — Sharpe and CVaR — therefore describe the *dispersion of outcomes across paths*
(an ensemble view), not the volatility of a single realised track record. They should be
read in that spirit.

- **CAGR** — the compound annual growth rate implied by the *mean* terminal value:
  $\left(\text{mean}(V_T)/V_0\right)^{1/\text{years}} - 1$.
- **Average maximum drawdown (Avg MDD)** — for each path, the deepest peak-to-trough decline
  of the daily mark-to-market equity curve is measured against a running peak; these per-path
  drawdowns are then averaged. (An earlier, cross-sectional approximation was replaced with
  this correct per-path computation.)
- **CVaR 95% (Conditional Value-at-Risk / Expected Shortfall)** — the mean total return over
  the worst 5% of paths (those at or below the 5th percentile of returns). It summarises the
  left tail: how bad things are *when* they go badly.
- **Sharpe ratio (annualised)** — mean excess return across paths (return minus `r·years`),
  divided by its cross-path standard deviation, annualised by `/√years`.
- **P(Wheel > Buy & Hold)** — the paired win-rate: the fraction of paths on which the Wheel
  ends with more wealth than Buy & Hold on the *same* path.

A Sortino ratio (downside-deviation analogue of Sharpe) is also computed in the notebook.

---

## Results

Metrics below are reproduced directly from the notebook (fixed seed, 10,000 paths per
regime) and represent the **flat-volatility baseline** — the configuration presented at the
conference. The volatility skew is an app-side extension: with the sidebar toggle switched
off, the app reproduces this baseline exactly. The pattern is consistent: the Wheel improves
the risk profile in every regime, and
improves *mean wealth* in the bear and sideways regimes — its cost is concentrated entirely
in the strong bull, where it caps the upside.

![Distribution of terminal portfolio values, Wheel vs Buy & Hold](terminal_value_distributions.png)

*Top row: histograms of terminal portfolio value (the dashed line is starting capital).
Bottom row: the same outcomes as empirical CDFs. The Wheel's distribution is consistently
compressed — it sheds both tails. In the bull market that compression costs the long right
tail; in the bear and sideways markets it mostly trims the left tail, which is exactly the
trade the strategy is built to make.*

**Bull market** (≈ 1000 trading days):

| Metric                       |     Wheel | Buy & Hold |
| ---------------------------- | --------: | ---------: |
| Mean terminal value          | \$152,913 | \$181,068 |
| CAGR                         |   11.04%  |   15.77%   |
| CVaR 95% (mean of worst 5%)  |  +14.54%  |   +8.08%   |
| Avg max drawdown             |   −10.1%  |   −14.0%   |
| Sharpe (annualised)          |    1.18   |    0.74    |
| P(Wheel > Buy & Hold)        |   21.9%   |     —      |

Buy & Hold wins on mean wealth — there is no cap on its upside, and a 15%-drift market
rewards simply staying invested. The Wheel still delivers the better *risk-adjusted* result
(Sharpe 1.18 vs 0.74) and shallower drawdowns, but it only beats Buy & Hold outright on ~22%
of paths. This is the regime where premium selling is genuinely a drag on return.

**Bear market** (≈ 300 trading days):

| Metric                       |    Wheel | Buy & Hold |
| ---------------------------- | -------: | ---------: |
| Mean terminal value          | \$88,973 | \$78,858  |
| CAGR                         |  −8.74%  | −16.96%    |
| CVaR 95% (mean of worst 5%)  | −56.72%  | −66.02%    |
| Avg max drawdown             |  −34.5%  |  −43.9%    |
| Sharpe (annualised)          |  −0.52   |  −0.74     |
| P(Wheel > Buy & Hold)        |  87.5%   |    —       |

Both strategies lose money in a −20%-drift, 35%-volatility market — premium selling is not
alchemy. But the premium meaningfully cushions the fall: the Wheel beats Buy & Hold on ~87%
of paths, its average drawdown is about 9 percentage points shallower, and its worst-5%
outcome is materially less severe. This is the cushion the strategy is designed to provide.

**Sideways market** (≈ 250 trading days):

| Metric                       |     Wheel | Buy & Hold |
| ---------------------------- | --------: | ---------: |
| Mean terminal value          | \$105,861 | \$103,307 |
| CAGR                         |   5.42%   |   3.06%    |
| CVaR 95% (mean of worst 5%)  |  −23.33%  |  −29.39%   |
| Avg max drawdown             |   −12.8%  |  −18.2%    |
| Sharpe (annualised)          |    0.12   |   −0.05    |
| P(Wheel > Buy & Hold)        |   70.8%   |     —      |

This is the Wheel's natural habitat. With little net drift to capture, Buy & Hold earns a
small return for taking full market risk (a negative Sharpe — the market is not rewarding
the risk taken). The Wheel converts the choppiness itself into return through repeated
premium collection: higher CAGR, a positive Sharpe, shallower drawdowns, and a ~71% paired
win-rate.

---

## Key findings

1. **The edge is conditional, not universal.** The Wheel beats Buy & Hold on a majority of
   paths in falling (~87%) and flat (~71%) markets, but on only ~22% of paths in a strong
   bull. Selling premium is a bet against large upside moves, and it pays off precisely when
   those moves don't happen.

2. **You pay for safety with upside.** In the strong bull the Wheel gives up return
   (≈11% vs ≈16% CAGR), but buys a much better risk profile — roughly Sharpe 1.18 vs 0.74
   and an average drawdown about 4 percentage points shallower. In every regime the Wheel's
   downside tail (CVaR 95%) and average drawdown are better than Buy & Hold's. The strategy
   trades the right tail of the distribution for a thinner left tail.

3. **The volatility risk premium is the engine.** Systematically writing options monetises
   the gap between implied and realised volatility. Within this model that gap is a
   structural, repeatable source of return — which is why the Wheel's advantage is most
   visible exactly where directional return is scarce.

**In one sentence:** the Wheel is a volatility-management tool. It makes the payoff
distribution more defensive and more predictable, at the cost of capping the upper tail of
returns — not a way to earn more on average.

---

## Assumptions and limitations

This is a simulation, and its conclusions are only as good as its assumptions. The honest
caveats matter as much as the headline numbers:

- **GBM has thin tails.** Geometric Brownian motion produces log-normal returns with no
  jumps and no volatility clustering, so it understates real crash risk (Black-Swan events).
  Because the Wheel's entire risk lives in the left tail (assignment into a falling market),
  thin-tailed GBM probably *flatters* the Wheel relative to reality.
- **No transaction costs.** Commissions, bid–ask spreads, and slippage are ignored. The
  Wheel trades far more often than Buy & Hold, so this omission **favours the Wheel**;
  real-world frictions would erode part of its measured edge.
- **Perfect liquidity and fills.** The model assumes an option can always be written at the
  Black–Scholes price plus a fixed 2% IV premium, at any moment — unrealistic in stressed
  markets, where spreads widen and premium is harder to capture on the seller's terms.
- **European pricing, expiry-only assignment.** Options are priced and exercised as European
  contracts, with assignment decided only at expiry. Real U.S. equity options are American
  and can be assigned early, making exercise path-dependent.
- **A constant additive VRP.** The 2% volatility risk premium is held fixed. Real VRP is
  time-varying and regime-dependent — it can spike in crises and occasionally invert.
- **The skew is static and stylised (app extension).** When enabled, the skew is a fixed
  function of log-moneyness applied only at the pricing step. Real skew is dynamic — it
  steepens in sell-offs — varies with tenor, and under a stochastic-volatility model would
  interact with the path dynamics themselves. Its parameters are typical equity-index
  magnitudes, not calibrated to market data.
- **No dividends; a single underlying; a single, fixed parameter set.** Delta (0.30) and
  tenor (21 steps) are fixed rather than optimised, and this is not a sensitivity sweep —
  results are conditional on these specific choices.
- **Stylised regimes.** The three scenarios illustrate behaviour across environments; they
  are not forecasts of, or calibrations to, any particular real market.
- **Ensemble metrics.** Sharpe and CVaR are computed over the cross-section of Monte Carlo
  terminal outcomes rather than from a single realised track record, and should be
  interpreted accordingly.

Several of these (no costs, perfect liquidity, thin tails) point the same way: they make the
Wheel look better than it would in practice. The qualitative story — premium selling trades
upside for a defensive, lower-variance payoff — is robust to all of them; the precise
out-performance figures are not.

---

## Repository structure

```
Options-Wheel-vs.-Buy-Hold-Monte-Carlo-Strategy-Comparison/
├── app.py                                        ← Streamlit UI (sidebar, charts, metrics table)
├── simulation.py                                 ← All mathematical logic (GBM, Black–Scholes, Wheel, metrics)
├── requirements.txt                              ← Python dependencies
├── README.md
├── PL_wheel_strategy_risk_analysis_mc.ipynb      ← Original research notebook (Polish)
└── Options_strategy_wheel_vs_Buy_paper_from_kkmf.pdf  ← Conference paper (Polish)
```

`app.py` and `simulation.py` live in the repository **root**, so the app runs — locally or on Streamlit Community Cloud — without any extra configuration.

---

## Interactive app

The app is live at **[wheel-strategy-montecarlo.streamlit.app](https://wheel-strategy-montecarlo.streamlit.app/)** — no setup required. It exposes every parameter — drift, volatility, volatility risk premium, option delta, the volatility-skew slope and curvature, number of paths — as sidebar controls, runs the simulation on demand, and renders the metrics table and distribution charts in the browser.

To run it locally instead:

```bash
git clone https://github.com/karolkudlacik/Options-Wheel-vs.-Buy-Hold-Monte-Carlo-Strategy-Comparison.git
cd Options-Wheel-vs.-Buy-Hold-Monte-Carlo-Strategy-Comparison
pip install -r requirements.txt
streamlit run app.py
```

Choose a scenario preset or switch to Custom to set μ and σ freely, then click **Run Simulation**; results are cached, so re-running with the same parameters is instant.

Beyond the metrics table and the distribution charts, the app renders:

- **A live implied-volatility smile** — σ(K) across moneyness for the current sidebar
  settings, drawn against the flat Black–Scholes line. It updates instantly as the skew
  sliders move, before any simulation runs.
- **Portfolio value over time** — the median path with a 10th–90th percentile band for the
  Wheel and Buy & Hold, showing *when* along the horizon the two strategies diverge, not
  just where they end up.

Toggling **Enable volatility skew** off reverts the pricing to the flat-vol baseline used in
the notebook and in the [Results](#results) above, which makes the flat-vs-skew comparison a
one-click experiment.

---

## Reproducing the study

For interactive exploration, use the [live app](#interactive-app) — the fixed random seed
makes every run reproducible, and switching **Enable volatility skew** off runs the identical
flat-vol logic as the notebook. To reproduce the exact numbers and figures in this README, run
the original research notebook (Python 3 with `numpy`, `scipy`, `matplotlib`):

```bash
pip install numpy scipy matplotlib jupyter
jupyter notebook PL_wheel_strategy_risk_analysis_mc.ipynb
```

Run the cells top to bottom. Re-parameterising the regimes (drift, volatility, horizon) or
the strategy (target delta, tenor) is a matter of editing the scenario dictionary and the
strategy constants near the top.

---

## Reading the results responsibly

This is an educational and research project, not investment advice. The figures describe the
behaviour of a strategy *inside a model*; the [limitations](#assumptions-and-limitations)
above — especially the thin-tailed price process and the absence of trading costs — mean the
real-world edge would be smaller than the simulation suggests, and could be negative once
frictions are included. The intended takeaway is structural and qualitative: **selling
option premium reshapes the distribution of returns toward a more defensive profile, and that
reshaping is most valuable in falling and sideways markets and most costly in strong bulls.**
