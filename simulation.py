"""
Wheel Strategy vs Buy & Hold — Monte Carlo core logic
=====================================================
Pure-logic functions only: Geometric Brownian Motion, Black-Scholes pricing,
the Wheel option-selling algorithm, Buy & Hold, and the performance metrics.

No matplotlib, no print statements, no Streamlit imports. The mathematical
logic is identical to the original notebook.

Both portfolios start on day 22 (START_DAY) with $100,000 of capital.
Buy & Hold buys shares on day 22 and holds to the end.
Wheel sells PUT/CALL options according to the algorithm in ``simulate_wheel``.
"""

import numpy as np
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────
# CONSTANTS (used as default arguments below)
# ─────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 100_000
START_DAY       = 22
R               = 0.04      # annual risk-free rate
CONTRACT_SIZE   = 100       # shares per option contract
TARGET_DELTA    = 0.30
OPTION_DAYS     = 21


# ─────────────────────────────────────────────────────────────
# BLOCK 1 — GEOMETRIC BROWNIAN MOTION
# ─────────────────────────────────────────────────────────────

def run_monte_carlo(S0, mu, sigma, days, num_paths):
    """
    Simulate asset price paths with discrete Geometric Brownian Motion.

    Parameters
    ----------
    S0        : initial asset price
    mu        : annual drift (expected return)
    sigma     : annual volatility
    days      : number of trading days
    num_paths : number of Monte Carlo paths

    Returns
    -------
    prices : price matrix (rows -> days, columns -> paths)
    """
    # Seed lives inside the function so that, given identical arguments, the
    # output is fully reproducible (required for correct caching downstream).
    np.random.seed(42)

    dt      = 1 / 252
    shocks  = np.random.normal(0, 1, (days, num_paths))
    returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks)
    prices  = np.zeros((days + 1, num_paths))
    prices[0] = S0
    prices[1:] = S0 * np.cumprod(returns, axis=0)
    return prices


# ─────────────────────────────────────────────────────────────
# BLOCK 2 — BLACK-SCHOLES
# ─────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option="put"):
    """Black-Scholes price of a European put or call option."""
    if T <= 0:
        return max(0.0, (K - S) if option == "put" else (S - K))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def find_strike(S, T, r, sigma, target_delta, option="put"):
    """Analytic inversion of Black-Scholes: solve for K given a target |delta|."""
    delta_call = (1 - target_delta) if option == "put" else target_delta
    d1 = norm.ppf(delta_call)
    return S * np.exp(-d1 * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)


# ─────────────────────────────────────────────────────────────
# BLOCK 2b — VOLATILITY SKEW (static, log-moneyness parametrisation)
#
# Equity index options do not trade at a single flat volatility: OTM puts
# carry higher implied vol than equidistant OTM calls (the "skew" / "smirk"),
# driven by the leverage effect and structural demand for downside hedges.
#
# We use a deliberately simple *static* parametrisation in log-moneyness:
#
#     sigma(K) = sigma_atm - slope * ln(K/S) + curvature * ln(K/S)^2
#
#     slope > 0     -> lower strikes (OTM puts) get HIGHER vol,
#                      higher strikes (OTM calls) get lower vol
#     curvature > 0 -> lifts both wings (the "smile")
#
# Scope / limitations (kept intentionally): this is a pricing-layer
# approximation only. The GBM paths are still generated at a single flat
# volatility, and the skew parameters are stylized (typical equity-index
# magnitudes), not calibrated to market data.
# ─────────────────────────────────────────────────────────────

VOL_FLOOR = 0.01   # numerical floor so extreme parameters cannot push vol <= 0


def skewed_vol(sigma_atm, S, K, skew_slope=0.0, smile_curvature=0.0):
    """Implied volatility at strike K under the static skew/smile model."""
    m   = np.log(K / S)
    vol = sigma_atm - skew_slope * m + smile_curvature * m**2
    return max(vol, VOL_FLOOR)


def find_strike_skewed(S, T, r, sigma_atm, target_delta, option="put",
                       skew_slope=0.0, smile_curvature=0.0, n_iter=5):
    """
    Strike for a target |delta| when vol depends on the strike itself.

    The strike-for-delta and the vol-at-that-strike depend on each other,
    so we solve the fixed point with a few iterations (convergence is
    geometric; 5 iterations leave a delta residual below ~1e-6 even for
    steep skews). With zero skew this reduces exactly to ``find_strike``.

    Returns
    -------
    (K, sigma_K) : the strike and the skew-adjusted vol at that strike
    """
    K = find_strike(S, T, r, sigma_atm, target_delta, option)
    for _ in range(n_iter):
        sigma_K = skewed_vol(sigma_atm, S, K, skew_slope, smile_curvature)
        K       = find_strike(S, T, r, sigma_K, target_delta, option)
    sigma_K = skewed_vol(sigma_atm, S, K, skew_slope, smile_curvature)
    return K, sigma_K


# ─────────────────────────────────────────────────────────────
# BLOCK 3 — IMPLIED VOLATILITY (rolling 21-day realized vol + VRP)
# ─────────────────────────────────────────────────────────────

def calculate_iv(prices, vrp):
    """
    Rolling 21-day realized volatility plus a volatility risk premium (VRP).

    Returns an IV array the same length as ``prices``. The first 21 values are
    0 (no history) — the agent starts on day 22, so this is fine.

    Parameters
    ----------
    prices : price path (1-D array)
    vrp    : volatility risk premium added to realized volatility
    """
    rets = np.log(prices[1:] / prices[:-1])
    iv   = np.zeros_like(prices)
    for t in range(21, len(prices)):
        realized_vol = np.std(rets[t-21:t], ddof=1) * np.sqrt(252)
        iv[t]        = realized_vol + vrp
    return iv


# ─────────────────────────────────────────────────────────────
# BLOCK 4 — BUY & HOLD
# ─────────────────────────────────────────────────────────────

def simulate_bh(price_path, start_day=START_DAY, initial=INITIAL_CAPITAL):
    """
    Buy shares on ``start_day`` with the full capital and hold to the end.

    Returns
    -------
    (daily value history, final value)
    """
    S_buy  = price_path[start_day]
    shares = initial / S_buy
    values = shares * price_path[start_day:]
    return values, float(values[-1])


# ─────────────────────────────────────────────────────────────
# BLOCK 5 — WHEEL STRATEGY
# ─────────────────────────────────────────────────────────────

def simulate_wheel(price_path, iv_path, r=R, target_delta=TARGET_DELTA,
                   option_days=OPTION_DAYS, start_day=START_DAY,
                   contract_size=CONTRACT_SIZE, initial=INITIAL_CAPITAL,
                   skew_slope=0.0, smile_curvature=0.0):
    # skew_slope / smile_curvature parametrise the static volatility skew
    # (see BLOCK 2b). With both at 0.0 the behaviour is identical to the
    # original flat-vol version.
    cash       = float(initial)
    shares     = 0
    K_assigned = None
    state      = "PUT"
    day        = start_day
    total_days = len(price_path) - 1
    T_exp      = option_days / 252
    history    = []

    while day <= total_days:
        S     = price_path[day]
        sigma = iv_path[day]

        if state == "PUT":
            K, sigma_K  = find_strike_skewed(S, T_exp, r, sigma, target_delta, "put",
                                             skew_slope, smile_curvature)
            n_contracts = int(cash / (K * contract_size))

            if n_contracts <= 0:
                # Not enough capital for another contract — stop trading, hold cash.
                for d in range(day, total_days + 1):
                    history.append(cash + shares * price_path[d])
                break

            premium  = bs_price(S, K, T_exp, r, sigma_K, "put") * contract_size * n_contracts
            cash    += premium

        else:  # state == "CALL"
            K_delta, _  = find_strike_skewed(S, T_exp, r, sigma, target_delta, "call",
                                             skew_slope, smile_curvature)
            K           = max(K_assigned, K_delta)
            # Price at the vol of the strike actually written (K may be the
            # assigned strike, not the delta-targeted one).
            sigma_K     = skewed_vol(sigma, S, K, skew_slope, smile_curvature)
            n_contracts = int(shares // contract_size)

            if n_contracts <= 0:
                # FIX: instead of `continue` (risk of an infinite loop), immediately
                # switch to writing a PUT on the same day.
                state = "PUT"
                K, sigma_K  = find_strike_skewed(S, T_exp, r, sigma, target_delta, "put",
                                                 skew_slope, smile_curvature)
                n_contracts = int(cash / (K * contract_size))
                if n_contracts <= 0:
                    for d in range(day, total_days + 1):
                        history.append(cash + shares * price_path[d])
                    break
                premium  = bs_price(S, K, T_exp, r, sigma_K, "put") * contract_size * n_contracts
                cash    += premium

            else:
                premium  = bs_price(S, K, T_exp, r, sigma_K, "call") * contract_size * n_contracts
                cash    += premium

        expiry_day = min(day + option_days, total_days)

        for d in range(day, expiry_day + 1):
            cash *= np.exp(r / 252)
            history.append(cash + shares * price_path[d])

        S_exp = price_path[expiry_day]
        if state == "PUT":
            if S_exp < K:
                cash      -= K * contract_size * n_contracts
                shares    += contract_size * n_contracts
                K_assigned = K
                state      = "CALL"
        else:
            if S_exp > K:
                cash      += K * contract_size * n_contracts
                shares    -= contract_size * n_contracts
                K_assigned = None
                state      = "PUT"

        day = expiry_day + 1

    final = history[-1] if history else float(initial)
    return np.array(history), final


# ─────────────────────────────────────────────────────────────
# BLOCK 6 — METRICS
#
# Notes:
#   1. Maximum Drawdown is computed along the time axis on each path
#      separately, then averaged — rather than an incorrect cross-sectional
#      approximation. This requires passing ``histories`` (list of history[]).
#
#   2. Sharpe Ratio is annualized by dividing by sqrt(years); the denominator
#      is excess.std() (not returns.std()).
# ─────────────────────────────────────────────────────────────

def max_drawdown_single(history):
    """Maximum drawdown for a single time-series of portfolio values."""
    arr         = np.array(history)
    peak        = np.maximum.accumulate(arr)
    drawdowns   = (arr - peak) / peak
    return drawdowns.min()


def compute_metrics(final_values, histories, years,
                    initial=INITIAL_CAPITAL, rf_annual=R):
    """
    Parameters
    ----------
    final_values : array of final portfolio values (one number per path)
    histories    : list of history[] arrays — the full history of each path
    years        : horizon length in years
    initial      : starting capital
    rf_annual    : annual risk-free rate
    """
    fv      = np.array(final_values)
    returns = (fv / initial) - 1           # total return of each path
    rf      = rf_annual * years            # benchmark over the whole period

    excess  = returns - rf

    # --- Annualized Sharpe ----------------------------------------------
    # Divide by sqrt(years) to bring it to an annual scale.
    sharpe = (excess.mean() / (excess.std(ddof=1) + 1e-9)) / np.sqrt(years)

    # --- Annualized Sortino ---------------------------------------------
    downside     = returns[returns < rf] - rf
    downside_std = np.sqrt((downside**2).mean()) if len(downside) > 0 else 1e-9
    sortino      = (excess.mean() / (downside_std + 1e-9)) / np.sqrt(years)

    # --- Maximum Drawdown -----------------------------------------------
    # Compute MDD for each path separately and average.
    mdd_values = np.array([max_drawdown_single(h) for h in histories])
    mdd        = mdd_values.mean()

    # --- CVaR 95% -------------------------------------------------------
    threshold = np.percentile(returns, 5)
    cvar_95   = returns[returns <= threshold].mean()

    return {
        "mean":    fv.mean(),
        "median":  np.median(fv),
        "std":     fv.std(ddof=1),
        "cagr":    (fv.mean() / initial) ** (1 / years) - 1,
        "p_loss":  (fv < initial).mean(),
        "cvar_95": cvar_95,
        "mdd":     mdd,
        "sharpe":  sharpe,
        "sortino": sortino,
    }
