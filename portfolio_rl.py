# =====================================================================================
# RL PIPELINE FOR PORTFOLIO ALLOCATION - built to avoid fooling myself
# =====================================================================================
# PHILOSOPHY (why I made each choice):
#
#  1. DELIBERATELY MODEST AGENT. In a domain with low signal and high noise, a more
#     expressive agent does not extract more signal, it just memorises the historical
#     path better. Simplicity (PPO + small MLP + few features + few timesteps) is the
#     regularisation that matters most. The strength is in the VALIDATION, not the model.
#
#  2. REWARD = RISK MANAGEMENT, not direction forecasting. My LSTM project already
#     showed (training loss stuck at ln(2)) that direction is unpredictable. The reward
#     uses the Differential Sharpe Ratio (Moody & Saffell, 1998), an online and stable
#     version of the Sharpe ratio, plus a drawdown penalty. It asks the agent to
#     MODULATE ITS RISK EXPOSURE (the only real signal: volatility clustering) instead
#     of guessing returns.
#
#  3. THREE-LEVEL VALIDATION:
#       - training: the agent learns its weights inside each window.
#       - walk-forward: I compare configurations / measure generalisation.
#       - SACRED HELD-OUT: a final block that does NOT exist during development;
#         it is run ONCE at the end. It is the only honest estimate.
#
#  4. PURGING + EMBARGO between train and test (Lopez de Prado): a time gap that removes
#     the autocorrelation coming from overlapping lookback windows.
#
#  5. MULTIPLE SEEDS + DEFLATED SHARPE RATIO: RL is stochastic, so a single number lies.
#     The DSR discounts the Sharpe by the number of trials, i.e. how many times I touched
#     the data myself (multiple testing is the real killer).
#
#  6. ROBUSTNESS TESTS:
#       - SYNTHETIC DATA (the ultimate judge): series with the same marginal properties
#         but NO time structure. If the agent "wins" here, the pipeline creates signal
#         out of noise and the real result is false.
#       - REGIME ANALYSIS: the edge should appear where theory says it should (high vol),
#         not scattered randomly. WHERE it wins is itself evidence.
#
# HONESTY NOTE: the ETFs below were chosen ex-post (survivorship/selection bias). There
# is no database of delisted ETFs here, so the bias is NAMED but not fully removed. In a
# production setting, the universe would be defined with the information available at
# the start.
#
# HOW TO RUN: in Google Colab, run all cells
#             (or locally: pip install -r requirements.txt && python portfolio_rl.py)
# Parameters are modest ON PURPOSE (see philosophy). Takes ~5-12 min on CPU.
# =====================================================================================


# =====================================================================================
# SECTION 1 - IMPORTS AND CONFIGURATION
# =====================================================================================
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import softmax

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import warnings, os, random
warnings.filterwarnings("ignore")

# ---- Universe and period ----
ETFS   = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG"]
START  = "2010-01-01"
END    = None

# ---- Market mechanics ----
TRANSACTION_COST = 0.001     # 10 bps per unit of turnover
LOOKBACK         = 20        # days used for the state features (average return, vol)

# ---- Reward (risk management) ----
ETA_DSR      = 0.02          # adaptation rate of the Differential Sharpe
DD_PENALTY   = 0.10          # weight of the drawdown penalty
REWARD_SCALE = 100.0         # scales the DSR so PPO gets a decent signal

# ---- Deliberately MODEST agent ----
TRAIN_TIMESTEPS = 20000      # modest on purpose (less memorisation)
N_SEEDS         = 3          # number of seeds per configuration
POLICY_KWARGS   = dict(net_arch=[64, 64])   # small MLP

# ---- Validation ----
EMBARGO_DAYS    = 5          # gap between train and test (purging/embargo)
HELDOUT_DAYS    = 378        # last ~18 months, SACRED (not touched until the very end)
WF_TRAIN_DAYS   = 756        # ~3 years of training per walk-forward window
WF_TEST_DAYS    = 189        # ~9 months of testing per window
WF_STEP_DAYS    = 189        # step between windows

REBAL_FREQ      = 21         # monthly rebalancing for the dynamic baselines
RF_ANNUAL       = 0.0        # risk-free rate (simplification)
PERIODS         = 252        # trading days per year

print("Configuration loaded. Agent is modest on purpose (see philosophy).")


# =====================================================================================
# SECTION 2 - DATA: download + split into DEV / SACRED HELD-OUT
# =====================================================================================
def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(-1)
    px = px[tickers].dropna()
    return px

prices = download_prices(ETFS, START, END)
print(f"Prices: {prices.shape[0]} days, {prices.shape[1]} ETFs "
      f"({prices.index.min().date()} -> {prices.index.max().date()})")

# SACRED SPLIT: the final held-out set does not exist during development.
dev_prices     = prices.iloc[:-HELDOUT_DAYS]
heldout_prices = prices.iloc[-(HELDOUT_DAYS + LOOKBACK + 1):]   # +lookback to build the state
print(f"DEV: {dev_prices.shape[0]} days | HELD-OUT (sacred): {HELDOUT_DAYS} days")


# =====================================================================================
# SECTION 3 - STATIONARY FEATURES (no price levels, no leakage)
# =====================================================================================
def to_returns(prices):
    """Daily log returns (stationary)."""
    return np.log(prices / prices.shift(1)).dropna()


# =====================================================================================
# SECTION 4 - ENVIRONMENT: Differential Sharpe + drawdown reward, action via softmax
# =====================================================================================
class PortfolioEnv(gym.Env):
    """
    State:   [recent average return (n), recent volatility (n), current weights (n)]
    Action:  logits -> softmax -> long-only weights that sum to 1 (no normalisation hacks)
    Reward:  Differential Sharpe Ratio (online, stable) + drawdown penalty,
             net of transaction costs. It does NOT reward direction; it rewards
             RISK-ADJUSTED return and exposure control.
    """
    metadata = {"render_modes": []}

    def __init__(self, returns, transaction_cost=0.001, lookback=20,
                 eta=0.02, dd_penalty=0.10, reward_scale=100.0):
        super().__init__()
        self.returns = returns.values if hasattr(returns, "values") else returns
        self.n = self.returns.shape[1]
        self.tc = transaction_cost
        self.lookback = lookback
        self.eta = eta
        self.dd_penalty = dd_penalty
        self.reward_scale = reward_scale

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n,), dtype=np.float32)
        # 3n features; wide ranges because I scale manually (no global stats => no leakage)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(self.n * 3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.weights = np.ones(self.n) / self.n
        # Differential Sharpe state
        self.A = 0.0
        self.B = 0.0
        # drawdown state
        self.nav = 1.0
        self.peak = 1.0
        return self._obs(), {}

    def _obs(self):
        window = self.returns[self.t - self.lookback:self.t]
        mean_r = window.mean(axis=0) * 100.0     # fixed scale (no leakage)
        vol_r  = window.std(axis=0)  * 100.0
        obs = np.concatenate([mean_r, vol_r, self.weights]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        w = softmax(np.asarray(action, dtype=np.float64))   # long-only, sums to 1

        # portfolio return on the next step, net of costs
        port_ret = float(np.dot(w, self.returns[self.t]))
        turnover = float(np.sum(np.abs(w - self.weights)))
        net_ret  = port_ret - turnover * self.tc

        # update NAV and drawdown
        self.nav *= (1.0 + net_ret)
        self.peak = max(self.peak, self.nav)
        drawdown = (self.nav - self.peak) / self.peak     # <= 0

        # ---- Differential Sharpe Ratio (Moody & Saffell) ----
        dA = net_ret - self.A
        dB = net_ret**2 - self.B
        var = self.B - self.A**2
        if var > 1e-12:
            dsr = (self.B * dA - 0.5 * self.A * dB) / (var**1.5)
        else:
            dsr = 0.0
        self.A += self.eta * dA
        self.B += self.eta * dB

        reward = self.reward_scale * dsr + self.dd_penalty * drawdown  # drawdown < 0 penalises

        self.weights = w
        self.t += 1
        done = self.t >= len(self.returns) - 1
        info = {"net_ret": net_ret, "weights": w, "nav": self.nav}
        return self._obs(), float(reward), bool(done), False, info


def rollout_weights(model, returns, lookback=LOOKBACK):
    """Runs the trained policy over `returns`, returns the net return series and the weights."""
    env = PortfolioEnv(returns, TRANSACTION_COST, lookback, ETA_DSR, DD_PENALTY, REWARD_SCALE)
    obs, _ = env.reset()
    done = False
    rets, ws = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rets.append(info["net_ret"]); ws.append(info["weights"])
    return np.array(rets), np.array(ws)


# =====================================================================================
# SECTION 5 - BASELINES (the yardstick; no leakage)
# =====================================================================================
def nav_from_returns(rets):
    return np.cumprod(1.0 + np.asarray(rets))

def bh_equal_weight(returns):
    """Buy & hold 1/N (no rebalancing)."""
    R = returns.values
    n = R.shape[1]
    w0 = np.ones(n) / n
    # weights drift with prices: NAV per asset
    growth = np.cumprod(1.0 + R, axis=0)
    port = (w0 * growth).sum(axis=1)
    rets = np.diff(port) / port[:-1]
    return np.concatenate([[0.0], rets])

def bh_6040(returns):
    R = returns.values; cols = list(returns.columns)
    w0 = np.zeros(R.shape[1])
    w0[cols.index("SPY")] = 0.6; w0[cols.index("AGG")] = 0.4
    growth = np.cumprod(1.0 + R, axis=0)
    port = (w0 * growth).sum(axis=1)
    rets = np.diff(port) / port[:-1]
    return np.concatenate([[0.0], rets])

def risk_parity(returns, lookback=60, rebal=REBAL_FREQ):
    """Inverse-vol weighting, rebalanced periodically (vol estimated using only the past)."""
    R = returns.values; T, n = R.shape
    out = np.zeros(T); w = np.ones(n)/n
    for t in range(T):
        if t >= lookback and t % rebal == 0:
            vol = R[t-lookback:t].std(axis=0) + 1e-8
            inv = 1.0 / vol
            w = inv / inv.sum()
        out[t] = float(np.dot(w, R[t]))
    return out

def mean_variance(returns_train, returns_eval, ridge=1e-3):
    """Markowitz: weights estimated ONLY on the training set, applied on eval (no leakage)."""
    Rtr = returns_train.values
    mu = Rtr.mean(axis=0)
    cov = np.cov(Rtr.T) + ridge * np.eye(Rtr.shape[1])
    w = np.linalg.solve(cov, mu)
    w = np.clip(w, 0, None)                  # long-only
    w = w / (w.sum() + 1e-12)
    Rev = returns_eval.values
    return Rev @ w

def random_alloc(returns, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(returns.shape[1]))
    return returns.values @ w


# =====================================================================================
# SECTION 6 - HONEST METRICS
# =====================================================================================
def metrics(rets, rf_annual=RF_ANNUAL, periods=PERIODS):
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return dict(total=0, annual=0, vol=0, sharpe=0, sortino=0, calmar=0, mdd=0, turn=np.nan)
    nav = nav_from_returns(r)
    total = nav[-1] / nav[0] - 1
    years = len(r) / periods
    annual = (nav[-1] / nav[0]) ** (1/years) - 1 if years > 0 else 0
    vol = r.std() * np.sqrt(periods)
    rf_daily = rf_annual / periods
    excess = r - rf_daily
    sharpe = excess.mean() / (r.std() + 1e-12) * np.sqrt(periods)   # ANNUALISED Sharpe
    downside = r[r < 0].std() if (r < 0).any() else 1e-12
    sortino = excess.mean() / (downside + 1e-12) * np.sqrt(periods)
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    calmar = annual / abs(mdd) if mdd < 0 else np.nan
    return dict(total=total*100, annual=annual*100, vol=vol*100, sharpe=sharpe,
                sortino=sortino, calmar=calmar, mdd=mdd*100)


# =====================================================================================
# SECTION 7 - DEFLATED / PROBABILISTIC SHARPE RATIO (Bailey & Lopez de Prado)
# =====================================================================================
# NOTE: the PSR formula needs the Sharpe PER PERIOD (here: daily), because it is
# combined with the number of daily observations. The rest of the code reports
# ANNUALISED Sharpes, so I convert them to daily inside these two functions.
# (Before this fix I passed the annualised Sharpe straight into the formula, which
# overstated the DSR.)
def probabilistic_sharpe(sr_hat, n_obs, sr_benchmark, skew, kurt):
    """
    PSR: P(true SR > sr_benchmark) given the observed SR and the return moments.
    ALL Sharpe inputs must be per-period (daily), not annualised.
    `kurt` is the regular (Pearson) kurtosis, where a normal distribution = 3.
    """
    num = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew*sr_hat + (kurt - 1)/4.0 * sr_hat**2)
    return float(stats.norm.cdf(num / (den + 1e-12)))

def deflated_sharpe(sharpe_hat_annual, n_obs, sharpe_trials_annual, skew, kurt,
                    periods=PERIODS):
    """
    DSR: corrects the best observed Sharpe for the number of trials (multiple testing).
    Inputs are ANNUALISED Sharpes (as printed by `metrics`); they are converted to daily
    Sharpes here before being used in the formula.
    Returns: (dsr, sr0_annual), where sr0_annual is the Sharpe expected by luck only.
    """
    to_daily = 1.0 / np.sqrt(periods)
    sr_hat_d = sharpe_hat_annual * to_daily
    trials_d = np.asarray(sharpe_trials_annual, dtype=float) * to_daily

    N = max(len(trials_d), 2)
    var_sr = np.var(trials_d, ddof=1) + 1e-12
    gamma = 0.5772156649  # Euler-Mascheroni
    z1 = stats.norm.ppf(1 - 1.0/N)
    z2 = stats.norm.ppf(1 - 1.0/(N*np.e))
    sr0_d = np.sqrt(var_sr) * ((1 - gamma)*z1 + gamma*z2)   # daily Sharpe expected by luck ONLY

    dsr = probabilistic_sharpe(sr_hat_d, n_obs, sr0_d, skew, kurt)
    return dsr, sr0_d * np.sqrt(periods)                     # sr0 back in annual units for printing


# =====================================================================================
# SECTION 8 - TRAIN + EVALUATE ONE WINDOW
# =====================================================================================
def train_agent(returns_train, seed, timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed); random.seed(seed)
    def make(): return PortfolioEnv(returns_train, TRANSACTION_COST, LOOKBACK,
                                    ETA_DSR, DD_PENALTY, REWARD_SCALE)
    venv = DummyVecEnv([make])
    model = PPO("MlpPolicy", venv, seed=seed, policy_kwargs=POLICY_KWARGS,
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
                gamma=0.99, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


# =====================================================================================
# SECTION 9 - WALK-FORWARD + MULTI-SEED (the validation engine)
# =====================================================================================
def walk_forward(dev_returns, label="DEV"):
    """
    Rolling windows with an EMBARGO between train and test. For each window, it trains
    N_SEEDS agents and aggregates them. Returns: the agent's out-of-sample return series
    (concatenated), and a dict with the baselines evaluated on the SAME segments.
    """
    T = len(dev_returns)
    agent_oos = []
    base_oos = {k: [] for k in ["equal", "6040", "rparity", "meanvar", "random"]}
    seed_sharpes = []   # annualised Sharpes, for the DSR

    start = 0
    win = 0
    while start + WF_TRAIN_DAYS + EMBARGO_DAYS + WF_TEST_DAYS <= T:
        tr0, tr1 = start, start + WF_TRAIN_DAYS
        te0 = tr1 + EMBARGO_DAYS                         # EMBARGO/purging
        te1 = te0 + WF_TEST_DAYS
        win += 1
        r_train = dev_returns.iloc[tr0:tr1]
        r_test  = dev_returns.iloc[te0:te1]

        # train N_SEEDS agents and average their OOS returns (more honest than best-seed)
        seed_rets = []
        for s in range(N_SEEDS):
            model = train_agent(r_train, seed=1000*win + s)
            rets, _ = rollout_weights(model, r_test)
            L = len(rets)
            seed_rets.append(rets)
            seed_sharpes.append(metrics(rets)["sharpe"])
        L = min(len(x) for x in seed_rets)
        agent_win = np.mean([x[:L] for x in seed_rets], axis=0)
        agent_oos.append(agent_win)

        # baselines on the SAME test segment (align length to L)
        base_oos["equal"].append(bh_equal_weight(r_test)[-L:])
        base_oos["6040"].append(bh_6040(r_test)[-L:])
        base_oos["rparity"].append(risk_parity(r_test)[-L:])
        base_oos["meanvar"].append(mean_variance(r_train, r_test)[-L:])
        base_oos["random"].append(random_alloc(r_test, seed=win)[-L:])

        print(f"  [{label}] window {win}: train {tr0}-{tr1}, test {te0}-{te1} "
              f"(embargo {EMBARGO_DAYS}d), L={L}")
        start += WF_STEP_DAYS

    agent_oos = np.concatenate(agent_oos) if agent_oos else np.array([])
    for k in base_oos:
        base_oos[k] = np.concatenate(base_oos[k]) if base_oos[k] else np.array([])
    return agent_oos, base_oos, seed_sharpes


# =====================================================================================
# SECTION 10 - ROBUSTNESS TESTS
# =====================================================================================
def synthetic_returns(returns, seed=0):
    """
    The ultimate judge: resamples WHOLE DAYS with replacement. It keeps the marginal
    distribution and the correlations between assets, but DESTROYS the time structure
    (autocorrelation, volatility clustering). By construction there is NOTHING for the
    agent to learn. If the agent "wins" here, the pipeline is creating signal out of noise.
    """
    rng = np.random.default_rng(seed)
    R = returns.values
    idx = rng.integers(0, len(R), size=len(R))
    synth = R[idx]
    return pd.DataFrame(synth, columns=returns.columns)

def regime_split(returns, lookback=20, q=0.66):
    """Classifies each day as HIGH vs LOW volatility (realised market vol)."""
    mkt = returns.mean(axis=1)
    vol = mkt.rolling(lookback).std()
    thr = vol.quantile(q)
    return (vol > thr).values   # True = high volatility


# =====================================================================================
# SECTION 11 - MAIN
# =====================================================================================
def main():
    dev_returns = to_returns(dev_prices)

    print("\n=== WALK-FORWARD (validation) ===")
    agent_oos, base_oos, seed_sharpes = walk_forward(dev_returns, "DEV")

    print("\n=== OOS METRICS (aggregated walk-forward) ===")
    rows = {"RL Agent": metrics(agent_oos),
            "Equal-Weight": metrics(base_oos["equal"]),
            "60/40": metrics(base_oos["6040"]),
            "Risk Parity": metrics(base_oos["rparity"]),
            "Mean-Variance": metrics(base_oos["meanvar"]),
            "Random": metrics(base_oos["random"])}
    hdr = f"{'Strategy':<16}{'Ret%':>8}{'Annual%':>9}{'Vol%':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD%':>8}"
    print(hdr); print("-"*len(hdr))
    for name, m in rows.items():
        print(f"{name:<16}{m['total']:>8.1f}{m['annual']:>9.1f}{m['vol']:>7.1f}"
              f"{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{(m['calmar'] if np.isfinite(m['calmar']) else 0):>8.2f}{m['mdd']:>8.1f}")

    # ---- Deflated Sharpe Ratio: does the agent beat luck, given the number of trials? ----
    a = agent_oos[np.isfinite(agent_oos)]
    sk = stats.skew(a); ku = stats.kurtosis(a, fisher=False)
    sr_annual = metrics(agent_oos)["sharpe"]
    dsr, sr0 = deflated_sharpe(sr_annual, len(a), seed_sharpes, sk, ku)
    print(f"\nDeflated Sharpe Ratio: trials={len(seed_sharpes)}, "
          f"Sharpe-by-luck-only~{sr0:.2f} (annualised), agent Sharpe={sr_annual:.2f} (annualised)")
    print(f"  DSR (probability that the edge is real) = {dsr:.3f}  "
          f"{'-> credible' if dsr > 0.95 else '-> NOT significant (within noise)'}")

    # ---- Ultimate judge test: synthetic data ----
    print("\n=== ROBUSTNESS: SYNTHETIC DATA (no time structure) ===")
    synth = synthetic_returns(dev_returns, seed=7)
    s_agent, s_base, _ = walk_forward(synth, "SYNTH")
    ms = metrics(s_agent)["sharpe"]; mb = metrics(s_base["rparity"])["sharpe"]
    print(f"Agent Sharpe on SYNTHETIC data: {ms:.2f} (risk parity: {mb:.2f})")
    print("  " + ("OK: the agent does not create signal out of noise." if ms < 0.5
                  else "WARNING: the agent 'wins' on pure noise -> the real result is suspicious."))

    # ---- Regime analysis ----
    print("\n=== ROBUSTNESS: PERFORMANCE BY REGIME ===")
    # align the regime to the length of the agent's OOS series (simple approximation)
    hv = regime_split(dev_returns)[-len(agent_oos):]
    hi = metrics(agent_oos[hv]); lo = metrics(agent_oos[~hv])
    print(f"  High vol: Sharpe={hi['sharpe']:.2f}  MaxDD={hi['mdd']:.1f}%")
    print(f"  Low vol:  Sharpe={lo['sharpe']:.2f}  MaxDD={lo['mdd']:.1f}%")
    print("  (expected from a real risk manager: bigger relative advantage in high vol)")

    # ---- SACRED HELD-OUT: a single pass, at the end ----
    print("\n=== SACRED HELD-OUT (single evaluation) ===")
    heldout_returns = to_returns(heldout_prices)
    # train on ALL of dev, evaluate on the held-out set (average of seeds)
    ho_rets = []
    for s in range(N_SEEDS):
        model = train_agent(dev_returns, seed=99000 + s)
        rets, _ = rollout_weights(model, heldout_returns)
        ho_rets.append(rets)
    L = min(len(x) for x in ho_rets)
    ho_agent = np.mean([x[:L] for x in ho_rets], axis=0)
    print(f"{'Strategy':<16}{'Sharpe':>8}{'Annual%':>9}{'MaxDD%':>8}")
    for name, series in [("RL Agent", ho_agent),
                         ("Equal-Weight", bh_equal_weight(heldout_returns)[-L:]),
                         ("Risk Parity", risk_parity(heldout_returns)[-L:]),
                         ("Mean-Variance", mean_variance(dev_returns, heldout_returns)[-L:])]:
        m = metrics(series)
        print(f"{name:<16}{m['sharpe']:>8.2f}{m['annual']:>9.1f}{m['mdd']:>8.1f}")

    # ---- Charts ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(nav_from_returns(agent_oos), label="RL Agent", lw=2)
    ax[0].plot(nav_from_returns(base_oos["rparity"]), label="Risk Parity", alpha=.8)
    ax[0].plot(nav_from_returns(base_oos["equal"]), label="Equal-Weight", alpha=.8)
    ax[0].set_title("Walk-forward OOS - NAV"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(nav_from_returns(ho_agent), label="RL Agent", lw=2)
    ax[1].plot(nav_from_returns(risk_parity(heldout_returns)[-L:]), label="Risk Parity", alpha=.8)
    ax[1].set_title("SACRED HELD-OUT - NAV"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=120, bbox_inches="tight")
    print("\n[chart saved as results.png]")

    print("\n" + "="*70)
    print("HONEST VERDICT: the most valuable result may be 'does not beat risk parity")
    print("significantly'. Reporting it with the DSR + synthetic data + held-out test")
    print("shows the evaluation was done carefully, which matters more than the number.")
    print("="*70)


if __name__ == "__main__":
    main()



