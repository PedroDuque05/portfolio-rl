# Portfolio allocation with reinforcement learning (PPO) on 6 ETFs.
# The agent is kept small on purpose, since there is very little signal in daily returns.
# I focused on validation: walk-forward, a held-out set, several seeds, a deflated Sharpe,
# a synthetic-data test and a bootstrap comparison against the best simple baselines.
# All strategies (agent and baselines) follow the same rules: simple returns, weights that
# drift with prices, and transaction costs on turnover.
# Run it in Google Colab (Runtime -> Run all). Takes around 5-15 min on CPU.
#
# Known limitations:
# - The ETFs were picked with hindsight (survivorship bias), I did not fix that here.
# - The high/low volatility threshold in the regime analysis uses the quantile of the whole
#   DEV period. It is only used to analyse results, never by the agent, so it is not leakage.

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

import warnings, random
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
ETFS  = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG"]
START = "2010-01-01"
END   = "2026-09-29"        # fixed so results are reproducible (yfinance treats END as exclusive)

TRANSACTION_COST = 0.001    # 10 bps per unit of turnover
LOOKBACK         = 20       # days used for the state (avg return, vol)

# reward
ETA_DSR      = 0.02         # adaptation rate of the differential Sharpe
REWARD_SCALE = 1.0          # the DSR term is ~O(1) per step, which suits PPO
DD_RATIO     = 0.2          # drawdown term sized to ~20% of the DSR term (0 = no drawdown penalty)

# agent (small on purpose)
TRAIN_TIMESTEPS = 20000
N_SEEDS         = 3
POLICY_KWARGS   = dict(net_arch=[64, 64])
RANDOM_START    = True      # training episodes start on a random day
MIN_EPISODE     = 126       # shortest training episode (~6 months)

# validation
EMBARGO_DAYS  = 5           # gap between train and test (not strictly needed, since the reward
                            # only uses same-day returns, but it costs little)
HELDOUT_DAYS  = 378         # last ~18 months, only used once at the end
WF_TRAIN_DAYS = 756         # ~3 years train per window
WF_TEST_DAYS  = 189         # ~9 months test per window
WF_STEP_DAYS  = 189

# baselines
REBAL_FREQ  = 21            # monthly rebalancing
RP_LOOKBACK = 60            # risk parity volatility window
RP_WARMUP   = int(np.ceil(RP_LOOKBACK / REBAL_FREQ)) * REBAL_FREQ   # 63: first rebalance lands on the first eval day
NEVER       = 10**9         # rebal value meaning buy and hold

RF_ANNUAL = 0.0             # risk-free rate, kept at 0 for simplicity
PERIODS   = 252


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(-1)
    return px[tickers].dropna()

def to_returns(prices):
    """Simple returns, so that (1 + r) products are exact everywhere."""
    return prices.pct_change().dropna()

prices = download_prices(ETFS, START, END)
print(f"Prices: {prices.shape[0]} days, {prices.shape[1]} ETFs "
      f"({prices.index.min().date()} -> {prices.index.max().date()})")

# the held-out set is separated here and not touched until the end
dev_prices     = prices.iloc[:-HELDOUT_DAYS]
heldout_prices = prices.iloc[-(HELDOUT_DAYS + LOOKBACK + 1):]   # extra days to build the first state
print(f"DEV: {dev_prices.shape[0]} days | HELD-OUT: {HELDOUT_DAYS} days")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class PortfolioEnv(gym.Env):
    """
    State:  recent avg return, recent vol and current weights (for each ETF)
    Action: logits -> softmax -> long-only target weights that sum to 1
    Reward: differential Sharpe ratio + drawdown penalty, after transaction costs
    """
    metadata = {"render_modes": []}

    def __init__(self, returns, transaction_cost=TRANSACTION_COST, lookback=LOOKBACK,
                 eta=ETA_DSR, dd_penalty=0.0, reward_scale=REWARD_SCALE,
                 random_start=False, min_episode=MIN_EPISODE):
        super().__init__()
        self.returns = returns.values if hasattr(returns, "values") else returns
        self.n = self.returns.shape[1]
        self.tc = transaction_cost
        self.lookback = lookback
        self.eta = eta
        self.dd_penalty = dd_penalty
        self.reward_scale = reward_scale
        self.random_start = random_start
        self.min_episode = min_episode

        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(self.n * 3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        last_start = len(self.returns) - self.min_episode
        if self.random_start and last_start > self.lookback:
            self.t = int(self.np_random.integers(self.lookback, last_start))
        else:
            self.t = self.lookback
        self.weights = np.ones(self.n) / self.n

        # warm start of the differential Sharpe with past data only, so the first
        # steps do not divide by a near-zero variance
        past = self.returns[self.t - self.lookback:self.t] @ self.weights
        self.A = float(past.mean())
        self.B = float((past**2).mean())

        self.nav = 1.0
        self.peak = 1.0
        return self._obs(), {}

    def _obs(self):
        window = self.returns[self.t - self.lookback:self.t]
        mean_r = window.mean(axis=0) * 100.0    # fixed scaling, no global stats
        vol_r  = window.std(axis=0) * 100.0
        obs = np.concatenate([mean_r, vol_r, self.weights]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        w = softmax(np.asarray(action, dtype=np.float64))
        r_t = self.returns[self.t]

        turnover = float(np.sum(np.abs(w - self.weights)))   # vs the weights after drift
        port_ret = float(np.dot(w, r_t))                      # gross return
        net_ret  = port_ret - turnover * self.tc

        self.nav *= (1.0 + net_ret)
        self.peak = max(self.peak, self.nav)
        drawdown = (self.nav - self.peak) / self.peak         # always <= 0

        # differential Sharpe ratio (Moody & Saffell)
        dA = net_ret - self.A
        dB = net_ret**2 - self.B
        var = self.B - self.A**2
        if var > 1e-12:
            dsr = (self.B * dA - 0.5 * self.A * dB) / (var**1.5)
        else:
            dsr = 0.0
        self.A += self.eta * dA
        self.B += self.eta * dB

        reward = self.reward_scale * dsr + self.dd_penalty * drawdown

        # weights drift with the day's returns before the next decision
        self.weights = w * (1.0 + r_t) / (1.0 + port_ret)
        self.t += 1
        done = self.t >= len(self.returns)                   # the last day is also used
        info = {"net_ret": net_ret, "weights": w, "nav": self.nav,
                "dsr": dsr, "dd": drawdown}
        return self._obs(), float(reward), bool(done), False, info


def calibrate_dd_penalty(returns_train, ratio=DD_RATIO):
    """
    Drawdown weight chosen so that, for an equal-weight portfolio on the TRAIN data,
    the drawdown term is on average `ratio` times the size of the DSR term.
    Uses training data only.
    """
    if ratio <= 0:
        return 0.0
    env = PortfolioEnv(returns_train)
    env.reset()
    done = False
    dsr_t, dd_t = [], []
    while not done:
        _, _, done, _, info = env.step(np.zeros(env.n))      # zeros -> equal weights
        dsr_t.append(abs(REWARD_SCALE * info["dsr"]))
        dd_t.append(abs(info["dd"]))
    mean_dd = np.mean(dd_t)
    return 0.0 if mean_dd < 1e-8 else ratio * np.mean(dsr_t) / mean_dd


def rollout_weights(model, returns):
    """Run the trained policy on `returns` and return the net returns and weights."""
    env = PortfolioEnv(returns)                             # evaluation: fixed start, reward unused
    obs, _ = env.reset()
    done = False
    rets, ws = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rets.append(info["net_ret"]); ws.append(info["weights"])
    return np.array(rets), np.array(ws)


# ---------------------------------------------------------------------------
# Baselines (same rules as the agent: drift, rebalancing, costs)
# ---------------------------------------------------------------------------
def simulate(returns, target_fn, rebal=REBAL_FREQ, tc=TRANSACTION_COST):
    """
    Weights drift with prices, are reset to target_fn every `rebal` days and pay tc
    on turnover. target_fn(R, t) may only use R[:t].
    """
    R = returns.values
    w = target_fn(R, 0)
    out = np.zeros(len(R))
    for t in range(len(R)):
        cost = 0.0
        if t > 0 and t % rebal == 0:
            new_w = target_fn(R, t)
            cost = np.abs(new_w - w).sum() * tc
            w = new_w
        gross = float(w @ R[t])
        out[t] = gross - cost
        w = w * (1.0 + R[t]) / (1.0 + gross)
    return out

def nav_from_returns(rets):
    return np.cumprod(1.0 + np.asarray(rets))

def bh_equal_weight(returns):
    """Buy and hold, 1/N, no rebalancing."""
    n = returns.shape[1]
    return simulate(returns, lambda R, t: np.ones(n) / n, rebal=NEVER)

def bh_6040(returns):
    cols = list(returns.columns)
    w = np.zeros(len(cols))
    w[cols.index("SPY")] = 0.6; w[cols.index("AGG")] = 0.4
    return simulate(returns, lambda R, t: w, rebal=NEVER)

def risk_parity(returns, lookback=RP_LOOKBACK, rebal=REBAL_FREQ):
    """Inverse volatility weights, rebalanced every `rebal` days using past data only."""
    def target(R, t):
        if t < lookback:
            return np.ones(R.shape[1]) / R.shape[1]
        inv = 1.0 / (R[t-lookback:t].std(axis=0) + 1e-8)
        return inv / inv.sum()
    return simulate(returns, target, rebal)

def mean_variance(returns_train, returns_eval, shrink=0.1, rebal=REBAL_FREQ):
    """Markowitz weights estimated on the train set only, then applied to the eval set."""
    Rtr = returns_train.values
    n = Rtr.shape[1]
    mu = Rtr.mean(axis=0)
    cov = np.cov(Rtr.T)
    cov = cov + shrink * np.trace(cov) / n * np.eye(n)   # ridge relative to the data's scale
    w = np.clip(np.linalg.solve(cov, mu), 0, None)       # long-only
    s = w.sum()
    w = w / s if s > 1e-12 else np.ones(n) / n           # all means <= 0 -> fall back to 1/N
    return simulate(returns_eval, lambda R, t: w, rebal)

def random_alloc(returns, seed=0, rebal=REBAL_FREQ):
    w = np.random.default_rng(seed).dirichlet(np.ones(returns.shape[1]))
    return simulate(returns, lambda R, t: w, rebal)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(rets, rf_annual=RF_ANNUAL, periods=PERIODS):
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return dict(total=0, annual=0, vol=0, sharpe=0, sortino=0, calmar=0, mdd=0)
    nav = nav_from_returns(r)
    total = nav[-1] - 1
    years = len(r) / periods
    annual = nav[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(periods)
    excess = r - rf_annual / periods
    sharpe = excess.mean() / (r.std() + 1e-12) * np.sqrt(periods)   # annualised
    downside = np.sqrt(np.mean(np.minimum(r, 0.0)**2))               # downside deviation
    sortino = excess.mean() / (downside + 1e-12) * np.sqrt(periods)
    peak = np.maximum.accumulate(np.concatenate([[1.0], nav]))[1:]
    mdd = ((nav - peak) / peak).min()
    calmar = annual / abs(mdd) if mdd < 0 else np.nan
    return dict(total=total*100, annual=annual*100, vol=vol*100, sharpe=sharpe,
                sortino=sortino, calmar=calmar, mdd=mdd*100)


# ---------------------------------------------------------------------------
# Probabilistic / deflated Sharpe ratio (Bailey & Lopez de Prado)
# The formula needs the Sharpe per period (daily here), so the annualised
# Sharpes are converted to daily inside deflated_sharpe.
# ---------------------------------------------------------------------------
def probabilistic_sharpe(sr_hat, n_obs, sr_benchmark, skew, kurt):
    """Probability that the true Sharpe is above sr_benchmark (all Sharpes daily)."""
    num = (sr_hat - sr_benchmark) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew * sr_hat + (kurt - 1) / 4.0 * sr_hat**2)
    return float(stats.norm.cdf(num / (den + 1e-12)))

def deflated_sharpe(sharpe_hat_annual, n_obs, sharpe_trials_annual, skew, kurt,
                    periods=PERIODS):
    """
    Adjusts the Sharpe for the number of trials. Each trial is one seed's full
    walk-forward track record. Takes annualised Sharpes, returns the DSR and the
    Sharpe expected by luck (annualised).
    Note: the honest number of trials also includes every configuration I tried
    while developing, which the code cannot know.
    """
    to_daily = 1.0 / np.sqrt(periods)
    sr_hat_d = sharpe_hat_annual * to_daily
    trials_d = np.asarray(sharpe_trials_annual, dtype=float) * to_daily

    N = max(len(trials_d), 2)
    var_sr = np.var(trials_d, ddof=1) + 1e-12
    gamma = 0.5772156649                      # Euler-Mascheroni constant
    z1 = stats.norm.ppf(1 - 1.0 / N)
    z2 = stats.norm.ppf(1 - 1.0 / (N * np.e))
    sr0_d = np.sqrt(var_sr) * ((1 - gamma) * z1 + gamma * z2)

    dsr = probabilistic_sharpe(sr_hat_d, n_obs, sr0_d, skew, kurt)
    return dsr, sr0_d * np.sqrt(periods)


def sharpe_diff_bootstrap(a, b, block=21, n_boot=2000, seed=0):
    """
    Block bootstrap of the Sharpe difference between two return series on the same days.
    Returns the observed difference and the share of bootstrap samples where it is <= 0
    (an approximate p-value for 'a has a higher Sharpe than b').
    """
    a = np.asarray(a); b = np.asarray(b)
    rng = np.random.default_rng(seed)
    T = len(a); n_blocks = max(T // block, 1)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, T - block, n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        diffs[i] = metrics(a[idx])["sharpe"] - metrics(b[idx])["sharpe"]
    return metrics(a)["sharpe"] - metrics(b)["sharpe"], float(np.mean(diffs <= 0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(returns_train, seed, dd_penalty, timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed); random.seed(seed)
    def make():
        return PortfolioEnv(returns_train, dd_penalty=dd_penalty, random_start=RANDOM_START)
    venv = DummyVecEnv([make])
    model = PPO("MlpPolicy", venv, seed=seed, policy_kwargs=POLICY_KWARGS,
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
                gamma=0.99, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def regime_split(returns, lookback=20, q=0.66):
    """True on high volatility days, False otherwise."""
    mkt = returns.mean(axis=1)
    vol = mkt.rolling(lookback).std()
    thr = vol.quantile(q)
    return (vol > thr).values

def walk_forward(returns, label="DEV"):
    """
    Rolling windows with an embargo between train and test. In each window I train
    N_SEEDS agents and average their out-of-sample returns. The baselines are
    evaluated on exactly the same days as the agent.
    """
    T = len(returns)
    hv_full = regime_split(returns)
    agent_oos, regime_oos = [], []
    base_oos = {k: [] for k in ["equal", "6040", "rparity", "meanvar", "random"]}
    seed_track = [[] for _ in range(N_SEEDS)]     # each seed's full track record

    start = 0
    win = 0
    while start + WF_TRAIN_DAYS + EMBARGO_DAYS + WF_TEST_DAYS <= T:
        tr0, tr1 = start, start + WF_TRAIN_DAYS
        te0 = tr1 + EMBARGO_DAYS
        te1 = te0 + WF_TEST_DAYS
        win += 1
        r_train = returns.iloc[tr0:tr1]
        r_test  = returns.iloc[te0:te1]
        dd_pen  = calibrate_dd_penalty(r_train)

        seed_rets = []
        for s in range(N_SEEDS):
            model = train_agent(r_train, seed=1000 * win + s, dd_penalty=dd_pen)
            rets, _ = rollout_weights(model, r_test)
            seed_rets.append(rets)
        L = min(len(x) for x in seed_rets)        # = WF_TEST_DAYS - LOOKBACK
        for s in range(N_SEEDS):
            seed_track[s].append(seed_rets[s][:L])
        agent_oos.append(np.mean([x[:L] for x in seed_rets], axis=0))
        regime_oos.append(hv_full[te1 - L:te1])   # same days as the agent's returns

        # baselines on the days the agent is actually evaluated on
        r_eval = r_test.iloc[-L:]
        base_oos["equal"].append(bh_equal_weight(r_eval))
        base_oos["6040"].append(bh_6040(r_eval))
        base_oos["rparity"].append(risk_parity(returns.iloc[te1 - L - RP_WARMUP:te1])[-L:])
        base_oos["meanvar"].append(mean_variance(r_train, r_eval))
        base_oos["random"].append(random_alloc(r_eval, seed=win))

        print(f"  [{label}] window {win}: train {tr0}-{tr1}, test {te0}-{te1} "
              f"(embargo {EMBARGO_DAYS}d), L={L}, dd_penalty={dd_pen:.2f}")
        start += WF_STEP_DAYS

    agent_oos = np.concatenate(agent_oos) if agent_oos else np.array([])
    regime_oos = np.concatenate(regime_oos) if regime_oos else np.array([], dtype=bool)
    for k in base_oos:
        base_oos[k] = np.concatenate(base_oos[k]) if base_oos[k] else np.array([])
    seed_sharpes = [metrics(np.concatenate(x))["sharpe"] for x in seed_track if x]
    return agent_oos, base_oos, seed_sharpes, regime_oos


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def synthetic_returns(returns, seed=0):
    """
    Resamples whole days with replacement. Keeps the distribution and the correlations
    between ETFs but destroys the time structure, so there is nothing to learn beyond
    the average drift. If the agent beats simple baselines here, something is wrong.
    """
    rng = np.random.default_rng(seed)
    R = returns.values
    idx = rng.integers(0, len(R), size=len(R))
    return pd.DataFrame(R[idx], columns=returns.columns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dev_returns = to_returns(dev_prices)

    print("\n=== WALK-FORWARD ===")
    agent_oos, base_oos, seed_sharpes, hv = walk_forward(dev_returns, "DEV")

    print("\n=== OUT-OF-SAMPLE METRICS ===")
    rows = {"RL Agent": metrics(agent_oos),
            "Equal-Weight": metrics(base_oos["equal"]),
            "60/40": metrics(base_oos["6040"]),
            "Risk Parity": metrics(base_oos["rparity"]),
            "Mean-Variance": metrics(base_oos["meanvar"]),
            "Random": metrics(base_oos["random"])}
    hdr = f"{'Strategy':<16}{'Ret%':>8}{'Annual%':>9}{'Vol%':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD%':>8}"
    print(hdr); print("-" * len(hdr))
    for name, m in rows.items():
        calmar = m["calmar"] if np.isfinite(m["calmar"]) else 0
        print(f"{name:<16}{m['total']:>8.1f}{m['annual']:>9.1f}{m['vol']:>7.1f}"
              f"{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{calmar:>8.2f}{m['mdd']:>8.1f}")
    print("\nPer-seed Sharpes (the agent row above is the average of the seeds):",
          np.round(seed_sharpes, 2))

    # deflated Sharpe: does the agent beat luck given the number of trials?
    a = agent_oos[np.isfinite(agent_oos)]
    sk = stats.skew(a); ku = stats.kurtosis(a, fisher=False)
    sr_annual = metrics(agent_oos)["sharpe"]
    dsr, sr0 = deflated_sharpe(sr_annual, len(a), seed_sharpes, sk, ku)
    print(f"\nDeflated Sharpe: trials={len(seed_sharpes)}, "
          f"luck-only Sharpe ~ {sr0:.2f}, agent Sharpe = {sr_annual:.2f} (both annualised)")
    print(f"  DSR = {dsr:.3f}  "
          f"{'-> significant' if dsr > 0.95 else '-> not significant, could be noise'}")

    # does the agent beat the simple baselines?
    print("\n=== AGENT VS BASELINES (block bootstrap) ===")
    for key, name in [("rparity", "Risk Parity"), ("equal", "Equal-Weight")]:
        d, p = sharpe_diff_bootstrap(agent_oos, base_oos[key])
        print(f"  vs {name:<13} Sharpe diff {d:+.2f}  (share of samples <= 0: {p:.2f})")

    # synthetic data test
    print("\n=== SYNTHETIC DATA TEST ===")
    synth = synthetic_returns(dev_returns, seed=7)
    s_agent, s_base, _, _ = walk_forward(synth, "SYNTH")
    ms = metrics(s_agent)["sharpe"]
    mb = max(metrics(s_base[k])["sharpe"] for k in ["equal", "rparity"])
    print(f"Agent Sharpe on synthetic data: {ms:.2f} (best simple baseline: {mb:.2f})")
    # the 0.2 margin is arbitrary, it only absorbs noise
    print("  " + ("Fine, the agent does not beat simple baselines on noise." if ms <= mb + 0.2
                  else "Warning: the agent beats the baselines on pure noise, so the results are suspicious."))

    # performance by volatility regime (no drawdown here: the days are not contiguous)
    print("\n=== BY REGIME ===")
    hi = metrics(agent_oos[hv]); lo = metrics(agent_oos[~hv])
    print(f"  High vol: Sharpe={hi['sharpe']:.2f}  Vol={hi['vol']:.1f}%  ({hv.sum()} days)")
    print(f"  Low vol:  Sharpe={lo['sharpe']:.2f}  Vol={lo['vol']:.1f}%  ({(~hv).sum()} days)")

    # held-out set, evaluated only once, with the same training setup as each walk-forward window
    print("\n=== HELD-OUT ===")
    heldout_returns = to_returns(heldout_prices)
    r_train_ho = dev_returns.iloc[-WF_TRAIN_DAYS:]
    dd_pen = calibrate_dd_penalty(r_train_ho)
    ho_rets = []
    for s in range(N_SEEDS):
        model = train_agent(r_train_ho, seed=99000 + s, dd_penalty=dd_pen)
        rets, _ = rollout_weights(model, heldout_returns)
        ho_rets.append(rets)
    L = min(len(x) for x in ho_rets)
    ho_agent = np.mean([x[:L] for x in ho_rets], axis=0)
    ho_eval = heldout_returns.iloc[-L:]
    rp_heldout = risk_parity(to_returns(prices.iloc[-(L + RP_WARMUP + 1):]))[-L:]

    print(f"{'Strategy':<16}{'Sharpe':>8}{'Annual%':>9}{'MaxDD%':>8}")
    for name, series in [("RL Agent", ho_agent),
                         ("Equal-Weight", bh_equal_weight(ho_eval)),
                         ("60/40", bh_6040(ho_eval)),
                         ("Risk Parity", rp_heldout),
                         ("Mean-Variance", mean_variance(r_train_ho, ho_eval))]:
        m = metrics(series)
        print(f"{name:<16}{m['sharpe']:>8.2f}{m['annual']:>9.1f}{m['mdd']:>8.1f}")

    # charts
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(nav_from_returns(agent_oos), label="RL Agent", lw=2)
    ax[0].plot(nav_from_returns(base_oos["rparity"]), label="Risk Parity", alpha=.8)
    ax[0].plot(nav_from_returns(base_oos["equal"]), label="Equal-Weight", alpha=.8)
    ax[0].set_title("Walk-forward out-of-sample NAV"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(nav_from_returns(ho_agent), label="RL Agent", lw=2)
    ax[1].plot(nav_from_returns(rp_heldout), label="Risk Parity", alpha=.8)
    ax[1].plot(nav_from_returns(bh_equal_weight(ho_eval)), label="Equal-Weight", alpha=.8)
    ax[1].set_title("Held-out NAV"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=120, bbox_inches="tight")
    print("\n[chart saved as results.png]")


if __name__ == "__main__":
    main()

