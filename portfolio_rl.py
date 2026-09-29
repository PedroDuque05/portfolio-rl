# Portfolio allocation with reinforcement learning (PPO) on 6 ETFs.
# The agent is kept small on purpose, since there is very little signal in daily returns.
# I focused on validation: walk-forward, a held-out set, several seeds and a deflated Sharpe.
# Run it in Google Colab (Runtime -> Run all). Takes around 5-12 min on CPU.
#
# Note: the ETFs were picked with hindsight (survivorship bias), I did not fix that here.

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
END   = None

TRANSACTION_COST = 0.001    # 10 bps per unit of turnover
LOOKBACK         = 20       # days used for the state (avg return, vol)

# reward
ETA_DSR      = 0.02         # adaptation rate of the differential Sharpe
DD_PENALTY   = 0.10         # drawdown penalty weight
REWARD_SCALE = 100.0

# agent (small on purpose)
TRAIN_TIMESTEPS = 20000
N_SEEDS         = 3
POLICY_KWARGS   = dict(net_arch=[64, 64])

# validation
EMBARGO_DAYS  = 5           # gap between train and test
HELDOUT_DAYS  = 378         # last ~18 months, only used once at the end
WF_TRAIN_DAYS = 756         # ~3 years train per window
WF_TEST_DAYS  = 189         # ~9 months test per window
WF_STEP_DAYS  = 189

REBAL_FREQ = 21             # monthly rebalancing for the baselines
RF_ANNUAL  = 0.0            # risk-free rate, kept at 0 for simplicity
PERIODS    = 252


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
    return np.log(prices / prices.shift(1)).dropna()

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
    Action: logits -> softmax -> long-only weights that sum to 1
    Reward: differential Sharpe ratio + drawdown penalty, after transaction costs
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
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(self.n * 3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.weights = np.ones(self.n) / self.n
        self.A = 0.0      # running averages for the differential Sharpe
        self.B = 0.0
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

        port_ret = float(np.dot(w, self.returns[self.t]))
        turnover = float(np.sum(np.abs(w - self.weights)))
        net_ret  = port_ret - turnover * self.tc

        self.nav *= (1.0 + net_ret)
        self.peak = max(self.peak, self.nav)
        drawdown = (self.nav - self.peak) / self.peak    # always <= 0

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

        self.weights = w
        self.t += 1
        done = self.t >= len(self.returns) - 1
        info = {"net_ret": net_ret, "weights": w, "nav": self.nav}
        return self._obs(), float(reward), bool(done), False, info


def rollout_weights(model, returns, lookback=LOOKBACK):
    """Run the trained policy on `returns` and return the net returns and weights."""
    env = PortfolioEnv(returns, TRANSACTION_COST, lookback, ETA_DSR, DD_PENALTY, REWARD_SCALE)
    obs, _ = env.reset()
    done = False
    rets, ws = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        rets.append(info["net_ret"]); ws.append(info["weights"])
    return np.array(rets), np.array(ws)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def nav_from_returns(rets):
    return np.cumprod(1.0 + np.asarray(rets))

def bh_equal_weight(returns):
    """Buy and hold, 1/N, no rebalancing."""
    R = returns.values
    w0 = np.ones(R.shape[1]) / R.shape[1]
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
    """Inverse volatility weights, rebalanced every `rebal` days using past data only."""
    R = returns.values; T, n = R.shape
    out = np.zeros(T); w = np.ones(n) / n
    for t in range(T):
        if t >= lookback and t % rebal == 0:
            vol = R[t-lookback:t].std(axis=0) + 1e-8
            inv = 1.0 / vol
            w = inv / inv.sum()
        out[t] = float(np.dot(w, R[t]))
    return out

def mean_variance(returns_train, returns_eval, ridge=1e-3):
    """Markowitz weights estimated on the train set only, then applied to the eval set."""
    Rtr = returns_train.values
    mu = Rtr.mean(axis=0)
    cov = np.cov(Rtr.T) + ridge * np.eye(Rtr.shape[1])
    w = np.linalg.solve(cov, mu)
    w = np.clip(w, 0, None)                 # long-only
    w = w / (w.sum() + 1e-12)
    return returns_eval.values @ w

def random_alloc(returns, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(returns.shape[1]))
    return returns.values @ w


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(rets, rf_annual=RF_ANNUAL, periods=PERIODS):
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return dict(total=0, annual=0, vol=0, sharpe=0, sortino=0, calmar=0, mdd=0)
    nav = nav_from_returns(r)
    total = nav[-1] / nav[0] - 1
    years = len(r) / periods
    annual = (nav[-1] / nav[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = r.std() * np.sqrt(periods)
    excess = r - rf_annual / periods
    sharpe = excess.mean() / (r.std() + 1e-12) * np.sqrt(periods)   # annualised
    downside = r[r < 0].std() if (r < 0).any() else 1e-12
    sortino = excess.mean() / (downside + 1e-12) * np.sqrt(periods)
    peak = np.maximum.accumulate(nav)
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
    Adjusts the Sharpe for the number of trials (seeds and windows I tried).
    Takes annualised Sharpes, returns the DSR and the Sharpe expected by luck (annualised).
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_agent(returns_train, seed, timesteps=TRAIN_TIMESTEPS):
    np.random.seed(seed); random.seed(seed)
    def make():
        return PortfolioEnv(returns_train, TRANSACTION_COST, LOOKBACK,
                            ETA_DSR, DD_PENALTY, REWARD_SCALE)
    venv = DummyVecEnv([make])
    model = PPO("MlpPolicy", venv, seed=seed, policy_kwargs=POLICY_KWARGS,
                learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
                gamma=0.99, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def walk_forward(dev_returns, label="DEV"):
    """
    Rolling windows with an embargo between train and test. In each window I train
    N_SEEDS agents and average their out-of-sample returns. The baselines are
    evaluated on the same test segments.
    """
    T = len(dev_returns)
    agent_oos = []
    base_oos = {k: [] for k in ["equal", "6040", "rparity", "meanvar", "random"]}
    seed_sharpes = []    # annualised Sharpes, used for the DSR

    start = 0
    win = 0
    while start + WF_TRAIN_DAYS + EMBARGO_DAYS + WF_TEST_DAYS <= T:
        tr0, tr1 = start, start + WF_TRAIN_DAYS
        te0 = tr1 + EMBARGO_DAYS
        te1 = te0 + WF_TEST_DAYS
        win += 1
        r_train = dev_returns.iloc[tr0:tr1]
        r_test  = dev_returns.iloc[te0:te1]

        seed_rets = []
        for s in range(N_SEEDS):
            model = train_agent(r_train, seed=1000 * win + s)
            rets, _ = rollout_weights(model, r_test)
            seed_rets.append(rets)
            seed_sharpes.append(metrics(rets)["sharpe"])
        L = min(len(x) for x in seed_rets)
        agent_oos.append(np.mean([x[:L] for x in seed_rets], axis=0))

        # baselines, cut to the same length as the agent
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


# ---------------------------------------------------------------------------
# Robustness checks
# ---------------------------------------------------------------------------
def synthetic_returns(returns, seed=0):
    """
    Resamples whole days with replacement. Keeps the distribution and the correlations
    between ETFs but destroys the time structure, so there is nothing to learn.
    If the agent does well here, something is wrong with the pipeline.
    """
    rng = np.random.default_rng(seed)
    R = returns.values
    idx = rng.integers(0, len(R), size=len(R))
    return pd.DataFrame(R[idx], columns=returns.columns)

def regime_split(returns, lookback=20, q=0.66):
    """True on high volatility days, False otherwise."""
    mkt = returns.mean(axis=1)
    vol = mkt.rolling(lookback).std()
    thr = vol.quantile(q)
    return (vol > thr).values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    dev_returns = to_returns(dev_prices)

    print("\n=== WALK-FORWARD ===")
    agent_oos, base_oos, seed_sharpes = walk_forward(dev_returns, "DEV")

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

    # deflated Sharpe: does the agent beat luck given how many runs I did?
    a = agent_oos[np.isfinite(agent_oos)]
    sk = stats.skew(a); ku = stats.kurtosis(a, fisher=False)
    sr_annual = metrics(agent_oos)["sharpe"]
    dsr, sr0 = deflated_sharpe(sr_annual, len(a), seed_sharpes, sk, ku)
    print(f"\nDeflated Sharpe: trials={len(seed_sharpes)}, "
          f"luck-only Sharpe ~ {sr0:.2f}, agent Sharpe = {sr_annual:.2f} (both annualised)")
    print(f"  DSR = {dsr:.3f}  "
          f"{'-> significant' if dsr > 0.95 else '-> not significant, could be noise'}")

    # synthetic data test
    print("\n=== SYNTHETIC DATA TEST ===")
    synth = synthetic_returns(dev_returns, seed=7)
    s_agent, s_base, _ = walk_forward(synth, "SYNTH")
    ms = metrics(s_agent)["sharpe"]; mb = metrics(s_base["rparity"])["sharpe"]
    print(f"Agent Sharpe on synthetic data: {ms:.2f} (risk parity: {mb:.2f})")
    print("  " + ("Fine, the agent does not find signal in noise." if ms < 0.5
                  else "Warning: the agent does well on pure noise, so the results are suspicious."))

    # performance by volatility regime
    print("\n=== BY REGIME ===")
    hv = regime_split(dev_returns)[-len(agent_oos):]    # rough alignment with the OOS series
    hi = metrics(agent_oos[hv]); lo = metrics(agent_oos[~hv])
    print(f"  High vol: Sharpe={hi['sharpe']:.2f}  MaxDD={hi['mdd']:.1f}%")
    print(f"  Low vol:  Sharpe={lo['sharpe']:.2f}  MaxDD={lo['mdd']:.1f}%")

    # held-out set, evaluated only once
    print("\n=== HELD-OUT ===")
    heldout_returns = to_returns(heldout_prices)
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

    # charts
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(nav_from_returns(agent_oos), label="RL Agent", lw=2)
    ax[0].plot(nav_from_returns(base_oos["rparity"]), label="Risk Parity", alpha=.8)
    ax[0].plot(nav_from_returns(base_oos["equal"]), label="Equal-Weight", alpha=.8)
    ax[0].set_title("Walk-forward out-of-sample NAV"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(nav_from_returns(ho_agent), label="RL Agent", lw=2)
    ax[1].plot(nav_from_returns(risk_parity(heldout_returns)[-L:]), label="Risk Parity", alpha=.8)
    ax[1].set_title("Held-out NAV"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("results.png", dpi=120, bbox_inches="tight")
    print("\n[chart saved as results.png]")


if __name__ == "__main__":
    main()


