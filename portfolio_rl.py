# =====================================================================================
# PIPELINE DE RL PARA ALOCACAO DE CARTEIRA — versao a prova de auto-ilusao
# =====================================================================================
# FILOSOFIA (o porque de cada escolha):
#
#  1. AGENTE DELIBERADAMENTE MODESTO. Em dominio de baixo sinal/ruido alto, um agente
#     mais expressivo nao extrai mais sinal — decora melhor o percurso historico.
#     Simplicidade (PPO + MLP pequena + poucas features + poucos timesteps) E a
#     regularizacao que mais importa. A potencia esta na VALIDACAO, nao no modelo.
#
#  2. REWARD = GESTAO DE RISCO, nao previsao de direcao. Ja provamos (modelo de
#     direcao preso em ln(2)) que a direcao e imprevisivel. A reward usa o
#     Differential Sharpe Ratio (Moody & Saffell, 1998) — versao online e estavel
#     do Sharpe — mais uma penalizacao de drawdown. Pede ao agente que MODULE
#     EXPOSICAO ao risco (o unico sinal real: clustering de volatilidade), nao que
#     adivinhe retornos.
#
#  3. VALIDACAO DE TRES NIVEIS:
#       - treino: o agente aprende pesos dentro de cada janela.
#       - walk-forward: comparamos configuracoes / medimos generalizacao.
#       - HELD-OUT SAGRADO: um bloco final que NAO existe durante o desenvolvimento;
#         corre-se UMA vez no fim. E a unica estimativa honesta.
#
#  4. PURGING + EMBARGO entre treino e teste (Lopez de Prado): gap temporal que mata
#     a autocorrelacao das janelas de lookback sobrepostas.
#
#  5. MULTIPLAS SEEDS + DEFLATED SHARPE RATIO: o RL e estocastico; um numero unico
#     mente. O DSR desconta o Sharpe pelo n de tentativas — mede quantas vezes o
#     nosso proprio cerebro tocou nos dados (o multiple-testing e o assassino real).
#
#  6. TESTES DE ROBUSTEZ:
#       - DADOS SINTETICOS (juiz supremo): series com as mesmas propriedades
#         marginais mas SEM estrutura temporal. Se o agente "ganha" aqui, o pipeline
#         fabrica sinal a partir de ruido e o resultado real e falso.
#       - ANALISE POR REGIME: o edge deve aparecer onde a teoria diz (alta vol),
#         nao espalhado ao acaso. O ONDE ganha e, ele proprio, evidencia.
#
# NOTA DE HONESTIDADE: os ETFs abaixo foram escolhidos ex-post (survivorship/selection
# bias). Nao ha aqui base de dados de ETFs delistados; o vies fica NOMEADO mas nao
# totalmente eliminado. Num cenario de producao, o universo seria definido com a
# informacao disponivel no inicio.
#
# COMO CORRER: pip install -r requirements.txt && python portfolio_rl.py
# Parametros modestos DE PROPOSITO (ver filosofia). Demora ~5-12 min em CPU.
# =====================================================================================


# =====================================================================================
# SECCAO 1 — IMPORTS E CONFIGURACAO
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

# ---- Universo e periodo ----
ETFS   = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG"]
START  = "2010-01-01"
END    = None

# ---- Mecanica de mercado ----
TRANSACTION_COST = 0.001     # 10 bps por unidade de turnover
LOOKBACK         = 20        # dias para features de estado (retorno medio, vol)

# ---- Reward (gestao de risco) ----
ETA_DSR     = 0.02           # taxa de adaptacao do Differential Sharpe
DD_PENALTY  = 0.10           # peso da penalizacao de drawdown
REWARD_SCALE = 100.0         # escala o DSR para dar sinal decente ao PPO

# ---- Agente DELIBERADAMENTE MODESTO ----
TRAIN_TIMESTEPS = 20000      # modesto de proposito (menos memorizacao)
N_SEEDS         = 3          # n de seeds por configuracao
POLICY_KWARGS   = dict(net_arch=[64, 64])   # MLP pequena

# ---- Validacao ----
EMBARGO_DAYS    = 5          # gap treino->teste (purging/embargo)
HELDOUT_DAYS    = 378        # ~18 meses finais SAGRADOS (nunca tocar ate ao fim)
WF_TRAIN_DAYS   = 756        # ~3 anos de treino por janela walk-forward
WF_TEST_DAYS    = 189        # ~9 meses de teste por janela
WF_STEP_DAYS    = 189        # avanco entre janelas

REBAL_FREQ      = 21         # rebalanceamento mensal dos baselines dinamicos
RF_ANNUAL       = 0.0        # taxa sem risco (simplificacao)

print("Configuracao carregada. Agente modesto de proposito (ver filosofia).")


# =====================================================================================
# SECCAO 2 — DADOS: download + split DEV / HELD-OUT SAGRADO
# =====================================================================================
def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = px.columns.get_level_values(-1)
    px = px[tickers].dropna()
    return px

prices = download_prices(ETFS, START, END)
print(f"Precos: {prices.shape[0]} dias, {prices.shape[1]} ETFs "
      f"({prices.index.min().date()} -> {prices.index.max().date()})")

# SPLIT SAGRADO: o held-out final nao existe durante o desenvolvimento.
dev_prices     = prices.iloc[:-HELDOUT_DAYS]
heldout_prices = prices.iloc[-(HELDOUT_DAYS + LOOKBACK + 1):]   # +lookback p/ formar estado
print(f"DEV: {dev_prices.shape[0]} dias | HELD-OUT (sagrado): {HELDOUT_DAYS} dias")


# =====================================================================================
# SECCAO 3 — FEATURES ESTACIONARIAS (sem niveis de preco, sem leakage)
# =====================================================================================
def to_returns(prices):
    """Log-retornos diarios (estacionarios)."""
    return np.log(prices / prices.shift(1)).dropna()


# =====================================================================================
# SECCAO 4 — AMBIENTE: reward de Differential Sharpe + drawdown, acao via softmax
# =====================================================================================
class PortfolioEnv(gym.Env):
    """
    Estado:  [retorno medio recente (n), volatilidade recente (n), pesos atuais (n)]
    Acao:    logits -> softmax -> pesos long-only que somam 1 (sem hacks de normalizacao)
    Reward:  Differential Sharpe Ratio (online, estavel) + penalizacao de drawdown,
             liquido de custos de transacao. NAO recompensa direcao; recompensa
             retorno AJUSTADO AO RISCO e controlo de exposicao.
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
        # 3n features; ranges largos pois escalamos manualmente (sem stats globais => sem leakage)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(self.n * 3,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.lookback
        self.weights = np.ones(self.n) / self.n
        # estado do Differential Sharpe
        self.A = 0.0
        self.B = 0.0
        # estado de drawdown
        self.nav = 1.0
        self.peak = 1.0
        return self._obs(), {}

    def _obs(self):
        window = self.returns[self.t - self.lookback:self.t]
        mean_r = window.mean(axis=0) * 100.0     # escala fixa (sem leakage)
        vol_r  = window.std(axis=0)  * 100.0
        obs = np.concatenate([mean_r, vol_r, self.weights]).astype(np.float32)
        return np.clip(obs, -10.0, 10.0)

    def step(self, action):
        w = softmax(np.asarray(action, dtype=np.float64))   # long-only, soma 1

        # retorno do portfolio no passo seguinte, liquido de custos
        port_ret = float(np.dot(w, self.returns[self.t]))
        turnover = float(np.sum(np.abs(w - self.weights)))
        net_ret  = port_ret - turnover * self.tc

        # atualizar NAV e drawdown
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

        reward = self.reward_scale * dsr + self.dd_penalty * drawdown  # dd<0 penaliza

        self.weights = w
        self.t += 1
        done = self.t >= len(self.returns) - 1
        info = {"net_ret": net_ret, "weights": w, "nav": self.nav}
        return self._obs(), float(reward), bool(done), False, info


def rollout_weights(model, returns, lookback=LOOKBACK):
    """Corre a politica treinada sobre `returns`, devolve serie de retornos liquidos e pesos."""
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
# SECCAO 5 — BASELINES (a vara de medir; sem leakage)
# =====================================================================================
def nav_from_returns(rets):
    return np.cumprod(1.0 + np.asarray(rets))

def bh_equal_weight(returns):
    """Buy & hold 1/N (sem rebalanceamento)."""
    R = returns.values
    n = R.shape[1]
    w0 = np.ones(n) / n
    # drift dos pesos com os precos: NAV por ativo
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
    """Inverse-vol weighting, rebalanceado periodicamente (vol estimada so com passado)."""
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
    """Markowitz: pesos estimados SO no treino, aplicados no eval (sem leakage)."""
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
# SECCAO 6 — METRICAS HONESTAS
# =====================================================================================
def metrics(rets, rf_annual=RF_ANNUAL, periods=252):
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
    sharpe = excess.mean() / (r.std() + 1e-12) * np.sqrt(periods)
    downside = r[r < 0].std() if (r < 0).any() else 1e-12
    sortino = excess.mean() / (downside + 1e-12) * np.sqrt(periods)
    peak = np.maximum.accumulate(nav)
    mdd = ((nav - peak) / peak).min()
    calmar = annual / abs(mdd) if mdd < 0 else np.nan
    return dict(total=total*100, annual=annual*100, vol=vol*100, sharpe=sharpe,
                sortino=sortino, calmar=calmar, mdd=mdd*100)


# =====================================================================================
# SECCAO 7 — DEFLATED / PROBABILISTIC SHARPE RATIO (Bailey & Lopez de Prado)
# =====================================================================================
def probabilistic_sharpe(sharpe_hat, n_obs, sr_benchmark, skew, kurt):
    """PSR: P(SR verdadeiro > sr_benchmark) dado o SR observado e os momentos."""
    num = (sharpe_hat - sr_benchmark) * np.sqrt(n_obs - 1)
    den = np.sqrt(1 - skew*sharpe_hat + (kurt - 1)/4.0 * sharpe_hat**2)
    return float(stats.norm.cdf(num / (den + 1e-12)))

def deflated_sharpe(sharpe_hat, n_obs, sharpe_trials, skew, kurt):
    """
    DSR: corrige o melhor Sharpe observado pelo n de tentativas (multiple testing).
    sharpe_trials = lista dos Sharpes (ANUALIZADOS) de todas as configs/seeds testadas.
    """
    N = max(len(sharpe_trials), 2)
    var_sr = np.var(sharpe_trials, ddof=1) + 1e-12
    gamma = 0.5772156649  # Euler-Mascheroni
    z1 = stats.norm.ppf(1 - 1.0/N)
    z2 = stats.norm.ppf(1 - 1.0/(N*np.e))
    sr0 = np.sqrt(var_sr) * ((1 - gamma)*z1 + gamma*z2)   # Sharpe esperado SO por sorte
    dsr = probabilistic_sharpe(sharpe_hat, n_obs, sr0, skew, kurt)
    return dsr, sr0


# =====================================================================================
# SECCAO 8 — TREINAR + AVALIAR UMA JANELA
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
# SECCAO 9 — WALK-FORWARD + MULTI-SEED (o motor de validacao)
# =====================================================================================
def walk_forward(dev_returns, label="DEV"):
    """
    Janelas deslizantes com EMBARGO entre treino e teste. Para cada janela, treina
    N_SEEDS agentes e agrega. Devolve: serie de retornos OOS do agente (concatenada),
    e dict com baselines avaliados nos MESMOS segmentos.
    """
    T = len(dev_returns)
    agent_oos = []
    base_oos = {k: [] for k in ["equal", "6040", "rparity", "meanvar", "random"]}
    seed_sharpes = []   # para o DSR

    start = 0
    win = 0
    while start + WF_TRAIN_DAYS + EMBARGO_DAYS + WF_TEST_DAYS <= T:
        tr0, tr1 = start, start + WF_TRAIN_DAYS
        te0 = tr1 + EMBARGO_DAYS                         # EMBARGO/purging
        te1 = te0 + WF_TEST_DAYS
        win += 1
        r_train = dev_returns.iloc[tr0:tr1]
        r_test  = dev_returns.iloc[te0:te1]

        # treinar N_SEEDS e fazer media dos retornos OOS (mais honesto que best-seed)
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

        # baselines no MESMO segmento de teste (alinhar comprimento L)
        base_oos["equal"].append(bh_equal_weight(r_test)[-L:])
        base_oos["6040"].append(bh_6040(r_test)[-L:])
        base_oos["rparity"].append(risk_parity(r_test)[-L:])
        base_oos["meanvar"].append(mean_variance(r_train, r_test)[-L:])
        base_oos["random"].append(random_alloc(r_test, seed=win)[-L:])

        print(f"  [{label}] janela {win}: treino {tr0}-{tr1}, teste {te0}-{te1} "
              f"(embargo {EMBARGO_DAYS}d), L={L}")
        start += WF_STEP_DAYS

    agent_oos = np.concatenate(agent_oos) if agent_oos else np.array([])
    for k in base_oos:
        base_oos[k] = np.concatenate(base_oos[k]) if base_oos[k] else np.array([])
    return agent_oos, base_oos, seed_sharpes


# =====================================================================================
# SECCAO 10 — TESTES DE ROBUSTEZ
# =====================================================================================
def synthetic_returns(returns, seed=0):
    """
    Juiz supremo: reamostra DIAS INTEIROS com reposicao. Mantem a distribuicao marginal
    e as correlacoes entre ativos, mas DESTROI a estrutura temporal (autocorrelacao,
    clustering de volatilidade). Por construcao NAO ha nada para o agente aprender.
    Se o agente "ganhar" aqui, o pipeline fabrica sinal a partir de ruido.
    """
    rng = np.random.default_rng(seed)
    R = returns.values
    idx = rng.integers(0, len(R), size=len(R))
    synth = R[idx]
    return pd.DataFrame(synth, columns=returns.columns)

def regime_split(returns, lookback=20, q=0.66):
    """Classifica cada dia em ALTA vol vs BAIXA vol (vol de mercado realizada)."""
    mkt = returns.mean(axis=1)
    vol = mkt.rolling(lookback).std()
    thr = vol.quantile(q)
    return (vol > thr).values   # True = alta volatilidade


# =====================================================================================
# SECCAO 11 — ORQUESTRACAO
# =====================================================================================
def main():
    dev_returns = to_returns(dev_prices)

    print("\n=== WALK-FORWARD (validacao) ===")
    agent_oos, base_oos, seed_sharpes = walk_forward(dev_returns, "DEV")

    print("\n=== METRICAS OOS (walk-forward agregado) ===")
    rows = {"RL Agent": metrics(agent_oos),
            "Equal-Weight": metrics(base_oos["equal"]),
            "60/40": metrics(base_oos["6040"]),
            "Risk Parity": metrics(base_oos["rparity"]),
            "Mean-Variance": metrics(base_oos["meanvar"]),
            "Random": metrics(base_oos["random"])}
    hdr = f"{'Estrategia':<16}{'Ret%':>8}{'Anual%':>8}{'Vol%':>7}{'Sharpe':>8}{'Sortino':>9}{'Calmar':>8}{'MaxDD%':>8}"
    print(hdr); print("-"*len(hdr))
    for name, m in rows.items():
        print(f"{name:<16}{m['total']:>8.1f}{m['annual']:>8.1f}{m['vol']:>7.1f}"
              f"{m['sharpe']:>8.2f}{m['sortino']:>9.2f}{(m['calmar'] if np.isfinite(m['calmar']) else 0):>8.2f}{m['mdd']:>8.1f}")

    # ---- Deflated Sharpe Ratio: o agente bate o ACASO dado o n de tentativas? ----
    a = agent_oos[np.isfinite(agent_oos)]
    sk = stats.skew(a); ku = stats.kurtosis(a, fisher=False)
    sr_annual = metrics(agent_oos)["sharpe"]
    dsr, sr0 = deflated_sharpe(sr_annual, len(a), seed_sharpes, sk, ku)
    print(f"\nDeflated Sharpe Ratio: tentativas={len(seed_sharpes)}, "
          f"Sharpe-so-por-sorte~{sr0:.2f}, Sharpe-agente={sr_annual:.2f}")
    print(f"  DSR (prob. de o edge ser real) = {dsr:.3f}  "
          f"{'-> credivel' if dsr > 0.95 else '-> NAO significativo (dentro do ruido)'}")

    # ---- Teste do juiz supremo: dados sinteticos ----
    print("\n=== ROBUSTEZ: DADOS SINTETICOS (sem estrutura temporal) ===")
    synth = synthetic_returns(dev_returns, seed=7)
    s_agent, s_base, _ = walk_forward(synth, "SYNTH")
    ms = metrics(s_agent)["sharpe"]; mb = metrics(s_base["rparity"])["sharpe"]
    print(f"Sharpe do agente em dados SINTETICOS: {ms:.2f} (risk parity: {mb:.2f})")
    print("  " + ("OK: agente nao fabrica sinal do ruido." if ms < 0.5
                  else "ALERTA: agente 'ganha' em ruido puro -> resultado real e suspeito."))

    # ---- Analise por regime ----
    print("\n=== ROBUSTEZ: DESEMPENHO POR REGIME ===")
    # alinhar regime ao comprimento do agente OOS (aproximacao simples)
    hv = regime_split(dev_returns)[-len(agent_oos):]
    hi = metrics(agent_oos[hv]); lo = metrics(agent_oos[~hv])
    print(f"  Alta vol:  Sharpe={hi['sharpe']:.2f}  MaxDD={hi['mdd']:.1f}%")
    print(f"  Baixa vol: Sharpe={lo['sharpe']:.2f}  MaxDD={lo['mdd']:.1f}%")
    print("  (esperado num gestor de risco real: vantagem relativa MAIOR em alta vol)")

    # ---- HELD-OUT SAGRADO: uma unica passagem, no fim ----
    print("\n=== HELD-OUT SAGRADO (uma unica avaliacao) ===")
    heldout_returns = to_returns(heldout_prices)
    # treina em TODO o dev, avalia no held-out (media de seeds)
    ho_rets = []
    for s in range(N_SEEDS):
        model = train_agent(dev_returns, seed=99000 + s)
        rets, _ = rollout_weights(model, heldout_returns)
        ho_rets.append(rets)
    L = min(len(x) for x in ho_rets)
    ho_agent = np.mean([x[:L] for x in ho_rets], axis=0)
    print(f"{'Estrategia':<16}{'Sharpe':>8}{'Anual%':>8}{'MaxDD%':>8}")
    for name, series in [("RL Agent", ho_agent),
                         ("Equal-Weight", bh_equal_weight(heldout_returns)[-L:]),
                         ("Risk Parity", risk_parity(heldout_returns)[-L:]),
                         ("Mean-Variance", mean_variance(dev_returns, heldout_returns)[-L:])]:
        m = metrics(series)
        print(f"{name:<16}{m['sharpe']:>8.2f}{m['annual']:>8.1f}{m['mdd']:>8.1f}")

    # ---- Visualizacao ----
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(nav_from_returns(agent_oos), label="RL Agent", lw=2)
    ax[0].plot(nav_from_returns(base_oos["rparity"]), label="Risk Parity", alpha=.8)
    ax[0].plot(nav_from_returns(base_oos["equal"]), label="Equal-Weight", alpha=.8)
    ax[0].set_title("Walk-forward OOS — NAV"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(nav_from_returns(ho_agent), label="RL Agent", lw=2)
    ax[1].plot(nav_from_returns(risk_parity(heldout_returns)[-L:]), label="Risk Parity", alpha=.8)
    ax[1].set_title("HELD-OUT sagrado — NAV"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("resultados.png", dpi=120, bbox_inches="tight")
    print("\n[grafico guardado em resultados.png]")

    print("\n" + "="*70)
    print("VEREDICTO HONESTO: o resultado mais valioso pode ser 'nao bate risk parity")
    print("de forma significativa' — e reporta-lo com DSR + dados sinteticos + held-out")
    print("prova que sabes onde estao os corpos enterrados. Isso e a peca de portfolio.")
    print("="*70)


if __name__ == "__main__":
    main()
