# Portfolio RL — Alocação de carteira à prova de auto-ilusão

Agente de aprendizagem por reforço (PPO) para alocação de carteira de ETFs, construído com um objetivo invulgar: **não me deixar enganar a mim próprio**. Num domínio de baixo sinal e ruído elevado como os mercados financeiros, a parte difícil não é treinar um modelo que pareça bom no histórico — é provar que esse desempenho é real e não um artefacto de sobreajuste ou de testar muitas configurações até uma "funcionar" por sorte.

Este projeto leva essa questão a sério. A maior parte do esforço não está no agente (que é deliberadamente simples), mas na **infraestrutura de validação** desenhada para detetar e refutar falsos positivos.

## Resultado

Após validação rigorosa, o agente **não supera, de forma estatisticamente significativa, baselines clássicos** como risk parity ou equal-weight:

```
Estrategia       Sharpe   Anual%   MaxDD%
-----------------------------------------
RL Agent          0.54     7.4     -33.0
Equal-Weight      0.52     6.9     -30.0
60/40             0.74     7.6     -22.9
Risk Parity       0.61     6.3     -24.0
Mean-Variance     0.67    10.5     -31.6
```

**Deflated Sharpe Ratio = 0.000** (contra 45 configuracoes/seeds testadas). O Sharpe do agente esta dentro do que se esperaria so por sorte, dado o numero de tentativas. No held-out sagrado (~18 meses nunca tocados durante o desenvolvimento), o agente empata com equal-weight (Sharpe 1.05 vs 1.06).

Este resultado negativo é o ponto central do projeto, não uma deceção. **Reportá-lo honestamente — com as ferramentas que provam que é robusto — é a competência que o projeto demonstra.** Num contexto real de gestão de ativos, saber quando *não* há edge é tão valioso como encontrá-lo, e muito mais raro.

## A metodologia anti-auto-ilusão

O que distingue este projeto é o conjunto de salvaguardas contra falsos positivos:

**Agente deliberadamente modesto.** PPO com uma MLP pequena (64x64), poucas features, poucos timesteps. Em domínios de baixo sinal, um agente mais expressivo não extrai mais sinal — apenas memoriza melhor o percurso histórico. A simplicidade *é* a regularização que mais importa.

**Reward de gestão de risco, não de previsão.** Em vez de pedir ao agente que adivinhe a direção dos retornos (já demonstrado, noutro projeto, ser imprevisível), a reward usa o **Differential Sharpe Ratio** (Moody & Saffell, 1998) — uma forma online e estável do Sharpe — mais uma penalização de drawdown. Pede ao agente que *module a exposição ao risco*, que é o único sinal genuíno disponível (clustering de volatilidade).

**Validação de três níveis com purging e embargo.** Walk-forward com janelas deslizantes para medir generalização, um **gap temporal (embargo)** entre treino e teste para matar a autocorrelação das janelas sobrepostas (López de Prado), e um **bloco held-out sagrado** que não existe durante o desenvolvimento e é avaliado uma única vez no fim.

**Deflated Sharpe Ratio.** Múltiplas seeds (o RL é estocástico; um número único mente) e correção do Sharpe pelo número de tentativas. O multiple-testing — quantas vezes o nosso próprio cérebro tocou nos dados — é o verdadeiro assassino da significância, e o DSR mede-o.

**Teste de dados sintéticos (o juiz supremo).** Reamostragem que preserva as distribuições marginais e correlações entre ativos mas **destrói a estrutura temporal**. Por construção, não há nada para o agente aprender. Se ele "ganhasse" aqui, provaria que o pipeline fabrica sinal a partir de ruído — e o resultado real seria falso. (O agente obteve Sharpe 0.44 no sintético, igual ao baseline: o pipeline **não** fabrica sinal.)

**Análise por regime.** O edge de um gestor de risco deve aparecer onde a teoria diz (alta volatilidade), não espalhado ao acaso. *Onde* o desempenho aparece é, ele próprio, evidência sobre se é real.

## Nota de honestidade

Os ETFs do universo foram escolhidos ex-post, o que introduz survivorship/selection bias. Sem uma base de dados de ETFs delistados, este viés fica *nomeado* mas não totalmente eliminado. Em produção, o universo seria definido com a informação disponível no início de cada janela.

## Stack

Python, Stable-Baselines3 (PPO), Gymnasium, yfinance, NumPy/SciPy, pandas.

## Como correr

```bash
pip install -r requirements.txt
python portfolio_rl.py
```

Parâmetros modestos por opção de desenho (ver a filosofia no topo do script). Demora ~5-12 min em CPU.

---
*Projeto de investigação. Não constitui aconselhamento financeiro.*

---
*Projeto de investigação. Não constitui aconselhamento financeiro.*
