# Portfolio allocation with reinforcement learning (PPO)

I trained a PPO agent to allocate between 6 ETFs (SPY, QQQ, IWM, EFA, EEM, AGG)
and compared it with simple strategies. Most of the work went into validation,
so I could tell if the agent actually learned something.

It didn't beat them. Out-of-sample it basically matched equal-weight.

| Strategy      | Sharpe | Annual return | Max drawdown |
|---------------|--------|---------------|--------------|
| RL Agent      | 0.63   | 8.7%          | -29.4%       |
| Equal-Weight  | 0.65   | 8.8%          | -28.0%       |
| 60/40         | 0.88   | 9.0%          | -21.3%       |
| Risk Parity   | 0.66   | 6.2%          | -22.9%       |
| Mean-Variance | 0.93   | 11.2%         | -28.0%       |

Walk-forward, 16 windows, 2013 to 2025.

<img width="1490" height="490" alt="download" src="https://github.com/user-attachments/assets/fe7df863-6751-4297-a539-202f1dc5a179" />


The 3 seeds gave almost the same Sharpe (0.62, 0.63, 0.63), so the agent most
likely ended up close to equal weights. On the held-out period (last 18 months,
used once) it was the same: 1.37 vs 1.38.

## Validation

- Walk-forward (3 years train, 9 months test) plus a held-out period
- 3 seeds per window
- Same rules for agent and baselines: transaction costs, weights drifting with prices
- Block bootstrap vs the baselines: -0.02 Sharpe vs equal-weight (worse in 95%
  of samples), -0.03 vs risk parity (no real difference)
- Synthetic test with the days shuffled: agent 0.04 vs 0.11 for the best
  baseline, so it doesn't find signal in noise

The deflated Sharpe says "significant", but only because the seeds were so
similar that the luck benchmark ended up near zero. It just means the Sharpe is
above zero, which any long-only portfolio got in this period.

## Limitations

The ETFs were picked with hindsight, only 3 seeds and one set of
hyperparameters, and 2010 to 2026 was a very good period for US stocks and bonds.


