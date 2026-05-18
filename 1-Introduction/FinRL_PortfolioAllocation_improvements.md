# Improving the FinRL Portfolio Allocation Workflow

A prioritised set of concrete changes to the `FinRL_PortfolioAllocation_NeurIPS_2020.ipynb` workflow, with expected impact, priority (1 = do first), and implementation difficulty (Easy / Medium / Hard) so you can plan the work.

The current setup hits ~31% annual return, Sharpe ~2.0 on Jul 2020 → Oct 2021. Most of that is the index itself. The improvements below are ordered roughly by impact-per-effort: do the cheap, high-impact things first.

---

## TL;DR ranking

| # | Change                                              | Impact     | Priority | Difficulty |
|---|-----------------------------------------------------|------------|----------|------------|
| 1 | Switch reward from portfolio *value* to log-return  | **High**   | 1        | Easy       |
| 2 | Add transaction-cost penalty to the reward          | **High**   | 1        | Easy       |
| 3 | Walk-forward (rolling-window) train/test            | **High**   | 2        | Medium     |
| 4 | Differential Sharpe / Sortino reward                | **High**   | 2        | Medium     |
| 5 | Multi-seed training + ensembled actions             | Medium     | 2        | Easy       |
| 6 | Early stopping on validation Sharpe                 | Medium     | 3        | Medium     |
| 7 | `VecNormalize` for observations (+ reward)          | Medium     | 3        | Easy       |
| 8 | Turbulence-gated risk-off rule                      | Medium     | 3        | Easy       |
| 9 | Turnover / action-smoothness penalty                | Medium     | 3        | Easy       |
| 10| Richer state: macro factors, regime features        | Medium     | 4        | Medium     |
| 11| Bayesian hyperparameter search (Optuna)             | Medium     | 4        | Medium     |
| 12| Algorithm swap: SAC / TD3 tuned properly            | Low–Medium | 4        | Easy       |
| 13| Larger universe (S&P 500 / sector-balanced subset)  | Medium     | 5        | Hard       |

---

## 1. Switch reward from portfolio value to log-return

**What.** In `env_portfolio.py`, replace:
```python
self.reward = new_portfolio_value
```
with:
```python
self.reward = np.log(1 + portfolio_return)
```

**Why.** The current target is non-stationary: portfolio value drifts from $1M to many millions over the 11.5-year training horizon, so late-period samples dominate the gradient signal. Log-returns are stationary, bounded in practice, and additive over time — exactly the property gradient-based optimisation likes. This is the single most impactful one-line change in the whole workflow.

**Expected impact.** High. Removes scale drift in the reward, makes training across the long horizon meaningful, often yields better generalisation to the trade slice.

**Priority.** 1.

**Difficulty.** Easy. One-line edit (plus possibly tweaking `reward_scaling` — log-returns are O(0.01) so the default 1e-4 scaling may need to come off).

---

## 2. Add a transaction-cost penalty to the reward

**What.** Track day-over-day weight changes and subtract their cost from the reward:
```python
prev_weights   = self.actions_memory[-2] if len(self.actions_memory) > 1 \
                 else np.ones(self.stock_dim) / self.stock_dim
turnover       = np.abs(np.array(weights) - np.array(prev_weights)).sum()
tc             = self.transaction_cost_pct * turnover
self.reward    = np.log(1 + portfolio_return) - tc
new_portfolio_value *= (1 - tc)   # also deduct from book equity
```

**Why.** The env already accepts `transaction_cost_pct=0.001` but does not currently deduct it from either the reward or the book value. The agent therefore churns weights for free every day. With a real cost, the policy learns to hold positions and only rebalance when the signal justifies the slippage.

**Expected impact.** High. Often shifts strategies from churn-prone to genuinely investable, improves out-of-sample Sharpe because realised costs were silently subsidising in-sample performance.

**Priority.** 1.

**Difficulty.** Easy. ~5 lines.

---

## 3. Walk-forward (rolling-window) training and evaluation

**What.** Instead of one fixed split (2009-01 → 2020-07 train, 2020-07 → 2021-10 trade), train and trade in rolling windows:

```
window 1:  train 2009-01..2014-12   →   eval 2015-01..2015-12
window 2:  train 2010-01..2015-12   →   eval 2016-01..2016-12
window 3:  train 2011-01..2016-12   →   eval 2017-01..2017-12
...
window k:  train 2014-01..2019-12   →   eval 2020-01..2020-12
window k+1:train 2015-01..2020-12   →   eval 2021-01..2021-10
```

Then stitch the per-window trade returns together into one continuous backtest.

**Why.** A single 16-month evaluation on a bull market is statistically meaningless — you cannot tell if you're observing skill or a lucky regime. Walk-forward gives you the agent's performance across multiple regimes (2015 correction, 2018 selloff, 2020 COVID crash, 2021 rally), and the train set always uses only data that was *actually available* before the eval window.

**Expected impact.** High for **credibility** of the result. You may discover the agent fails in non-bull regimes — that's a real finding, not a failure of the approach.

**Priority.** 2 (do after the reward function fixes so you're walk-forwarding a sensible objective).

**Difficulty.** Medium. Mostly a wrapper loop around the existing notebook code. Watch out for two things: (i) re-running `FeatureEngineer` and `cov_list` computation cleanly per window without leaking, (ii) collecting per-window outputs into one daily-return series.

Code sketch:
```python
results = []
for train_start, train_end, eval_start, eval_end in walk_forward_windows():
    train = data_split(df, train_start, train_end)
    eval_ = data_split(df, eval_start, eval_end)
    env   = StockPortfolioEnv(df=train, **env_kwargs)
    agent = DRLAgent(env=env.get_sb_env()[0])
    model = agent.get_model("ppo", model_kwargs=PPO_PARAMS, seed=42)
    trained = agent.train_model(model=model, tb_log_name=f"ppo_{eval_start}",
                                total_timesteps=80_000)
    daily_ret, actions = DRLAgent.DRL_prediction(model=trained,
                                                 environment=StockPortfolioEnv(df=eval_, **env_kwargs))
    results.append(daily_ret)
full_returns = pd.concat(results, ignore_index=True)
```

---

## 4. Differential Sharpe / Sortino as the reward

**What.** Reward the agent with an incremental update to the *running Sharpe* (or Sortino) rather than a raw return. Moody & Saffell's Differential Sharpe Ratio:
```python
# state held in the env between steps
A_prev, B_prev = self.A, self.B
dA = portfolio_return - A_prev
dB = portfolio_return**2 - B_prev
self.A = A_prev + eta * dA
self.B = B_prev + eta * dB
denom = (B_prev - A_prev**2) ** 1.5
self.reward = (B_prev * dA - 0.5 * A_prev * dB) / max(denom, 1e-8)
```
with `eta` ≈ 1/252 (adaptation rate, one trading year).

**Why.** Trains the agent to *care about volatility* directly. Log-return reward already helps, but DSR makes risk-adjustment first-class. Sortino is similar but only penalises downside deviation — useful if max drawdown is your real concern.

**Expected impact.** High when you actually evaluate across regimes (i.e., paired with walk-forward). On the current 16-month bull-only slice the effect will be muted because there is little volatility to penalise.

**Priority.** 2.

**Difficulty.** Medium. ~20 lines in the env, plus careful initialisation of `A` and `B` at episode reset.

---

## 5. Multi-seed training and ensembled actions

**What.** Train N (e.g. 5–10) identical PPO policies with different seeds. At trade time, run each policy on the current state, take the softmax of each, and **average the resulting weight vectors** before applying.

```python
weights = np.mean([m.predict(obs, deterministic=True)[0] for m in models], axis=0)
weights = np.exp(weights) / np.sum(np.exp(weights))
```

**Why.** Single-seed PPO has huge run-to-run variance — the $0.04 Sharpe difference you saw earlier between 80k and 150k is well inside seed noise. Ensembling averages out seed-specific local optima and is one of the cheapest reliable boosts in DRL.

**Expected impact.** Medium. Typically +0.1–0.3 Sharpe and noticeably smaller drawdowns, mostly from variance reduction rather than alpha generation.

**Priority.** 2.

**Difficulty.** Easy. A for-loop around training, a list comprehension at prediction time.

---

## 6. Early stopping on validation Sharpe

**What.** Split the training window into train / **validation** (e.g., last 6 months of the train range). Every K updates (or every M timesteps), evaluate the current model on the validation slice and compute its Sharpe. Save the best checkpoint. Stop when validation Sharpe hasn't improved for `patience` evaluations.

Use SB3's `EvalCallback`:
```python
from stable_baselines3.common.callbacks import EvalCallback

val_env = DummyVecEnv([lambda: StockPortfolioEnv(df=validation, **env_kwargs)])
eval_cb = EvalCallback(val_env,
                       best_model_save_path="./best/",
                       eval_freq=10_000,
                       n_eval_episodes=1,
                       deterministic=True)
trained = agent.train_model(model=model_ppo, total_timesteps=200_000,
                            tb_log_name="ppo", callback=eval_cb)
```
Then load `./best/best_model.zip` for the final trade evaluation.

**Why.** Right now you're guessing `total_timesteps` (80k? 150k? 200k?). Early stopping picks the best model objectively. Crucial once you turn on a noisier reward (DSR) because PPO can overshoot a good policy and never get back.

**Expected impact.** Medium. Prevents the common failure mode of "trained too long, overfit, lost the good policy".

**Priority.** 3.

**Difficulty.** Medium. The callback is one-liner, but you need a clean validation split that doesn't leak into the trade window, and you need an `EvalCallback` reward metric that matches what you care about (annualised Sharpe, not raw env reward).

---

## 7. `VecNormalize` for observations and reward

**What.** Wrap the training env in `VecNormalize(norm_obs=True, norm_reward=True)`, save the running statistics, and apply the **same** statistics at trade time. You already wrote a `SelectiveVecNormalize` for the Stock_NeurIPS2018 notebook (see git history — `7dcf3d8` and `c95ce55`) — the same idea applies here.

**Why.** Your 945-dim observation has wildly different scales: covariance entries are O(1e-3), MACD is O(1), RSI is 0–100, Bollinger Bands are at price level (tens to hundreds). MLPs train badly on heterogeneous scales; running normalisation fixes that. Reward normalisation also stabilises PPO when the reward distribution is changing (e.g., during regime shifts in walk-forward).

**Expected impact.** Medium. Smoother training curves, often slightly better final policy.

**Priority.** 3.

**Difficulty.** Easy if you copy the `SelectiveVecNormalize` pattern you already have. The subtlety is **saving the running stats** with the model and reloading them at trade time — if you skip that, the trade env sees un-normalised inputs and the agent's predictions go off-distribution. You hit this bug already on the Stock_NeurIPS2018 backtest.

---

## 8. Turbulence-gated risk-off rule

**What.** Re-enable the turbulence index (`FeatureEngineer(use_turbulence=True)`), pass a `turbulence_threshold` to the env, and force the agent to **equal-weight cash / minimum-risk position** when turbulence exceeds the threshold:

```python
if self.turbulence > self.turbulence_threshold:
    weights = np.zeros(self.stock_dim)        # or move to a defensive subset
```

**Why.** A simple non-learned safety valve. The 2020 COVID crash is the obvious test case: a turbulence gate would cut exposure in late February, avoiding the worst weeks. Most of the academic DRL trading papers that report "beats DJIA" rely on this gate.

**Expected impact.** Medium. Significant on long evaluation windows that include crashes; negligible on a pure bull window.

**Priority.** 3.

**Difficulty.** Easy. The infrastructure (`turbulence_threshold` parameter, turbulence column) already exists in FinRL — you just need to flip it on and tune the threshold.

---

## 9. Turnover / action-smoothness penalty

**What.** Add an L1 or L2 penalty on weight changes between consecutive steps:
```python
turnover_penalty = lambda_ * np.linalg.norm(weights - prev_weights, 1)
self.reward     -= turnover_penalty
```

**Why.** Even with item (2), you may want stronger smoothing to keep the policy trading slowly. Helps interpretability (you can read the weights and see strategy logic) and is a standard regulariser in production allocation systems.

**Expected impact.** Medium. Mostly improves the *quality* of the strategy (lower realised costs, lower trade-attribution noise) rather than headline Sharpe.

**Priority.** 3.

**Difficulty.** Easy. 2 lines.

---

## 10. Richer state: macro / regime features

**What.** Beyond the 8 price-derived indicators, add:
- **VIX level and its 1-week change** (broad risk regime).
- **Term spread** (10Y - 2Y Treasury yield) — recession proxy.
- **Dollar index (DXY) return**.
- **Sector ETF returns** (XLK, XLF, XLE, …) as cross-sectional context.
- **Cross-sectional momentum and dispersion** (top-N minus bottom-N return over the last 20 days).

**Why.** The current state tells the agent everything about *individual stocks* and nothing about the *market environment*. Macro/regime features give the policy a way to condition its allocation on the regime, which is exactly what makes a tactical allocator different from buy-and-hold.

**Expected impact.** Medium. Bigger states are not automatically better — but in this case the missing information genuinely limits what the agent can express. Combined with walk-forward, this is where you might actually see a Sharpe lift vs DJIA.

**Priority.** 4.

**Difficulty.** Medium. New data fetches (FRED for macro, Yahoo for ETFs), alignment to trading dates, broadcasting market-level features into the per-stock observation tensor.

---

## 11. Hyperparameter search with Optuna

**What.** Replace hand-tuned `PPO_PARAMS` with an Optuna study over `learning_rate`, `n_steps`, `batch_size`, `ent_coef`, `clip_range`, `gamma`, `gae_lambda`. Objective = validation Sharpe (not training reward). 30–100 trials.

**Why.** PPO has many interacting hyperparameters; hand-tuning four of them is mostly cargo-cult. A modest Optuna run usually finds a configuration 0.2–0.4 Sharpe better than the default, *and* gives you the variance across configurations so you know how seed-sensitive your results are.

**Expected impact.** Medium. Diminishing returns once you've fixed the reward function — but worth doing once the rest of the pipeline is correct.

**Priority.** 4 (after items 1–4, otherwise you tune to the wrong objective).

**Difficulty.** Medium. Optuna wrapper is standard, but each trial trains a model — wall-clock cost is real. Use a smaller `total_timesteps` per trial and final retrain at full budget on the winning config.

---

## 12. Algorithm swap: SAC / TD3 tuned properly

**What.** The notebook already trains SAC and TD3 but never uses them. Swap `trained_a2c` → `trained_sac` in `DRL_prediction` and compare. Tune SAC's `gradient_steps`, `train_freq`, `target_entropy`.

**Why.** SAC's max-entropy objective is natively risk-aware (entropy bonus = exploration even after convergence), and for continuous-action allocation problems it usually outperforms PPO with much less hyperparameter sensitivity. TD3 is more conservative but more sample-efficient than DDPG.

**Expected impact.** Low to Medium. Algorithm changes alone rarely beat reward/feature changes. But on a tight setup, SAC can give +0.1–0.2 Sharpe over PPO.

**Priority.** 4.

**Difficulty.** Easy. Already in the notebook; just use the trained model.

---

## 13. Larger universe (S&P 500 / sector-balanced subset)

**What.** Replace the Dow 30 list with the S&P 500 or a sector-balanced 100-stock subset. Bigger action space (100 → 500 logits) and more diversification opportunities.

**Why.** Dow 30 is a tiny, mega-cap, US-only universe — the upside of any allocation policy is limited by how concentrated and correlated those names are. A larger universe genuinely gives the agent more to do.

**Expected impact.** Medium. Diminishing returns past ~100 stocks, and computational cost grows quadratically with the covariance matrix.

**Priority.** 5.

**Difficulty.** Hard. Data download (S&P 500 has survivorship-bias issues — historical members change), memory (covariance is O(N²) and the observation matrix scales accordingly), training time. Also, the softmax over 500 actions makes exploration much slower; you may need a different action parameterisation (e.g., Dirichlet policy or top-K selection).

---

## Suggested execution order

If you wanted a concrete plan over, say, a couple of weeks of effort:

**Week 1 — Fix the objective function**
- [ ] Item 1: log-return reward.
- [ ] Item 2: transaction-cost penalty.
- [ ] Item 5: multi-seed runs (use this from now on as your evaluation baseline).
- [ ] Item 7: `VecNormalize` with persisted statistics (copy from `SelectiveVecNormalize`).

By end of week 1 you should have a credible reward function and a stable training pipeline.

**Week 2 — Make the evaluation credible**
- [ ] Item 3: walk-forward windows (5–6 windows from 2015 onward).
- [ ] Item 6: `EvalCallback` early stopping on validation Sharpe.
- [ ] Item 8: turbulence gate (so 2020-Q1 isn't a disaster).
- [ ] Item 9: turnover penalty.

By end of week 2 you have an honest walk-forward backtest with realistic costs.

**Week 3+ — Squeeze more performance (optional)**
- [ ] Item 4: differential Sharpe reward.
- [ ] Item 10: macro features.
- [ ] Item 11: Optuna tuning.
- [ ] Items 12 / 13 if you still want to push.

Most of the headline Sharpe improvement comes from weeks 1–2. Weeks 3+ are the long tail.

---

## What to *measure* after every change

To avoid the "did this help or am I just rolling dice on seeds" trap, report each candidate change against the same evaluation protocol:

- Mean and std of **annualised Sharpe** across ≥5 seeds.
- Mean and std of **annualised return**, **max drawdown**, **annual volatility**.
- **Turnover** (avg sum of |Δweights| per day).
- Performance against **DJIA** and against **classical Min-Variance** baselines over the full walk-forward window.

If the mean Sharpe improvement is less than the std across seeds, the change is noise — keep it only if it improves something else (e.g., drawdown, turnover) or roll it back.
