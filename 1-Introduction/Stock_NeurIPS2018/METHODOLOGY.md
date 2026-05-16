# Stock NeurIPS2018 — Methodology Notes

A working reference covering the end-to-end pipeline of the `Stock_NeurIPS2018` tutorial, the inner workings of the trading environment, training-log interpretation for the SB3 algorithms, common errors, and known limitations of the default design.

---

## 1. Project Overview

The tutorial reproduces *"Practical Deep Reinforcement Learning Approach for Stock Trading"* (Yang et al., NeurIPS 2018). A reinforcement-learning agent learns a daily portfolio-allocation policy across the Dow Jones 30 tickers, trained on 2009–2020 data and back-tested on 2020–2021 against classical baselines (Mean-Variance Optimization, DJIA buy-and-hold).

The project is split across **three notebooks** that communicate via CSV artifacts:

| Stage | Notebook | Output |
|---|---|---|
| 1. Data | `Stock_NeurIPS2018_1_Data.ipynb` | `train_data.csv`, `trade_data.csv` |
| 2. Train | `Stock_NeurIPS2018_2_Train.ipynb` | `trained_models/agent_<algo>.zip` |
| 3. Backtest | `Stock_NeurIPS2018_3_Backtest.ipynb` | Equity curves, performance plots |

---

## 2. Reinforcement Learning Algorithms in This Project

FinRL exposes five Stable-Baselines3 algorithms — **A2C, DDPG, PPO, TD3, SAC**. They differ along several axes: on-policy vs off-policy, deterministic vs stochastic policy, single vs twin critics, exploration mechanism, sample efficiency, and stability. The notebooks let you flip each one on with a boolean (`if_using_ppo = True`, etc.). This section gives a one-paragraph overview of each, plus a comparison table.

### 2.1 A2C (Advantage Actor-Critic)
Synchronous variant of A3C. On-policy, actor-critic. The actor outputs a stochastic policy; the critic estimates state values. Uses **advantage** (`A = R − V(s)`) instead of raw returns to reduce variance. Updates happen every few env steps (small `n_steps=5` by default). Strengths: simple, fast wall-clock per update, low memory. Weaknesses: sample-inefficient, noisy gradients, easily outperformed by PPO. Continuous or discrete actions. Best for: quick baselines, low-budget experiments.

### 2.2 DDPG (Deep Deterministic Policy Gradient)
Off-policy, deterministic actor + Q-critic. Designed for continuous actions. The actor maps state → single deterministic action (no distribution); the critic learns `Q(s, a)`. Trained from a **replay buffer** with Polyak-averaged target networks for stability. Exploration via additive action noise (Ornstein-Uhlenbeck or Gaussian). Strengths: sample-efficient compared to on-policy methods. Weaknesses: notoriously unstable — Q-value overestimation, sensitive to hyperparameters, can diverge silently. Continuous actions only. Largely superseded by TD3 and SAC.

### 2.3 PPO (Proximal Policy Optimization)
On-policy, stochastic actor-critic with a trust-region clip. Collects a rollout (`n_steps=2048`), then does multiple epochs of mini-batch updates with the **clipped surrogate objective** (`clip_range=0.2`) that prevents policy updates from straying too far. Diagnostics: `approx_kl`, `clip_fraction`. Strengths: robust default behavior (works on most problems without heavy tuning and rarely diverges), runs on continuous and discrete actions, the de facto baseline in RL. Weaknesses: needs more steps than off-policy methods, weak when constraints frequently clip its actions. Best for: a strong default with minimal tuning.

### 2.4 TD3 (Twin Delayed DDPG)
DDPG fixed. Three tricks: (1) **twin critics** — train two Q-networks and use the minimum to compute targets (combats overestimation); (2) **delayed policy updates** — update actor every k=2 critic updates; (3) **target-policy smoothing** — add noise to the target action so the critic doesn't overfit narrow Q-peaks. Off-policy, deterministic, continuous-only. Strengths: much more stable than DDPG with similar sample efficiency. Weaknesses: still deterministic (less exploration than SAC). Often the right choice when SAC is overkill.

### 2.5 SAC (Soft Actor-Critic)
Off-policy, stochastic actor + twin Q-critics with maximum-entropy objective. Optimizes expected return plus an **entropy bonus** (`α · H(π)`) — explicitly rewarding exploration. Has a learned temperature `α` that auto-tunes entropy. Continuous actions. Strengths: state-of-the-art sample efficiency on most continuous-control benchmarks; very robust; minimal tuning. Weaknesses: more compute per step than DDPG/TD3, replay buffer memory cost. Often the best practical choice when sample efficiency matters.

### 2.6 Comparison table

| Property | **A2C** | **DDPG** | **PPO** | **TD3** | **SAC** |
|---|---|---|---|---|---|
| Family | Actor-Critic | Actor-Critic (Q-based) | Actor-Critic | Actor-Critic (Q-based) | Actor-Critic (Q-based, max-ent) |
| Policy type | Stochastic | Deterministic | Stochastic | Deterministic | Stochastic |
| On/Off-policy | On | Off | On | Off | Off |
| Action space | Continuous/Discrete | Continuous | Continuous/Discrete | Continuous | Continuous |
| Buffer | Rollout (small) | Replay (large) | Rollout (medium) | Replay (large) | Replay (large) |
| Critics | 1 V(s) | 1 Q(s,a) | 1 V(s) | 2 Q(s,a) twin | 2 Q(s,a) twin |
| Exploration | Stochastic policy | External noise | Stochastic policy | External noise + target smoothing | Entropy bonus (auto-tuned) |
| Sample efficiency | Low | High | Medium | High | High |
| Stability | Medium | Low | High | High | Very High |
| Tuning difficulty | Easy | Hard | Easy | Medium | Easy |
| Memory cost | Low | High | Low–Medium | High | High |
| Steps to converge (FinRL) | Highest | Medium | Medium-High | Medium | Medium |
| Best when | Cheap baseline | (legacy) | Robust default | Stable continuous control | Best sample efficiency, less tuning |

**Practical ranking for FinRL stock trading**: SAC ≈ TD3 > PPO > DDPG > A2C, with PPO usually the most reproducible if compute is limited.

> **Note on "robust default" (PPO).** It means the algorithm works reasonably on most problems without careful tuning and rarely fails catastrophically — not that it produces the *best* final policy. SAC and TD3 often beat PPO on continuous control, but they require more careful setup. PPO is the lower-variance-of-outcomes choice.

---

## 3. End-to-End Workflow (PPO Example)

### Stage 1 — Data Preparation
1. Download daily OHLCV bars for DJI-30 tickers from Yahoo Finance (2009-01-01 → 2021-10-29).
2. Engineer features: technical indicators (MACD, RSI, CCI, ADX, Bollinger bands, SMAs) and a VIX-based turbulence index.
3. Chronological split into train and trade sets (no shuffling — time series).
4. Persist as CSV.

### Stage 2 — RL Training
1. Load `train_data.csv`.
2. Build the trading environment (`StockTradingEnv`, a Gymnasium env).
3. Instantiate the algorithm via FinRL's `DRLAgent` wrapper:
   ```python
   agent = DRLAgent(env=env_train)
   model_ppo = agent.get_model("ppo")
   trained_ppo = agent.train_model(model=model_ppo, tb_log_name='ppo',
                                   total_timesteps=50000)
   ```
4. Save policy to `trained_models/agent_ppo.zip`.

### Stage 3 — Backtest
1. Load `trade_data.csv` and the trained policy.
2. Build a fresh env over the trade period, with `turbulence_threshold=70` as a risk filter.
3. Run deterministic inference (`DRLAgent.DRL_prediction`) → daily portfolio values + actions.
4. Compute baselines (MVO, DJIA).
5. Plot equity curves of all algorithms vs. baselines, all starting at $1M.

**Mental model**: train a neural policy to map `(cash, holdings, prices, indicators)` → `(buy/sell amounts)` by simulating thousands of trading days on historical data, then evaluate on a held-out period.

---

## 4. The Trading Environment in Detail

Source: `finrl/meta/env_stock_trading/env_stocktrading.py` (class `StockTradingEnv`).

### 4.1 State vector
```
state = [cash,
         price_1, ..., price_N,
         shares_1, ..., shares_N,
         indicator_1_stock_1, ..., indicator_K_stock_N]
```
Dimensions: `state_space = 1 + 2·stock_dim + n_indicators·stock_dim`. For the tutorial (N=30 stocks, K=8 indicators): `1 + 60 + 240 = 301`.

### 4.2 Action vector
Continuous, one value per stock in `[-1, +1]`. `spaces.Box(low=-1, high=1, shape=(stock_dim,))`.

### 4.3 Reward
```
reward = (end_total_asset − begin_total_asset) × reward_scaling
```
Raw daily P&L in dollars, then numerically scaled.

### 4.4 Step lifecycle (one trading day)
1. Receive `action` from the agent.
2. Scale: `actions = (action × hmax).astype(int)` — intended trade sizes in integer shares.
3. **Turbulence override**: if `turbulence >= turbulence_threshold`, replace actions with `[-hmax] × N` (sell everything).
4. Compute `begin_total_asset = cash + Σ(price × shares)`.
5. Sort actions: execute **sells first** (frees cash), then **buys** (uses freed cash).
6. `_sell_stock`: clip sell-quantity to current holdings; credit cash net of fee; debit shares.
7. `_buy_stock`: clip buy-quantity to `cash // (price × (1 + fee))`; debit cash; credit shares.
8. Advance to next day (`self.day += 1`), refresh `self.data`, rebuild state via `_update_state`.
9. Compute `end_total_asset`; reward = ΔP&L × `reward_scaling`.
10. Return `(state, reward, terminal, truncated, info)`.

### 4.5 Key parameters (defined in your notebook's `env_kwargs`)

| Parameter | Tutorial value | Meaning |
|---|---|---|
| `hmax` | 100 | Max shares per trade per stock. Caps how aggressive a single-day action can be. |
| `initial_amount` | 1,000,000 | Starting cash. |
| `buy_cost_pct` / `sell_cost_pct` | `[0.001] × 30` | 0.1% transaction fee per trade. Discourages churn. |
| `reward_scaling` | 1e-4 | Multiplier on raw $ P&L reward. Pure numerical conditioning for stable value-function fitting. Does not change which actions are optimal. |
| `turbulence_threshold` | 70 (backtest), `None` (train) | Forced-liquidation trigger using `risk_indicator_col` (VIX). Hand-engineered circuit breaker. |
| `state_space` | 301 | Computed from `stock_dim` and `len(INDICATORS)`. |
| `action_space` | 30 | One scalar per stock. |
| `tech_indicator_list` | `INDICATORS` | Imported from `finrl.config`. |

---

## 5. Indicators

### Where the list is defined
`finrl/config.py`:
```python
INDICATORS = [
    "macd", "boll_ub", "boll_lb",
    "rsi_30", "cci_30", "dx_30",
    "close_30_sma", "close_60_sma",
]
```
These are just string keys.

### Where they are computed
`finrl/meta/preprocessor/preprocessors.py` — class `FeatureEngineer`, method `add_technical_indicator` (~line 200). The implementation delegates to the third-party `stockstats` library (`StockDataFrame`), which parses each string and computes the corresponding indicator on the fly. To add a new indicator, append a valid `stockstats` name to `INDICATORS` — no code change required.

---

## 6. Neural Network Architecture (PPO / SB3 Defaults)

### 6.1 Default PPO network (SB3 `MlpPolicy`)
No explicit network parameters are passed in the notebook because FinRL's `agent.get_model("ppo")` resolves to SB3's `PPO(policy="MlpPolicy", ...)` — a default architecture:

- **Type**: feedforward MLP
- **Body**: 2 hidden layers × 64 units, `Tanh` activation
- **Actor head**: linear → mean per action dim; learned `log_std` per action dim (the `std: 1.06` in training logs)
- **Critic head**: linear → scalar value

```
state (301)
   │
[Linear 301→64] → Tanh
[Linear  64→64] → Tanh
   ├──► [Linear 64→30] ──► action means  (actor)
   │    + learned log_std
   └──► [Linear 64→1]  ──► state value   (critic)
```

### 6.2 Overriding the architecture
```python
import torch.nn as nn
model_ppo = agent.get_model("ppo", policy_kwargs={
    "net_arch": dict(pi=[256, 256], vf=[256, 256]),
    "activation_fn": nn.ReLU,
})
```

### 6.3 PPO training hyperparameters (FinRL defaults)
Live in `finrl/config.py` under `PPO_PARAMS`: `n_steps=2048`, `ent_coef=0.01`, `learning_rate=0.00025`, `batch_size=64`.

---

## 7. Interpreting Training Logs

### 7.1 Common fields (all algos)
| Field | Meaning |
|---|---|
| `fps` | env steps per second |
| `total_timesteps` | env steps consumed so far |
| `time_elapsed` | wall-clock seconds |
| `entropy_loss` | exploration term; large negative = highly stochastic policy |
| `value_loss` | critic's regression error on returns |
| `explained_variance` | fraction of return variance the critic explains. 1.0 = perfect, ≤0 = worse than mean |
| `learning_rate` | optimizer LR |
| `reward_mean / min / max` | episode reward statistics |
| `std` | width of policy Gaussian; should narrow over training as policy commits |

### 7.2 PPO-specific
| Field | Meaning |
|---|---|
| `approx_kl` | KL divergence between old and new policy; healthy range ≈ 0.01–0.02 |
| `clip_fraction` | fraction of samples where PPO's trust-region clip activates |
| `clip_range` | PPO's epsilon (default 0.2) |
| `policy_gradient_loss` | the PPO surrogate objective |

### 7.3 What healthy training looks like
- `reward_mean` trending up
- `explained_variance` climbing toward ≥ 0.5
- `value_loss` falling
- `std` narrowing (policy committing to a strategy)
- `clip_fraction` low and stable (PPO)

### 7.4 What this project's runs actually showed (50k–200k steps)
- A2C (50k): `reward_mean ≈ 0.13`, `explained_variance ≈ 0`. Policy barely above random.
- PPO (200k): `reward_mean` stuck at ~0.17 across 196k extra steps; `clip_fraction` rose 0.14 → 0.38, `std` widened 1.00 → 1.15. Trust-region indicators all moved the wrong way — **policy was destabilizing, not converging**.
- DDPG: trained successfully despite noisy callback errors (see §8).

**Takeaway**: 50k timesteps is far below what FinRL agents typically need. References usually recommend 200k–1M for meaningful convergence. Even then, the training curve is a weak proxy — the **backtest equity curve** is the metric that matters.

---

## 8. Errors Encountered & Fixes

### 8.1 `Logging Error: 'rollout_buffer'` (DDPG/SAC/TD3)
**Cause**: FinRL's custom `TensorboardCallback` reads `self.locals["rollout_buffer"]` to log per-step reward. Off-policy algos (DDPG/SAC/TD3) use a `replay_buffer` instead, so the lookup raises `KeyError`, caught and printed as `Logging Error: 'rollout_buffer'`.

**Impact**: Cosmetic. Training itself is unaffected; only the custom reward log is missing from TensorBoard. Standard SB3 logs (`train/actor_loss`, `train/critic_loss`) still appear.

**Fixes**:
- Ignore (recommended)
- Suppress: `logging.getLogger().setLevel(logging.ERROR)` before training
- Patch FinRL's callback to check for both `rollout_buffer` and `replay_buffer`

### 8.2 Replay buffer memory warning
```
UserWarning: This system does not have apparently enough memory to store
the complete replay buffer 2.54GB > 1.26GB
```
**Cause**: DDPG default `buffer_size=1_000_000`. Combined with the 301-dim observation, the buffer demands ~2.5 GB.

**Fix**: Shrink the buffer to match actual training budget:
```python
model_ddpg = agent.get_model("ddpg", model_kwargs={
    "buffer_size": 100_000,   # was 1_000_000
})
```
A 50k-step run cannot fill more than 50k transitions anyway — the larger buffer is purely waste.

### 8.3 MVO `ValueError: operands could not be broadcast together with shapes (29,) (30,)`
**Cause**: Hardcoded `range(29)` in the MVO weights cell of `Stock_NeurIPS2018_3_Backtest.ipynb`:
```python
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(29)])
```
The DJI added Dow Inc. in 2019, bringing the index to 30 tickers. The tutorial code still assumes 29.

**Fix**: Use `stock_dimension` (already defined earlier in the notebook):
```python
mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_dimension)])
```

### 8.4 `!pip install` cells
Required only on **Google Colab** (clean VM per session). For a local `.venv`, run once then comment out — re-running just refetches the same packages and wastes a minute.

---

## 9. Known Limitations of the Default Environment

### 9.1 Action clipping bias
The policy outputs an action `a`; the env executes `a' = clip(a)` (bounded by cash, holdings, turbulence). PPO is updated using the **raw** `a`, not the executed `a'`. Consequences:

- Gradient credit assignment uses the wrong action when constraints bind.
- If sampling `+1.0`, `+2.0`, `+3.0` all yield the same executed trade ("buy as much as cash allows"), the policy cannot distinguish them — pressure on the mean to drift outward without a learning signal pulling it back.
- This is the well-known **action-saturation problem** in continuous control RL.

**Symptoms in our PPO run**: rising `clip_fraction` (0.14 → 0.38), widening `std` (1.00 → 1.15), `reward_mean` flat for 150k+ steps.

**Mitigations** (ordered by effort):
1. Lower `hmax` (less headroom for infeasible asks)
2. Add a `|requested − executed|` penalty to the reward
3. Re-design the action space to be cash-conditional (e.g., action ∈ [-1,1] = fraction of available cash to use)

### 9.2 Turbulence override hides agent agency
On high-VIX days the env replaces the agent's action with a forced full liquidation. PPO loses control entirely, so the policy never learns to handle market stress — it only learns the conditional behavior given the override.

### 9.3 Reward = pure ΔP&L
No drawdown penalty, no Sharpe shaping, no transaction-frequency penalty beyond the bps fee. The agent optimizes for raw return, not risk-adjusted return.

### 9.4 Hardcoded ticker counts in tutorial code
See §8.3. The repo's MVO baseline assumes the pre-2019 DJI composition.

### 9.5 Train-once / no retraining (alpha decay)
The author of the tutorial explicitly notes: training happens once on 2009-01 to 2020-07, then the policy trades 2020-07 to 2021-10 with no retraining and no hyperparameter re-tuning. In production, quant teams retrain weekly/monthly/quarterly because markets change — the longer you trade after the last training cutoff, the further the live data drifts from the training distribution, and the strategy's edge ("alpha") decays. Expect performance to degrade toward the end of the trade window for this reason alone.

### 9.6 Unnormalized state vector
The default state mixes features on wildly different scales: cash (~10⁶), prices (10–10³), share holdings (0 to thousands), bounded indicators like RSI (0–100), unbounded indicators like CCI (±300), and price-scale indicators like Bollinger bands and SMAs. Feeding this directly into an MLP with `Tanh` activations is poor practice — the large-magnitude features saturate the activations (zero gradient), while the small ones dominate the learnable signal. The network's effective capacity collapses to whatever features happen to be accidentally well-scaled.

**Why FinRL ships it this way**: historical inertia — the original 2018 paper didn't normalize, and the maintainers kept the convention for reproducibility. The single existing mitigation, `reward_scaling = 1e-4`, scales only the reward, not the observation.

**Symptoms in our runs that point to this**:
- PPO `explained_variance` stuck near 0 for 200k steps — critic cannot fit returns when inputs are unscaled.
- `value_loss` swinging from 12 → 278 → 67 — numerical instability in the critic, exacerbated by both unscaled inputs and the magnitude of raw $ rewards.
- Slow / non-convergent reward curves across A2C and PPO — common signature of input-scale problems in deep RL.

**Conceptual options** (no implementation here):
1. **Running mean/std observation normalization** (e.g., SB3's `VecNormalize`): the standard, low-effort fix. The wrapper maintains running statistics of every observation dimension during training, z-scores them on the fly, and persists the stats so the same transform applies at backtest time. Pitfall: the statistics file must be saved with the model and reused at inference — fitting a fresh normalizer on trade data would leak lookahead.
2. **Static z-score inside the env**: compute mean/std of each feature group from the training set once, then apply in `_initiate_state` / `_update_state`. More explicit and debuggable but requires you to manage the stats yourself.
3. **Re-design the feature set** (the "quanty" approach): replace raw prices with log-returns, replace cash/holdings with portfolio weights summing to 1. Features become naturally bounded and stationary, but you also have to rethink the action space (allocation weights instead of share counts). Highest payoff, highest effort.

**Why this matters here**: addressing scale is likely the single largest practical improvement available to this project — larger than tuning learning rates or training longer. Combined with reducing `hmax` (per §9.1), it would directly attack the two biggest weaknesses in the default FinRL setup.

---

## 10. Tuning Recommendations

If reusing this project as a starting point:

| Goal | Change |
|---|---|
| Train longer | `total_timesteps=300_000` to `1_000_000` for PPO; A2C needs more |
| Stabilize PPO | `learning_rate=1e-4`, `n_steps=4096`, possibly linear LR schedule |
| Reduce action clipping | Lower `hmax` from 100 to 50; or add an explicit penalty |
| Reduce DDPG memory | `buffer_size=100_000` instead of default 1M |
| Normalize observations | Wrap env with `VecNormalize` (norm_obs + norm_reward); save stats with the model (see §9.6) |
| Compare fairly | Train all algos with the same step budget, then judge by **backtest equity curve and Sharpe**, not by training logs |

---

## 11. File Map (Quick Reference)

| Path | What lives there |
|---|---|
| `finrl/config.py` | `INDICATORS`, default hyperparameters for each algo |
| `finrl/meta/env_stock_trading/env_stocktrading.py` | `StockTradingEnv` (state, action, reward, step) |
| `finrl/meta/preprocessor/preprocessors.py` | `FeatureEngineer` (indicator computation via `stockstats`) |
| `finrl/agents/stablebaselines3/models.py` | `DRLAgent` wrapper, `TensorboardCallback` |
| `1-Introduction/Stock_NeurIPS2018/Stock_NeurIPS2018_1_Data.ipynb` | Data download + feature engineering |
| `1-Introduction/Stock_NeurIPS2018/Stock_NeurIPS2018_2_Train.ipynb` | Model training |
| `1-Introduction/Stock_NeurIPS2018/Stock_NeurIPS2018_3_Backtest.ipynb` | Backtest + baselines + plots |
