# PPO hyperparameters — tuning guide

This document covers every knob you can tune for the PPO model in the portfolio-allocation pipeline, what each one does, how it interacts with the others, and what to expect when you change it. Examples are concrete and intended to be runnable by editing a single line in `config.json`.

The pipeline trains PPO through Stable-Baselines3 (SB3). Most of the parameters below are SB3 PPO constructor arguments, surfaced in `config.json` under `models.ppo.model_kwargs` and `models.ppo.policy_kwargs`. The rest are env-level or training-loop knobs (reward shaping, total budget, early stopping) that affect what the PPO optimiser actually sees.

---

## Where the knobs live

Every knob is in `<folder>/config.json`:

| JSON path                                | What it controls                                                |
|------------------------------------------|-----------------------------------------------------------------|
| `models.ppo.model_kwargs.*`              | SB3 PPO constructor args (the "inner" RL hyperparameters)       |
| `models.ppo.policy_kwargs.*`             | Network architecture and activation function                    |
| `models.ppo.total_timesteps`             | Max training budget per training run                            |
| `env.*`                                  | Reward shaping (drives what PPO is optimising)                  |
| `early_stopping.*` (folder 5+)           | Effective stop criterion when ES is on                          |
| `training.seed`                          | RNG seed for reproducibility                                    |

A single change in `config.json`, re-run `02_train.py`, and the new behaviour is in effect for the next training run.

---

## 1. learning_rate

How big each gradient step is during optimisation. SB3 PPO uses Adam under the hood.

| Setting | What happens |
|---|---|
| Very low (1e-5)        | Training crawls; needs huge `total_timesteps` to converge. |
| Low (1e-4, current)    | Conservative; safe but slow. |
| Medium (3e-4, SB3 default) | Balanced — good starting point for most problems. |
| High (5e-4 to 1e-3)    | Faster initial progress, higher risk of divergence. |
| Very high (>1e-3)      | Likely to overshoot and not recover; value loss explodes. |

Recommended starting point: **0.0003**.
Best paired with: higher `clip_range` (0.3) and higher `ent_coef` (>=0.01) when going aggressive.
Note: with early stopping enabled (folder 5+), a higher LR is safer because ES catches divergence.

Expected effect of bumping 1e-4 -> 3e-4: stage 2 stdout shows the policy loss decreasing faster in the first few thousand steps; validation Sharpe usually peaks earlier (~10–30k steps instead of 40–60k).

---

## 2. n_steps

Number of env steps collected per rollout before each policy update. Total policy updates over training = `total_timesteps / n_steps`.

| Setting | What happens |
|---|---|
| Small (256, 512)        | Many updates, fast adaptation, noisier per-update gradients. |
| Medium (1024)           | Good balance; 2× more updates than the current default. |
| Large (2048, current)   | Smoother gradients, fewer total updates. |
| Very large (4096+)      | Very smooth but slow to learn. Rare for daily RL. |

With ~1512 days per walk-forward training window and `n_steps=2048`, each rollout covers ~1.35 full episodes. `n_steps=1024` covers ~0.67 episodes per rollout but gives 2× the update count within the same budget.

Recommended: **1024 or 2048**.

Expected effect of 2048 -> 1024 paired with `batch_size=64`: stage 2 runs more update iterations per training timestep, so the policy adapts to validation Sharpe improvements faster. Early stopping usually triggers a few thousand timesteps sooner.

---

## 3. batch_size

Minibatch size for SGD inside each policy update. Number of minibatches per update = `n_steps / batch_size`.

| Setting | What happens |
|---|---|
| 32                | Very noisy per-minibatch gradients; sometimes helps generalisation. |
| 64 (SB3 default)  | Standard. |
| 128 (current)     | Smoother per-minibatch gradients. |
| 256+              | Smooths too much; close to full-batch gradient. |

Recommended: **64**.

Constraint: `batch_size` should divide `n_steps` cleanly. With n_steps=1024 and batch_size=64 you get 16 minibatches per update; with n_steps=2048 and batch_size=128 you get the same 16. The total compute is similar; the difference is in gradient variance per minibatch.

---

## 4. n_epochs

How many SGD passes over each rollout's data before discarding it.

| Setting | What happens |
|---|---|
| 3                | Minimal data reuse, very on-policy, slow to learn but stable. |
| 5                | Moderate reuse, less overfitting per rollout. |
| 10 (SB3 default, currently implicit) | Standard. |
| 20+              | High reuse; risk of policy drift from the old data distribution. |

For noisy rewards (DSR/Sortino): try **5**. For log_return without TC penalty: 10 is fine.

Expected effect of 10 -> 5: training is slightly slower per timestep but the policy is less prone to overfitting to a single rollout's quirks. Validation Sharpe trajectory in `models/agent_ppo_w*.history.json` becomes smoother.

---

## 5. ent_coef

Weight on the policy-entropy bonus added to the PPO objective. Larger value = wider action distribution = more exploration.

| Setting | What happens |
|---|---|
| 0.0 (SB3 default) | No entropy bonus. Policy can collapse to deterministic too quickly. |
| 0.001             | Light exploration push. |
| 0.005 (current)   | Moderate exploration. |
| 0.01 - 0.02       | Strong exploration. Useful for noisy DSR rewards. |
| 0.05+             | Action distribution too random; policy refuses to commit. |

Recommended for DSR/Sortino: **0.01 to 0.02**. For log_return: 0.005 is fine.

Expected effect of 0.005 -> 0.02:
- Stage 3 `df_actions` (the post-softmax weights) show entropy ~ln(27) for the first ~10k timesteps, then specialise.
- Less prone to collapse onto a single concentrated position (e.g., "all in AAPL").
- The trade-off is slower convergence to the final specialised weights.

---

## 6. clip_range

PPO's signature trick — clips the policy-update ratio between old and new policy to `[1 - clip, 1 + clip]`. Bounds how far the policy can move in a single update.

| Setting | What happens |
|---|---|
| 0.1               | Very conservative updates. Slow but very stable. |
| 0.2 (SB3 default) | Standard. |
| 0.3               | Larger updates per step; pair with higher LR. |
| 0.4+              | PPO loses its guarantees; close to vanilla policy gradient. Unstable. |

Recommended for aggressive training: **0.3** paired with `learning_rate=3e-4` and `ent_coef=0.02`.

Expected effect: with clip=0.3, you'll occasionally see the `clip_fraction` in tensorboard logs go above 0.2 (some fraction of updates are actually being clipped). That's fine — it means the optimiser is taking large enough steps for clipping to bite, which is what we want.

---

## 7. gamma

Discount factor. The agent values a reward `t` steps in the future at `gamma^t`. Effective planning horizon ≈ `1 / (1 - gamma)`.

| Setting | Effective horizon | Notes |
|---|---|---|
| 0.9               | 10 days   | Very short-sighted. |
| 0.95              | 20 days   | Short horizon. Reasonable for daily rebalancing. |
| 0.99 (SB3 default, current) | 100 days  | Standard for continuous control. |
| 0.995             | 200 days  | Closer to "annual" planning. |
| 0.999             | 1000 days | Far too long for daily decisions on noisy markets. |

Recommended for portfolio allocation: **0.95 or 0.99**.

Expected effect of 0.99 -> 0.95: the agent stops weighting decisions by their effect on rewards 100 days out. With DSR specifically, the running EMA already provides longer-horizon context, so a shorter discount can work fine.

---

## 8. gae_lambda

GAE λ controls the bias-variance tradeoff in advantage estimation. λ=0 reduces to TD(0); λ=1 reduces to Monte Carlo returns.

| Setting | What happens |
|---|---|
| 0.85              | Lower variance, higher bias. Faster but rougher learning. |
| 0.9               | Slight noise reduction. Useful for noisy DSR. |
| 0.95 (SB3 default, current) | Standard. |
| 0.99              | Closer to Monte Carlo. Higher variance, often slower. |

Touch only after the higher-impact knobs (LR, ent_coef, n_steps) have been tuned.

---

## 9. vf_coef

Weight on the value-function loss inside PPO's combined loss.

| Setting | What happens |
|---|---|
| 0.25              | Policy dominates the loss. |
| 0.5 (SB3 default, current) | Balanced. |
| 1.0               | Value function dominates; can help if value loss is huge. |

Tune only if tensorboard shows `train/value_loss` orders of magnitude larger or smaller than `train/policy_loss`. Otherwise leave at 0.5.

---

## 10. max_grad_norm

Gradient clipping threshold. Caps the L2 norm of the gradient before applying the update.

| Setting | What happens |
|---|---|
| 0.5 (SB3 default) | Standard, almost never needs tuning. |
| 1.0               | Allows larger updates. |
| None / 1e9        | No clipping. Risky with noisy rewards. |

Leave at default unless training diverges visibly (NaN losses, exploding actions).

---

## 11. policy_kwargs.net_arch

Shape of the MLP that maps observation -> action mean / value estimate. The observation is 945-dimensional (27×27 covariance block + 8×27 indicator rows, flattened).

| Setting | What happens |
|---|---|
| `[64, 64]` (SB3 default; `null` -> this) | Small, fast, possibly underfits a 945-dim observation. |
| `[128, 128]`                              | Bigger; usually no overfitting for this dataset size. |
| `[256, 256]`                              | Larger; needs more timesteps; may overfit per window. |
| `[128, 128, 64]`                          | Deeper, narrowing. |
| `{"pi": [128, 128], "vf": [128, 128]}`    | Separate policy and value heads. Often best for actor-critic. |

Recommended: **`{"pi": [128, 128], "vf": [128, 128]}`**.

Concrete JSON value:
```
"policy_kwargs": {
  "net_arch":      {"pi": [128, 128], "vf": [128, 128]},
  "activation_fn": "Tanh"
}
```

Expected effect: training time goes up ~20–30% per step (bigger forward/backward passes), but the agent can express more nuanced allocation policies. Worth it for the 945-dim observation.

---

## 12. policy_kwargs.activation_fn

Nonlinearity in the MLP hidden layers.

| Setting | What happens |
|---|---|
| `"Tanh"` (SB3 default, recommended for PPO) | Bounded outputs, stable with normalised observations. |
| `"ReLU"`                                    | Faster but can produce dead neurons; less stable here. |
| `"ELU"`                                     | Smoother variant of ReLU. Compromise. |

Stick with **Tanh** for PPO portfolio allocation. ReLU is mostly for image-based RL.

---

## 13. total_timesteps

Training budget. With early stopping enabled (folder 5+), this is the **maximum** — actual training usually stops earlier.

| Setting | Notes |
|---|---|
| 50,000              | Often too short for the network to settle. |
| 80,000 (current)    | Reasonable budget per walk-forward window. |
| 150,000             | Recommended ceiling when ES is on. Gives headroom for late-improving windows. |
| 200,000             | Only useful if you've also raised `patience`. |

Recommended with ES: **150,000**. ES handles when training actually stops; this is just the cap.

---

## 14. seed

RNG seed. Affects initial network weights, action sampling, env shuffling.

A single seed gives one point estimate. RL is high-variance, so headline metrics depend heavily on the seed. To draw real conclusions, train 3–5 seeds and report mean ± std. The plumbing isn't wired up in this pipeline by default but can be added by looping `02_train.py` over seeds and ensembling at trade time (improvement #5 in `FinRL_PortfolioAllocation_improvements.md`).

Reproducibility note: same seed + same machine + same library versions = byte-identical training trajectory. Different machines (different BLAS, different PyTorch version) can produce slightly different results even with the same seed.

---

## 15. env.reward_scaling

Multiplicative scale applied to the reward inside our subclassed env.

| `reward_mode`   | Recommended `reward_scaling` | Reason |
|---|---|---|
| `"value"`       | 1e-4                          | Raw value is O(1e6); aggressive scaling required. |
| `"log_return"`  | 1.0                           | Log returns are O(1e-2)/day. |
| `"diff_sharpe"` | 1.0                           | DSR is O(1) after EMA warm-up. |
| `"diff_sortino"`| 1.0                           | DDR is O(1) after warm-up. |

Wrong scaling failure modes:
- Too high in value mode -> loss explosion.
- Too low in log_return mode -> warning printed, gradient starves.
- Wrong for DSR/DDR -> training diverges quietly.

---

## 16. env.reward_mode

What the per-step reward signal looks like. Full discussion in folder-specific README appendices.

| Mode             | Reward formula                            | Notes |
|---|---|---|
| `"value"`        | `new_portfolio_value`                     | Non-stationary, dominated by late-period samples. Avoid for new work. |
| `"log_return"`   | `log(1 + r) * scaling`                    | Stable, additive, simple. Good baseline. |
| `"diff_sharpe"`  | DSR contribution (Moody & Saffell)        | Trains running-Sharpe maximisation. Noisier. |
| `"diff_sortino"` | DDR contribution                          | Same idea but only penalises downside. |

Default in folder 5: `"diff_sharpe"`.

---

## 17. env.transaction_cost_penalty

Per-unit-turnover cost applied during training. Independent of the broker-level commission in stage 4.

| Setting | What happens |
|---|---|
| 0.0   (current) | Free rebalancing during training. Stage 4 still charges. |
| 0.001           | Matches default broker commission. |
| 0.002           | Stronger penalty; smoother weight trajectories. |
| 0.005           | Aggressive; encourages large, infrequent rebalances. |

Expected effect of 0.0 -> 0.002: stage 3's `df_actions` change less day-over-day, stage 4's `trade_log.csv` has fewer fills, the friction tax (stage 3 vs stage 4 equity gap) shrinks.

---

## 18. env.diff_ratio_eta

EMA decay for the running A/B/D moments in DSR/DDR. Effective half-life is approximately `1 / eta` days.

| Setting           | Half-life      | Notes |
|---|---|---|
| 1/63 ≈ 0.016      | ~quarter        | Fast adaptation, noisy DSR. |
| 1/126 ≈ 0.008     | ~half-year      | Compromise. |
| 1/252 ≈ 0.004 (current) | ~year     | Standard. |
| 1/504 ≈ 0.002     | ~2 years        | Smooth but slow to track regime changes. |

Only relevant when `reward_mode` is `diff_sharpe` or `diff_sortino`. Otherwise the value is ignored.

---

## 19. early_stopping.* (folder 5 only)

| Knob          | Current | Effect of raising it |
|---|---|---|
| `val_fraction`| 0.1     | Larger validation slice -> more stable Sharpe estimate, less training data. |
| `eval_freq`   | 5000    | Less frequent evals -> cheaper but slower to react. |
| `patience`    | 5       | More tolerance for noise -> longer training, possibly better fits. |
| `min_delta`   | 0.01    | Stricter "improvement" threshold -> earlier stops. |

For DSR (which produces volatile validation Sharpe), try `patience=8` or `10` to avoid stopping on a noise dip.

---

## Suggested experiment ladder

Don't change everything at once — RL is too noisy to attribute combined effects. Run each as a separate ablation against the current baseline:

| # | Change                                                    | One-line edit                                                                                          | Expected effect |
|---|---|---|---|
| 0 | (baseline)                                                | (no change)                                                                                            | Reference run. |
| 1 | Bigger LR                                                 | `"learning_rate": 0.0003`                                                                              | Faster convergence; usually hits ES earlier. |
| 2 | More entropy                                              | `"ent_coef": 0.02`                                                                                     | Spreads weights early, less collapse risk. |
| 3 | Faster updates                                            | `"n_steps": 1024, "batch_size": 64`                                                                    | More updates per timestep; smoother validation Sharpe. |
| 4 | Shorter discount                                          | `"gamma": 0.95`                                                                                        | Agent values nearer-term returns more. |
| 5 | Bigger net                                                | `"policy_kwargs": {"net_arch": {"pi":[128,128],"vf":[128,128]}, "activation_fn":"Tanh"}`              | More capacity; ~25% slower per step. |
| 6 | Combined "tuned"                                          | Apply all five above + `total_timesteps=150000`                                                        | Best single-run config; you won't know which knob did what. |
| 7 | More patience                                             | `"early_stopping": {"patience": 10, ...}`                                                              | Less aggressive ES; longer training, possibly better. |
| 8 | Different reward                                          | `"reward_mode": "diff_sortino"`                                                                        | Downside-only risk. Different policy character. |
| 9 | TC penalty on                                             | `"transaction_cost_penalty": 0.002`                                                                    | Lower turnover, smaller friction tax. |

After each run:
- Look at `models/agent_ppo_w*.history.json` to see the validation Sharpe trajectory.
- Run stage 3 and compare equity curves / Sharpe / MaxDD against baseline.
- Run stage 4 to see how the friction tax changes.

---

## Concrete starting config (the "tuned" PPO)

A reasonable single config to try as the next experiment, combining the high-impact tweaks:

```
"ppo": {
  "use": true,
  "total_timesteps": 150000,
  "model_kwargs": {
    "n_steps":       1024,
    "ent_coef":      0.02,
    "learning_rate": 0.0003,
    "batch_size":    64,
    "n_epochs":      5,
    "clip_range":    0.3,
    "gamma":         0.95
  },
  "policy_kwargs": {
    "net_arch":      {"pi": [128, 128], "vf": [128, 128]},
    "activation_fn": "Tanh"
  }
}
```

And, in folder 5, in the `early_stopping` block: `"patience": 8`.

---

## Method note

Single-seed RL comparisons can be misleading by 0.2–0.5 in Sharpe without anything actually changing. Either:
- Run 3–5 seeds per config and report mean ± std.
- Or use a long evaluation window (the 7-year walk-forward stitch in folder 3+ helps — DSR variance averages out over more days).

If a hyperparameter change produces a Sharpe shift smaller than the seed std, it's noise. Don't chase it.

---

## What is NOT a PPO hyperparameter (but affects training)

- `data.*` — universe, dates, indicators. Changing these changes the problem, not the optimiser.
- `walk_forward.*` — splits the training data into multiple slices. PPO is trained independently per slice.
- `paths.*` — where outputs land. No effect on training.
- `training.seed` — RNG only.
- `models.<other>.use` — `a2c`, `ddpg`, `sac`, `td3`. Different algorithms with their own hyperparameter spaces. Most of the principles in this doc carry over but specific defaults differ (e.g., off-policy algorithms have `buffer_size` instead of `n_steps`).
