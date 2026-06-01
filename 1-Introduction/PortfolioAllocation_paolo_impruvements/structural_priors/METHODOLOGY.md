# Residual-Policy Methodology

How a structural prior is injected into a PPO portfolio allocator, why it
works, what every parameter means, and how to test it end-to-end.

This document is the technical companion to `STRUCTURAL_PRIORS.md` (which is
the "what's in this folder" summary). Read this when you want to know how
the technique actually works and how to evaluate it cleanly.

---

## 1. Problem statement

Pure deep reinforcement learning applied to portfolio allocation suffers a
characteristic failure mode: the policy collapses toward uniform weights
(`1/N`) and stays there, regardless of architecture, reward function, or
hyperparameters. The 47-experiment sweep in `Article_priority_1` confirmed
this empirically — every PPO configuration landed within 0.10 of
EqualWeight's Sharpe, and `inspect_policy.py` reported "the allocation
signal is weak; given freedom to concentrate, it declines to."

The root cause is **data scarcity relative to model capacity**. PPO with a
~17 k-parameter encoder has to discover the entire mapping
`observation → weights` from ~2,900 daily bars per training window. The
cross-sectional return dispersion at daily frequency is small relative to
noise, so the policy gradient signal pushing it away from uniform is weak.
PPO defaults to the maximum-entropy point (`softmax(0) = 1/N`) because
nothing teaches it otherwise.

Classical practitioner allocators (risk parity, Black-Litterman, the AQR /
Bridgewater family) don't have this problem because they don't *learn*
allocation from observation. They derive weights from theoretically-grounded
rules (closed-form, 1-2 free parameters), which makes them data-efficient
by construction. The residual-policy technique imports that strength into
DRL: keep the closed-form prior, let the network learn only the small
deviations from it.

---

## 2. Theoretical framework

### 2.1 The mathematical change

Standard DRL with softmax allocation:

```
w = softmax(policy_output)
```

Residual policy with structural prior:

```
w = softmax(policy_output + alpha * log(w_prior(state)))
```

where `w_prior(state)` is a closed-form weight vector computed from the
trailing-252-day covariance matrix (already part of the state, no
look-ahead).

### 2.2 Why log-space addition is the right operation

Multiplicative interpretation:

```
softmax(a + log(p)) = exp(a + log(p)) / sum_j exp(a_j + log(p_j))
                   = p * exp(a) / sum_j p_j * exp(a_j)
```

So adding `log(w_prior)` to the pre-softmax action is equivalent to
**multiplying** the prior weights by `exp(action)`, then renormalising. A
zero action multiplies by 1 — the prior is unchanged. A positive action on
asset `i` tilts the portfolio toward asset `i` by a factor `exp(a_i)`. This
is precisely the Bayesian posterior structure:

```
log(posterior) = log(prior) + log(likelihood) - const
```

i.e. the agent's action plays the role of a `log-likelihood` of relative
attractiveness, and the prior shapes the "default" allocation. This is the
canonical way to inject a prior into a categorical distribution and is
mathematically clean (unlike, say, a convex combination
`alpha * w_agent + (1 - alpha) * w_prior`, which has no probabilistic
interpretation and breaks differentiability properties at the boundary).

### 2.3 Gradient flow

PPO records `(state, action, reward)` tuples and updates the policy
network's parameters so that the policy increases the probability of high-
reward actions. The env's transformation
`action -> action + alpha * log(w_prior(state))` is **deterministic given
the state**, so from the network's perspective:

> Given state `s`, output `a` such that `softmax(a + alpha * log(w_prior(s)))` is profitable.

The policy network learns this *implicitly*. The action it emits is
internally re-interpreted by the env as a tilt; the network's gradient
correctly flows through softmax, through the addition (which is
gradient-transparent), and into the network parameters. PPO does not need
to be aware of the prior — only the env does.

### 2.4 Limit behaviour

- `alpha = 0` (or `policy_prior.enabled = false`): identical to standard
  PPO. No prior; the agent learns from scratch.
- `alpha = 1` (the canonical "residual policy" form): at network
  initialisation (random weights yielding mean-zero actions), the resulting
  allocation is exactly `w_prior`. The agent starts deployment at a known-
  sensible allocation and learns deviations.
- `alpha -> infinity`: the prior dominates regardless of the network's
  output. The policy effectively becomes the prior itself; PPO updates have
  no influence.
- `0 < alpha < 1`: prior is anchored but the agent has more leverage to
  deviate (it takes a "smaller" action to produce the same tilt magnitude
  in weight space).

---

## 3. Implementation

### 3.1 Where the modification lives

A single ~5-line addition in `LogReturnPortfolioEnv.step()` (in `utils.py`),
called **before** the parent class's existing `softmax_normalization`. Two
new helper methods on the env class encapsulate the change:

| Method | What it does |
|---|---|
| `_compute_prior_weights()` | Reads the current `self.covs` (trailing-252-day covariance, set by the parent during the previous step — no look-ahead), computes the prior weight vector per `self.policy_prior_type`, returns a length-`stock_dim` array summing to 1. |
| `_apply_policy_prior(actions)` | Returns `actions + alpha * log(w_prior)`. Called at the top of `step()`. |

The factory `make_portfolio_env` reads the `policy_prior` config block and
threads the four new constructor arguments through to
`LogReturnPortfolioEnv`. No changes are required in `02_train.py`,
`03_backtest.py`, `04_backtrader_replay.py`, `05_quantstats_report.py`,
`policies.py`, `features.py`, or any stage-1 code path.

### 3.2 The four prior types

| Type | Formula | Properties |
|---|---|---|
| `none` | (disabled) | Equivalent to `Article_priority_1`. |
| `equal_weight` | `w_i = 1 / N_risky` for risky names; `cash_share` for cash | Floor case. Tests whether *any* non-trivial prior helps. |
| `inverse_vol` | `w_i ∝ 1 / sqrt(diag(Sigma)_i)`, normalised across risky names | Closed-form, no solver. Diagonal approximation to risk parity. Default. |
| `risk_parity` | Iterative solver such that `w_i * (Sigma @ w)_i = const` | Full off-diagonal-aware risk parity on the risky sub-covariance. Solver in `_solve_risk_parity` (typical convergence: 10-50 iterations to `tol=1e-7`). |

Cash is always held out of the covariance-based calculation. Its synthetic
zero variance would either send `inverse_vol` to infinity (`1 / 0 -> inf`)
or make the risk-parity sub-covariance singular. Cash gets `cash_share`
(default `1 / stock_dim` for equal share with risky names); the risky
weights are normalised to sum to `1 - cash_share`.

### 3.3 Order of operations per training step

1. Policy network outputs raw action `a` (logits-like).
2. Env's `step(a)` is called.
3. **NEW**: env replaces `a` with `a + alpha * log(w_prior(self.covs))`.
4. Parent class's `softmax_normalization(a_modified)` produces weights `w`.
5. Parent class advances `self.day`, recomputes `self.covs` for the next
   step, computes portfolio return.
6. **Risk-off gate**: if `turbulence > threshold`, override `w` to
   `[0, ..., 1.0 at cash_idx, ..., 0]` and book the cash bar return.
7. Transaction cost penalty applied (using either `naive` or
   `drift_adjusted` turnover per config).
8. Shaped reward (`diff_sharpe` / `article_absolute` / etc.) computed.
9. PPO records `(obs, a, reward)`; on rollout end the policy gradient
   updates the network.

Note that the gate runs **after** the prior, not before — the prior is a
soft anchor on normal days; the gate is a hard safety on stress days. They
do not interfere.

---

## 4. Configuration parameters

All live under the top-level `policy_prior` block in `config.json`.

### 4.1 `policy_prior.enabled`

| | |
|---|---|
| Type | boolean |
| Default | `true` |
| Effect | Master switch. When `false`, the env behaves identically to the `Article_priority_1` baseline regardless of the other three settings. When `true` and `type != "none"`, the residual mechanism is active. |

### 4.2 `policy_prior.type`

| | |
|---|---|
| Type | string |
| Default | `"inverse_vol"` |
| Allowed | `"none"`, `"equal_weight"`, `"inverse_vol"`, `"risk_parity"` |
| Effect | Selects the formula used to compute `w_prior` from the trailing covariance. See section 3.2 for the formulas. |

Recommendation: start with `"inverse_vol"` (closed-form, no solver
overhead, the standard practitioner default for multi-asset baskets where
asset-class diversity dominates pairwise correlation effects). Test
`"risk_parity"` if pairwise correlations between your assets are large.

### 4.3 `policy_prior.alpha`

| | |
|---|---|
| Type | float |
| Default | `1.0` |
| Practical range | `[0.25, 4.0]` |
| Effect | Blending strength in log-space (see section 2.4). |

Higher `alpha` makes the prior "stickier" (the policy has to output larger
actions to move weights the same distance from the prior). Lower `alpha`
makes the prior "softer" (the policy can deviate easily). `alpha = 1` is
canonical residual form: action `= 0` produces exactly `w_prior`.

### 4.4 `policy_prior.cash_share`

| | |
|---|---|
| Type | float or `null` |
| Default | `null` (use `1 / stock_dim`, i.e. equal share with risky) |
| Range | `[0.0, 1.0]` |
| Effect | Target cash allocation in the prior. Cash is held out of the covariance-based calculation; the risky names share `1 - cash_share` per the chosen `type`. |

Set explicitly (e.g. `0.10`) if you want the prior to consistently hold a
fixed cash fraction independent of the universe size.

---

## 5. Testing protocol

A five-stage protocol that decomposes the contribution of each piece.
Stages 1 and 2 are sanity / smoke tests; stages 3 - 5 are the actual
research questions. Run them in order; do not proceed to a later stage
until earlier ones pass.

### 5.1 Stage 1 — code-correctness sanity

A quick numerical check that the helpers do what they should. Run a Python
session and verify:

- `_solve_risk_parity` on a hand-picked 3-asset covariance returns weights
  summing to 1 with approximately-equal risk contributions
  (`w_i * (Sigma @ w)_i / total ≈ 1/3`).
- `_compute_prior_weights` on the trained env's first observation returns
  a length-`stock_dim` vector summing to 1 with cash at the configured
  `cash_share` and risky weights aligned with the chosen `type`.
- `softmax(0 + log(w_prior))` numerically equals `w_prior` (the action-=-0
  identity).

### 5.2 Stage 2 — initial-policy smoke test

Train a deliberately under-trained model (`total_timesteps = 5000`) and
inspect:

- `inspect_policy.py` per-asset weight `mean` should match `w_prior`
  almost exactly. At 5 k steps the policy has barely moved from
  initialisation; what you see is the prior with negligible learned tilt.
- Cash weight `mean` should match `cash_share`.

If this fails, the prior is not flowing through `step()` correctly.

### 5.3 Stage 3 — marginal-contribution test (the headline experiment)

Train two models with full `total_timesteps = 150000`:

| Experiment | Override | Question answered |
|---|---|---|
| `baseline` | (none — uses default config) | Residual PPO performance |
| `prior_off` | `policy_prior.enabled = false`, `policy_prior.type = "none"` | Pure-DRL performance (= `Article_priority_1` equivalent) |

Compare on:

- **`env_sharpe`** — does the prior add Sharpe?
- **`ann_turnover`** — does the prior anchor the policy and reduce churn?
- **Per-asset weight `std`** in `inspect_policy.py` — does the policy
  meaningfully deviate from the prior, or does it just sit on it?

Expected: `baseline` should at minimum match `prior_off` on Sharpe and
beat it on turnover. A meaningful Sharpe improvement (`>= 0.02`) is the
strong outcome.

### 5.4 Stage 4 — prior-type comparison

Run all three non-none types:

| Experiment | Override |
|---|---|
| `prior_equal_weight` | `policy_prior.type = "equal_weight"` |
| `baseline` *(inverse_vol)* | (default) |
| `prior_risk_parity` | `policy_prior.type = "risk_parity"` |

Expected ordering (general practitioner wisdom):

- `equal_weight < inverse_vol <= risk_parity` on Sharpe, with `inverse_vol
  ≈ risk_parity` when pairwise correlations are weak (diagonal-dominant
  covariance, common in diversified multi-asset baskets).
- If `equal_weight` matches or beats the others, the universe's
  cross-sectional vol dispersion is too small to exploit — the prior
  itself is contributing little and the marginal value is elsewhere.

### 5.5 Stage 5 — alpha sweep

Sweep `policy_prior.alpha` on the best-performing prior type:

| Experiment | Override |
|---|---|
| `alpha_a0.25` | `policy_prior.alpha = 0.25` |
| `alpha_a0.5` | `policy_prior.alpha = 0.5` |
| `alpha_a1.0` (= baseline) | (default) |
| `alpha_a2.0` | `policy_prior.alpha = 2.0` |

Expected: a smooth curve with a maximum near `alpha = 1` or slightly
higher. A monotone curve (lower alpha is always better) suggests the
prior is hurting; a monotone curve (higher alpha is always better)
suggests the prior is dominating and the agent is learning nothing useful.

### 5.6 Optional — gate-vs-prior decomposition

To attribute the total system performance to the gate vs the prior, run
four conditions:

| Experiment | `risk_off.enabled` | `policy_prior.enabled` |
|---|---|---|
| `gate_off_prior_off` | false | false |
| `gate_off` | false | true |
| `prior_off` | true | false |
| `baseline` | true | true |

Then:

- Prior contribution: `gate_off - gate_off_prior_off`
- Gate contribution: `prior_off - gate_off_prior_off`
- Joint contribution: `baseline - gate_off_prior_off`

If `prior + gate ≈ joint`, the effects are additive. If `joint < prior + gate`,
there's a saturation effect (the gate is doing what the prior would have
done on those days, or vice versa).

---

## 6. Caveats and known interactions

### 6.1 The trained model is inseparable from the prior

The policy network is trained to output **tilts**, not raw weights. At
inference (stages 3, 4, 5 and `predict_tomorrow.py`) the env reads the
`policy_prior` config block and re-applies the same transformation. **Do
not change `policy_prior.type` or `policy_prior.alpha` between training
and inference** — the policy's actions will be miscalibrated and the
resulting allocations meaningless.

### 6.2 The right benchmark in stage 3 is not (yet) present

Stage 3 currently reports `MinVariance`, `EqualWeight`, and
`EqualWeight_w_Cash`. The most informative benchmark for a residual
policy — **a static, daily-rebalanced portfolio of `w_prior` itself with
no learning** — is not computed. Without it, you cannot directly attribute
the system's Sharpe lift to "the prior alone" vs "the agent's learned tilts
on top of the prior." This is a known gap; adding it is a few lines in
`03_backtest.py` and is a recommended follow-up.

### 6.3 Walk-forward and the prior re-estimate

Under walk-forward training, each window builds its own trailing covariance
and therefore its own prior. This is correct behaviour but means the prior
is technically window-specific. If you train on a single split, the prior
is computed from one trajectory of covariances and may be more "fitted" to
that window.

### 6.4 Cash gate dominance

When the risk-off gate fires (`turbulence > 80` per default config), it
overrides the prior-modified softmax weights with 100 % CASH. On those
days the prior is moot. This is by design — gate is hard safety, prior is
soft anchor.

---

## 7. References

| Topic | Reference |
|---|---|
| Risk parity (theoretical foundation) | Maillard, S., Roncalli, T., & Teïletche, J. (2010). *The Properties of Equally Weighted Risk Contribution Portfolios.* Journal of Portfolio Management, 36(4), 60-70. |
| Bayesian prior + tilts in portfolio construction | Black, F., & Litterman, R. (1992). *Global Portfolio Optimization.* Financial Analysts Journal, 48(5), 28-43. |
| Trading toward an analytically-derived target with friction | Garleanu, N., & Pedersen, L. H. (2013). *Dynamic Trading with Predictable Returns and Transaction Costs.* Journal of Finance, 68(6), 2309-2340. |
| The practitioner workflow this folder mimics | Pedersen, L. H. (2015). *Efficiently Inefficient: How Smart Money Invests and Market Prices Are Determined.* Princeton University Press, chapters 3-4. |
| The pure-DRL approach this folder departs from | Kashif, A., & Slepaczuk, R. (2025). *Deep Reinforcement Learning Framework for Diversified Portfolio Management Across Global Equity Markets.* arXiv:2605.17307. (The `Articles/` folder in this repo.) |
| The differential Sharpe reward used in training | Moody, J., & Saffell, M. (2001). *Learning to Trade via Direct Reinforcement.* IEEE Transactions on Neural Networks, 12(4), 875-889. |

---

## 8. Summary

The residual-policy technique replaces "let PPO discover the entire
allocation function" with "let PPO discover small corrections to a closed-
form practitioner allocation." Mathematically it is a single addition in
log-space, applied inside the env before softmax, with three configuration
knobs (`type`, `alpha`, `cash_share`) and one master switch. Engineering
cost is minimal (~30 lines in `utils.py`); the conceptual lift is
substantial because the policy starts deployment at a known-sensible point
and trains only the residual.

The technique is the canonical fix for the "data scarcity + collapse to
uniform" failure mode observed in `Article_priority_1`. Whether it actually
improves out-of-sample Sharpe on your universe is an empirical question
answered by the stage-3 marginal-contribution test in section 5.3. If it
does, you have a deployable hybrid system (`risk parity + RL tilts`) that
is materially closer to what production systematic allocators actually run.
If it does not, you have learned that the data limit is below what any
prior can overcome on this problem — which is itself a clean research
result.
