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

### 5.1 Stage 1 — code-correctness sanity (`test_priors.py`)

A quick numerical check that the helpers do what they should. Run this
**before** burning training time — it catches every bug in the prior /
softmax math layer without needing a trained model, in under a second.

```bash
# from the structural_priors/ folder, in the same shell you train in:
python test_priors.py
```

The script runs three tests in order, prints inputs / outputs / a
`PASS`-`FAIL` line for each, prints a summary line, and exits with code
`0` on all-pass / `1` on any failure (suitable for use as a CI gate):

1. **`_solve_risk_parity` on a hand-picked 3-asset covariance** — verifies
   the iterative solver converges to weights whose risk contributions
   `w_i * (Sigma @ w)_i / total` are each approximately `1/3`. Tests the
   only piece of code that does anything non-trivial.

2. **`compute_prior_weights_from_cov` returns a valid weight vector** —
   verifies the helper produces a length-`stock_dim` vector summing to
   1, with cash at the configured `cash_share` and risky weights aligned
   with the configured `type` (against a hand-built 4-asset toy: 3 risky
   + 1 synthetic zero-variance cash). **Reads your `config.json`**, so
   the test adapts to whatever `policy_prior.type` / `cash_share` is
   currently configured. `LogReturnPortfolioEnv._compute_prior_weights`
   is a two-line wrapper around this helper — testing the helper tests
   the env's training-time prior too.

3. **`softmax(0 + 1.0 * log(w_prior)) == w_prior` exactly** — verifies
   the algebraic identity that justifies the residual-policy
   formulation: at `action = 0` the agent's allocation IS the prior. For
   context the script also prints what `alpha = 0.0, 0.5, 2.0` produce
   so you can see the role of the blending knob visually.

If any test fails, **do not proceed to stage 2** (smoke training) — the
prior math is broken somewhere and the resulting model will be
miscalibrated. Investigate by editing the inputs in `test_priors.py`
(they are 5 lines of hand-built covariances) until the failure isolates
to a specific helper.

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

### 6.2 The static-prior benchmark across stages 3, 4, and 5

Stage 3 reports `MinVariance`, `EqualWeight`, `EqualWeight_w_Cash`, AND
`Prior_{type}` — a daily-rebalanced *static* portfolio of `w_prior`
computed via the same `compute_prior_weights_from_cov` helper the env
uses at training time (single source of truth for the prior formula).
This is the right comparator for a residual-policy PPO: lift over
`Prior_{type}` is the agent's learned tilt; lift over `EqualWeight` that
`Prior_{type}` *also* shows is the prior alone.

The static-prior baseline propagates downstream automatically:

- **Stage 3** (`03_backtest.py`): adds a `Prior_{type}` row in the
  printed summary table and a `Prior_{type}` column in
  `results/equity_curves.csv`. The curve also appears in
  `results/equity_plot.png`.
- **Stage 4** (`04_backtrader_replay.py`): overlays the static prior on
  the backtrader equity plot as a dash-dot line labelled
  `Prior ({type}, static)`, alongside the existing dashed `EqualWeight`
  and dotted `EqualWeight w/ Cash` overlays.
- **Stage 5** (`05_quantstats_report.py`): emits a third HTML report
  `report_vs_prior.html` (+ `metrics_vs_prior.csv`) using the static
  prior as the QuantStats benchmark.

The three QuantStats reports together answer three distinct questions:

| Report | Benchmark | Question |
|---|---|---|
| `report.html` | `EqualWeight` | "Did the deployable system beat the dumb passive comparator?" |
| `report_w_cash.html` | `EqualWeight_w_Cash` | "After correcting for the cash drag, any allocation skill?" |
| `report_vs_prior.html` | `Prior_{type}` | "Did the agent's *learned tilts* add value over the prior alone?" |

The propagation is gated by `benchmark.show_prior_baseline` (default
`true`) AND by `policy_prior` being active (`enabled = true` AND
`type != "none"`). Setting `show_prior_baseline = false` suppresses just
the stage-5 report; the stage-3 column and stage-4 overlay still appear
as long as `policy_prior` is active. Setting `policy_prior.enabled =
false` (the `prior_off` control experiment) auto-suppresses the prior
baseline everywhere — there is no prior to benchmark against.

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

---

## Appendix A — Env-time rebalance cadence (planned, not yet implemented)

**Status**: documented design, no implementation. As of the current
commit the pipeline rebalances *daily* everywhere (env step, stage-3
backtest, stage-4 backtrader replay, `predict_tomorrow.py`). This
appendix captures the env-time approach to weekly / monthly rebalancing
("Approach B" in the design discussion) so the design is fixed before
implementation begins.

### A.1 Motivation

Practitioner allocators (AQR, Bridgewater, BlackRock systematic)
rebalance monthly because transaction costs dominate at higher frequency
and the allocation signal does not change meaningfully day-to-day. They
achieve this by decoupling the **signal** frequency (daily, used for
estimation) from the **decision** frequency (monthly, used for action).
The current pipeline conflates the two: every daily bar is an
opportunity for a fresh allocation decision, which generates churn the
backtrader replay then has to penalise via execution costs.

The env-time cadence implements the decoupling: the policy network still
*sees* daily observations and learns from daily transitions, but the
*allocation decision* it produces is only acted upon at the configured
cadence (e.g. Friday close, or first trading day of month). Between
rebalance bars the weights drift naturally with price changes.

The cheaper alternative (the "Approach A" inference-only cadence) is
discussed in section A.5 and is recommended as a *diagnostic* but not as
the production solution.

### A.2 Mechanism

Inside `LogReturnPortfolioEnv.step()`, **before** the existing
structural-prior injection and **before** the parent class's softmax, a
new gate fires:

> "If the current bar is NOT a rebalance day, replace the agent's action
> with the action stored at the most recent rebalance bar."

The agent's network still emits a fresh action every bar — but the env
silently discards non-rebalance-day actions and reuses
`self._last_rebalance_action`. The structural prior then applies to that
(possibly stale) action, the softmax produces weights, the parent class
advances the day, the turbulence cash gate runs as today, and reward is
computed against the next bar's returns.

On rebalance days the agent's new action is captured into
`self._last_rebalance_action` for replay on subsequent non-rebalance
bars.

### A.3 Configuration (proposed)

A new top-level block in `config.json`:

```
"rebalance": {
  "cadence":     "weekly",       // "daily" | "weekly" | "monthly"
  "weekly_day":  "FRI",          // ISO weekday code; used when cadence = "weekly"
  "monthly_day": 1,              // calendar day of month (1-28); used when cadence = "monthly"
  "_notes":      "..."
}
```

Default for backward compatibility: `cadence: "daily"`, so the mechanism
is opt-in and the current pipeline keeps behaving identically. The
holiday-adjacent fallback rule: if the configured day is not a trading
day, the **next available trading day** in the calendar is used (so a
weekly Friday rebalance shifts to Monday if Friday is a market holiday).

### A.4 Impact on training

Three structural effects on PPO.

**A.4.1 Sparser policy-gradient signal.** PPO records
`(state, action, reward)` on every step. With weekly rebalancing only
~1 in 5 steps produces a fresh action; the remaining 4 steps record the
same action with new state and the next-day return. From the agent's
perspective, the non-rebalance bars are essentially zero-gradient: its
current output never moved the weights, so its contribution to the
reward at those bars is constant (zero derivative w.r.t. its parameters).
Effective number of *informative* samples per training year drops from
~252 to ~52 (weekly) or ~12 (monthly).

Practical consequence: for a fixed `total_timesteps = 150000`, the
policy receives roughly `1/5` (weekly) or `1/21` (monthly) the actual
policy-gradient updates. To compensate, `total_timesteps` should be
scaled up roughly proportionally — `750000` for weekly, `~3M` for
monthly. Walk-forward window length should also be reconsidered, because
each window now contains far fewer decision points.

**A.4.2 Reward magnitude and variance per decision event.** On any
single bar the booked return is still a single-bar return regardless of
cadence, but the *effective* return between two rebalance decisions is
the compounded N-bar return. `diff_sharpe`'s running A (mean) and B
(mean of `R^2`) moments accumulate from daily returns regardless of
cadence — but those moments now characterise the *churned* daily return
series rather than the agent's *decision-frequency* return series.

This is acceptable as a first implementation but introduces a mismatch
between training reward and the metric you really care about (the
weekly Sharpe of the deployed system). A stricter implementation would
compute reward **only on rebalance bars** and use the compounded
multi-bar return as the single per-event reward — much higher
per-event magnitude and variance. In that case `diff_ratio_eta` would
need to scale roughly with the cadence: `~1/52` for weekly,
`~1/12` for monthly, so the running half-life stays at one calendar
year.

**A.4.3 Implicit robustness pressure.** The agent learns under a hard
constraint: any weight choice will be held for N bars regardless of how
the state evolves intra-period. It cannot rely on next-day correction;
a high-conviction tilt on a noisy short-term feature carries cost for
the full holding window. This is a useful form of implicit
regularisation — the policy is pushed toward persistent,
regime-stable allocations rather than short-horizon noise tilts. The
expected behavioural effect: lower turnover (already low under the
prior; should fall further), smoother weight trajectories, and likely
better generalisation to live trading where noise-tilting is a
well-known overfitting failure mode.

### A.5 Why env-time, not backtrader-only

The cheaper alternative — apply the cadence only in stage 4 at
inference, leaving training unchanged — creates a **training-inference
distribution mismatch**. The agent was trained assuming its action
would be applied every day; at deployment its action is applied weekly.
The policy gradient never saw the holding constraint; its choices are
calibrated for fresh-action-every-day reward dynamics, not for
"survive-being-held" dynamics.

In practice this mismatch manifests as a divergence between the
"claimed" Sharpe (env metric from stage 3, daily-rebalance) and the
deployed Sharpe (stage 4 with weekly cadence). Useful as a *diagnostic*
to detect whether cadence matters; not the canonical production
solution. The env-time approach is the principled answer because it
puts the constraint inside the gradient flow.

### A.6 Interactions

**A.6.1 Structural prior.** The prior is computed and applied before
the parent class's softmax regardless of cadence. The agent's effective
action is whatever was stored at the last rebalance bar; the env still
adds `alpha * log(w_prior)` to that stored action every step before
softmax, so weights drift slightly with the prior even between
rebalance days as the covariance changes.

Whether this is desirable is a design choice. The alternative is to
also freeze the prior between rebalance bars (use the prior from the
last rebalance day), in which case weights remain exactly constant
until drift adjustments from price moves. The recommended default is
the first behaviour: the prior is a *soft anchor*, and on non-rebalance
bars the anchor stays active even if the action does not change.

**A.6.2 Risk-off cash gate.** Two sensible policies:

- **Gate wins (recommended).** Turbulence above threshold flips weights
  to 100 % cash on any bar regardless of cadence. After a gate-cash
  bar, the next rebalance day's action determines the next allocation.
  This preserves the gate's hard-safety semantics, which the prior
  sweep has confirmed as the single largest source of edge over
  EqualWeight.
- **Cadence wins.** The gate only evaluates on rebalance bars. Cleaner
  in principle but materially weakens the gate's protective value
  during the four trading days between rebalances. Not recommended for
  deployment; worth a single ablation run to measure the cost of the
  cleaner semantics.

**A.6.3 Per-ticker no-trade band (backtrader).** The band is applied as
today on rebalance days only (since no rebalance order is sent on
non-rebalance days, the band has nothing to evaluate). On a rebalance
day the band may still skip individual tickers whose proposed weight
change falls below the threshold, even when the cadence has authorised
a rebalance. The band and the cadence are orthogonal mechanisms.

### A.7 Testing protocol (mirrors section 5)

1. **Code-correctness sanity.** Verify the `is_rebalance_day(date,
   cadence)` helper returns the expected boolean for known fixtures
   (a Friday is a weekly rebalance day; January 1st is a monthly
   rebalance day; a holiday-adjacent day falls back to the next trading
   day). Verify the env stores and replays
   `self._last_rebalance_action` correctly across multiple bars.
2. **Smoke training.** Train 5k timesteps under each cadence and
   confirm that on non-rebalance days the env's `actions_memory[-1]`
   matches `actions_memory[-2]` (the action was held), while on
   rebalance days it differs (the action was refreshed).
3. **Marginal contribution** (the headline experiment). Train two
   models to convergence at the same *effective* sample count: one
   daily, one weekly, each with appropriately scaled `total_timesteps`.
   Compare env Sharpe, bt Sharpe, turnover, and `n_trades`. Expected:
   weekly should have ~5× lower turnover and modestly different
   Sharpe; the direction is the empirical question.
4. **Cadence comparison.** Extend to monthly. Plot all three on the
   same equity chart.
5. **Gate × cadence ablation.** For the best cadence, run gate-wins
   vs cadence-wins. The expected effect is small if turbulence is rare
   during the test window and large if it overlaps with crisis periods.

### A.8 Why this is documented but not implemented

The implementation cost is non-trivial: env modification (one new gate
+ one new instance variable), config block (~10 lines), date-handling
utilities (trading-calendar interaction), retraining at scaled
`total_timesteps` (5-21× longer per run), and a new sweep grid (cadence
+ ablations). The decision to invest in it should be informed by the
cheaper inference-only diagnostic (Approach A, also not implemented),
which is roughly an afternoon of engineering.

The honest recommended sequence is:

1. Implement and run Approach A first (backtrader-only cadence). Cheap
   diagnostic: does cadence matter for net-of-cost results on this
   universe?
2. If yes (deployed Sharpe materially improves under weekly / monthly),
   invest in Approach B per this appendix.
3. If no, do not invest in B. The cadence is not the binding
   constraint and the daily-rebalance behaviour is acceptable.

Approach B becomes the canonical implementation only if the diagnostic
in step 1 confirms it is worth the engineering and retraining cost.
