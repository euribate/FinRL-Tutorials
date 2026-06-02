# Structural Priors — residual policy on a rules-based base

This folder is a fork of `Article_priority_1`. Everything that's *not* about
the structural-prior change is identical: same data pipeline, same env (minus
the surgical addition described below), same per-asset encoder, same reward
modes, same backtrader replay, same QuantStats report.

## The single architectural change

Today's allocation (`Article_priority_1`):

```
weights = softmax(actions)
```

This folder:

```
weights = softmax(actions + alpha * log(w_prior))
```

`w_prior` is a closed-form allocation (equal-weight, inverse-vol, or
risk-parity) computed each bar from the trailing-252-day covariance already
present in the state (no look-ahead). The policy network's effective job
collapses from *"discover the entire allocation function from observation"*
to *"find small tilts away from a sensible base"*. At `actions = 0` the
resulting weights are exactly `w_prior` — the policy starts deployment from
a known-sensible point and has to *earn* every deviation.

## Why this should help (and the academic basis)

The Article_priority_1 sweep concluded with a clear ceiling near
`EqualWeight` and a turnover that never broke 0.85x/yr — the agent kept
collapsing toward uniform because allocation alpha on liquid ETFs at daily
frequency is weak relative to noise. Classical practitioner approaches
(risk parity: Maillard, Roncalli & Teiletche 2010; Black-Litterman: Black
& Litterman 1992; the AQR / Bridgewater allocator family more broadly)
solve this by injecting strong, theory-derived priors and only learning
small corrections. PPO with a per-asset encoder makes no such assumption
out of the box; this folder bolts a closed-form prior into the env so the
RL policy learns the *residual*.

References for the change:

- Maillard, S., Roncalli, T., & Teiletche, J. (2010). *The Properties of
  Equally Weighted Risk Contribution Portfolios.* Journal of Portfolio
  Management.
- Black, F., & Litterman, R. (1992). *Global Portfolio Optimization.*
  Financial Analysts Journal.
- Garleanu, N., & Pedersen, L. H. (2013). *Dynamic Trading with Predictable
  Returns and Transaction Costs.* Journal of Finance — formal "trade toward
  an analytically-derived target" framework.

Note: this is **not** what Kashif & Slepaczuk's paper (the article this
project derives from) does. They run pure end-to-end DRL with no prior. The
prior is a deliberate departure from the article.

## Files changed vs Article_priority_1

| File | Change |
|---|---|
| `utils.py` | New module-level `_solve_risk_parity` helper. `LogReturnPortfolioEnv.__init__` gains `policy_prior_enabled / type / alpha / cash_share` parameters. New methods `_compute_prior_weights` (closed-form per-step prior) and `_apply_policy_prior` (log-space residual). `step()` calls `_apply_policy_prior(actions)` BEFORE the parent's softmax_normalization. `make_portfolio_env` reads the new `policy_prior` config block and passes it through. |
| `config.json` | New top-level `policy_prior` block: `enabled`, `type`, `alpha`, `cash_share`. Defaults are `enabled=true`, `type="inverse_vol"`, `alpha=1.0` — so the empty-override "baseline" experiment IS the residual policy. |
| `experiments.json` | Replaced the legacy architecture sweep with a focused 6 + 4 = 10-experiment grid testing the prior's marginal contribution (on/off), type (eqw/invvol/rp), and alpha. |
| `STRUCTURAL_PRIORS.md` | This file. |

Everything else — stages 1-5, `predict_tomorrow.py`, `run_experiments.py`,
`inspect_policy.py`, `policies.py`, `features.py` — is unchanged. The
residual lives entirely inside the env.

## The four prior types

- **`none`** — disables the prior; the env behaves identically to
  `Article_priority_1`. Use this as the control.
- **`equal_weight`** — `w_prior = 1/N` across every weight (risky + cash).
  Trivial. Establishes the floor: does *any* prior help?
- **`inverse_vol`** (DEFAULT) — `w_i \propto 1/sqrt(diag(Sigma)_i)` across
  the risky names; cash held out at `cash_share` (default `1/N`).
  Closed-form, no solver. The "diagonal risk-parity" approximation.
- **`risk_parity`** — iterative fixed-point solver on the risky-only
  sub-covariance such that each risky asset contributes equally to
  portfolio risk. The full version of `inverse_vol`. Cash still held out
  at `cash_share`.

## How `alpha` works

`alpha` is the blending strength applied to `log(w_prior)`:

| alpha | Effect |
|---|---|
| `0.0` | Prior disabled (the agent runs free, identical to `Article_priority_1`). |
| `0.5` | Prior anchors the agent's weights "softly"; the agent can move significantly. |
| `1.0` (DEFAULT) | At `actions = 0` the resulting weights are EXACTLY `w_prior`. Canonical residual form. |
| `2.0` | Tilts get amplified; agent has more leverage to deviate. Use sparingly — can become unstable. |

## How to run

### Step 0 — sanity-check the prior math (sub-second, no model needed)

Before burning training time, run the three-test harness that verifies
the prior helpers are working correctly. See `METHODOLOGY.md` section
5.1 for what each test does.

```bash
python test_priors.py
```

Exits with code `0` on all-pass, `1` on any failure (suitable as a CI
gate). If a test fails, **do not proceed** — the prior math is broken
and any model you train will be miscalibrated. The script reads your
`config.json`, so it auto-adapts to whatever `policy_prior.type` /
`cash_share` you have configured.

### Stages 1 - 5 — the standard pipeline

The data layer is unchanged from `Article_priority_1`, so:

```bash
# 1. Generate the dataset for this folder's universe.
python 01_get_data.py

# 2. Train the residual-policy PPO (baseline = inverse_vol, alpha=1.0).
python 02_train.py

# 3. Env backtest with FOUR baselines: MinVariance, EqualWeight,
#    EqualWeight_w_Cash, AND the static Prior_{type} portfolio.
python 03_backtest.py

# 4. Realistic-execution check with the static prior overlay.
python 04_backtrader_replay.py

# 5. HTML report - emits three variants: vs EqualWeight, vs
#    EqualWeight_w_Cash, and vs Prior_{type}.
python 05_quantstats_report.py
```

Or to sweep the focused experiment grid:

```bash
python run_experiments.py --experiments experiments.json --with-backtrader
```

## What to look for in the results

A successful structural prior should show, vs `Article_priority_1`'s
winner (`enc_hidden128_emb_dim1`, env Sharpe 0.774 / bt Sharpe 0.782 /
0.85x turnover):

1. **`prior_off` matches Article_priority_1's number** on this universe.
   Sanity check that nothing else changed.
2. **`baseline` (inverse_vol + alpha=1.0) beats `prior_off`** on
   `env_sharpe_minus_eqw`. If yes, the prior is providing real lift.
3. **Per-asset weight variation in `inspect_policy.py` shows non-uniform
   means** that align with the prior (e.g. bonds/TLT and gold/GLD
   overweighted vs equity under `inverse_vol`). The agent is now tilting
   around a non-uniform anchor instead of collapsing toward uniform.
4. **Turnover is materially lower** — the agent has less reason to churn
   when its action floor is already sensible.
5. **The alpha grid shows a clean curve** rather than chaos: lower alpha
   should converge toward `prior_off`'s behaviour; higher alpha toward
   "the prior IS the policy."

If `baseline` does NOT beat `prior_off`, the conclusion is honest: priors
don't help here either, and the ceiling holds. That is itself a clean
research result.

## Caveats and known interactions

- **The risk-off cash gate (`risk_off.enabled=true`) still overrides
  everything on high-turbulence bars.** The prior is a soft anchor that
  shapes normal-day behaviour; the gate is hard safety on stress days. They
  do not interfere with each other.
- **CASH is held out of the covariance-based priors.** Its synthetic zero
  variance would otherwise collapse `inverse_vol` (1/sigma -> infinity) and
  destabilise `risk_parity` (singular sub-covariance). Cash gets `cash_share`
  (default `1/N`) and the risky weights are renormalised to sum to
  `1 - cash_share`.
- **The trained model and the data folder must be regenerated for this
  folder.** `data/` and `models/` were excluded from the clone.
- **The cash-inclusive EqualWeight benchmark** (added in `Article_priority_1`
  and gated on `benchmark.show_cash_inclusive`) is inherited unchanged and
  works the same way in stages 3/4/5.

## Honest expectation

This is a single-knob architectural change with strong theoretical backing
and minimal engineering cost. Realistic outcomes:

- Best case: Sharpe improvement of 0.02-0.05 over `Article_priority_1`'s
  winner, with materially lower turnover (because the agent's deviations
  from a sensible base are smaller than its softmax exploration from
  uniform). Turns the deployable PPO into a competitive `risk parity +
  RL tilts` system.
- Modal case: Sharpe roughly matches `Article_priority_1`; turnover drops.
  Result: same return profile but cheaper to execute.
- Worst case: the prior dominates and the agent learns nothing useful on
  top of it. In this case `baseline` and `prior_equal_weight` will both
  approximate a static `inverse_vol` / equal-weight portfolio's Sharpe.
  That is a deployable answer too — just a non-RL one.

None of these outcomes are bad. The experiment is cheap, attributes cleanly,
and either way it's the last *architectural* lever worth pulling on this
problem.
