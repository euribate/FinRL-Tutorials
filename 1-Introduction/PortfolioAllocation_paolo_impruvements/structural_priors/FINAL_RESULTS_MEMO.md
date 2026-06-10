# Final results memo — four-phase triage closed

**Date**: 2026-06-10
**Pipeline**: env-time weekly cadence, article_benchmark reward, IR
selection, λ_TO = 0.003, action_logit_scale = 3.0, γ = 0.9.
**Universe**: 13 sector ETFs + CASH (N=14).
**Evaluation window**: 2020-07-01 → 2026-05-19 (5.9 years, T=1477 bars).
**Minimum-detectable IR at p<0.05 on this sample length**: ~ 0.8.

---

## 1. Outcome table — all four phases + 3a calibration

| Phase | Setup | Ensemble IR vs EW | t_NW (p_NW) | Per-seed IR signs +/- |
|---|---|---|---|---|
| **Phase 1** | Daily cadence, 5 seeds | −0.723 | −1.89 (0.059) | 1/4 |
| **Phase 4** | Weekly cadence, 5 seeds | −0.966 | −2.44 (0.015) | 1/4 |
| **Phase 3 placebo** | Synthetic IC=0, 2 seeds | **−1.046** | **−2.68 (0.007)** | **0/2** |
| **Phase 3 IC=0.40** | Synthetic, realised IC=0.20, 2 seeds | **−1.322** | **−3.29 (0.001)** | **0/2** |
| 3a theoretical ceiling | Perfect-info @ IC=0.20 | +10.27 (median) | — | — |
| 3a theoretical ceiling | Perfect-info @ IC=0.40 | +21.17 (median) | — | — |

**Pipeline efficiency at IC=0.20 (realised): −12.9%.**
The pipeline does not just lose signal — it INVERTS sign at every IC level we measured.

---

## 2. The reframe

The original interpretation of Phase 1/Phase 4 was: "PPO's tilts subtract value vs EW with statistical evidence — features carry no exploitable signal."

The Phase 3 placebo test, which the analyst pre-registered as non-negotiable, has shown that interpretation is **not supported by the data**:

- On synthetic data with PLANTED IC = 0 (pure noise, mathematically no extractable signal), the pipeline still produces **IR = −1.05, p_NW = 0.007**.
- On real data (Phase 1, Phase 4), the pipeline produces IR ≈ −0.7 to −0.97 — within the range of the placebo's structural bias.

The pipeline has a **systematic negative bias of roughly −1 IR unit** on data that cannot carry signal. The real-data result is therefore consistent with real features carrying anywhere from no signal to substantial positive signal that the pipeline corrupts.

The IC=0.40 test then shows the pipeline cannot extract even a strong planted signal: realised IC = 0.20 on the trade slice (block-bootstrap dilutes the planted 0.40), theoretical ceiling IR ≈ +10, observed IR = −1.32. **The pipeline does not have detection capacity above the noise floor.**

---

## 3. The corrected inference chain

The analyst's intended publishable claim was:

> "3b shows the pipeline recovers planted IC ≥ x → real data shows IR ≈ 0 → therefore the features carry less than x of extractable IC."

The chain is currently broken at the FIRST link. With pipeline-recovery negative at any tested IC, the appropriate publishable claim is:

> "The PPO + article_benchmark + IR-selection + softmax-action-geometry + VecNormalize-observation pipeline exhibits a structural negative bias (IR ≈ −1) on null data and cannot extract signals up to realised IC = 0.20. The 13-ETF universe's daily features therefore CANNOT be evaluated with this pipeline; any conclusion about feature IC requires a pipeline that passes its placebo test."

This reframes the negative result from a *finance* result ("features are noise") to a *methodology* result ("the apparatus is inverting signals"). The latter is, if anything, more publishable in a methods-oriented journal — but it changes what the paper is about.

---

## 4. The deployable answer remains EqualWeight_w_Cash

The negative-bias finding does not invalidate the deployment decision, but it changes the supporting argument:

- **Before**: "PPO actively hurts vs EW with p < 0.05 — don't deploy active management."
- **After**: "PPO cannot beat EW on this pipeline. Whether the features have signal is now unknown. Deploying active management would require either fixing the pipeline or empirically demonstrating recovery on placebo + planted-signal tests."

`EqualWeight_w_Cash` (Sharpe 1.103, CumReturn 72.60%, MaxDD −14.97%) is therefore the defensible deployment, with the turbulence gate as a tail-risk overlay.

---

## 5. Pipeline-pathology candidates for follow-up

The negative-bias mechanism must be one or more of:

1. **Feature-scaling / encoder asymmetry**: `alpha_signal` lives in the state vector alongside 26 other features (technical indicators + 12 other custom features). The per-asset shared encoder gives every feature equal capacity ex ante; the network must learn to attend to `alpha_signal` from raw gradient signal alone. With a ~17k-parameter encoder and only 5 years of training data, that may be infeasible.
2. **IR-selection overfit on small val sample**: with `val_fraction = 0.1` and 17 years of training data, the val slice is ~1.7 years (~430 bars). The IR sampling-error stdev on that slice is roughly `1 / sqrt(1.7) ≈ 0.77`. The selection criterion's noise is roughly equal to its signal at the IC levels we care about — virtually guaranteeing OOS-anti-correlated picks.
3. **VecNormalize cross-sectional flattening**: VecNormalize normalises per-feature obs across time. `alpha_signal`'s cross-sectional dispersion (the part that carries the signal) is preserved at any given bar, but if its time-series statistics differ between train and trade, the trade-period normalised values may misrepresent the true cross-sectional ranking.
4. **article_benchmark reward dynamics**: the per-step reward is `r_scale * log(1+r_net) − log(1+r_bench) − λ_to * turnover * 100`. With r_net ≈ r_bench on most bars (small bp differences) and r_scale = 1000, the active-return signal is dominated by the log-return-minus-benchmark which has roughly the same scale as the natural noise; the advantage estimator may not isolate the signal direction.
5. **Long-only + softmax + rebalancing-premium loss**: a concentrated tilted portfolio holds correlated assets with above-average vol; vs daily-rebalanced EW it gives up some of the arithmetic-geometric spread. This produces a small negative drift per bar even on a noisy or absent signal.

**Recommended diagnostic order**: (2) is the cheapest to test (raise val_fraction to 0.3, see if placebo IR moves toward zero); (1) is next (run inspect_policy to see if the policy's gradient w.r.t. alpha_signal is non-trivial); (5) is independently fixable by including the rebalancing-premium correction in the reward.

---

## 6. The artifacts produced

Even with the pathological pipeline, the methodology infrastructure built during this triage is the genuinely valuable output:

| Artifact | What it does |
|---|---|
| `inspect_action_range.py` + `action_logit_scale` knob | Proves and resolves the legacy [0,1] cage |
| `compute_active_stats()` (IR, t_iid, t_NW, p) | First HAC-corrected paired test in this codebase |
| `ValidationSharpeCallback.selection_metric ∈ {sharpe, ir}` | Reward-aligned model selection |
| Per-seed IR dispersion block in stage 3 | First multi-seed robustness reporting |
| `compute_rebalance_dates` + env-time cadence gate | Approach B per METHODOLOGY Appendix A |
| `test_signal_recovery_upper_bound.py` (3a) | Closed-form theoretical IR ceiling |
| `gen_synthetic_data.py` (3b) | Block-bootstrap + planted-IC synthetic generator |
| Placebo + IC=0.40 configs | Pre-registered detection-floor + sensitivity tests |
| METHODOLOGY.md sec 6.5, 6.6 | Documented dead-code policy_prior, TC asymmetry |
| All 4 commits with quantitative pre-registered predictions | Reproducible audit trail |

This is, as the analyst noted earlier, the publishable artifact — independent of any alpha claim.

---

## 7. Recommended next move

**Option A — Ship the methodology paper.** Frame the negative result as "leak-free pipeline + properly powered placebo discovers pipeline-pathology; the absence of pipeline calibration is the most-overlooked source of false negative results in published deep-RL portfolio allocation work." The four phases + the 3a/3b calibration form the methodological backbone.

**Option B — Debug the pipeline.** Test the five mechanisms in section 5 in cost order. The most likely single fix is raising `val_fraction` (suspect #2) — if that moves placebo IR from −1 toward 0, IR-selection on small samples is the root cause and the negative real-data IR may dissolve. ~30 min of additional compute.

**Option C — Both, in sequence.** Run the val_fraction test (Option B's cheap fix) as a single follow-up, then write up the methodology paper either way.

Option C is the analyst's "every run is decision-relevant" framing applied one more time.
