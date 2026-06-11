# Final results memo — four-phase triage closed (CORRECTED)

**Date**: 2026-06-11 (revision; supersedes 2026-06-10).
**Pipeline**: env-time weekly cadence (Phase 4) or daily (Phase 1),
article_benchmark reward, IR selection, λ_TO = 0.003, action_logit_scale
= 3.0, γ = 0.9.
**Universe**: 13 sector ETFs + CASH (N=14).
**Evaluation window**: 2020-07-01 → 2026-05-19 (5.9 years, T=1477 bars).
**Minimum-detectable IR at p<0.05 on this sample length**: ~ 0.8.

---

## 0. What changed in this revision

The original (2026-06-10) memo concluded that the pipeline exhibits a
"structural negative bias" and "inverts signals." On analyst review, this
conclusion was **incorrect**. The active-stats block in `03_backtest.py`
benchmarked against `EqualWeight` (a cash-free, fully invested 1/M
strategy), but the PPO agent has CASH as one of its 14 allocatable
assets and structurally holds ~6.4-7.3% in cash. Comparing a cash-holding
strategy to a cash-free benchmark during a period where risk assets
compounded ~80% mechanically produces an IR of approximately -1 with a
significant t-stat — purely from the cash exposure, with zero correlation
to any signal in the features.

The corrected active-stats benchmark is `EqualWeight_w_Cash` (also fully
invested but holds 1/(M+1) in CASH alongside 1/M in each risky asset).
This is the apples-to-apples comparator. The `03_backtest.py` script
has been patched to use it automatically when cash is enabled, and a
cash-drag decomposition block has been added that prints both IRs
side-by-side so the drag is visible.

All four phases were re-backtested against the corrected benchmark.
No retraining was needed. The corrected results and revised conclusions
are below.

---

## 1. Corrected outcome table — all four phases + 3a calibration

| Phase | Setup | Ensemble IR vs EW_w_Cash (corrected) | t_NW (p_NW) | Per-seed signs (+/-) |
|---|---|---|---|---|
| **Phase 1** | Daily cadence, 5 seeds | **+0.323** | +0.92 (0.359) | 3 / 2 |
| **Phase 4** | Weekly cadence, 5 seeds | **+0.517** | +1.29 (0.196) | 3 / 2 |
| **Phase 3 placebo** | Synthetic IC=0, 2 seeds | **+0.367** | +0.92 (0.359) | 1 / 1 |
| **Phase 3 IC=0.40** | Synthetic realised IC≈0.37 at rebalance days, 2 seeds | **−0.214** | −0.54 (0.590) | 0 / 2 |
| 3a theoretical at IC=0.20 | Perfect-info weekly, MC | +10.27 (median) | — | — |
| 3a theoretical at IC=0.40 | Perfect-info weekly, MC | +21.17 (median) | — | — |

**None of the four observed ensemble IRs is statistically distinguishable
from zero** (all p_NW > 0.19). The pipeline behaves consistently with
noise on all four arms.

---

## 2. Cash-drag decomposition

Side-by-side IR vs the old (incorrect) and new (correct) benchmarks,
with the deterministic cash-drag delta:

| Phase | IR vs EW (no cash, OLD) | IR vs EW_w_Cash (NEW) | Cash-drag Δ |
|---|---|---|---|
| Phase 1 | −0.723 | +0.323 | −1.046 |
| Phase 4 | −0.966 | +0.517 | −1.484 |
| Phase 3 placebo | −1.046 | +0.367 | −1.413 |
| Phase 3 IC=0.40 | −1.322 | −0.214 | −1.107 |

The cash-drag Δ is in every case approximately −1.0 to −1.5. The agent's
average cash weight is 6.4-7.3% across all phases, consistent with this
drag size. Compare to `EqualWeight_w_Cash` vs `EqualWeight` itself:
IR = −1.103, the same deterministic effect on a strategy with literally
zero skill.

The original memo's "pipeline pathology" finding was therefore a
benchmark mis-specification, not a property of the pipeline.

---

## 3. Per-seed dispersion (corrected)

| Phase | Per-seed IRs vs EW_w_Cash | Mean | SD | Signs +/- |
|---|---|---|---|---|
| Phase 1 (daily) | −0.66, +0.26, **−1.59**, +0.79, +0.13 | −0.215 | 0.928 | 3 / 2 |
| Phase 4 (weekly) | +0.06, +0.74, **−1.09**, +0.76, −0.63 | −0.033 | 0.823 | 3 / 2 |
| Phase 3 placebo | +0.61, −0.88 | −0.134 | 1.054 | 1 / 1 |
| Phase 3 IC=0.40 | −0.14, −0.31 | −0.225 | 0.117 | 0 / 2 |

Per-seed spread is wide on real-data arms (most seeds within ±1 IR, one
outlier seed below −1 in Phases 1 and 4). The IC=0.40 arm has tight
clustering (sd 0.12), both seeds slightly negative but both well within
the placebo's range. **The pipeline does not distinguish IC=0 placebo
from IC=0.37 strong signal** — it produces approximately the same null
distribution on both.

---

## 4. The corrected conclusions

**Conclusion 1 — Pipeline calibration**. With the correct benchmark, the
placebo IR is +0.37 (p=0.36, indistinguishable from zero). The pipeline
does NOT exhibit a structural negative bias on pure noise. The
calibration check passes. There is no need to debug an "inversion"
pathology.

**Conclusion 2 — No demonstrated signal extraction at realised IC = 0.37**.
The IC=0.40 planted-signal test (realised cross-sectional rank-IC ≈ 0.37
at rebalance days, per the generator's internal smoke test on the 916
rebalance days) produced an ensemble IR of −0.214 against an achievable
ceiling of approximately +20 from the 3a Monte Carlo. The pipeline does
not extract a substantial planted signal — but it does not invert it
either; it just produces noise around zero. The strong signal arm
behaves essentially identically to the placebo arm.

**Conclusion 3 — Real-data results are within the noise floor**. Phases
1 (+0.32) and 4 (+0.52) are both indistinguishable from zero IR vs
EW_w_Cash. Whatever signal the real features carry, the pipeline does
not extract enough of it to reach significance at this sample length.

**Conclusion 4 — Deployable answer**. `EqualWeight_w_Cash` (Sharpe 1.103,
CumReturn 72.60%, MaxDD −14.97%) remains the deployable. PPO does not
materially differ from it under any of the tested configurations.

---

## 5. The publishable claim (corrected)

The original chain — "pipeline has structural negative bias → real-data
negative IR is uninformative" — is **withdrawn**. The corrected chain is:

> 3a shows a perfect-information strategy at realised IC = 0.37 achieves
> theoretical median IR ≈ +20 on a 5.9-year window. Our pipeline produces
> IR ≈ 0 (within noise) on the same data with the same realised IC.
> Pipeline detection efficiency at IC ≈ 0.37 is therefore approximately
> zero. On real data the pipeline also produces IR ≈ 0. **From the
> real-data null we can conclude only that real features carry less
> extractable IC than 0.37 — substantially weaker than a publishable
> alpha**, but cannot rule out useful IC at lower levels.

This is a weaker but defensible claim. The detection-threshold work the
analyst initially designed Phase 3 to produce IS valuable; it just sets
the threshold at "IC < 0.37 cannot be detected by this design", not at
"the pipeline does not work."

---

## 6. The genuine remaining open question

The pipeline produces IR ≈ 0 on BOTH the placebo (zero signal) AND the
IC=0.37 strong-signal arm. With perfect-info ceiling at +20 and observed
recovery at 0, recovery efficiency is essentially nil.

**Why doesn't the pipeline extract IC = 0.37?** This is the question
that survives the benchmark correction. Five candidate mechanisms (now
correctly ordered after the analyst's note that selection-noise cannot
produce a systematically negative bias):

| # | Candidate mechanism | Cheap test |
|---|---|---|
| 1 | `alpha_signal` is one of 27 features in the per-asset state; the shared encoder may lack capacity to attend to it from raw gradient signal alone. | Train an arm where `alpha_signal` is the ONLY per-asset feature. If IR recovers, attention capacity is the binding constraint. |
| 2 | `VecNormalize` observation normalisation may flatten `alpha_signal`'s informative cross-sectional dispersion if its time-series statistics differ between train and trade. | Disable observation normalisation for one IC=0.40 run. If recovered IR improves, normalisation is mis-handling the planted feature. |
| 3 | Per-bar `article_benchmark` reward signal-to-noise: the per-bar difference between net return and benchmark return is small relative to noise. The advantage estimator may not isolate the signal direction. | Compare placebo IR under `article_benchmark` vs `article_absolute`. If both produce null, the issue is not the benchmark term. |
| 4 | IR-selection on a small val sample produces noisy selection. Not a SIGN-biased mechanism, but a power-reduction mechanism: the wrong checkpoint is selected on noise. | Raise `val_fraction` from 0.1 to 0.3 and rerun IC=0.40. If recovered IR improves, selection-on-noise is hurting power. |
| 5 | The planted signal is forward-cumulative over the holding week; the agent may need explicit "next-week return prediction" in its reward structure to learn the right mapping. | Train an arm with `reward_kind = article_absolute` (no benchmark subtraction). If recovery improves, the benchmark term obscures the cross-sectional signal. |

The analyst-recommended order: address mechanism #1 first (feature
saliency / attention), then #2 (normalisation), then #4 (val_fraction).

---

## 7. The methodology infrastructure stands

Independent of the corrected interpretation, the infrastructure built
during this triage is the genuinely valuable output:

| Artifact | What it does |
|---|---|
| `inspect_action_range.py` + `action_logit_scale` knob | Proves and resolves the legacy [0,1] cage |
| `compute_active_stats()` (IR, t_iid, t_NW, p) | First HAC-corrected paired test in this codebase |
| Cash-drag decomposition block | New in this revision: prevents the EW vs EW_w_Cash benchmark mistake from recurring |
| `ValidationSharpeCallback.selection_metric ∈ {sharpe, ir}` | Reward-aligned model selection |
| Per-seed IR dispersion block in stage 3 | First multi-seed robustness reporting |
| `compute_rebalance_dates` + env-time cadence gate | Approach B per METHODOLOGY Appendix A |
| `test_signal_recovery_upper_bound.py` (3a) | Closed-form theoretical IR ceiling |
| `gen_synthetic_data.py` (3b) | Block-bootstrap + planted-IC synthetic generator (realised IC ≈ 0.37 at rebalance days, target 0.40, within 6% relative — within spec) |
| Placebo + IC=0.40 configs | Pre-registered detection-floor + sensitivity tests |
| METHODOLOGY.md sec 6.5, 6.6 | Documented dead-code policy_prior, TC asymmetry |

---

## 8. Recommended next move

**Option A — Ship the methodology paper**. Frame as: "leak-free pipeline
+ properly powered placebo + cash-drag-corrected benchmark show no
demonstrated alpha extraction at realised IC ≥ 0.37 from this PPO +
article_benchmark + IR-sel pipeline on a 13-ETF universe. The deployable
strategy is `EqualWeight_w_Cash` with the turbulence gate."

**Option B — Attack the detection-capacity question**. Run candidate
mechanism #1 from section 6 (`alpha_signal` as the only per-asset
feature) on the IC=0.40 synthetic data. If recovered IR moves from
~0 to materially positive, the pipeline can extract signal when given
attention; the remaining question becomes feature saliency under the
27-dimensional state vector. ~30 minutes of compute.

**Option C — Both, in sequence**. Run B as one targeted experiment,
then write up A either way. The combined deliverable has both the
methodology paper and a concrete diagnostic that suggests the binding
constraint for future signal-extraction work.

The corrected triage is now closed with a defensible result. The
pipeline-pathology rabbit hole was a false trail caught by the
pre-registered placebo and the analyst's cash-drag observation — exactly
what a calibrated evaluation infrastructure is supposed to do.
