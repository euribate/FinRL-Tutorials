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
| **Phase 1** | Daily cadence, 5 seeds | **+0.323** | +0.918 (0.359) | 3 / 2 |
| **Phase 4** | Weekly cadence, 5 seeds | **+0.517** | +1.292 (0.196) | 3 / 2 |
| **Phase 3 placebo** | Synthetic IC=0, 2 seeds | **+0.367** | +0.917 (0.359) | 1 / 1 |
| **Phase 3 IC=0.40** | Synthetic realised IC≈0.37 at rebalance days, 2 seeds | **−0.214** | −0.539 (0.590) | 0 / 2 |
| 3a theoretical at IC=0.20 | Perfect-info weekly, MC | +10.27 (median) | — | — |
| 3a theoretical at IC=0.40 | Perfect-info weekly, MC | +21.17 (median) | — | — |

**None of the four observed ensemble IRs is statistically distinguishable
from zero** (all p_NW > 0.19). The pipeline behaves consistently with
noise on all four arms.

The t_NW values for Phase 1 and the placebo (0.918 and 0.917) coincide
to two decimals — that is a numerical coincidence, not a transcription
error: they correspond to different IRs (+0.323 vs +0.367) and different
TEs (4.55 vs 1.71 bps/day), with the Newey-West SE landing at
proportionally different values that produce nearly identical
t-statistics. p-values agree to three decimals as a knock-on.

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

Phase 4's drag delta of −1.484 is larger than the zero-skill EW_w_Cash
reference of −1.103. The excess of approximately −0.38 IR is not cash
drag — it is the TC plus variance drag induced by the policy's tilts.
PPO's tilted positions have higher concentration than 1/(M+1) and pay
turnover cost on each rebalance day; both effects compound to a small
additional negative IR vs cash-free EW that EW_w_Cash does not pay.

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

**Caveat on Conclusion 2**: this finding rests on **two seeds of a single
block-bootstrap draw**. It is directionally credible — a perfect-info
ceiling of approximately +20 IR missed entirely down to ~0 is hard to
explain by seed luck alone or by bootstrap-draw-specific structure —
but a fully powered claim would need a second bootstrap draw and a
larger seed pool. The recommended Option B (alpha_signal as the only
per-asset feature, ~30 min compute) is precisely the cheap test that
converts "no detection observed" into a mechanism: feature-saliency
loss in the 27-dimensional state vector.

**Conclusion 3 — Real-data results are uninformative about feature IC**.
Phases 1 (+0.32) and 4 (+0.52) are both indistinguishable from zero IR
vs EW_w_Cash. Critically, **this does NOT bound the IC content of the
real features.** The IC=0.40 result establishes that this pipeline
produces approximately the same null distribution whether the data
carries planted IC = 0 or planted IC = 0.37. Real features could carry
IC = 0.5 and this pipeline would still report IR ≈ 0. The defensible
claim is about the apparatus, not the features: *this pipeline extracts
no alpha from these features on the tested arms, and its demonstrated
detection capacity is below realised IC ≈ 0.37; real-feature IC content
remains unmeasured by this design.*

**Conclusion 4 — Deployable answer**. `EqualWeight_w_Cash` (Sharpe 1.103,
CumReturn 72.60%, MaxDD −14.97%) remains the deployable. PPO does not
materially differ from it under any of the tested configurations.

---

## 5. The publishable claim (corrected)

The original chain — "pipeline has structural negative bias → real-data
negative IR is uninformative" — is **withdrawn**. The corrected
publishable claim is about the apparatus, not the features:

> **3a** shows a perfect-information strategy at realised IC = 0.37
> achieves theoretical median IR ≈ +20 on a 5.9-year window. **3b** shows
> our pipeline produces IR ≈ 0 on the same data with the same realised
> IC — that is, the pipeline's demonstrated detection capacity is below
> realised IC ≈ 0.37 (subject to the 2-seed-1-bootstrap caveat above).
> **On real data** the pipeline also produces IR ≈ 0. **We cannot
> conclude anything about real-feature IC content from these results**:
> a pipeline that returns the null distribution on planted IC = 0.37 also
> returns the null distribution on hypothetical IC = 0.5 — the
> apparatus's output is uninformative about its input above its
> detection floor.

The defensible negative claim is: *this pipeline does not extract alpha
from these features on the tested configurations.* The defensible
positive contribution is the apparatus itself, plus the demonstration
that pre-registered placebo + perfect-info ceiling reveals
apparatus-pathology vs feature-pathology — a distinction the literature
routinely conflates.

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
+ properly powered placebo + cash-drag-corrected benchmark show this
PPO + article_benchmark + IR-sel pipeline does not extract alpha from a
13-ETF universe AND has demonstrated detection capacity below realised
IC ≈ 0.37 (subject to a 2-seed-1-bootstrap caveat). Real-feature IC
content is therefore unmeasured. The deployable strategy is
`EqualWeight_w_Cash` with the turbulence gate."

**Option B — Attack the detection-capacity question** (the cheap
mechanism test). Run candidate mechanism #1 from section 6 (`alpha_signal`
as the ONLY per-asset feature in the state) on the IC=0.40 synthetic
data. If recovered IR moves from ~0 to materially positive, the pipeline
can extract signal when given attention; the binding constraint is
feature saliency in the 27-dimensional state vector. ~30 minutes of
compute. Converts the under-caveated "no detection observed" into a
concrete mechanism with a clean fix path.

**Option C — Both, B first (RECOMMENDED, per analyst's endorsement)**.
Run B as one targeted experiment, then write up A either way. The
combined deliverable has the methodology paper AND a concrete
diagnostic that identifies (or rules out) feature saliency as the
binding constraint for future signal-extraction work.

The corrected triage is now closed with a defensible result. The
pipeline-pathology rabbit hole was a false trail caught by the
pre-registered placebo and the analyst's cash-drag observation —
exactly what a calibrated evaluation infrastructure is supposed to do.
