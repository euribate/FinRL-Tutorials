"""One-shot generator for the session summary Word document.

Produces FINAL_RESULTS_SUMMARY.docx in the same folder.
"""
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH


def add_heading(doc: Document, text: str, level: int) -> None:
    h = doc.add_heading(text, level=level)
    h.style.font.name = "Calibri"


def add_para(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text)
    p.style.font.name = "Calibri"
    p.style.font.size = Pt(11)


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Light Grid Accent 1"
    hdr_cells = tbl.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        for para in hdr_cells[i].paragraphs:
            for run in para.runs:
                run.bold = True
    for r_idx, row in enumerate(rows, start=1):
        for c_idx, cell_text in enumerate(row):
            tbl.rows[r_idx].cells[c_idx].text = str(cell_text)
    doc.add_paragraph()  # spacing


def add_monospace(doc: Document, text: str) -> None:
    """Add a block of text in a monospace font (for command outputs)."""
    for line in text.split("\n"):
        p = doc.add_paragraph()
        run = p.add_run(line)
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    doc.add_paragraph()


# ─────────────────────────────────────────────────────────────────────────
# Build the document
# ─────────────────────────────────────────────────────────────────────────

doc = Document()
for sec in doc.sections:
    sec.left_margin   = Cm(2.0)
    sec.right_margin  = Cm(2.0)
    sec.top_margin    = Cm(2.0)
    sec.bottom_margin = Cm(2.0)

# Title
title = doc.add_heading("PPO Portfolio Allocation Triage — Session Summary", level=0)
sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.LEFT
sub_run = sub.add_run(
    "Universe: 13 sector ETFs plus a synthetic CASH asset (N=14). "
    "Evaluation window: 2020-07-01 to 2026-05-19 (5.9 years, T=1477 daily bars). "
    "Generated 2026-06-10."
)
sub_run.italic = True


# 1. Executive summary
add_heading(doc, "1. Executive Summary", 1)
add_para(doc,
    "This session ran a four-phase triage to determine whether a PPO-based "
    "active portfolio allocator can outperform a daily-rebalanced equal-weight "
    "(EW) baseline on a 13-ETF plus CASH universe. The triage closed every "
    "structural layer that could plausibly explain the agent's persistent "
    "near-EW behaviour: action-space geometry, reward shape, discount horizon, "
    "model selection criterion, decision frequency, and turnover penalisation. "
    "It then ran the analyst-mandated calibration tests on synthetic data with "
    "known properties.")
add_para(doc,
    "The headline finding is not what was expected. On synthetic data with "
    "PLANTED information coefficient (IC) of zero — that is, pure noise with "
    "no extractable cross-sectional signal — the pipeline produces an "
    "Information Ratio (IR) of negative 1.05 with a Newey-West p-value of "
    "0.007. The pipeline therefore exhibits a structural negative bias on "
    "data that mathematically cannot carry signal. Both real-data results "
    "(daily Phase 1 IR = negative 0.72; weekly Phase 4 IR = negative 0.97) "
    "sit within this negative-bias band. The earlier interpretation that "
    "'PPO actively hurts vs EW' is therefore not supported: the real-data "
    "negative IR is consistent with anywhere from no signal in the features "
    "to substantial positive signal that the pipeline corrupts.")
add_para(doc,
    "An additional test with a strong planted signal (target IC 0.40, "
    "realised on the trade slice 0.20) produced IR negative 1.32. The "
    "theoretical achievable IR at realised IC 0.20 from a perfect-information "
    "weekly strategy is plus 10.27 (Monte Carlo over 2000 paths). The "
    "pipeline's recovery efficiency is therefore negative 12.9 percent at "
    "the strong-signal level: not just losing signal, but inverting its sign.")
add_para(doc,
    "The deployable answer is EqualWeight_w_Cash (annualised Sharpe 1.103, "
    "cumulative return 72.60 percent, maximum drawdown negative 14.97 "
    "percent), supported now by methodology rather than by an 'active "
    "management is anti-edge' claim that the placebo cannot support.")


# 2. Scripts modified / added
add_heading(doc, "2. Scripts Modified or Added", 1)
add_para(doc,
    "Every code change in this session is captured in eight commits to the "
    "structural_priors folder. The list below summarises what each script "
    "now does relative to the start of the session.")

add_table(doc,
    headers=["Script", "Status", "What was changed"],
    rows=[
        ["utils.py", "modified",
         "Added action_logit_scale parameter on LogReturnPortfolioEnv (widens the "
         "softmax action box from [0,1] to [-s,+s]). Added compute_rebalance_dates "
         "helper for env-time weekly cadence. Added cadence and weekly_day kwargs "
         "on the env: step() now gates the agent's action before super().step() so "
         "the action only takes effect on rebalance days (typically Fridays, with "
         "holiday fallback). ValidationSharpeCallback gained selection_metric in "
         "{sharpe, ir} so model selection can be aligned with the article_benchmark "
         "reward. make_portfolio_env reads all new config blocks."],

        ["02_train.py", "modified",
         "Now passes the early_stopping.selection_metric value through to the "
         "callback so IR-based selection can be turned on per-config."],

        ["03_backtest.py", "modified",
         "Added compute_active_stats(): per-strategy daily active return mean (bps), "
         "tracking error (bps), annualised IR, iid paired t and two-sided p, and "
         "Newey-West HAC-adjusted t and p with lag floor(N^(1/4)). The stage 3 "
         "output now prints an 'Active-return statistics vs EqualWeight' block and, "
         "when multiple seeds are trained, a 'Per-seed IR distribution' block with "
         "min, max, sd, and +/- sign count across seeds."],

        ["inspect_action_range.py", "added",
         "Diagnostic that reads a weights_<algo>.csv and reports max/min weight, "
         "implied logit spread, L1 distance from equal weight, and turnover "
         "percentiles. Verdicts on whether the legacy box cap was binding. Used "
         "throughout the triage as the action-geometry instrument."],

        ["features.py", "modified",
         "Registered alpha_signal as a per_asset pass-through feature. It expects "
         "the column to be present in the dataframe (planted by gen_synthetic_data.py) "
         "and raises KeyError on real data so it cannot accidentally leak into "
         "production runs."],

        ["test_priors.py", "modified",
         "Added Test 4: an empirical proof that softmax(a + alpha*log(w_eq)) equals "
         "softmax(a) for any a and any alpha (the equal-weight prior is "
         "softmax-shift-invariant and therefore a mathematical no-op). All 4 tests "
         "pass."],

        ["test_cadence.py", "added",
         "Three unit tests for env-time cadence: 'daily' as identity, 'weekly' "
         "selecting Fridays with holiday-fallback to Thursday, and a real env "
         "stepping that confirms 98 percent of non-rebalance bars correctly "
         "replay the held action."],

        ["test_signal_recovery_upper_bound.py", "added",
         "Monte Carlo (2000 paths) of the theoretical IR achievable by a "
         "perfect-information weekly long-only strategy at planted IC in "
         "{0, 0.05, 0.10, 0.20, 0.40} on a 5.9-year window. Provides the "
         "denominator for pipeline efficiency."],

        ["gen_synthetic_data.py", "added",
         "Phase 3b generator. Loads the real OHLCV panel, block-bootstraps it on "
         "the date index (block length 20 days) to preserve cross-sectional "
         "correlation and volatility clustering, plants an alpha_signal column "
         "whose cross-sectional rank-correlation with the next-week cumulative "
         "return is set to a target IC, recomputes all derived features through "
         "the same pipeline as 01_get_data.py, and writes synthetic_*.pkl. Prints "
         "the realised IC for sanity validation."],

        ["METHODOLOGY.md", "modified",
         "Added section 6.5 (policy_prior is currently REMOVED from env-time "
         "injection; equal_weight is a mathematical no-op; alpha is the prior "
         "temperature, NOT a tilt-leverage knob) and section 6.6 (TC asymmetry: "
         "agent leg is net-of-TC while benchmark leg is gross, documented as a "
         "design choice)."],

        ["config.json + 5 scale configs", "modified",
         "Added action_logit_scale env knob (default 3.0 on the main config). "
         "Added early_stopping.selection_metric (default 'sharpe'; set to 'ir' on "
         "the three article_benchmark configs). Flipped policy_prior.enabled to "
         "false as the shipped default (avoids the equal_weight no-op footgun). "
         "Rewrote policy_prior docstring to reflect the actual current "
         "behaviour."],

        ["New configs", "added",
         "config_scale1.json, config_scale3.json, config_scale3_bench.json, "
         "config_scale3_bench_g09.json, config_scale3_bench_g09_lto003.json "
         "(triage sweep), config_phase1_5seeds.json (multi-seed daily robustness), "
         "config_phase4_weekly.json (weekly cadence combined arm), "
         "config_phase3_placebo_IC0.json, config_phase3_synth_IC040.json "
         "(synthetic data tests)."],

        ["Article_priority_1/utils.py", "modified",
         "Ported action_logit_scale support to the parallel folder so the "
         "geometry fix is consistent across implementations."],

        ["Article_priority_1/04_backtrader_replay.py", "modified",
         "Added an UNMAINTAINED banner pointing to structural_priors/ as the "
         "maintained version (the alignment-fix port was not done here)."],
    ])


# 3. Pre-registered experiments
add_heading(doc, "3. Pre-Registered Experiments", 1)
add_para(doc,
    "The triage proceeded in four pre-registered phases. Each phase had a "
    "concrete hypothesis, a concrete prediction, and a concrete decision rule "
    "for what to do next. Phases were run sequentially; the decision after "
    "each phase determined the next.")

add_table(doc,
    headers=["Phase", "Hypothesis tested", "Decision rule"],
    rows=[
        ["1 — Daily multi-seed rider",
         "Multi-seed averaging of the daily IR-selected arm produces 5 IRs "
         "straddling zero (analyst's pre-registration). Establishes the robust "
         "EqualWeight_w_Cash sign-off baseline.",
         "If 5 IRs straddle zero with ensemble p > 0.05, sign off on EW_w_Cash."],

        ["2 — Implement env-time weekly cadence",
         "Existence-only; the implementation work itself (not a hypothesis). "
         "Verifies the gate mechanism with three unit tests.",
         "Pass unit tests, proceed to Phase 4."],

        ["3a — Theoretical IR ceiling",
         "What is the maximum IR a perfect-information weekly strategy can "
         "achieve at planted IC = {0, 0.05, 0.10, 0.20, 0.40} on a 5.9-year "
         "window?",
         "Calibration only. Provides the denominator for Phase 3b efficiency."],

        ["3b — Planted-signal recovery (placebo + IC 0.40)",
         "Analyst's adaptive plan: placebo IR ~ 0 (calibrates the pipeline); "
         "IC = 0.40 IR > 0 with material magnitude (confirms recovery capacity). "
         "If both pass, bisect to find detection threshold.",
         "If placebo IR > 0 with significance, pipeline reports false positives. "
         "If IC = 0.40 fails to recover, pipeline is broken; debug, do not "
         "grid-search intermediate ICs."],

        ["4 — Combined arm (weekly cadence, all knobs stacked)",
         "Pre-registered success criteria: ensemble IR > 0, TE per |IR| below "
         "the daily arm's ratio, Sharpe at least 1.05. Not significance "
         "(unreachable on 5.9 years per the MDE arithmetic).",
         "If criteria met, revisit deployment decision. Otherwise, EW_w_Cash "
         "stays the deployable."],
    ])


# 4. Headline results
add_heading(doc, "4. Headline Results", 1)
add_para(doc,
    "All phases ran to completion. The following table consolidates the "
    "ensemble outcomes against the analyst's pre-registrations.")

add_table(doc,
    headers=["Phase", "Setup", "Ensemble IR vs EW", "t_NW (p_NW)",
             "Per-seed IR signs (+/-)"],
    rows=[
        ["Phase 1",
         "Daily cadence, 5 seeds",
         "-0.723",
         "-1.89 (0.059)",
         "1 / 4"],

        ["Phase 4",
         "Weekly cadence, 5 seeds",
         "-0.966",
         "-2.44 (0.015)",
         "1 / 4"],

        ["Phase 3 placebo",
         "Synthetic IC = 0, 2 seeds",
         "-1.046",
         "-2.68 (0.007)",
         "0 / 2"],

        ["Phase 3 IC = 0.40",
         "Synthetic realised IC = 0.20, 2 seeds",
         "-1.322",
         "-3.29 (0.001)",
         "0 / 2"],

        ["3a theoretical at IC 0.20",
         "Perfect-info weekly, Monte Carlo",
         "+10.27 (median)",
         "Not applicable",
         "Not applicable"],

        ["3a theoretical at IC 0.40",
         "Perfect-info weekly, Monte Carlo",
         "+21.17 (median)",
         "Not applicable",
         "Not applicable"],
    ])

add_para(doc,
    "Key per-seed numbers for the multi-seed arms are below. The Phase 1 "
    "and Phase 4 arms each show a wide spread across seeds, with most seeds "
    "individually negative at p_NW below 0.05, and a single positive outlier "
    "in each. The Phase 3 arms show tightly clustered negative values around "
    "negative 1, consistent with a structural pipeline effect rather than "
    "seed-specific overfit.")

add_table(doc,
    headers=["Phase", "Per-seed IRs", "Mean", "SD"],
    rows=[
        ["Phase 1 (daily)",
         "-1.00, -1.16, -1.92, +0.41, -0.44",
         "-0.822",
         "0.868"],
        ["Phase 4 (weekly)",
         "-0.86, -0.85, -1.65, +0.28, -1.23",
         "-0.862",
         "0.717"],
        ["Phase 3 placebo",
         "-0.68, -1.25",
         "-0.965",
         "0.402"],
        ["Phase 3 IC = 0.40",
         "-1.30, -1.18",
         "-1.241",
         "0.081"],
    ])


# 5. Raw outputs
add_heading(doc, "5. Raw Output Excerpts", 1)
add_para(doc,
    "The following are excerpts of the actual stage-3 output for each phase, "
    "shown verbatim from the run logs.")

add_para(doc, "Phase 1 (daily 5-seed):")
add_monospace(doc,
"============================================================\n"
"Strategy          CumReturn     Sharpe      MaxDD\n"
"------------------------------------------------------------\n"
"PPO                  74.64%      1.084    -15.21%\n"
"MinVariance          16.61%      0.613    -12.77%\n"
"EqualWeight          79.67%      1.103    -16.05%\n"
"EqualWeight_w_Cash       72.60%      1.103    -14.97%\n"
"============================================================\n"
"\n"
"PPO active vs EqualWeight (ensemble):\n"
"  Active_bps = -0.20  TE_bps = 4.46  IR = -0.723  t_NW = -1.89  p_NW = 0.059"
)

add_para(doc, "Phase 4 (weekly 5-seed):")
add_monospace(doc,
"============================================================\n"
"Strategy          CumReturn     Sharpe      MaxDD\n"
"------------------------------------------------------------\n"
"PPO                  74.77%      1.093    -15.45%\n"
"MinVariance          16.61%      0.613    -12.77%\n"
"EqualWeight          79.67%      1.103    -16.05%\n"
"EqualWeight_w_Cash       72.60%      1.103    -14.97%\n"
"============================================================\n"
"\n"
"PPO active vs EqualWeight (ensemble):\n"
"  Active_bps = -0.20  TE_bps = 3.30  IR = -0.966  t_NW = -2.44  p_NW = 0.015"
)

add_para(doc, "Phase 3 placebo (synthetic IC = 0):")
add_monospace(doc,
"============================================================\n"
"Strategy          CumReturn     Sharpe      MaxDD\n"
"------------------------------------------------------------\n"
"PPO                  92.15%      1.198    -13.39%\n"
"MinVariance          16.35%      0.669    -11.81%\n"
"EqualWeight         100.39%      1.193    -14.16%\n"
"EqualWeight_w_Cash       91.08%      1.193    -13.20%\n"
"============================================================\n"
"\n"
"PPO active vs synthetic EqualWeight (ensemble):\n"
"  Active_bps = -0.31  TE_bps = 4.74  IR = -1.046  t_NW = -2.68  p_NW = 0.007"
)

add_para(doc, "Phase 3 IC = 0.40 (realised IC on trade slice = 0.20):")
add_monospace(doc,
"============================================================\n"
"Strategy          CumReturn     Sharpe      MaxDD\n"
"------------------------------------------------------------\n"
"PPO                  89.86%      1.165    -14.27%\n"
"MinVariance          16.35%      0.669    -11.81%\n"
"EqualWeight         100.39%      1.193    -14.16%\n"
"EqualWeight_w_Cash       91.08%      1.193    -13.20%\n"
"============================================================\n"
"\n"
"PPO active vs synthetic EqualWeight (ensemble):\n"
"  Active_bps = -0.39  TE_bps = 4.68  IR = -1.322  t_NW = -3.29  p_NW = 0.001"
)

add_para(doc, "Phase 3a theoretical IR ceiling (excerpt):")
add_monospace(doc,
"    IC   median IR         5%        95%   P(IR>0)   P(|t|>1.96)\n"
"--------------------------------------------------------------------\n"
"  0.00      -0.019     -0.684      0.695     47.8%          5.5%\n"
"  0.05       2.544      1.848      3.247    100.0%        100.0%\n"
"  0.10       5.119      4.430      5.792    100.0%        100.0%\n"
"  0.20      10.270      9.555     11.027    100.0%        100.0%\n"
"  0.40      21.173     20.270     22.116    100.0%        100.0%"
)


# 6. Diagnostic comparisons
add_heading(doc, "6. Action-Geometry Diagnostic Across Phases", 1)
add_para(doc,
    "The inspect_action_range.py diagnostic was run on every saved weights "
    "file. The headline columns track how aggressively the policy tilts "
    "vs equal weight: implied logit spread at the 99th percentile (1.0 was "
    "the legacy box cap), L1 distance from equal weight, and the fraction "
    "of days where weights sit within 0.05 L1 of equal weight.")

add_table(doc,
    headers=["Phase / configuration", "Spread p99", "L1 p99",
             "% within L1 < 0.05 of EW"],
    rows=[
        ["Historical (pre-fix, alpha_iv arm)",         "1.000", "n/a",   "n/a"],
        ["Triage 2A (scale=1, log_return)",            "0.323", "0.083", "86.6%"],
        ["Triage 2B (scale=3, log_return)",            "3.345", "0.663", "0.0%"],
        ["Triage 3 (scale=3, article_bench, sharpe-sel)","0.211", "0.053", "97.2%"],
        ["Triage 4 (scale=3, art_bench, g=0.9, sharpe-sel)","0.280","0.075", "63.2%"],
        ["Test 3 IR-sel (single seed)",                "3.193", "0.792", "0.0%"],
        ["Test 4 IR-sel (single seed)",                "3.070", "0.739", "0.0%"],
        ["Phase 1 (daily 5-seed)",                     "0.676", "0.156", "2.6%"],
        ["Phase 4 (weekly 5-seed)",                    "0.531", "0.110", "3.0%"],
        ["Phase 3 placebo (synthetic IC=0, weekly)",   "0.727", "n/a",   "31.9%"],
        ["Phase 3 IC=0.40 (weekly)",                   "0.719", "n/a",   "6.2%"],
    ])

add_para(doc,
    "Two observations from this table. First, the policy DOES tilt under "
    "article_benchmark plus IR-selection — Triage rows for the single-seed "
    "IR-selected arms show large active tilts (L1 around 0.7) compared to "
    "the sharpe-selected arms (L1 around 0.05). The IR selection switch "
    "worked as intended at the action-geometry level. Second, the weekly "
    "cadence regularised the policy compared to the daily arm: L1 dropped "
    "from 0.156 to 0.110, and the near-EW share rose slightly. The cadence "
    "is doing what METHODOLOGY Appendix A.4.3 predicted (implicit "
    "robustness pressure) — but it does not move the IR.")


# 7. The pipeline-pathology finding
add_heading(doc, "7. The Pipeline-Pathology Finding", 1)
add_para(doc,
    "The Phase 3 placebo test was pre-registered by the analyst as "
    "non-negotiable: if the pipeline reports a non-zero IR on data "
    "synthesised with planted IC equal to zero, no other result is "
    "interpretable until the pipeline is calibrated. The placebo result "
    "(ensemble IR = negative 1.046, p_NW = 0.007) shows the pipeline does "
    "not pass that test.")
add_para(doc,
    "The placebo IR is statistically distinguishable from zero. The "
    "ensemble mean is negative; the two individual seeds (-0.68 and -1.25) "
    "are both negative. The action-geometry diagnostic confirms the "
    "policy IS tilting (logit spread p99 = 0.73, near-EW share = 31.9 "
    "percent), so this is not a degenerate-policy artifact — the "
    "negative IR is produced by an actively-tilting policy whose tilts "
    "happen to systematically lose value vs the synthetic EW benchmark.")
add_para(doc,
    "The IC = 0.40 follow-up confirms the diagnosis. The realised "
    "cross-sectional rank-IC on the trade slice (after block bootstrap "
    "dilution) was +0.20, against which the 3a Monte Carlo says a "
    "perfect-information strategy would achieve median IR around +10. "
    "The pipeline's recovered IR was negative 1.32. Pipeline efficiency "
    "at the realised IC is therefore negative 12.9 percent — the "
    "pipeline does not just lose signal, it inverts it.")

add_para(doc,
    "The candidate mechanisms for this negative bias, in approximate cost-of-"
    "debug order, are listed below. These are documented hypotheses; none "
    "has been ruled in or out by this session.")

add_table(doc,
    headers=["#", "Candidate mechanism", "Cheap test"],
    rows=[
        ["1",
         "IR-selection overfits on a small validation sample. With "
         "val_fraction = 0.1 and 17 years of training data, the val slice is "
         "roughly 1.7 years (~430 daily bars). The IR sampling-error stdev "
         "on that slice is around 1/sqrt(1.7) ~ 0.77, comparable in magnitude "
         "to the signal it selects on; selected checkpoints likely "
         "anti-correlate OOS.",
         "Raise val_fraction to 0.3; rerun the placebo. If placebo IR moves "
         "toward zero, root cause is selection-on-noise."],

        ["2",
         "Long-only softmax weights give up the daily-rebalancing arithmetic-"
         "geometric premium. A concentrated portfolio of correlated, "
         "above-average-vol assets has a structural geometric-mean disadvantage "
         "vs daily-rebalanced 1/N — present even on random tilts.",
         "Compare IR of a constant random-tilt portfolio held weekly vs daily "
         "EW on the placebo data. Quantify the rebalancing-premium gap."],

        ["3",
         "Feature scaling or encoder asymmetry: alpha_signal is one of 27 "
         "features in the state. The per-asset shared encoder must learn to "
         "attend to it from raw gradient signal alone, which may exceed the "
         "encoder's capacity on a 5-year training window.",
         "Train an arm where alpha_signal is the ONLY per-asset feature in "
         "the state. If IR recovers materially, attention capacity is the "
         "binding constraint."],

        ["4",
         "VecNormalize observation normalisation may flatten alpha_signal's "
         "informative cross-sectional dispersion if its time-series stats "
         "drift between train and trade.",
         "Disable observation normalisation for one synthetic-IC run; if "
         "recovered IR improves, normalisation is mis-handling the planted "
         "feature."],

        ["5",
         "article_benchmark per-step reward signal-to-noise ratio: the per-bar "
         "difference between net return and benchmark return is small in "
         "absolute terms relative to the natural daily-return noise. The "
         "advantage estimator may not isolate the signal direction even when "
         "the policy is theoretically able to see it.",
         "Compare placebo IR under article_benchmark vs article_absolute. "
         "If the absolute reward also produces negative-bias on noise, the "
         "issue is not the benchmark term."],
    ])


# 8. Conclusions
add_heading(doc, "8. Conclusions and Deployable Decision", 1)
add_para(doc,
    "Three substantive conclusions from this session, in decreasing order "
    "of confidence.")

add_para(doc,
    "First, the deployable answer is EqualWeight_w_Cash with the turbulence "
    "gate as a tail-risk overlay. Annualised Sharpe 1.103, cumulative return "
    "72.60 percent, maximum drawdown negative 14.97 percent. PPO does not "
    "beat that on this pipeline, and any active-management claim would now "
    "require evidence that the pipeline can detect signals at the IC levels "
    "the features plausibly carry — evidence the Phase 3 tests do not "
    "provide.")

add_para(doc,
    "Second, the negative IR observed on real data (Phase 1 and Phase 4) "
    "should NOT be reported as 'active management leaks value on these "
    "features'. The placebo test establishes that the pipeline produces "
    "approximately the same negative IR on data that mathematically cannot "
    "carry any signal. The real-data result is consistent with anywhere "
    "from zero to substantial positive IC in the features; the pipeline "
    "is the binding constraint, not the universe.")

add_para(doc,
    "Third, the genuinely valuable artifact produced in this session is "
    "the methodology infrastructure rather than any alpha claim. The "
    "leak-free walk-forward harness, drift-aware transaction costs, "
    "VecNormalize stats snapshot, IR-aligned model selection, HAC-corrected "
    "paired inference, per-seed dispersion reporting, env-time weekly "
    "cadence gate, theoretical-IR Monte Carlo, and block-bootstrap "
    "planted-signal generator — together with the pre-registered placebo "
    "test that uncovered the pipeline pathology — constitute a "
    "reproducible apparatus for evaluating deep-RL portfolio allocators. "
    "Few published comparisons in this literature run their pipelines "
    "against a calibrated placebo.")


# 9. Recommended next move
add_heading(doc, "9. Recommended Next Move", 1)
add_para(doc,
    "Three options, ordered by analyst-flagged information value.")

add_table(doc,
    headers=["Option", "Effort", "Decision it informs"],
    rows=[
        ["A. Ship the methodology paper.",
         "Writeup only",
         "Frames the deliverable as 'leak-free pipeline + placebo discovers "
         "pipeline-pathology' rather than as a finance result. The four phases "
         "plus the 3a/3b calibration form the backbone."],

        ["B. Debug the pipeline pathology.",
         "30 to 60 minutes per candidate",
         "Five candidate mechanisms in section 7. Cheapest single test: "
         "raise val_fraction from 0.1 to 0.3, rerun the placebo, see whether "
         "the negative IR moves toward zero. If it does, IR-selection overfit "
         "on small val sample is the root cause."],

        ["C. Both, in sequence (recommended).",
         "Option B first, then Option A",
         "Run the val_fraction debug as a final calibration test. If placebo "
         "IR moves to ~0, redo Phase 4 with the larger val and produce a "
         "single combined writeup with the calibrated pipeline. If placebo "
         "stays negative, ship Option A with the candidate mechanisms as "
         "open questions."],
    ])

add_para(doc,
    "The four-phase triage as designed and executed is complete. The "
    "pipeline-pathology finding was not the expected outcome but it is "
    "the cleanest result of the session — the kind of finding that only "
    "shows up because the placebo was pre-registered and the calibration "
    "was treated as non-negotiable rather than as an optional sanity check.")


# Save
out = "FINAL_RESULTS_SUMMARY.docx"
doc.save(out)
print(f"Wrote {out}")
