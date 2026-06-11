# structural_priors_improved

Clean-slate restructure of `../structural_priors`, applying the analyst's
roadmap from the FINAL_RESULTS_MEMO triage. Three load-bearing changes:

1. **The environment is vendored.** No FinRL dependency at runtime; the
   `[0,1]` action cage and the earn-date/decision-date convention bugs
   from the legacy code are eliminated by construction in `env.py`.
2. **One reward mode, one cost knob, one cadence switch.** The legacy
   five reward modes, two turnover-accounting paths, and ad-hoc
   policy_prior block are collapsed into `cost_bps` + `cadence` + a
   benchmark-relative log-return reward.
3. **The calibration harness survives verbatim.** `compute_active_stats`
   (NW HAC + paired t), the cash-drag decomposition, the synthetic
   block-bootstrap generator, and the perfect-information Monte Carlo
   ceiling are ported untouched. They were the crown jewel of the
   triage; they remain the crown jewel here.

## Layout

```
structural_priors_improved/
├── README.md
├── requirements.txt        # pinned: sb3, gymnasium, torch, numpy, pandas, scipy
├── config.json             # ONE config; experiment variants override at call site
├── data.py                 # download + feature pipeline (consolidates 01_get_data + utils helpers)
├── features.py             # feature registry (@register decorators)
├── env.py                  # NEW: vendored PortfolioEnv (~250 LOC)
├── policy.py               # PerAssetSharedEncoder (kept; sound)
├── callbacks.py            # ValidationIRCallback (IR-selection only)
├── train.py                # PPO loop (single-split, multi-seed)
├── backtest.py             # ensemble + active stats + cash-drag block
├── synthetic.py            # block-bootstrap + planted-signal generator (verbatim port)
├── ceiling.py              # Monte Carlo theoretical IR upper bound (verbatim)
├── diagnostics.py          # inspect_action_range (updated for symmetric box)
├── tests/
│   ├── test_env.py         # 6 tests: cadence, softmax invariance,
│   │                       #   decision-date convention, cost math,
│   │                       #   env-curve == recompute regression
│   └── test_stats.py       # 3 tests: NW math + iid/AR sanity
├── METHODOLOGY.md          # ported from legacy folder
├── FINAL_RESULTS_MEMO.md   # ported, institutional memory
└── FINAL_RESULTS_SUMMARY.docx
```

## Step-0 acceptance gate (passes)

```
$ python tests/test_env.py
TEST 1 — daily rebalance dates = identity         PASS
TEST 2 — weekly picks Friday + Thursday fallback  PASS
TEST 3 — softmax shift-invariance                 PASS
TEST 4 — decision-date convention                 PASS  ← bug killed by construction
TEST 5 — cost math (bps round-trip on L1)         PASS
TEST 6 — env-curve == recompute regression        PASS  ← max diff 0.000e+00

$ python tests/test_stats.py
TEST 1 — identical curves -> NaN guard            PASS
TEST 2 — strong +5 bps edge -> IR > 0             PASS
TEST 3 — AR(0.3) -> |t_NW| < |t_iid|              PASS
```

The regression test (Test 6, `env-curve == returns_from_weights`) at
machine precision proves that what the env books during training and
what `backtest.py` reconstructs from the saved weights are byte-identical.
That's the structural correctness proof — the placebo IR ≈ 0 calibration
runs as Step-0's empirical follow-up.

## What's gone vs. the legacy folder

Deleted: `04_backtrader_replay.py`, `05_quantstats_report.py`,
`run_experiments.py`, `replay_cadence_sweep.py`, `predict_tomorrow.py`,
`filter_seeds.py`, all `inspect_*` scripts except `diagnostics.py`,
`utils_old.py`, `_gen_summary_docx.py`, all ~12 experiment configs,
all model/results/tensorboard directories from prior runs, and every
legacy README/USAGE doc except the two memos that constitute the
project's institutional memory.

Total LOC: ~1,200 (was ~9,400). Half of those LOC are verbatim ports
of validated helpers — `compute_active_stats`, the synthetic generator,
the Monte Carlo ceiling — and the other half is fresh code in the
vendored env and slim training/backtest loops.

## Migration order (per analyst's brief)

* **Step 0** (here): port the harness, run placebo + planted-signal
  through the new env with the OLD feature treatment. Required:
  placebo IR ≈ 0 and the single-seed regression test passing.
  **Status: structural tests passing. Empirical placebo run is the
  first experiment to run on this skeleton.**
* **Step 1**: rank normalisation + feature diet (6–8 features), tested
  on the planted-signal arm. This is where `data.py` and `features.py`
  get their first substantive refactor.
* **Step 2**: per-decision weekly reward refinement (METHODOLOGY A.4.2
  stricter implementation: reward computed only at rebalance bars on
  compounded multi-bar return).
* **Step 3**: only if needed, action mapping and exploration tweaks.

One change per step. Planted-signal arm as the judge. Placebo as the
guard.
