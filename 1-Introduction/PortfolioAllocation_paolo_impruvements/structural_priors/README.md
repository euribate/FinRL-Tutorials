# 8.Multi-seed training and ensembled actions (improvement #8 on top of #1–#7)

Variant of the base pipeline (`../0.original/`) that layers improvement #8 on top of the seven prior improvements:
- **#1–#7** carried over from the previous folders.
- **#8** (new here): instead of training one policy per walk-forward window, train **N policies with different random seeds** per window. At inference, **average their post-softmax weights** row-by-row to produce a smoother, lower-variance allocation. Compute portfolio returns from the averaged weights against the eval slice.

### Why it matters

RL training has high seed variance. With one seed, the policy can land in a local optimum that looks great or terrible by luck — a single-seed comparison between two configs is usually noisier than the effect you're trying to measure. Ensembling 3 seeds typically reduces equity-curve std across re-runs by 30–50%, often raises mean Sharpe by 0.1–0.3, and produces smoother day-over-day weight trajectories (different policies disagree → averaging cancels the disagreement).

The cost is linear in the number of seeds: N seeds = N× stage-2 wall-clock. Stage 3 also runs N times per window but each is short. Stages 4 and 5 are unchanged.

See operative notes in `multi_seed_design_notes.md` for the implementation specifics.

### How it works

1. **Stage 2 trains one model per (window, seed) pair.** Outer loop over walk-forward windows, inner loop over `seeds.list`. Each combination gets a fresh model trained from scratch with its own seed, saved as `models/agent_<algo>_w<i>_s<seed>.zip`. Default `seeds = [42, 1337, 7]`.
2. **Stage 3 ensembles per window.** For each window, load all `N` seed models, run each through a fresh env over the eval slice, capture each seed's `df_actions` (post-softmax weights, date × tickers). Average row-by-row → `ensemble_actions`. Re-normalise rows to sum to 1.
3. **Portfolio return is recomputed from averaged weights, NOT averaged across seed returns.** Naively averaging the per-seed `daily_return` series would be wrong — each seed's equity curve drifts independently. Instead, `daily_return_from_weights(ensemble_actions, eval_slice)` computes `return_t = sum((close_t / close_{t-1} - 1) * w_t)` against the actual eval prices.
4. **Stitching across windows** is identical to single-seed walk-forward — concatenate per-window outputs in chronological order, dedupe boundary days.
5. **Stages 4 and 5 require no code changes.** They read `results/weights_<algo>.csv` which now contains the ensemble-averaged weights.

Setting `seeds.list = [42]` (singleton) reproduces single-seed behaviour, modulo the model filename suffix `_s42`. Removing the `seeds` block entirely falls back to `training.seed`.

**Stage 4 (backtrader) is still NOT walk-forward aware.** It loads only the last window's model and replays the full out-of-sample period through it. See appendix A.13.

---

## Folder layout

```
PortfolioAllocation_paolo_impruvements/
    config.json                  walk-forward VALIDATION config (default 7 windows)
    config_production.json       single-split DEPLOYMENT config (candidate pool + ensemble_size)
    backtrader_config.json       backtrader-specific replay settings
    quantstats_config.json       quantstats report settings
    utils.py                     shared helpers (config IO, env_kwargs, covariance,
                                                 daily_return_from_weights, pick_ensemble_seeds)
    01_get_data.py               stage 1: download + indicators + covariance + split
    02_train.py                  stage 2: train each algo flagged use=true (per seed)
    03_backtest.py               stage 3: backtest + Min-Variance + DJIA baselines
    04_backtrader_replay.py      stage 4: replay one agent through backtrader
    05_quantstats_report.py      stage 5: QuantStats HTML tearsheet
    predict_tomorrow.py          DAILY INFERENCE for live trading (HOLD/REBALANCE decision)
    inspect_ensemble.py          diagnostic: per-seed weights side-by-side + agreement
    filter_seeds.py              diagnostic: rank trained seeds by training convergence
    inspect_turbulence.py        diagnostic: turbulence distribution + threshold sensitivity
    inspect_fills.py             diagnostic: backtrader fill count + implied slippage
    inspect_env_turnover.py      diagnostic: env-vs-backtrader turnover comparison
    README.md                    this file
    USAGE_LIVE_TRADING.md        operations manual for the live trading workflow
    multi_seed_design_notes.md   design notes for improvement #8
    data/                        train_data.pkl, trade_data.pkl                  (created)
    models/                      agent_<algo>_[w<i>_]s<seed>.zip                 (created)
    results/                     equity_curves.csv, equity_plot.png,
                                 target_weights_<date>.csv                       (created)
    results_backtrader/          backtrader outputs                              (created)
    results_quantstats/          report.html, metrics.csv                        (created)
    tensorboard/                 TB logs per algo                                (created)
```

### Two configs, two purposes

| Config                    | Purpose                                                                                                  |
|---|---|
| `config.json`             | Walk-forward VALIDATION: 7 train/eval windows producing the OOS Sharpe distribution. Run when you change hyperparameters, the reward shape, or the universe. |
| `config_production.json`  | DEPLOYMENT: single train/val split on the full history, multi-seed candidate pool, no held-out test slice. The trained models go into live trading via `predict_tomorrow.py`. |

The walk-forward pipeline proves the recipe is robust. The production pipeline trains the model you actually trade. See `USAGE_LIVE_TRADING.md` for the daily ops loop.

---

## Pipeline

```
       config.json
            |
            v
+-----------------------+      data/train_data.pkl
| 01_get_data.py        | ---> data/trade_data.pkl
+-----------------------+
            |
            v
+-----------------------+      models/agent_<algo>.zip
| 02_train.py           | ---> tensorboard/<algo>/
+-----------------------+
            |
            v
+-----------------------+      results/equity_curves.csv
| 03_backtest.py        | ---> results/equity_plot.png
+-----------------------+
            |
            v   (uses backtrader_config.json)
+-----------------------+      results_backtrader/equity_backtrader.csv
| 04_backtrader_replay  | ---> results_backtrader/summary.json
+-----------------------+      results_backtrader/transactions.csv (optional)
            |
            v   (uses quantstats_config.json)
+-----------------------+      results_quantstats/report.html
| 05_quantstats_report  | ---> results_quantstats/metrics.csv
+-----------------------+
```

Each stage reads `config.json` and writes its outputs into the paths configured under the `paths` section.

---

## Quick start

```bash
cd 1-Introduction/PortfolioAllocation_paolo_impruvements

# 1. Download data, compute indicators, attach rolling-252d covariance, split.
python 01_get_data.py --config config.json

# 2. Train every algorithm flagged "use": true in config.json (default: A2C only).
python 02_train.py   --config config.json

# 3. Backtest enabled algos + Min-Variance + DJIA baselines.
python 03_backtest.py --config config.json

# 4. (Optional) Replay the trained agent through backtrader for a realistic
#    broker simulation. Requires `pip install backtrader`.
python 04_backtrader_replay.py --config backtrader_config.json

# 5. (Optional) Generate a QuantStats HTML tearsheet from stage 4's equity
#    curve, comparing the strategy against the configured benchmark.
#    Requires `pip install quantstats`.
python 05_quantstats_report.py --config quantstats_config.json
```

### Production deployment (live trading)

The standard pipeline above (stages 1–5 on `config.json`) is the VALIDATION harness. To actually trade with the model, switch to `config_production.json`:

```bash
# Train a candidate pool of seeds on the full history (no held-out test slice)
python 01_get_data.py --config config_production.json
python 02_train.py   --config config_production.json

# Inspect convergence and ensemble health before deploying
python filter_seeds.py
python inspect_ensemble.py --config config_production.json

# Daily, before 4 pm market close:
python predict_tomorrow.py --config config_production.json \
                           --current-weights current_weights.csv
```

The daily script outputs either `HOLD` (no trades) or `REBALANCE` (with the per-ticker trade list) based on the execution filters in `config_production.json`. See `USAGE_LIVE_TRADING.md` for the full operations manual.

---

## `config.json` — what each section controls

### `data`
- `ticker_list` — the 10 ETF universe (IVE, QQQ, TLT, SHY, LQD, HYG, GLD, EEM, VNQ, XLE). HYG is included only because `download_start_date` is `2007-05-01` (after HYG's 2007-04-11 inception); an earlier start makes FinRL silently drop HYG. When `cash.enabled=true` a synthetic `CASH` asset is added on top, so the env actually sees 11 assets.
- `download_start_date` — must be ≥ 1 year before `train_start_date`, otherwise the 252-day covariance loop produces no rows.
- `train_start_date` / `train_end_date` — training slice.
- `trade_start_date` / `trade_end_date` — out-of-sample backtest slice.
- `indicators` — eight stockstats technical indicators added per ticker per day. These are the *base* indicators; `custom_features` below adds more.
- `custom_features` (new in this folder) — extra per-asset and global features computed by `features.py` (see dedicated subsection below).
- `lookback` — covariance window in trading days (default 252).
- `use_turbulence` — `true` in this folder (required by `risk_off` and not otherwise harmful).

### `data.custom_features` (new in this folder)
Feature engineering beyond the stockstats `indicators`, computed in stage 1 by `features.py` and appended to the env state. Two lists of registry KEYS:
- `per_asset` — features with a different value per ticker per date (one env-state row each). Each key may emit several columns: `mom` → `mom_1/5/20/60`, `vol` → `std_5/20`, plus `bb_pctb`, `dist_high_20`, `meanrev_20`, `beta_60`, and the suggested extras `xsec_mom_rank`, `downside_semidev_20`, `drawdown_60`.
- `global` — features broadcast equally across all tickers per date (the paper's market-wide signals): `vix` → `vix/vix_chg5`, `xsec_avg_ret` → `xsec_avg_ret/xsec_avg_ret_vol5`, `breadth`, `mkt_ret` → `mkt_ret_5/20`.
- `params` — optional overrides of builder defaults (e.g. `{"beta_window": 90}`).

To add a feature: write a builder in `features.py`, decorate with `@register("name", "per_asset"|"global")`, then list its key here. Set both lists to `[]` to fall back to the indicators-only state of folder 8. `beta_60` and `mkt_ret` use the benchmark from the `benchmark` block; `vix` needs the `^VIX` download stage 1 performs when a `vix` feature is requested. The combined env state is `cov(N) + (indicators + custom columns)` rows × N columns.

### `env`
Direct passthrough to `StockPortfolioEnv` (or `LogReturnPortfolioEnv` when `reward_mode = "log_return"`):
- `hmax` — max share units per trade (unused once softmax normalisation is applied, but kept for env contract).
- `initial_amount` — starting cash.
- `transaction_cost_pct` — passthrough to the upstream env constructor. The upstream env stores it but never uses it in the active reward/value computation, so it's effectively dead config. Cost shaping is controlled by `transaction_cost_penalty` below; cost realisation in stage 4 is controlled by `backtrader_config.json`. Kept here because the upstream constructor requires it.
- `reward_scaling` — multiplicative scaling of the reward signal. Default in this folder is **1.0**.
- `reward_mode` — six values supported in this folder:
  - `"value"` (upstream `self.reward = new_portfolio_value`) — appendix A.11 baseline.
  - `"log_return"` (`self.reward = log(1 + portfolio_return) * reward_scaling`) — appendix A.11.
  - `"diff_sharpe"` — Moody & Saffell DSR (risk-adjusted). See appendix A.14.
  - `"diff_sortino"` — same construction but using only the downside second moment.
  - `"article_absolute"` (**default in this folder**) — paper's absolute reward: `return_scale·log(1+r_net) − λ_TO·turnover·100 − λ_conc·(HHI − 1/N)·100`. Coefficients in `article_reward` below.
  - `"article_benchmark"` — same but the return term is `return_scale·(log(1+r_net) − log(1+r_bench))`, using the benchmark from the `benchmark` block.
- `transaction_cost_penalty` — per-unit-turnover cost rate deducted from the realised return (`r_net`) for any shaped reward, **regardless of reward mode**. Default `0.001`. This is the "real" cost; the article modes ALSO add a separate `λ_TO` shaping penalty on top (turnover charged twice, by design — see `article_reward`). See appendix A.12.
- `turnover_mode` (new in this folder) — how turnover is measured for BOTH the cost deduction and the `λ_TO` penalty:
  - `"naive"` (**default**) — `|w_target − w_previous_target|`, the article-faithful definition (change in the agent's target weights).
  - `"drift_adjusted"` — `|w_target − w_drift|` where `w_drift` is the previous weights after one bar of price drift; charges the real rebalancing trade and matches a broker (folder-8 behaviour), but deviates from the article. Keep `naive` for paper fidelity (backtrader / stage 4 is the realistic execution check).
- `diff_ratio_eta` — EMA decay for the running A / B / D moments used by `diff_sharpe` / `diff_sortino`. Default **0.00396825 = 1/252** (one trading year). No effect for the other reward modes.

### `env.article_reward` (new in this folder)
Coefficients used ONLY when `reward_mode` is `"article_absolute"` or `"article_benchmark"` (ignored otherwise):
- `return_scale` — multiplier on the log-return term. Default `1000.0`. The return term ≈ ±10/day at ±1% daily return.
- `lambda_to` — turnover penalty coefficient. Penalty = `λ_TO · turnover · 100`. Default `0.003`. The break-even daily edge a trade must beat ≈ `λ_TO·100/return_scale` per unit turnover. RAISE this to make the policy trade-cautious (it's threshold-like: < ~0.5 churns, ≥ ~2 freezes the policy to equal-weight). Tune via `inspect_policy.py`.
- `lambda_conc` — concentration penalty coefficient. Penalty = `λ_conc · (HHI − 1/N) · 100`, where `HHI = Σwᵢ²`. Default `0.1`. Pushes toward equal weight (`0.0` = no pressure, `0.5` = strong). NOTE: with this universe the agent self-selects ~equal-weight even at `λ_conc=0` — the allocation signal is weak.

### `models.<algo>`
One block per algorithm (`a2c`, `ppo`, `ddpg`, `sac`, `td3`):
- `use` — boolean. Stages 2 and 3 iterate only over algorithms where this is true.
- `total_timesteps` — training horizon for the algorithm.
- `model_kwargs` — passed straight to the SB3 constructor. Default values match the notebook.
- `policy_kwargs` — passed to the SB3 policy network. `null` uses SB3 defaults. Activation functions can be specified as strings (e.g., `"Tanh"`, `"ReLU"`); they are resolved to `torch.nn` classes at construction time.

Default: A2C `use=true`, everything else `use=false` — reproduces the notebook's trading run.

### `training`
- `seed` — passed as the top-level `seed` argument to `DRLAgent.get_model()`. Do not also put `seed` inside `model_kwargs` — FinRL would then pass it twice and SB3 would raise `TypeError: got multiple values for keyword argument 'seed'`.

### `baselines`
- `min_variance.weight_bounds` — bounds for `pypfopt.EfficientFrontier.min_volatility`. The notebook uses `[0, 0.1]` (long-only, max 10% per stock).
- `dji_ticker` — Yahoo ticker for the index baseline (default `^DJI`).

### `paths`
- `data_dir`, `model_dir`, `results_dir`, `tensorboard_dir` — relative to this folder unless absolute.

### `walk_forward` (new in this folder)
- `enabled` — boolean. When `true`, stages 2 and 3 use walk-forward windows. When `false`, falls back to the single-split behaviour of `../2.transaction_cost/`.
- `auto.train_years` / `auto.eval_years` / `auto.step_years` — used when `windows` is `null`. Slides a `(train_years, eval_years)` window forward by `step_years` from `data.train_start_date`, capped by `data.trade_end_date`.
- `windows` — optional explicit list of `[train_start, train_end, eval_start, eval_end]` 4-tuples (ISO date strings). When provided (non-null), overrides `auto`.

Per-window training writes `models/agent_<algo>_w<i>.zip` plus a `models/windows.json` manifest. Stage 3 reads the manifest and stitches per-window predictions back into a single continuous equity curve. See appendix **A.13** for the rationale, the no-look-ahead invariant, and the limitation on stages 4/5.

### `early_stopping` (new in this folder)
- `enabled` — boolean. When `true`, training is monitored on a held-out validation slice and the best-so-far checkpoint is saved. When `false`, falls back to the fixed-`total_timesteps` behaviour of `../4.Sharpe Sortino Ratio/`.
- `val_fraction` — fraction of unique trading dates from each training slice to reserve for validation. Default `0.1`. With a 6-year walk-forward window that's ~6 months.
- `eval_freq` — evaluate every N training timesteps. Default `5000`. With PPO `n_steps=2048`, that's ~2–3 rollouts between evaluations.
- `patience` — number of consecutive evaluations without improvement before stopping. Default `5`.
- `min_delta` — Sharpe must improve by at least this amount to count as "improved". Default `0.01`.

For every model that gets trained (one in single-split mode, N in walk-forward mode), a history JSON is written next to the model: `models/agent_<algo>[_w<i>].history.json`. Each row records `timesteps`, `sharpe`, `improved`, `best_so_far`. Inspect it to verify the early-stopping cadence matched expectations and to spot windows that flat-lined immediately versus windows that kept improving. See appendix **A.15** for the full rationale, the validation-leakage discussion, and the SB3 callback wiring.

### `normalization` (new in this folder)
- `enabled` — boolean. When `true`, wraps the training env in `VecNormalize` and saves running stats next to each model. When `false`, falls back to the un-normalised path of `../5.Early stopping on validation Sharpe/`.
- `norm_obs` — normalise observations. Default `true`. For the 945-dim heterogeneous-scale observation in this pipeline, almost always wanted.
- `norm_reward` — normalise rewards by running std of discounted returns. Default `true`. Helps stabilise PPO when reward distribution shifts across walk-forward windows.
- `clip_obs` — after normalisation, observations are clipped to `[-clip_obs, +clip_obs]`. Default `10.0`.
- `clip_reward` — same idea for rewards. Default `10.0`.
- `epsilon` — added to running variance before sqrt to avoid division by zero. Default `1e-8`. Almost never needs tuning.
- `gamma` — discount used inside VecNormalize's return tracker (for `ret_rms`). Default `0.99`. Should match PPO's `gamma` for best results.

Stats are persisted as `models/vecnormalize_<algo>[_w<i>].pkl` (typically <100 KB per file, *not* SB3's default ~200 MB pickled-venv format). Stage 3 reloads them via `load_vecnormalize_stats(...)`, then does a manual rollout that mirrors what `DRLAgent.DRL_prediction` would do — required because the upstream FinRL helper doesn't know about VecNormalize wrappers and would feed un-normalised observations to a policy trained on normalised ones (producing junk weights).

See appendix **A.16** for the full discussion of why per-model stats files matter in walk-forward, why early stopping needs frozen-stats validation envs, and why stages 4–5 require no changes.

### `risk_off` (new in this folder)
- `enabled` — boolean. When `true`, `LogReturnPortfolioEnv.step()` overrides weights to zero on any bar where turbulence > `turbulence_threshold`. When `false`, the gate is a no-op and the env behaves identically to `../6.VecNormalize for observations (+ reward)/`.
- `turbulence_threshold` — Kritzman & Li turbulence value above which the gate fires. Default `70.0`. Typical normal-market values are 30–60; recent crises spike to 100–200+.

Prerequisite: `data.use_turbulence` must be `true` (default in this folder). If it's false, stage 1 won't emit the `turbulence` column and the env's `_risk_off_active()` will silently return `False` for every bar (gate effectively off, with a one-time warning printed when the env is built). When `cash.enabled=true`, the gate routes 100% into the CASH asset instead of zeroing weights.

See appendix **A.17** for the rationale, the train-vs-trade threshold discussion, and why the gate interacts with TC penalty.

### `benchmark` (new in this folder)
Defines the benchmark used by THREE things: the `article_benchmark` reward, the per-asset `beta_60` feature, and the global `mkt_ret_5/20` features. Stage 1 writes the result to a `benchmark_return` column consumed downstream, and stages 3/4/5 use the same series for their comparison line/overlay/tearsheet.
- `type` — `"equal_weight"` (**default**) builds a daily-rebalanced equal-weight portfolio of the universe tickers and uses its daily return; `"ticker"` downloads `ticker` and uses its return.
- `ticker` — Yahoo ticker used when `type="ticker"` (e.g. `^DJI`, `^GSPC`, `SPY`).

This is separate from `baselines.dji_ticker`, which is only a fallback for the stage-3 overlay. With `type="equal_weight"`, stage 3 shows an **EqualWeight** baseline line (replacing DJIA); with `type="ticker"` it shows that ticker.

### `cash` (new in this folder)
Adds an explicit CASH position so the agent allocates across N risky ETFs **plus** cash (N+1 weights summing to 1).
- `enabled` — boolean. When `true`, stage 1 injects a synthetic `CASH` asset.
- `risk_free_rate` — ANNUAL rate the cash earns. Default `0.0` (cash earns nothing, paper-style). The synthetic asset's price compounds at this rate, so its variance/covariance are ~0.
- `ticker` — name of the synthetic asset (default `"CASH"`).

Implemented as a **synthetic asset** (not env surgery): because CASH is just another ticker in the data, the action space, softmax, covariance state, and return accounting all become N+1 automatically. Downstream handling: stage 3's Min-Variance baseline EXCLUDES CASH (its zero variance would otherwise capture 100% of the min-vol weight); stage 4 backtrader does NOT submit a CASH order (the broker holds the un-invested fraction as real cash). **IMPORTANT:** with `cash.enabled=true`, `backtrader_config.json` `execution.cash_buffer` MUST be `0.0`, else cash is double-counted (a guard in stage 4 errors if not). Set `enabled=false` to reproduce the fully-invested N-asset behaviour of folder 8.

---

## Switching `reward_mode`

All four reward shapes live in the same env class (`LogReturnPortfolioEnv`) and are selected by `env.reward_mode` in `config.json`. To switch:

1. Edit `env.reward_mode`.
2. Adjust `env.reward_scaling` if needed (see table).
3. Re-run **stage 2** (the reward shapes the policy — the model files in `models/` need to be retrained).
4. Re-run stage 3 and onwards. Stages 4 and 5 are reward-agnostic — they consume the recorded weights / equity from stage 3.

### Recommended `reward_scaling` per mode

| `reward_mode` | What the reward is | Recommended `reward_scaling` | Why |
|---|---|---:|---|
| `"value"` | `new_portfolio_value` (upstream) | **`1e-4`** | Raw value is O(1e6); aggressive scaling needed to keep gradients sane. |
| `"log_return"` | `log(1 + r_net) * scaling` | **`1.0`** | Log returns are O(1e-2)/day; 1.0 keeps the signal in a healthy range. |
| `"diff_sharpe"` | DSR contribution per step | **`1.0`** | DSR is O(1) once the EMAs warm up. |
| `"diff_sortino"` | DDR contribution per step | **`1.0`** | DDR is O(1) once D has accumulated some downside. |
| `"article_absolute"` / `"article_benchmark"` | scaled log-return − turnover − concentration penalties | **`1.0`** | Magnitude is set by `article_reward.return_scale` (default 1000), not `reward_scaling`; leave `reward_scaling=1.0` and tune `return_scale` / `lambda_to` / `lambda_conc` instead. |

Wrong scaling fails loudly in `"log_return"` mode (a warning is printed if `reward_scaling < 0.1`). For the differential ratios and article modes it fails quietly — the agent just trains slower or diverges. If training looks unstable, reduce `reward_scaling`; if the value loss looks flat, increase it. For the article modes the relevant magnitude knob is `article_reward.return_scale`, not `reward_scaling`.

### Common transitions

| Want to do this | Change in `config.json` | Retrain? |
|---|---|---|
| Switch DSR → Sortino | `"reward_mode": "diff_sortino"` | yes |
| Switch DSR → plain log-return | `"reward_mode": "log_return"` | yes |
| Switch DSR → notebook-original raw value | `"reward_mode": "value"`, `"reward_scaling": 1e-4` | yes |
| Add transaction-cost penalty on top of DSR | `"transaction_cost_penalty": 0.001` (or whatever rate) | yes |
| Speed up the running-Sharpe adaptation | `"diff_ratio_eta": 0.0079`  (= 1/126, half-year) | yes |
| Disable walk-forward to isolate the reward effect | `"walk_forward": {"enabled": false, ...}` | yes |

### What you do NOT need to redo

- **Stage 1** (`01_get_data.py`): data pickles are independent of reward.
- **Stage 4** (`04_backtrader_replay.py`): consumes `results/weights_<algo>.csv` from stage 3, which is regenerated for you.
- **Stage 5** (`05_quantstats_report.py`): consumes stage 4's equity CSV.
- `backtrader_config.json`, `quantstats_config.json`: no reward-related fields.

To keep multiple trained policies for direct comparison, back up `models/` between runs (e.g., `mv models models_diff_sharpe && mkdir models`) — the per-window zips in `models/agent_<algo>_w<i>.zip` get overwritten on every retraining.

---

## `backtrader_config.json` — replay specifics

The replay re-uses everything from `config.json` (tickers, indicators, dates, env params) by pointing at it via `source_config`. It only overrides things specific to the broker simulation:

- `model.algorithm` — which trained agent to replay (defaults to `a2c`).
- `broker.commission`, `broker.slippage` — realistic execution costs.
- `execution.min_weight_delta` — suppresses rebalances when the new target weight is within this absolute distance of the current portfolio weight (prevents fee-burn from near-zero adjustments).
- `analyzers` — which backtrader analyzers to attach (Sharpe, DrawDown, Returns, TradeAnalyzer, Transactions).
- `output` — where to write the equity CSV, transactions CSV, summary JSON, and the plot PNG.

The strategy reconstructs the same observation matrix the env produced during training: covariance block read from the `cov_list` column in the pickled trade DataFrame, indicator rows read live from the data feeds.

---

## `quantstats_config.json` — report specifics

Reads the equity-curve CSV from stage 4 and renders a QuantStats HTML tearsheet comparing the strategy against a benchmark. Inherits the benchmark ticker from the main config via the chain `quantstats_config.json` → `backtrader_config.json` → `config.json`.

- `inputs.equity_csv` — path to a CSV with `date` index and an `equity` column. Defaults to `results_backtrader/equity_backtrader.csv` (stage 4's output).
- `benchmark.use_source` — when `true`, read the benchmark ticker from the main config's `baselines.dji_ticker`.
- `benchmark.ticker` — used when `use_source` is `false` (e.g., `"^GSPC"` for S&P 500, `"SPY"` for the ETF).
- `output.report_dir` / `html_filename` / `metrics_csv` — where the HTML tearsheet and the long-form metrics CSV go. Set `metrics_csv` to `null` to skip the CSV.
- `report.strategy_name` — column label QuantStats uses for the strategy.
- `report.title` — HTML `<title>` and report header.
- `report.risk_free_rate` — annualised, used in Sharpe/Sortino calculations.

The script pre-fetches the benchmark via FinRL's `YahooDownloader` and passes a returns `Series` to QuantStats. This is **important**: if you instead let QuantStats download the benchmark internally (its default behaviour when you pass a ticker string), an intermittent DNS failure to `fc.yahoo.com` will silently produce a report with an empty benchmark series, and you'll see warnings like `No non-zero returns found for win rate calculation` and `Beta is zero, cannot calculate Treynor ratio`. The pre-fetch fails loudly on download problems instead of producing a meaningless comparison.

---

## Reproducing the notebook results

With the default `config.json` (A2C `use=true`, seed 42):

```bash
python 01_get_data.py
python 02_train.py
python 03_backtest.py
```

The headline numbers from the notebook (`end_total_asset ≈ $1.44M`, Sharpe ≈ 2.14 over Jul 2020 → Oct 2021) should be reproducible up to seed effects. Note the notebook does **not** seed PPO/A2C; the scripts do (seed 42), so results will be close but not byte-identical to the notebook output.

To swap to another algorithm: flip `models.a2c.use` to `false` and `models.ppo.use` (or any other) to `true`. Stages 2 and 3 will pick it up automatically.

---

## Notes and gotchas

- **Pickle, not CSV, for the data.** `cov_list` and `return_list` are object-typed columns (numpy arrays and DataFrames). CSV would stringify them irreversibly, so all stage 1 outputs use `pandas.to_pickle`. Stage 2/3/4 use `pandas.read_pickle`. If you ever need a portable CSV of the *numeric* features only, drop those two columns first.

- **Ticker count is determined at runtime.** The notebook lists 30 Dow names; Yahoo may fail to download some (WBA was failing as of the original notebook run). The actual `stock_dim` is whatever survives the download. Everything downstream — `state_space`, `action_space`, the covariance dimension — is computed from `train_df.tic.nunique()`, not hard-coded.

- **The PortfolioAllocation observation is a 2-D matrix.** Shape is `(stock_dim + n_indicators, stock_dim)` — the first `stock_dim` rows are the covariance matrix, the remaining rows are the indicator vectors. SB3's MLP policy flattens this internally. The backtrader replay rebuilds the same 2-D matrix and lets SB3 do the same flatten.

- **The `seed` parameter belongs at `get_model()` level, not inside `model_kwargs`.** See the explanation under `training` above.

- **`hmax` is in the env contract but unused** by `StockPortfolioEnv` because actions are softmaxed into weights. It is kept in the config so the env constructor signature matches the notebook.

- **Backtrader rebalancing model.** The replay calls `order_target_percent` per ticker per bar to converge to the SB3-derived target weights. This matches the conceptual model of the training env (which directly applies weights) but introduces realistic commissions, slippage, and fractional-share rounding into the backtest.

---

## Appendix: design considerations from notebook to scripts

This appendix records the decisions taken to convert the notebook into a reusable script pipeline, and the trade-offs behind each one. None of these are "improvements" to the model — they are structural choices that the move from notebook to scripts forced.

### A.1 Pickle vs CSV for intermediate data

The notebook generates two object-typed columns during preprocessing:

- `cov_list[t]` — the 27×27 sample covariance matrix as of day `t`.
- `return_list[t]` — the 252×27 trailing-returns DataFrame as of day `t`.

Pandas writes these to CSV using `str()`, which produces unparseable text. A round-trip via CSV would either lose the columns or require regex-parsing them back. We chose `pandas.to_pickle` / `pandas.read_pickle` instead. Trade-offs:

- **Pro:** lossless, fast, single line to read/write.
- **Con:** not human-readable, and pickle files are sensitive to NumPy/Pandas version drift. Anyone reproducing this pipeline should regenerate the pickles rather than commit them.

For this reason `data/` is treated as transient output and not committed.

### A.2 `use=true` flag vs single hard-coded algorithm

The notebook trains five algorithms in sequence and then trades only A2C. We surfaced this as a per-algorithm `use` flag so the same script can either reproduce the notebook (default: A2C only) or run a multi-algorithm comparison (set several flags to true and stage 3 will backtest all of them). Cost: an extra layer of indirection in `02_train.py` and `03_backtest.py`. Benefit: the same code runs every experimental configuration.

### A.3 `seed` placement

FinRL's `DRLAgent.get_model()` exposes `seed` as a dedicated parameter rather than expecting it inside `model_kwargs`. The first attempt put it inside `model_kwargs` and SB3 raised `TypeError: got multiple values for keyword argument 'seed'`. The fix is to read `config.training.seed` and pass it as a top-level argument. Documented under "Notes and gotchas" because anyone editing the config is likely to hit it.

### A.4 No `normalization` block

The reference `Stock_NeurIPS2018/improvements_paolo` pipeline supports three observation-normalisation modes (off / indicators / all) via `SelectiveVecNormalize`. The portfolio-allocation notebook does not normalise observations, so the corresponding block and the SelectiveVecNormalize plumbing have been omitted from `config.json` and `utils.py`. If observation normalisation is added later as an improvement, the block can be added back with the same shape.

### A.5 Min-Variance baseline (not Max-Sharpe)

The reference uses `pypfopt.EfficientFrontier.max_sharpe()`. The portfolio-allocation notebook uses `min_volatility()` with `weight_bounds=(0, 0.1)`, rebalanced daily from `return_list[t].cov()`. We mirrored the notebook's choice exactly — different baseline, same library. The choice is parameterised in `config.json` under `baselines.min_variance` so the bounds can be tightened or relaxed without code edits.

### A.6 Env signature differences from `StockTradingEnv`

`StockPortfolioEnv` accepts a smaller `env_kwargs` dict than `StockTradingEnv`:

- Single `transaction_cost_pct` instead of separate `buy_cost_pct` / `sell_cost_pct`.
- No `num_stock_shares` (the portfolio env has no inventory concept; everything is expressed as weights).
- No `turbulence_threshold` / `risk_indicator_col` (no turbulence gating built in).
- `state_space` is just `stock_dim` (the env internally builds the `(stock_dim + n_indicators, stock_dim)` observation).

`utils.build_env_kwargs` reflects this. If both pipelines ever share a common utils module, this function needs an `env_type` switch.

### A.7 Backtrader replay: weight-based vs share-based actions

The reference `04_backtrader_replay.py` for Stock_NeurIPS2018 issues `buy(size=…)` / `sell(size=…)` calls keyed off the SB3 action vector interpreted as share quantities scaled by `hmax`. That maps directly to `StockTradingEnv`'s mechanics.

For portfolio allocation the action vector is logits → softmax → portfolio weights, so the natural broker call is `order_target_percent(data, target=weight)` per ticker per bar. We adapted the strategy accordingly. Consequences:

- No `hmax` scaling at trade time.
- No inventory bookkeeping in the strategy; backtrader's `order_target_percent` handles deltas vs current holdings.
- Added a `min_weight_delta` filter so trivial weight changes don't churn commissions.

### A.8 Covariance reconstruction in the backtrader strategy

`StockPortfolioEnv` constructs the observation matrix every bar by reading the pre-computed `cov_list` column from the DataFrame. The backtrader replay reproduces this exactly by building a `pd.Timestamp -> ndarray` lookup at startup and indexing into it inside `RLPortfolioStrategy.next()`. We considered re-computing covariance on the fly inside `next()` from a rolling return buffer, but that would (a) introduce numerical drift from the values used during training, and (b) duplicate logic already in `01_get_data.py`. The lookup keeps the replay byte-aligned with what the model saw during training.

### A.9 Ticker order

Both `StockPortfolioEnv` and the backtrader replay assume **alphabetical ticker order** in the observation matrix columns. The notebook achieves this implicitly via `df.sort_values(['date','tic'])`; the scripts make it explicit in `compute_cov_features()` (sort) and in `RLPortfolioStrategy.__init__` (assertion that the data-feed order matches `ticker_order`). If you ever pass a custom `ticker_list` that isn't already alphabetical, this assertion is what protects you from silently feeding a permuted observation to the model.

### A.10 Stage 5: QuantStats with a pre-fetched benchmark

QuantStats accepts the benchmark either as a ticker string (and downloads it internally via `yfinance.download()`) or as a returns `pd.Series`. The first path is more convenient but couples the report to QuantStats's network behaviour, and the failure mode is silent: a DNS failure to `fc.yahoo.com` (the host yfinance uses for cookie/crumb retrieval) returns an empty DataFrame, QuantStats fills it with zeros, and the report renders with degenerate benchmark stats (Beta 0, zero correlations, Treynor undefined). I hit this on the first run.

The fix in `05_quantstats_report.py` is to pre-fetch the benchmark with FinRL's `YahooDownloader` — the same path stages 3 and 4 already use successfully — and pass the resulting daily-returns series to QuantStats. If the download fails, the script raises a `RuntimeError` immediately instead of producing a misleading report. Side effect: the column label in the metrics CSV changes from `Benchmark (^DJI)` (QuantStats's auto-label when given a ticker) to plain `Benchmark` (the default when given a Series). Cosmetic only — all metrics are correct.

### A.11 Improvement #1: log-return reward

The upstream `StockPortfolioEnv.step()` ends with `self.reward = new_portfolio_value`. That target is non-stationary — over the 11.5-year training horizon it drifts from $1M to many millions, so the magnitude of the gradient signal is dominated by late-period rollouts. Log-returns are stationary, bounded in practice, and additive over time, which is exactly what actor-critic methods like A2C/PPO/DDPG/SAC/TD3 expect.

**Implementation choice.** We *did not* fork FinRL. Instead `utils.py` defines `LogReturnPortfolioEnv(StockPortfolioEnv)` that overrides `step()` like this:

```python
def step(self, actions):
    obs, _reward, done, truncated, info = super().step(actions)
    if not done and self.portfolio_return_memory:
        r = float(self.portfolio_return_memory[-1])
        self.reward = float(np.log(1.0 + r) * self.reward_scaling)
    return obs, self.reward, done, truncated, info
```

This is intentionally a post-hook on `super().step()` rather than a copy of the parent body. The upstream method does many things besides set the reward (advances `self.day`, rebuilds `self.covs`, updates `self.state`, appends to `self.portfolio_return_memory`, etc.). Re-implementing all of that would make us brittle to upstream FinRL changes; reading `portfolio_return_memory[-1]` after the parent runs is one line and keeps us in sync forever.

**Config knob.** `env.reward_mode` is added with two values:
- `"value"` — upstream behaviour, byte-compatible with `0.original/`.
- `"log_return"` — the new behaviour, default in this folder.

The factory `make_portfolio_env(df, config, stock_dim)` in `utils.py` dispatches between the two; stages 2 and 3 call it instead of constructing `StockPortfolioEnv` directly. Stages 1, 4, 5 are untouched — data prep doesn't see rewards, backtrader reconstructs observations manually and only reads `model.predict()`, QuantStats consumes the post-trade CSV.

**`reward_scaling` is now load-bearing.** Upstream commented out the line that multiplies by `reward_scaling`, so in value mode the knob is dead config. In log-return mode our override applies it, so the value matters. Log returns are O(0.01) per day; the default `1e-4` (sized for portfolio values in the millions) would shrink the signal to ~1e-6 and starve the gradient. `make_portfolio_env` prints a warning when it sees that combination. This folder ships with `reward_scaling = 1.0` to match the new default mode.

**Model files.** Stage 2 still saves to `models/agent_<algo>.zip`. Re-training with the new reward overwrites the previous model in the same folder. That's fine here because each folder (`0.original/`, `1.reward_returns/`, ...) has its own `models/` directory — A/B comparison happens at the folder level, not via filename suffixes.

**Expected effect.** More stable training curves, less late-period overfitting, similar or slightly better out-of-sample Sharpe. The improvements doc estimates "high impact" mostly because log-returns are a precondition for further reward-shaping work (transaction-cost-aware reward, Sharpe-style reward) — those changes are only meaningful once the base reward is on a sensible scale.

### A.12 Improvement #2: transaction-cost penalty

The upstream env accepts `transaction_cost_pct` in its constructor but never deducts it in `step()` — the cost is essentially dead config. As a result the agent in `0.original/` and `1.reward_returns/` learns to rebalance every day for free, which is unrealistic and unstable.

**What changed.** `LogReturnPortfolioEnv` is extended with a keyword-only `tc_penalty` constructor argument (default `0.0`). When `tc_penalty > 0`, `step()` does the following after `super().step()` returns:

```python
w_curr = actions_memory[-1]            # weights just applied (post-softmax)
w_prev = actions_memory[-2]            # weights from previous bar
turnover     = |w_curr - w_prev|.sum()
tc_fraction  = tc_penalty * turnover
net_return   = (1 + gross_return) * (1 - tc_fraction) - 1
# rewrite the entries upstream just appended:
portfolio_return_memory[-1] = net_return
portfolio_value             = asset_memory[-2] * (1 + net_return)
asset_memory[-1]            = portfolio_value
# and set the reward from the net return:
reward = log(1 + net_return) * reward_scaling
```

When `tc_penalty == 0`, that whole block is skipped and the env is byte-identical to `LogReturnPortfolioEnv` in `../1.reward_returns/`. This is the "zero penalty = same result as before" guarantee.

**Why rewrite the memories, not just the reward.** A pure reward penalty would teach the policy to avoid churn but stage 3 would still report gross performance — `DRL_prediction` builds its daily-return DataFrame from `portfolio_return_memory`. By overwriting those memories in place, stages 3/4/5 all see realistic post-cost equity curves and metrics.

**Why the exact formula and not the approximation.** `gross - tc_fraction` is fine for small costs but breaks down when turnover spikes (early training when weights bounce randomly). `(1 + gross)(1 - tc) - 1` is one extra multiplication and is correct for all magnitudes.

**`actions_memory` initialisation.** Upstream resets `actions_memory = [[1/N]*N]`. The very first step appends the first softmax-normalised weight vector, so `actions_memory[-2]` exists from step 1 onward — no edge case to handle. Turnover on day 1 is measured against the uniform `1/N` allocation, which is reasonable: the agent starts equal-weighted and pays for its first move away from that.

**Config knob.** Added `env.transaction_cost_penalty` (float). Default in this folder is `0.001` (matches the broker-level commission rate used in `backtrader_config.json`). Set to `0.0` to disable. The factory `make_portfolio_env()` reads it and passes it to the env constructor.

**Interaction with stage 4 (backtrader) costs.** Backtrader applies its own commission (0.1%) and slippage (0.05%) on every fill during the replay. The env-level TC penalty shapes the *policy* during training; the broker-level cost realises actual *cash* during the replay. These live in different layers and don't double-count:
- Stage 3 uses the env directly → equity curve reflects the env-level penalty.
- Stage 4 uses the trained policy via backtrader → equity curve reflects backtrader's broker costs (the env-level penalty has no effect at inference time because `DRL_prediction` doesn't read `self.reward`).

**`value` reward mode is out of scope.** `make_portfolio_env()` prints a warning and ignores `transaction_cost_penalty` when `reward_mode == "value"`. Adding a TC-aware value-reward env would require a `NetValuePortfolioEnv` subclass and isn't needed for any improvement currently on the roadmap.

### A.13 Improvement #3: walk-forward training and evaluation

**Motivation.** A single 16-month evaluation on a near-monotone bull run (Jul 2020 → Oct 2021) is statistically meaningless — we cannot tell whether the agent's apparent edge is skill or a lucky regime. Walk-forward gives the agent's policy multiple shots at unseen data across multiple regimes (2015 correction, 2018 selloff, COVID 2020, 2021 rally), and the train set always uses *only* data that was actually available before the eval window. No look-ahead.

**Sliding default windows.** Anchored on `data.train_start_date = 2009-01-01`, with `train_years=6` and `eval_years=1` stepping forward by `step_years=1`:
```
window 0:  train 2009-01-01..2014-12-31   eval 2015-01-01..2015-12-31
window 1:  train 2010-01-01..2015-12-31   eval 2016-01-01..2016-12-31
window 2:  train 2011-01-01..2016-12-31   eval 2017-01-01..2017-12-31
window 3:  train 2012-01-01..2017-12-31   eval 2018-01-01..2018-12-31
window 4:  train 2013-01-01..2018-12-31   eval 2019-01-01..2019-12-31
window 5:  train 2014-01-01..2019-12-31   eval 2020-01-01..2020-12-31
window 6:  train 2015-01-01..2020-12-31   eval 2021-01-01..2021-10-31
```
7 windows total. With A2C @ 50k timesteps each, this is ~7 minutes of training on CPU. Stage 3 then runs 7 short predictions (~seconds each) and concatenates them.

**Data flow.** Stage 1 now also persists `data/full_data.pkl` — the entire processed history (~98k rows) with covariance features computed once. Stages 2 and 3 read this single source of truth and slice it per window via `slice_by_dates(full_df, start, end)`. The legacy `train_data.pkl` and `trade_data.pkl` are kept so stages 4/5 keep working in non-walk-forward configs, and so that setting `walk_forward.enabled=false` is a true byte-compatible fallback to `../2.transaction_cost/`.

**Slice-and-reindex requirement.** `StockPortfolioEnv.df.loc[self.day, :]` needs the integer index to start at 0 and be dense. After a date-range slice, the original factorised index is fragmented. `slice_by_dates()` re-factorises the date column after the slice so `df.loc[0]` is the first day of the window. Without this, training crashes on the second `step()`.

**Per-window model files.** Stage 2 writes `models/agent_<algo>_w<i>.zip` (e.g., `agent_a2c_w0.zip`, `agent_a2c_w1.zip`, ...) plus a `models/windows.json` manifest containing the 4-tuple for each window. Stage 3 reads the manifest, iterates over windows, loads each model, and runs prediction over the corresponding eval slice.

**Stitching.** Each per-window `daily_return` DataFrame is concatenated in chronological order. Duplicate dates that occur at window boundaries (e.g., when window N's eval_end equals window N+1's eval_start) are de-duplicated by keeping the first occurrence — that's the prediction made *before* the next window's model takes over. After stitching, `daily_return_to_equity()` applies a single `cumprod` to get a continuous equity curve.

**Baselines now cover the full out-of-sample period.** Min-Variance is daily-rebalanced over the slice `[windows[0].eval_start, windows[-1].eval_end]` (the union of all eval ranges). DJIA is fetched from Yahoo over the same range. So all three series — DRL agent(s), Min-Variance, DJIA — are commensurable on the same date axis.

**Stages 4 and 5: deliberate scope limitation.** Stage 4 (backtrader) calls `model.predict()` inside `bt.Strategy.next()` once per bar — but it currently knows only about one model. Faithful walk-forward in backtrader requires either (a) holding a list of `(date_range, model)` pairs and switching inside `next()` based on the current bar date, or (b) running cerebro once per window and stitching the equity curves. Both are tractable but non-trivial; for now stage 4 in this folder loads only `agent_<algo>_w<last>.zip` and replays the full out-of-sample period through it. This gives a useful end-of-training picture but is **not** a fair walk-forward replay. Stage 5 (QuantStats) consumes whatever stage 4 produces, so it inherits the same limitation. Full integration is logged as a follow-up.

**Falling back to single split.** Set `walk_forward.enabled=false` in `config.json`. Stages 2/3 then revert to reading `train_data.pkl` / `trade_data.pkl` and writing `agent_<algo>.zip` exactly like `../2.transaction_cost/`. Useful for direct A/B comparison of the same agent against itself with/without walk-forward training.

### A.14 Improvement #4: differential Sharpe / Sortino reward

**Motivation.** The log-return reward (improvement #1) trains the agent to maximise *expected* return. There is no explicit volatility penalty in the reward — the agent has to discover diversification benefits indirectly through the policy gradient. For most practical portfolios we care about *risk-adjusted* return: same expected return with lower variance is strictly better. Moody & Saffell (1998) showed that the per-step contribution to a running Sharpe ratio is a clean differentiable signal that an RL agent can be trained on directly, with much faster convergence to risk-aware policies than reward shaping via a value-function penalty.

**The math.** Let `r_t` be the effective (post-TC) return at step `t`. The env keeps three exponentially-weighted moments updated by the same constant `eta`:

```
A_t = A_{t-1} + eta * (r_t       - A_{t-1})        # running E[R]
B_t = B_{t-1} + eta * (r_t^2     - B_{t-1})        # running E[R^2]
D_t = D_{t-1} + eta * (R_down^2  - D_{t-1})        # running E[downside R^2]
                       where R_down = min(r_t, 0)
```

Treating the Sharpe ratio `S_t = A_t / sqrt(B_t - A_t^2)` as a function of the new observation `r_t` and Taylor-expanding around `(A_{t-1}, B_{t-1})`, the first-order term is:

```
DSR_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^{1.5}
        where dA = r_t - A_{t-1},  dB = r_t^2 - B_{t-1}
```

This is exactly the per-step *change* in the running Sharpe ratio induced by `r_t`. Rewarding the agent with DSR is therefore equivalent to rewarding it for "improving the long-run Sharpe of the portfolio it has held so far".

For Sortino, the same derivation on `S^d_t = A_t / sqrt(D_t)` (downside Sortino) yields:

```
DDR_t = (D_{t-1} * dA - 0.5 * A_{t-1} * dD) / D_{t-1}^{1.5}
        where dD = R_down^2 - D_{t-1}
```

The denominator uses only the downside second moment, so upside variance no longer counts as risk — useful when max drawdown is the metric you actually care about.

**Implementation choices.**

- **One class, three modes.** `LogReturnPortfolioEnv` is generalised with a `reward_kind` parameter (`"log_return"` | `"diff_sharpe"` | `"diff_sortino"`). The TC math (improvement #2) is identical in all three modes and runs first; the reward formula is selected after. Setting `reward_kind="log_return"` produces the exact same byte stream as `../3. Walk-forward/`.
- **State held on the env, reset on `reset()`.** `self._A`, `self._B`, `self._D` are zero at the start of every episode. The `eta=1/252` decay means they take ~252 steps to converge to their stationary values, so the first ~year of each rollout has noisy DSR/DDR rewards. SB3's PPO/A2C are robust to this; if you want a warmer start, set `eta` higher.
- **Numerical guards.** Both denominators (`(B - A^2)^{1.5}` for Sharpe, `D^{1.5}` for Sortino) can be near zero in the first few steps. We clamp via `max(denom, 1e-8)` to avoid divide-by-zero, and clamp `B - A^2` to `[0, ∞)` so it never goes negative due to floating-point error.
- **`eta` interpretation.** `1/252` corresponds to a one-trading-year half-life. Larger `eta` (e.g., `1/63` for a quarter) gives faster adaptation but more reward noise; smaller `eta` (e.g., `1/504` for two years) gives smoother rewards but slower regime tracking. Exposed via `env.diff_ratio_eta`.
- **Reward scale.** DSR/DDR are dimensionless per-step contributions; magnitudes are typically `O(1)` once the EMAs converge but can spike during warm-up. `reward_scaling = 1.0` is the sensible default; reduce it if training diverges, increase if gradients vanish.

**What changed and what didn't.**

| File | Change |
|---|---|
| `utils.py` | `LogReturnPortfolioEnv`: new `reward_kind` + `diff_ratio_eta` params, `_A`/`_B`/`_D` state, `reset()` clears them, `step()` dispatches to `_diff_sharpe_reward()` or `_diff_sortino_reward()`. `make_portfolio_env()` accepts the two new reward_mode values. |
| `config.json` | `env.reward_mode = "diff_sharpe"` (default). Added `env.diff_ratio_eta`. |
| `02_train.py` | Startup print includes `diff_ratio_eta` when shaped-ratio modes are active. |
| `03_backtest.py` | No change. The reward signal is only used at training time; inference reads the same softmaxed actions. |
| `04_backtrader_replay.py`, `05_quantstats_report.py`, both `*_config.json` | No change. The replay reads recorded weights, which are determined by the trained policy regardless of which reward shaped it. |

**Comparing modes.** The four folders (`0.original` / `1.reward_returns` / `2.transaction_cost` / `3. Walk-forward` / `4.Sharpe Sortino Ratio`) now form a clean ablation:
- Same data, same hyperparameters, same architecture, same evaluation protocol.
- Only the reward function changes (plus walk-forward in the last two).
- A/B comparison is just "run stage 3 in each folder and compare equity curves, Sharpe, max drawdown".

If you want a pure improvement-#4 comparison, set `walk_forward.enabled = false` here and in `../3. Walk-forward/`; both will use the single-split protocol and the only difference will be the reward shape.

### A.15 Improvement #5: early stopping on validation Sharpe

**Motivation.** Until this folder, `total_timesteps` in `config.json` was a hand-picked number (50k for A2C, 80k–150k for PPO, etc.). Three things were wrong with that:

1. **Wrong number, wrong outcome.** Pick too low → the policy hasn't converged. Pick too high → on noisy rewards (especially DSR from improvement #4) the policy overshoots a good plateau and degrades from there. Either failure is silent — the final saved model is just "whatever state the optimiser was in when the clock ran out".
2. **No principled stopping criterion.** Loss curves are dominated by reward noise; raw episode reward conflates "agent is improving" with "the market regime in the training data happens to be favourable". You need a metric that measures policy quality on data the agent didn't train on.
3. **Compute waste.** In walk-forward with 7 windows × 150k timesteps you're spending ~70% of the budget after the best policy has already been seen.

**The fix.** Reserve the last `val_fraction` of every training slice for validation. Every `eval_freq` training steps, run the current policy deterministically through the validation slice and compute annualised Sharpe from `portfolio_return_memory`. Save the best-so-far checkpoint; stop after `patience` consecutive evaluations without improvement.

**No leakage invariant.** The validation slice is taken from the *training* range (the last `val_fraction` of it), NOT from the trade/eval window. In walk-forward this is crucial:

```
Window i:  train_only [..............]  validation [...]  eval [..............]
                                        <-- never seen during training -->
                                                          <-- never seen during validation -->
```

The eval slice the model is ultimately backtested on (stage 3) is strictly after `train_end`, while the validation slice for early stopping is strictly before `train_end`. The model selection criterion never touches data after the train cutoff, so there's no peeking.

**Why a custom callback, not SB3's `EvalCallback`.** SB3's built-in `EvalCallback` evaluates by running episodes and taking the mean episode reward. With our shaped rewards (DSR/DDR/log_return) the mean episode reward is not the metric the user actually cares about — the *annualised Sharpe of the realised returns* is. The `ValidationSharpeCallback` in `utils.py` builds a fresh `make_portfolio_env(val_df, …)`, runs the model deterministically end-to-end, calls `env_method("save_asset_memory")` just before the terminal step (to capture `portfolio_return_memory` before DummyVecEnv's auto-reset wipes it), and computes Sharpe from those returns directly. It's ~80 lines and gives a metric you can actually trust.

**FinRL plumbing.** `DRLAgent.train_model` always wraps `model.learn()` with `TensorboardCallback` and doesn't expose a callback parameter. In early-stopping mode we bypass `DRLAgent.train_model` and call `model.learn()` directly with `CallbackList([TensorboardCallback(), ValidationSharpeCallback(...)])`. Both tensorboard logging and early stopping work.

**Best-vs-final on disk.** SB3's `model.learn()` returns the model in its *final* state (not the best). The callback writes the best to disk as soon as it sees an improvement, so the file on disk is always the best so far, even after early termination. After learn() returns, `train_one()` does NOT save the model again — that would overwrite the best with the (possibly worse) final state. The one exception is when training completes its full `total_timesteps` budget without ever triggering an "improved" save (no checkpoint ever met `min_delta`); `_on_training_end()` then saves the final policy so stage 3 still has something to load.

**Effect on training time.** With `total_timesteps=150_000`, `eval_freq=5000`, `patience=5`:
- best case: training stops at ~25k (best at 0, patience expires by 25k).
- typical case: ~50–80k per window.
- worst case: never improves enough → runs the full 150k budget.

Across 7 walk-forward windows that's a 50–70% reduction in wall-clock training time.

**History JSON.** After each training run (per algo, per window), a `models/agent_<algo>[_w<i>].history.json` is written. Each entry is `{timesteps, sharpe, improved, best_so_far}` — useful to see which window's best policy was found early vs late, and how stable the Sharpe trajectory was.

**What changed.**

| File | Change |
|---|---|
| `utils.py` | `split_train_validation()` helper (date-based, re-factorises index). `ValidationSharpeCallback(BaseCallback)` class. |
| `02_train.py` | `train_one()` now takes `model_path`. Branches on `early_stopping.enabled`: if on, splits train_df, builds the callback, calls `model.learn(callback=CallbackList(...))`; if off, keeps the legacy `DRLAgent.train_model` path. Callers (`train_single_split`, `train_walk_forward`) no longer save the model themselves — `train_one` always handles the save. |
| `config.json` | New `early_stopping` block. |
| `03_backtest.py`, `04_backtrader_replay.py`, `05_quantstats_report.py`, both `*_config.json` | No change. Stage 3 just loads whatever's at `models/agent_<algo>_w<i>.zip` — which is the early-stopping best when this is enabled. |

**Independence from prior improvements.** Early stopping is orthogonal to reward mode, TC penalty, and walk-forward. The five folders now form a clean ablation cube — you can isolate any single improvement by toggling its flag and leaving the others alone.

### A.16 Improvement #6: VecNormalize for observations and reward

**Motivation.** The 945-dim flattened observation that `StockPortfolioEnv` produces mixes wildly heterogeneous-scale features:

| Block | Approximate scale |
|---|---|
| Covariance entries (27×27 = 729 of them) | O(1e-3) |
| MACD                                     | O(1) |
| Bollinger Bands (`boll_ub`, `boll_lb`)   | $30–$400 (price level) |
| RSI-30                                   | 0–100 |
| CCI-30                                   | -300 to +300 |
| DX-30                                    | 0–100 |
| 30/60-day SMA                            | price level |

An MLP fed those raw inputs has to spend the first few thousand gradient steps just discovering that "1 unit of MACD" is 100,000 times more meaningful than "1 unit of covariance entry". `VecNormalize` does this in a closed-form way: it tracks running mean and variance per dimension, applies `(x - mu) / sqrt(sigma^2 + eps)` clipped to `[-clip_obs, +clip_obs]`, and lets the policy network see a unit-variance observation from step 1.

Reward normalisation is the same idea applied to the per-step reward — particularly useful with shaped rewards (DSR/Sortino from improvement #4) that can shift in magnitude across walk-forward windows.

**Implementation choices.**

- **Per-window stats files.** In walk-forward mode, every window trains its own model with its own training data distribution. Mean / variance of the observation differ across windows (different regimes, different prices). So each window gets its own `models/vecnormalize_<algo>_w<i>.pkl` file, saved alongside `models/agent_<algo>_w<i>.zip`. Stage 3 loads the right pair when predicting on that window's eval slice.
- **Custom save/load instead of SB3's default.** `VecNormalize.save()` pickles `self.__dict__` — which includes `self.venv`, the wrapped vec env, which includes the StockPortfolioEnv's full DataFrame (cov_list / return_list object columns, ~200 MB for a 6-year window). For 7 walk-forward windows that's >1 GB of stats files. We instead persist a small dict with only `obs_rms`, `ret_rms`, and the config flags (`clip_obs`, `norm_obs`, etc.). Typical file size: <100 KB. `load_vecnormalize_stats()` rebuilds a fresh `VecNormalize` around a freshly-built venv and copies the stats in.
- **Frozen stats for early-stopping validation rollouts.** The training env's stats update on every step. If we evaluated the validation slice with its own running stats, the policy would see a different normalisation than at training time — off-distribution. So `ValidationSharpeCallback._evaluate_sharpe()` calls `wrap_eval_env_with_ref_stats(sb_env, self.ref_vn)` which deep-copies the *current* `obs_rms` and `ret_rms` from the training wrapper and applies them to the validation env with `training=False`. The eval rollout cannot mutate the training stats, and the policy sees on-distribution observations.
- **Save stats matched to the saved checkpoint.** With early stopping, the model checkpoint on disk is the best-so-far — but training continues until patience expires, so the running stats keep evolving. If we saved stats at end-of-training they'd be a few thousand steps "off" from the saved model. To keep them matched, the callback also calls `save_vecnormalize_stats(self.ref_vn, self.vn_save_path)` every time it saves a new best model. The .zip and .pkl always correspond to the same training step.
- **`norm_reward=False` at inference and validation.** Only relevant during training. At eval/inference we want raw rewards so realised Sharpe / equity curves are honest.
- **`gamma` parameter.** VecNormalize uses an internal `gamma` to track the running variance of discounted returns (for reward normalisation). This should match PPO's `gamma` for best behaviour. Default 0.99 in both places.

**Why stages 4–5 require no changes.**

Stage 3 already produces `results/weights_<algo>.csv` — post-softmax target weights, one row per trading day. These weights are the SAME regardless of whether the policy was trained with normalised observations or raw ones (the softmax is invariant to the absolute scale of the raw logits). Stage 4 (backtrader) just replays these weights through the broker; stage 5 (QuantStats) reads stage 4's equity curve. Neither stage invokes the policy, so neither needs to know about normalisation.

**Why `DRLAgent.DRL_prediction` doesn't work with normalised models.**

`DRL_prediction` builds an env, wraps it as `DummyVecEnv`, calls `model.predict()` per bar, captures memories. It does NOT wrap with `VecNormalize` and has no API for loading saved stats. If you feed it a model trained on normalised observations, the model receives raw `obs_unnorm` and applies its weights as if they were normalised inputs — the resulting actions are essentially random. Hence `predict_one()` in stage 3 branches on `normalization.enabled`: off → `DRL_prediction`; on → manual rollout with `load_vecnormalize_stats(...)`.

**What changed.**

| File | Change |
|---|---|
| `utils.py` | `build_vecnormalize`, `save_vecnormalize_stats`, `load_vecnormalize_stats`, `vecnormalize_path_for`, `wrap_eval_env_with_ref_stats`. `ValidationSharpeCallback` gains `ref_vecnormalize` and `vn_save_path` parameters; saves stats when it saves a new best model. |
| `02_train.py` | Branches on `normalization.enabled` in `train_one`: wraps env, plumbs the wrapper to the callback, saves stats. Bypasses `DRLAgent.train_model` when normalisation is on without ES (because the FinRL helper doesn't preserve the VecNormalize wrapping). |
| `03_backtest.py` | `predict_one()` branches: when normalisation is on, does a manual rollout with `load_vecnormalize_stats`. Same `(daily_return, df_actions)` output, so all downstream stitching / baselines / CSV writing is unchanged. |
| `config.json` | New `normalization` block. |
| `01_get_data.py`, `04_backtrader_replay.py`, `05_quantstats_report.py`, `backtrader_config.json`, `quantstats_config.json` | No changes. |

**Expected effect.** Smoother training curves in tensorboard, particularly for the actor loss in the first ~10k timesteps. Often a slightly higher best-validation Sharpe at the early-stopping checkpoint. In walk-forward, the reduction in cross-window variance is usually visible — windows that previously had wildly different best-Sharpe values become more comparable because the policy sees a normalised view of each window's data.

### A.17 Improvement #7: turbulence-gated risk-off rule

**Motivation.** RL policies are shaped by training-period statistics. When the live market enters a regime that looks nothing like training (COVID March 2020, GFC October 2008, August 2015 flash crash, Feb 2022 inflation shock), the policy can produce confidently-wrong allocations that compound losses fast. A non-learned safety valve overrides the policy with "go to cash" whenever a market-wide stress indicator spikes. The FinRL NeurIPS2018 paper uses exactly this trick at trade time and credits it for most of the "beats DJIA" claim — without it, DRL stock-trading agents typically don't beat the index out-of-sample.

**The turbulence index.** Kritzman & Li (2010, FAJ). Define the cross-sectional mean and covariance of daily returns over a trailing window (FinRL uses 252 days). For each day, compute the Mahalanobis distance of that day's return vector from the trailing mean:

```
turbulence_t = (r_t - mu_train)^T  Sigma_train^-1  (r_t - mu_train)
```

This is one scalar per day, regardless of how many tickers are in the universe. Empirically:

| Regime                          | Typical turbulence |
|---|---|
| Calm equity markets             | 20–60             |
| Mild volatility                 | 60–80             |
| Earnings-season chop            | 70–90             |
| Single-day shock                | 100–150           |
| Crisis (2008-Q4, 2020-Q1)       | 200+              |

A threshold of 70 cuts roughly the worst 1–3% of trading days on the Dow universe.

**Where the gate runs (and why it's last among #2/#7 in semantic order, but first in `step()`).** The order of operations inside `LogReturnPortfolioEnv.step()` is:

```
super().step(actions)              # upstream env: softmax actions → weights → compute portfolio_return
# ---- improvement #7 ----
if risk_off_active:                # check turbulence; if above threshold:
    weights ← 0                    #   wipe weights
    portfolio_return ← 0           #   wipe return
    portfolio_value ← prior value  #   roll equity unchanged
# ---- improvement #2 ----
if tc_penalty > 0:                 # cost on whatever turnover survives
    turnover = |weights - prev|
    return ← return * (1 - tc * turnover)
# ---- improvement #1 / #4 ----
reward = f(return)                 # log(1+r), DSR, or DDR
```

The gate fires before TC because TC needs to charge turnover on the realised position change — which is "everything to cash" when the gate fires. If we ran TC first, we'd charge cost on the agent's *attempted* trades, then wipe them — incorrect double-accounting.

**Training vs inference.** The same threshold is applied in both modes. The agent's gradient on gated days is therefore zero-reward (or slightly negative under TC) and uninformative — the agent learns it can't influence turbulent days. Gradients on the remaining ~97–99% of days continue to refine the policy normally.

The Stock_NeurIPS2018 paper takes a different choice: gate-off in training, gate-on in inference. Their argument is that during training you want the agent to see all data so the value function generalises better; during inference the gate is a safety post-hoc override. That works for them but introduces a train/test mismatch we'd rather avoid. Our choice (same gate everywhere) is the cleaner default; a future improvement could add `risk_off.turbulence_threshold_train` and `risk_off.turbulence_threshold_trade` separately if the asymmetry turns out to matter.

**Subtle interaction with DSR (improvement #4).** DSR's running A/B EMAs include the gated zero-return days. The running mean A is pulled toward 0 on gated days, slightly dampening the DSR signal on the following day. Sortino is less affected because gated days don't contribute to the downside second moment D. For typical threshold values (70+) and typical turbulence frequencies (~1–3% of days), the effect is small.

**Subtle interaction with VecNormalize (improvement #6).** The reward distribution shifts when the gate fires often (lots of zeros). VecNormalize's `ret_rms` adapts its variance estimate downward, which can amplify the rescaled reward magnitude on non-gated days. This is mostly harmless but worth being aware of: if you crank the threshold very low (say 40) you might see PPO's `train/value_loss` behave more erratically.

**What changed.**

| File | Change |
|---|---|
| `utils.py` | `LogReturnPortfolioEnv.__init__` accepts `risk_off_enabled` + `turbulence_threshold`. New `_risk_off_active()` reads `self.data["turbulence"]`. `step()` applies the gate as the first post-`super()` operation. `make_portfolio_env()` reads `config.risk_off`, warns when enabled but turbulence column is missing, and threads params through. |
| `config.json` | `data.use_turbulence` flipped to `true`. New `risk_off` block. |
| `02_train.py` | Startup print extended to show `risk_off=on(thr=70.0)` or `risk_off=off`. |
| `01_get_data.py`, `03_backtest.py`, `04_backtrader_replay.py`, `05_quantstats_report.py`, `backtrader_config.json`, `quantstats_config.json` | **No code changes.** Stage 1 already calls `FeatureEngineer(use_turbulence=data_cfg["use_turbulence"])`, so flipping the config bit is enough. Stages 3/4/5 are insulated from the gate by the env's internal handling. |

**Effect on results.** Expected effect over the 2015–2021 walk-forward eval window:
- A few gated days per year (~3–10), concentrated around Feb–Mar 2020.
- Slightly lower cumulative return (cash earns 0% on gated days while index has positive drift on average).
- Materially smaller max drawdown (the worst COVID week is partially or fully avoided).
- Higher Sharpe and Calmar — both numerator (return variance shrinks) and denominator (drawdown shrinks) move favourably.

The exact magnitude depends on which days the index turbulence happens to spike on; this is data-dependent and seed-independent. Look at the new stage 3 stdout: it'll print one line per algorithm and you can compare CumReturn / Sharpe / MaxDD against `../6.VecNormalize for observations (+ reward)/results/` directly.

### A.18 Improvement #8: multi-seed training and ensembled actions

**Motivation.** A single PPO training run is a high-variance experiment. Two runs with the same hyperparameters and different seeds can produce out-of-sample Sharpe values 0.2–0.4 apart. That noise floor is bigger than most hyperparameter effects you're trying to measure. Without seed averaging, you can't tell whether a change "helped" or just landed on a lucky seed.

The standard fix in RL practice is to train N independent policies with different seeds and report mean ± std. For trading, an even more useful variant is to **ensemble the policies at inference**: average their actions and treat the ensemble as a single (smoother, lower-variance) strategy. That's what this folder implements.

**Two design choices that matter.**

1. **Average POST-softmax weights, not pre-softmax logits.** Both are mathematically valid, but they differ in semantics:
   - Average of softmax outputs = "weighted average of each policy's allocation". Stays on the simplex (non-negative, sums to 1).
   - Softmax of average logits = a single sharper distribution. Can concentrate weight if one policy strongly prefers a stock and others are indifferent.

   Averaging post-softmax is closer to the "ensemble votes equally" mental model and produces more diversified, lower-turnover allocations. That's what we use.

2. **Compute portfolio_return from the averaged weights, not by averaging per-seed returns.**

   Suppose seed A produces weights `[0.7, 0.3]` on day t with return `+0.02`, and seed B produces `[0.2, 0.8]` with return `−0.01`. Averaging the seed returns gives `+0.005`. But the **ensemble** would hold `[0.45, 0.55]` on day t, which produces a different return depending on the underlying ticker moves.

   So in `predict_walk_forward`, we average the actions first, then recompute the return against the eval data using `daily_return_from_weights()`. This produces the equity curve a real trader following the ensemble would have seen.

**Implementation.**

| File | Change |
|---|---|
| `utils.py` | `get_seeds(config)` reads `seeds.list` with fallback to `training.seed`. `average_seed_actions(per_seed_dfs)` averages row-by-row and re-normalises. `daily_return_from_weights(weights, trade_df)` does the price-weighted return calculation. |
| `02_train.py` | Outer `for s in seeds:` loop wraps the existing per-window training. Per-seed model filename `agent_<algo>[_w<i>]_s<seed>.zip`. ES history files become `agent_<...>_s<seed>.history.json`; VecNormalize stats become `vecnormalize_<...>_s<seed>.pkl`. |
| `03_backtest.py` | In `predict_walk_forward`, for each window load all N seeds, average actions, recompute return. In single-split mode, same but skipping the window loop. |
| `config.json` | New `seeds.list` block. Default `[42, 1337, 7]`. |
| `01_get_data.py`, `04_backtrader_replay.py`, `05_quantstats_report.py` | No changes. Stages 4 and 5 read the saved ensemble-averaged weights CSV from stage 3 — they don't know about seeds at all. |

**Compute cost.** Stage 2 wall-clock scales linearly with `len(seeds.list)`. With 3 seeds × 7 walk-forward windows × ~15 min per training run on CPU, expect ~5 hours total. Early stopping (improvement #5) helps — typical individual runs stop at 30–60k timesteps, so the multiplier is more like 2–3x rather than the worst-case linear.

Stage 3 also scales linearly with N (each window runs N inferences) but each inference is seconds, not minutes. Negligible cost.

Stages 4 and 5 don't scale with N at all.

**Variance reduction in practice.** With the diff_sharpe reward + VecNormalize + ES, three-seed ensembling typically:
- Reduces equity-curve std across re-runs by 30–50%.
- Raises mean out-of-sample Sharpe by 0.1–0.3 over the median single-seed run.
- Materially smooths the per-day weight trajectory (you can see this in `daily_trades.csv` — turnover per day drops because different seeds disagreeing get averaged out).
- Reduces max drawdown by a few hundred bps in stressful regimes (different seeds rarely all panic on the same day).

The improvement is "free alpha" in the sense that it costs only compute, requires no new ideas about the market, and removes a known noise source from the evaluation.

**Common variations (one-line edits in `config.json`).**

| Config | Effect |
|---|---|
| `"list": [42]` | Singleton ensemble — identical to single-seed runs. Useful as a sanity check. |
| `"list": [42, 1337, 7]` (default) | Three seeds. Recommended starting point. |
| `"list": [0, 1, 2, 3, 4]` | Five seeds. Further variance reduction at 5/3 = 1.67× more compute. |
| `"list": [42, 42, 42]` | Repeated seed — pointless (all three policies identical). Don't do this. |

**Subtle gotcha with early stopping.** Each seed has its own validation Sharpe trajectory and stops at a different timestep. That's fine — when the ensemble runs at inference, each seed contributes its own best checkpoint. But the per-seed `history.json` files won't be aligned in length; they're inspected individually. The stage 2 stdout prints the best Sharpe per seed per window.

**Subtle gotcha with VecNormalize.** Each seed has its own running statistics, persisted as a separate `vecnormalize_<algo>_w<i>_s<seed>.pkl`. At inference, each seed's `predict_one()` loads its own stats. The ensemble averaging happens AFTER each policy has produced its softmaxed weights from its own normalised view of the observations. That's the correct behaviour — you don't want to average raw observations across seeds.

### A.19 What was deliberately *not* changed

To keep this pipeline a faithful replication, the following potential improvements were intentionally *not* applied. They are catalogued in `FinRL_PortfolioAllocation_improvements.md` for the next iteration:

- Reward function (still raw portfolio value).
- Transaction-cost-aware reward.
- Observation normalisation.
- Walk-forward training/eval windows.
- Differential Sharpe / Sortino reward.
- Turbulence-gated risk-off rule.
- Multi-seed ensembling.
- Early stopping on validation Sharpe.

Each of those can be slotted into this scaffolding by adding a new section to `config.json` and a corresponding branch in the relevant stage script.

### A.20 Production deployment pipeline

The standard walk-forward pipeline (A.13) is a VALIDATION harness — its job is to prove the recipe is robust across regimes via 7 OOS windows. It is NOT the model you deploy. The deployment pipeline lives alongside the validation pipeline in three independent additions.

**A.20.1 Two configs**

`config.json` (existing, unchanged) runs the walk-forward validation. `config_production.json` (new) runs a single-split training on the full history through ~yesterday, with no held-out test slice. The validation slice for ES is the last 10% of training dates (~1.7 years of recent data). The deployment models go into `models/agent_ppo_s<seed>.zip` (no `_w` suffix).

**A.20.2 Candidate pool + filter**

PPO training is initialisation-dependent. Some seeds escape the near-uniform initial policy and learn meaningful allocations; others stay stuck, with the saved "best" checkpoint being the un-learned policy at step 2,500. Empirically about 1 in 3 seeds gets stuck on this state space.

Rather than fighting PPO's training dynamics, deployment treats this as routine:

| Config knob               | Role                                                                                           |
|---|---|
| `seeds.list`              | Candidate POOL trained by stage 2. Default 8 seeds. Stage 2 trains every entry.                |
| `seeds.ensemble_size`     | How many to use at INFERENCE. Default 3. Lower than the pool size enables filtering.            |

`utils.pick_ensemble_seeds(config, model_dir)` reads each `agent_ppo_s<seed>.history.json` and ranks seeds by (n_improvements desc, best_sharpe desc). The top `ensemble_size` participate in the deployed ensemble. No retraining required to change `ensemble_size` — the ranking happens at inference time.

`filter_seeds.py` prints the ranking explicitly. `inspect_ensemble.py` compares the chosen seeds' weight vectors side-by-side with pairwise L1 / Jensen-Shannon disagreement metrics and an auto-classified verdict.

**A.20.3 Critical bug fix: `daily_return_from_weights` was friction-free**

A long-standing bug in `predict_walk_forward` / `predict_one_split` (stage 3) caused the equity curve in `equity_plot.png` to overstate returns by ~1.2 pp per year. Root cause:

- The env's `LogReturnPortfolioEnv.step()` correctly applies a drift-adjusted TC penalty during training and per-seed rollouts.
- But stage 3 in walk-forward + multi-seed mode does NOT use the env's portfolio_return — it averages per-seed action vectors first, then recomputes returns via `utils.daily_return_from_weights(...)`.
- That function used to compute `(rets * w).sum()` without any TC penalty. The drift-adjusted TC formula in the env was bypassed entirely at evaluation.

Symptom: stage 3 reported PPO at +208.6% over 11.4 years (CAGR 10.4%), while the same weights replayed through backtrader (stage 4) produced +118.8% (CAGR 7.1%). The 90 pp gap is decomposable as ~35 pp from the missing env TC + ~55 pp from real backtrader execution friction (discrete-event fills, integer share rounding, sequential rebalances).

**Fix.** `daily_return_from_weights` now accepts `tc_penalty` and applies the exact same drift-adjusted formula as the env:

```
growth   = 1.0 + rets                          # per-ticker growth t-1 -> t
drifted  = w.shift(1).fillna(0.0) * growth     # post-drift unnormalised
total    = drifted.sum(axis=1)
w_drift  = drifted.div(total.where(total > 0.0, 1.0), axis=0)
turnover = (w - w_drift).abs().sum(axis=1)
tc_drag  = (tc_penalty * turnover).clip(upper=1.0)
port_ret = (1 + gross_return) * (1 - tc_drag) - 1
```

Vectorised against `w.shift(1)`. Identical to the env's per-step formula. The function is backward-compatible — `tc_penalty=0.0` (the new default) reproduces the friction-free behaviour.

`03_backtest.py` now passes `tc_penalty = config["env"]["transaction_cost_penalty"]` to both call sites (walk-forward stitching and single-split). After this fix, the stage 3 equity curve drops from $308.6M to ~$273M (CAGR 9.3%). The gap to stage 4 collapses to ~54 pp = pure backtrader execution friction (~3 pp/yr).

**Implication for kill-switch tripwires.** The walk-forward distribution computed BEFORE this fix was inflated by ~1 pp/yr CAGR and ~0.1–0.2 Sharpe. Re-run the walk-forward after the fix; use the new distribution for live monitoring.

**A.20.4 Execution filters — HOLD vs REBALANCE (Option B)**

PPO trained with `transaction_cost_penalty=0.001` still produces ~5% L1 turnover per day = ~600% annualised one-way. That signal IS the model's view, but executing every tiny daily delta is wasteful. Two knobs in `config_production.json` decide whether the daily inference results in trades or a no-op:

| Knob                          | Default | Semantics |
|---|---|---|
| `execution.rebalance_band`    | 0.02    | Portfolio-level L1 drift threshold. If `sum(|target - current|) < band`, skip the day entirely (HOLD). |
| `execution.min_weight_delta`  | 0.02    | Per-ticker freeze. Any ticker whose `|target - current| < min_weight_delta` is pinned at current. Mirrors backtrader's `execution.min_weight_delta`. |

Both apply ONLY when `predict_tomorrow.py` is called with `--current-weights <csv>`. The user provides their actual broker portfolio weights each evening; the script compares them to the model's proposed target and outputs one of:

| Condition                                                            | Decision  | Trades |
|---|---|---|
| L1 drift < `rebalance_band`                                          | HOLD      | 0 |
| L1 drift ≥ band AND every per-ticker delta < `min_weight_delta`      | HOLD      | 0 |
| L1 drift ≥ band AND ≥ 1 per-ticker delta ≥ `min_weight_delta`        | REBALANCE | N |

Frozen tickers stay EXACTLY at their current weight (not at the model's target). Renormalisation absorbs only into the unfrozen subset, so no nuisance ±0.1% drift creeps into frozen positions.

Empirically with `0.02` for both knobs and ~5% daily turnover, this turns ~40–60% of trading days into HOLDs and substantially reduces execution friction in production. The savings compound over time vs always executing every daily delta.

**A.20.5 New scripts and helpers**

| File | Role |
|---|---|
| `config_production.json` | Single-split deployment config + execution filter knobs |
| `predict_tomorrow.py` | Daily inference: fetch live Yahoo data, recompute features, run ensemble, apply execution filters, emit HOLD/REBALANCE |
| `inspect_ensemble.py` | Per-seed weight comparison + agreement verdict before deploying |
| `filter_seeds.py` | Rank trained candidates by convergence (for choosing `ensemble_size`) |
| `utils.pick_ensemble_seeds()` | Filter top-N converged seeds from `seeds.list` at inference time |
| `utils.daily_return_from_weights(..., tc_penalty=...)` | Drift-adjusted TC formula now applied to multi-seed ensemble curves |

See `USAGE_LIVE_TRADING.md` for the full operations manual: one-time setup, daily 3:30–3:55 pm workflow, retraining cadence, kill-switch metrics, failure modes, and the 30–90 day paper trading phase.

### A.21 Priority-1: article features, reward, benchmark, and cash

**Source.** This folder ports the most promising ideas from Kashif & Slepaczuk, *"Deep Reinforcement Learning Framework for Diversified Portfolio Management Across Global Equity Markets"* (arXiv 2605.17307, 2026) onto the folder-8 scaffolding. Four additions, each independently toggleable: a richer state (custom per-asset + global features), the paper's reward formulations, a configurable benchmark, and an explicit cash position. A full review of the paper and the port decisions lives in `Articles/2605.17307v1_review_and_comparison.md`. Dirichlet policies and SAC were deliberately NOT ported (the review explains why); we keep PPO + softmax.

**A.21.1 Custom features (`features.py`).** Stage 1 computes extra features beyond the eight stockstats indicators, driven by `data.custom_features`. `features.py` is a small registry: each builder is decorated `@register("key", "per_asset"|"global")` and returns one or more named columns; the orchestrator `add_custom_features()` applies the requested keys and returns the added column names. `build_env_kwargs` then assembles `tech_indicator_list = indicators + features.expected_columns(...)`, so the new columns become extra rows of the env state automatically.
- Per-asset (one value per ticker per date → one state row each): `mom` (log-returns at 1/5/20/60d), `vol` (rolling std 5/20d), `bb_pctb`, `dist_high_20`, `meanrev_20`, `beta_60` (vs the benchmark), plus three suggested extras not in the paper — `xsec_mom_rank` (scale-free cross-sectional momentum rank), `downside_semidev_20`, `drawdown_60`.
- Global (one value broadcast across all tickers per date — the paper's market-wide signals): `vix`/`vix_chg5` (needs the `^VIX` download), `xsec_avg_ret`/`xsec_avg_ret_vol5`, `breadth`, `mkt_ret_5`/`mkt_ret_20` (benchmark cumulative returns).

Warmup NaNs (longest window 60d) are well inside the 252-day covariance drop, so no NaN reaches the env. To extend: add a builder, register it, list its key in config — no edits to stage 1.

**A.21.2 Article reward formulations.** Two new `reward_mode` values:
```
article_absolute:  r = return_scale·log(1+r_net) − λ_TO·turnover·100 − λ_conc·(HHI − 1/N)·100
article_benchmark: r = return_scale·(log(1+r_net) − log(1+r_bench)) − (same penalties)
```
`r_net` is the return already net of the real transaction cost; `HHI = Σwᵢ²` is concentration (`1/N` = equal weight). Coefficients live in `env.article_reward` (`return_scale=1000`, `λ_TO=0.003`, `λ_conc=0.1`). Turnover is charged TWICE on purpose, exactly as in the paper — once as the real cost inside `r_net`, once as the `λ_TO` shaping penalty. The old `log_return`/`diff_sharpe`/`diff_sortino` modes remain; `article_absolute` is the folder default.

**A.21.3 `turnover_mode` (naive vs drift-adjusted).** Folder 8 charged *drift-adjusted* turnover (`|w_target − w_drift|`, the real broker rebalancing trade) to match backtrader. The article uses *naive* turnover (`|w_target − w_previous_target|`). This folder defaults to **naive** for article fidelity, with `drift_adjusted` available. Rationale: the env is the article-faithful training signal; stage 4 (backtrader, with its 2% no-trade band) is the realistic execution check — so the env doesn't need to mimic the broker. The mode drives both the `r_net` cost deduction and the `λ_TO` penalty, in the env (`step()`) and in stage 3's `daily_return_from_weights()`.

**A.21.4 Configurable benchmark.** The `benchmark` block (`type: equal_weight | ticker`) defines ONE benchmark-return series that stage 1 stores as a `benchmark_return` column and that everything downstream reuses: the `article_benchmark` reward, the `beta_60` feature, the `mkt_ret` features, the stage-3 baseline line, the stage-4 overlay, and the stage-5 QuantStats benchmark (`utils.load_benchmark_returns` / `benchmark_label`). Default `equal_weight` (daily-rebalanced equal weight of the universe). With it, stage 3 shows an **EqualWeight** baseline instead of DJIA; `baselines.dji_ticker` is only a fallback.

**A.21.5 Cash as a synthetic asset (N+1).** Rather than surgery on the env's action/state/return code, cash is a synthetic ticker injected by stage 1 (`features.inject_cash_asset`): its price compounds at `cash.risk_free_rate` (so variance/covariance ≈ 0), per-asset features are zeroed, global features copied per date. Because cash is "just another ticker," the action space, softmax, covariance state, and return accounting all become N+1 automatically. The weight vector follows alphabetical ticker order, so `CASH` is NOT last — the env locates it by name (`cash_idx`). Downstream exclusions: stage 3 Min-Variance drops CASH (its zero variance would capture 100% of min-vol weight); stage 4 backtrader submits no CASH order (the broker holds the un-invested fraction) and **errors if `execution.cash_buffer ≠ 0`** (double-count guard); the risk-off gate routes to 100% CASH instead of zeroing weights. Set `cash.enabled=false` to revert to folder-8's fully-invested behaviour.

**A.21.6 Empirical findings (turnover tuning).** Tuning `λ_TO` / `λ_conc` produced a clear, somewhat sobering result, all diagnosable via `inspect_policy.py`:
- `λ_TO` is **threshold-like, not a smooth dial**: ≲ 0.5 → the policy churns (~400–660%/yr turnover, fidgeting around equal weight); ≳ 2 → it FREEZES to exactly equal weight (per-asset std ~0.009, ~130 trades over 11y, all from the risk-off gate). There may be no wide "cautious but active" middle.
- **The risk-off gate, not `λ_TO`, dominates turnover** when the policy is sticky: ~90% of turnover on a frozen policy is the gate flipping to cash and back (non-learned, immune to `λ_TO`).
- **The allocation signal is weak.** With `λ_conc=0` (zero concentration penalty — nothing forcing diversification), the time-averaged risky weights are *still* equal weight (cross-sectional std ~0.003). Given full freedom to concentrate, the agent declines — it has no persistent per-asset edge. This reproduces the paper's own central finding (RL barely beats equal weight) on this universe. The strategy's edge over plain equal weight comes mainly from the gate's cash timing, not from asset selection.

Practical upshot: govern executed-trade count with the execution band (`min_weight_delta`, no retrain) and the gate threshold; treat `λ_TO`/`λ_conc` as coarse regime switches, not fine dials; and for any clean comparison use multi-seed (single-seed run-to-run variance swamps the coefficient effect in the low-`λ_TO` range).

**A.21.7 New files and the tuning loop.**

| File | Role |
|---|---|
| `features.py` | Feature-builder registry + `add_benchmark_return` + `inject_cash_asset` |
| `inspect_policy.py` | Reads `results/weights_ppo.csv`; reports per-asset std, turnover, regime drift, cash-by-year, a combined ACTIVE/FROZEN verdict, and a stale-weights guard |
| `utils.resolve_indicator_list` | Combines indicators + custom feature columns for the env state |
| `utils.load_benchmark_returns` / `benchmark_label` | Shared benchmark series/label for stages 3/4/5 |

The mandatory tuning cycle (each λ change): **edit `config.json` → `02_train.py` → `03_backtest.py` → `inspect_policy.py`**. `λ_TO`/`λ_conc` only affect TRAINING, so re-running stage 3 alone (or `inspect_policy` alone) shows the *previous* policy — `inspect_policy.py` prints a loud STALE-WEIGHTS warning when `weights_ppo.csv` is older than the newest model to catch exactly this.

### A.22 Per-asset shared encoder (optional policy)

**Why.** The A.21.6 diagnostics showed the default policy emits near-equal weights *uncorrelated with the per-asset features* (`inspect_policy.py`: weight↔`mom_20` correlation ≈ 0) — under *every* reward. Root cause is architectural, not the objective: SB3's default `MlpPolicy` flattens the (n_rows, n_assets) state into one long vector, scattering each asset's feature column across the input and diluting it among the 121 covariance entries. The network has no notion of "asset", so the easy optimum is constant logits → equal weight.

**The state's structure.** Column *j* of the (39, 11) observation is asset *j*'s full feature vector (its covariance row + 28 indicators). The fix keeps that structure instead of flattening it away.

**`PerAssetSharedEncoder` (`policies.py`).** A custom SB3 `BaseFeaturesExtractor` selected via `policy_kwargs`:
```
obs (B, 39, 11) → transpose → (B, 11 assets, 39 features)
  → shared MLP f: 39 → hidden → hidden → emb_dim   (ONE set of weights, applied to every asset)
  → (B, 11, emb_dim)
  → [optional] subtract cross-asset mean (cross-sectional comparison)
  → flatten → (B, 11·emb_dim)  = features
```
With `emb_dim=1` it emits one score per asset (features_dim = 11). The shared weights mean the network learns a single "features → score" rule (updated with 11 assets' worth of gradient per step) and asset *j*'s score depends only on asset *j*'s features — so it *can* differentiate. This is the per-asset / cross-sectional half of the article's encoder, minus the expensive learned temporal part (see `Articles/2605.17307v1_review_and_comparison.md` for why the temporal LSTM/Transformer is deferred: ~1000× compute, a full observation rewrite, more overfitting, and the paper's own results show it barely beats equal-weight).

**How to enable.** Default is `models.ppo.policy_kwargs: null` (plain MLP, so prior results reproduce). Copy the `_policy_kwargs_example` block into `policy_kwargs`:
```json
"policy_kwargs": {
  "features_extractor": "per_asset",
  "features_extractor_kwargs": { "hidden": 32, "emb_dim": 1, "cross_sectional": true, "layers": 2 },
  "net_arch": { "pi": [], "vf": [64, 64] }
}
```
`net_arch.pi=[]` keeps the policy head a minimal `Linear(11→11)` on the already-per-asset, cross-sectionally-centred scores; `vf=[64,64]` gives the value head capacity. `parse_policy_kwargs` resolves `features_extractor: "per_asset"` to the class; `null`/`"mlp"` keep the default.

**Caveat.** SB3's final action head is still a dense `Linear`, so a small residual cross-asset mix remains at the very last layer (with `emb_dim=1` it's an 11×11 map the optimiser can drive toward identity). A fully per-asset action head needs a custom policy *class*; this extractor is the cheap, `policy_kwargs`-selectable first step.

**A/B test.** Train with `policy_kwargs: null` and again with the per_asset block, then compare `inspect_policy.py`. The decisive signal is the weight↔feature correlation and the within-day spread: if they move meaningfully off ~0, the per-asset structure unlocked signal-driven allocation. If they *still* collapse to equal-weight even with the right architecture, that's strong evidence the per-asset edge genuinely isn't there for these 10 ETFs daily (the paper's conclusion, now properly earned rather than confounded by a weak network). It only affects TRAINING, so use the full cycle: edit → `02_train` → `03_backtest` → `inspect_policy`.
