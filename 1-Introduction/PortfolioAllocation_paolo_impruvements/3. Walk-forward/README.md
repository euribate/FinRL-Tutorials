# 3.Walk-forward — rolling train/eval windows (improvement #3 on top of #1 + #2)

Variant of the base pipeline (`../0.original/`) that layers improvement #3 on top of improvements #1 and #2:
- **#1** (from `../1.reward_returns/`): reward = `log(1 + portfolio_return) * reward_scaling`.
- **#2** (from `../2.transaction_cost/`): subtract per-unit-turnover TC penalty from both reward and book equity.
- **#3** (new here): instead of one fixed train (2009-07 → 2020-07) / trade (2020-07 → 2021-10) split, **train a fresh model per rolling window** and stitch the per-window out-of-sample predictions into one continuous backtest.

Default sliding windows (auto-generated from `walk_forward.auto`):
```
window 0:  train 2009-01-01..2014-12-31   eval 2015-01-01..2015-12-31
window 1:  train 2010-01-01..2015-12-31   eval 2016-01-01..2016-12-31
window 2:  train 2011-01-01..2016-12-31   eval 2017-01-01..2017-12-31
...
window 6:  train 2015-01-01..2020-12-31   eval 2021-01-01..2021-10-31
```
Each window's eval slice only ever sees a model trained on data strictly preceding it — no look-ahead. The 7 per-window equity curves are stitched in time order to form a single continuous out-of-sample backtest spanning ~7 years instead of 16 months.

Activated by `walk_forward.enabled = true` in `config.json` (default in this folder). Setting it to `false` falls back to the single-split behaviour of `../2.transaction_cost/`.

**Stage 4 (backtrader) is currently NOT walk-forward aware.** It loads only the last window's model and replays the full out-of-sample period through it. See appendix A.13 for the rationale and a sketch of what full integration would require.

---

## Folder layout

```
PortfolioAllocation_paolo_impruvements/
    config.json                  data + env + per-algorithm hyperparameters
    backtrader_config.json       backtrader-specific replay settings
    quantstats_config.json       quantstats report settings
    utils.py                     shared helpers (config IO, env_kwargs, covariance)
    01_get_data.py               stage 1: download + indicators + covariance + split
    02_train.py                  stage 2: train each algo flagged use=true
    03_backtest.py               stage 3: backtest + Min-Variance + DJIA baselines
    04_backtrader_replay.py      stage 4: replay one agent through backtrader
    05_quantstats_report.py      stage 5: QuantStats HTML tearsheet
    README.md                    this file
    data/                        train_data.pkl, trade_data.pkl       (created)
    models/                      agent_<algo>.zip                     (created)
    results/                     equity_curves.csv, equity_plot.png   (created)
    results_backtrader/          backtrader outputs                   (created)
    results_quantstats/          report.html, metrics.csv             (created)
    tensorboard/                 TB logs per algo                     (created)
```

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

---

## `config.json` — what each section controls

### `data`
- `ticker_list` — the 30 Dow Jones names (one or two may fail to download from Yahoo on a given day; the env auto-drops them).
- `download_start_date` — must be ≥ 1 year before `train_start_date`, otherwise the 252-day covariance loop produces no rows.
- `train_start_date` / `train_end_date` — training slice.
- `trade_start_date` / `trade_end_date` — out-of-sample backtest slice.
- `indicators` — eight technical indicators added per ticker per day.
- `lookback` — covariance window in trading days (default 252).
- `use_turbulence` — kept false to match the notebook.

### `env`
Direct passthrough to `StockPortfolioEnv` (or `LogReturnPortfolioEnv` when `reward_mode = "log_return"`):
- `hmax` — max share units per trade (unused once softmax normalisation is applied, but kept for env contract).
- `initial_amount` — starting cash.
- `transaction_cost_pct` — passthrough to the upstream env constructor. The upstream env stores it but never uses it in the active reward/value computation, so it's effectively dead config. Cost shaping is controlled by `transaction_cost_penalty` below; cost realisation in stage 4 is controlled by `backtrader_config.json`. Kept here because the upstream constructor requires it.
- `reward_scaling` — multiplicative scaling of the reward signal. Default in this folder is **1.0** because `reward_mode = "log_return"` is also the default; the upstream-compatible value would be `1e-4`.
- `reward_mode` — `"value"` (upstream `self.reward = new_portfolio_value`) or `"log_return"` (our `self.reward = log(1 + portfolio_return) * reward_scaling`). Default in this folder is `"log_return"`. See appendix A.11 for the rationale.
- `transaction_cost_penalty` — per-unit-turnover cost rate used to shape the reward when `reward_mode = "log_return"`. Default in this folder is **0.001**. Set to **0.0** to disable the penalty and reproduce `../1.reward_returns/` exactly. See appendix A.12 for the math, semantics, and how it interacts with the broker-level costs in stage 4. Combining `reward_mode="value"` with a non-zero penalty prints a warning and the penalty is ignored (this combination is out of scope for the current implementation).

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

### A.14 What was deliberately *not* changed

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
