# Live trading guide — how to use the model day to day

This document is the operations manual for taking the multi-seed PPO ensemble into production. It covers:

1. The deployment recipe (one-time setup with candidate-pool training).
2. The daily intraday workflow (run before 4 pm market close).
3. The monthly retraining cadence.
4. Risk monitoring and the kill-switch.
5. Failure modes and what to do.
6. Paper trading phase.

The model itself, the env, the reward shaping, the risk-off gate, and the multi-seed ensemble are all described in `README.md` and `multi_seed_design_notes.md`. **This document focuses purely on running the model live.**

---

## 0. Mental model

### 0.1 Two pipelines: validation vs deployment

There are two distinct configs / pipelines in this folder:

| Pipeline                       | Config file                | Purpose                                                                          |
|---|---|---|
| Walk-forward validation        | `config.json`              | Out-of-sample evaluation across 7 historical windows. Proves the recipe is robust. |
| Production deployment          | `config_production.json`   | Single-split training on ALL available data. Produces the model that trades live. |

- The walk-forward pipeline is your **methodology validation**. Re-run it whenever you change hyperparameters, the reward shape, or the universe — it tells you whether the recipe still generalises.
- The production pipeline is your **deployment binary**. It trains a candidate pool of seeds on the full history, with no held-out test slice (you've already done OOS testing via walk-forward). A subset of the trained candidates becomes the live ensemble.

**Never trade with a walk-forward window model directly.** The last walk-forward window's model was trained on data ending in 2020-12 — it has not seen the regimes from 2021 onwards. Use `config_production.json` to train a fresh model on everything.

### 0.2 Candidate pool + filter — why we don't just train 3 seeds

PPO training is **initialisation-dependent**. Some seeds escape the near-uniform initial policy and learn a meaningful allocation. Other seeds stay stuck — the saved checkpoint is from the very first evaluation (step 2,500), essentially an untrained policy whose softmax happens to land at `1/N` per asset. These stuck checkpoints would dilute the ensemble back toward uniform.

The fix is structural: **train a larger candidate pool than you need, then automatically select the best-converged subset for deployment.** With 8 candidates trained, you typically get 5–7 converged and 1–3 stuck; the filter keeps only the top N by convergence quality, where N = `seeds.ensemble_size`. The stuck seeds' compute is sunk cost, but the resulting ensemble is what you want: N independently-learned policies whose averaging actually reduces variance.

This is controlled by **two distinct config knobs**:

| Knob                       | Where in config                  | Role                                                                                 |
|---|---|---|
| `seeds.list`               | candidate pool                   | The seeds that get trained. Stage 2 processes every entry. Typical: 6–8 seeds.       |
| `seeds.ensemble_size`      | deployment ensemble size         | How many of the trained candidates to use at inference. Picks the top N by convergence. Typical: 3–5. |

You only change `seeds.list` when you want to retrain. You can change `seeds.ensemble_size` at any time with no retraining — the inference scripts re-rank the existing trained models each run.

---

## 1. One-time setup: train the production ensemble

### 1.1 Update `config_production.json`

Bump `data.train_end_date` to the most recent fully-closed trading day (yesterday is safe; check Yahoo has a bar at that date). Also update `data.trade_start_date` and `data.trade_end_date` to the same value (the trade slice is intentionally empty — single-split mode does not use `data/trade_data.pkl`).

```
"data": {
  ...
  "train_end_date":   "2026-05-20",
  "trade_start_date": "2026-05-20",
  "trade_end_date":   "2026-05-20",
  ...
}
```

Confirm the seeds block looks like:

```
"seeds": {
  "list":          [42, 1338, 17, 100, 999, 12345, 31415, 7777],
  "ensemble_size": 3
}
```

8 candidates trained, top 3 deployed. Leave the rest of the config alone — every other hyperparameter is the one validated by the walk-forward run.

### 1.2 Download data

```bash
python 01_get_data.py --config config_production.json
```

Produces:
- `data/full_data.pkl` — full processed history with indicators + turbulence + rolling covariance.
- `data/train_data.pkl` — slice from 2009-01-01 to `train_end_date`.
- `data/trade_data.pkl` — empty (single-split deployment).

### 1.3 Train the candidate pool

```bash
python 02_train.py --config config_production.json
```

This trains 8 independent PPO models — one per seed in `seeds.list`. For each:
- Train slice is split into train_only (first 90% of dates) + validation (last 10%, ~1.7 years).
- Trains with `total_timesteps=150_000` and ES on validation Sharpe (`eval_freq=2500`, `patience=20`, `min_delta=0.001`).
- Saves the best-by-val-Sharpe checkpoint as `models/agent_ppo_s<seed>.zip` + `models/vecnormalize_ppo_s<seed>.pkl`.
- Writes a per-seed validation trajectory to `models/agent_ppo_s<seed>.history.json`.

**Wall-clock**: ~7–15 minutes for 8 seeds. Counter-intuitively, this is fast because most stuck seeds ES early (50k–85k steps after the patience counter exhausts). Only 1–2 seeds typically train the full 145k. Don't be alarmed if it finishes faster than you expect — that's the candidate-pool strategy working as designed.

### 1.4 Inspect convergence with `filter_seeds.py`

```bash
python filter_seeds.py
```

You'll see a table like:

```
   seed  evals  imp     best  best@   last@    delta  status
   ----  -----  ---  ------- ------ -------  -------  ------
     42     58   11    1.973  95000  145000   +0.022  converged
    999     57   10    1.841  92500  142500   -0.044  converged
  31415     29    7    1.860  22500   72500   -0.016  converged
   1338     27    5    1.698  17500   67500   -0.063  mild
     17     23    3    1.729   7500   57500   -0.103  mild
    100     23    3    1.719  17500   57500   -0.172  mild
  12345     34    3    1.667  35000   85000   -0.247  mild
   7777     21    1    1.722   2500   52500   -0.228  STUCK
```

Healthy training run looks like:
- **At least N ≥ ensemble_size seeds with `imp ≥ 3`** (converged).
- The top N have `best@ ≥ 20,000` (the model's best checkpoint came AFTER meaningful training, not at step 2,500 = first eval = untrained baseline).
- At most 1–2 "STUCK" seeds at the bottom (best@ = 2,500).

**Unhealthy training run**: fewer than `ensemble_size` converged seeds. If that happens, see section 5.3 ("not enough converged seeds").

### 1.5 Verify ensemble health with `inspect_ensemble.py`

```bash
python inspect_ensemble.py --config config_production.json
```

This reuses the inference pipeline (fresh Yahoo pull, full feature recomputation) and prints side-by-side weights for the **selected ensemble** (top `ensemble_size`), plus pairwise disagreement metrics.

What you want to see:
- **All seeds produce non-uniform weights** — top holding > 15%, smallest holding < 8%. A uniform portfolio scores 11.1% per asset for a 9-ticker universe; a learned policy should depart from this clearly.
- **Pairwise L1 distances in the 0.15–0.35 range** — the "sweet spot" labelled `close` to `diverging` in the script's output. Too small (< 0.10) means seeds converged on identical policies (ensemble adds nothing); too large (> 0.50) means seeds are random.
- **Top-3 overlap of at least 1 ticker** across all seeds — there's some shared consensus on the strongest bet.
- **Verdict**: `SEEDS AGREE` or `MODERATE DISAGREEMENT` is fine for deployment. `SEEDS DISAGREE STRONGLY` means something is off — retrain.

### 1.6 No stage 3 backtest

Skip `03_backtest.py` — the trade slice is empty in `config_production.json` so there's nothing to test against. Your OOS metrics come from the walk-forward config. Stage 3 is only for the validation pipeline.

### 1.7 Smoke-test the inference pipeline

```bash
python predict_tomorrow.py --config config_production.json
```

Should print at the top:

```
ensemble_size=3: using 3 of 8 candidate seeds.
  selected: [42, 999, 31415]
  skipped:  [1338, 17, 100, 12345, 7777]
asof: 2026-05-22   algo: ppo   seeds: [42, 999, 31415]
```

Then later:

```
=== Target portfolio for execution at the 2026-05-22 close ===
  Ensemble: 3 seeds ([42, 999, 31415])    algo: PPO
  Turbulence on 2026-05-22: 4.23   threshold: 80.00   risk_off: no
  Holding period: 2026-05-22 close -> next trading day close

  ticker     target     model
  --------  --------  --------
  EEM          9.5%     9.5%
  GLD         19.8%    19.8%
  IVE          9.6%     9.6%
  ...
  SUM        100.0%   100.0%
```

If anything looks wrong (NaNs, weights not summing to 1, missing tickers), debug before deploying. Otherwise the output CSV at `results/target_weights_<YYYYMMDD>.csv` is ready.

---

## 2. Daily workflow (live trading)

This is the loop you run every trading day from now on.

### 2.1 Pre-market (optional, before 9:30 am ET)

Nothing required. You can optionally pre-run `predict_tomorrow.py` to see what the model would have suggested at yesterday's close.

### 2.2 Intraday — between 3:30 pm and 3:55 pm ET

1. **Run the prediction**:

   ```bash
   cd "/Users/paolobortolotti/FinRL-Tutorials/1-Introduction/PortfolioAllocation_paolo_impruvements/8.Multi-seed training and ensembled actions"
   python predict_tomorrow.py --config config_production.json
   ```

   The script will:
   - Pull the last ~500 calendar days of OHLC from Yahoo for the 9 ETFs (yfinance's intraday quote for today is treated as the close).
   - Recompute indicators + turbulence + rolling 252-day covariance.
   - Auto-select the top `ensemble_size` trained seeds by convergence quality.
   - Load each selected seed's model + VecNormalize stats.
   - Roll each model deterministically through the recent history; capture the LAST action vector (the one for today → tomorrow).
   - Softmax each seed's raw action → per-seed weight vector.
   - Average across seeds → ensemble model weights.
   - Apply the risk-off gate: if today's turbulence > threshold (80.0 by default), override target weights to zero (= cash).
   - Print a summary and save `results/target_weights_<YYYYMMDD>.csv`.

   Total runtime: ~30–90 seconds depending on Yahoo's responsiveness.

2. **Sanity-check the output**:
   - Weights sum to 1.00 (or 0.00 if risk-off triggered).
   - Top holding ≤ 30% (excessive concentration suggests something's off).
   - `turbulence` is a reasonable number (typically 5–50 in calm regimes, 80–200 during stress).
   - The "risk_off" flag matches your expectation given the turbulence value.
   - The "selected" seed list matches what you expect from `filter_seeds.py`.

3. **Compute the rebalance trade list**: compare today's target weights to your current portfolio's actual weights.

   For each ticker:
   - `desired_dollar = target_weight × total_portfolio_value`
   - `current_dollar = current_position × current_price`
   - `delta_dollar = desired_dollar - current_dollar`
   - `delta_shares = round(delta_dollar / current_price)`

   Sells first, then buys (to free up cash).

4. **Execute orders before 4 pm**:
   - **MOC (Market On Close)** orders: submit before the broker's MOC cutoff (usually 3:50 pm ET for retail). The exchange fills at the official 4 pm closing price. **This is the recommended execution method** because it matches the backtest assumption that you trade at the close.
   - **VWAP in the last 15 minutes**: alternative if MOC isn't available.
   - **Avoid market orders earlier in the day**: you'll drift from the intraday-derived target.

5. **Record the predicted weights**: keep the day's `target_weights_<YYYYMMDD>.csv` file.

### 2.3 Post-close (after 4 pm)

1. **Verify fills**: check that all orders filled at or near the 4 pm close.
2. **Record actual weights** (after end-of-day mark-to-market).
3. **Log to a tracking spreadsheet/database**: date, target weights, executed weights, total portfolio value, daily PnL, daily return.

---

## 3. Retraining cadence

Retrain monthly or quarterly. The walk-forward setup used a 1-year slide, so the model has been validated under the assumption that "training data ≤ 12 months stale" is acceptable.

- **Conservative**: retrain monthly. ~10–15 min/month of compute.
- **Balanced**: retrain quarterly. ~10–15 min/quarter.
- **Aggressive**: retrain every six months. Risk: 6+ months without new training data is more stale than the walk-forward validated.

### 3.1 Retraining procedure

1. Edit `config_production.json`:
   - `data.train_end_date` → today's date - 1 day.
   - `data.trade_start_date` → same.
   - `data.trade_end_date` → same.

2. Re-run stages 1 and 2:
   ```bash
   python 01_get_data.py --config config_production.json
   python 02_train.py --config config_production.json
   ```

3. The new `models/agent_ppo_s<seed>.zip` files overwrite the old ones.

4. Verify convergence and ensemble health:
   ```bash
   python filter_seeds.py
   python inspect_ensemble.py --config config_production.json
   ```

5. If the retrain has at least `ensemble_size` converged seeds AND the ensemble verdict is not `SEEDS DISAGREE STRONGLY`, you're good. Otherwise see section 5.3.

6. From the next trading day, `predict_tomorrow.py` automatically uses the newly trained models.

### 3.2 When to ALSO re-validate the recipe

If you've been retraining for 6–12 months and want confidence the recipe still generalises:

1. Re-run the walk-forward pipeline (`config.json`, not the production config):
   ```bash
   python 02_train.py --config config.json
   python 03_backtest.py --config config.json
   ```
2. Check the latest walk-forward window's eval metrics (Sharpe, max drawdown, total return) against the historical distribution from previous walk-forward runs.
3. If metrics are within the historical range, the recipe is still working. If they've degraded significantly, investigate before continuing to deploy.

### 3.3 Changing `ensemble_size` does NOT require retraining

If you decide 3 isn't enough smoothing, you can raise `ensemble_size` to 4, 5, etc. — up to `len(seeds.list)`. No retraining; the inference scripts re-rank the existing trained models each run. Lower it back to 3 the same way.

---

## 4. Risk monitoring and the kill switch

Live performance must be tracked against the backtest distribution. If live performance materially diverges, stop trading and investigate.

### 4.1 Metrics to log daily

| Metric                    | How to compute                                                           |
|---|---|
| Daily return              | `(end_value - start_value) / start_value`                                |
| 60-day rolling Sharpe     | `sqrt(252) × mean(daily_returns_last_60) / std(daily_returns_last_60)`   |
| Rolling max drawdown      | Peak-to-trough drawdown over the last 252 days                           |
| Realised turnover         | `0.5 × sum(|target_weight - actual_weight_yesterday|)` per day           |

### 4.2 Reference distribution from walk-forward

Pull the per-window metrics from `results/equity_curves.csv` (produced by the walk-forward `03_backtest.py`). For each window, compute annualised Sharpe and max drawdown. The 7 walk-forward windows give a distribution of "what reasonable looks like".

### 4.3 Kill switch criteria

**Stop trading immediately if any of the following happens:**

1. **60-day live Sharpe drops below the minimum walk-forward window Sharpe.**
2. **Current drawdown exceeds the worst walk-forward window's max drawdown by more than 5 percentage points.**
3. **Three consecutive months of negative returns**, regardless of Sharpe.
4. **Risk-off gate triggers but you don't see the expected protection.**
5. **predict_tomorrow.py output looks degenerate** for multiple consecutive days (single ticker > 60%, NaNs, sum != 1).

**Halt procedure**:
1. Stop running `predict_tomorrow.py`.
2. Hold the current portfolio.
3. Investigate: pull recent logs, check Yahoo data integrity, re-run the walk-forward, compare hyperparameters.
4. Resume only after identifying and fixing the cause, AND running the walk-forward on current data successfully.

---

## 5. Failure modes and what to do

### 5.1 Yahoo download fails or returns stale data

**Symptom**: `predict_tomorrow.py` prints a "latest available bar is YYYY-MM-DD" warning where the date is not today.

**Action**: re-run after 60 seconds. If still failing after several retries and Yahoo's UI is up, the API issue is transient. If Yahoo itself is down, do not trade today (skip the rebalance; hold current positions).

### 5.2 A model file is missing

**Symptom**: `FileNotFoundError: models/agent_ppo_s<seed>.zip not found`.

This means the auto-filter picked a seed that exists in `seeds.list` but wasn't trained. Either retrain (`python 02_train.py --config config_production.json`) or remove the missing seed from `seeds.list`.

### 5.3 Not enough converged seeds after retraining

**Symptom**: `filter_seeds.py` shows fewer than `ensemble_size` seeds with `imp ≥ 3`.

**Diagnosis**: The candidate pool was too small for this particular run's initialisation luck.

**Actions in order of preference**:

1. **Expand the candidate pool**. Add 4–8 more seeds to `seeds.list` (e.g., `[42, 1338, 17, 100, 999, 12345, 31415, 7777, 271828, 161803, 222, 8675309]`) and retrain. The filter will still pick the best top-N. ~1–2 minutes per added seed.

2. **Lower `ensemble_size` temporarily**. If you have 2 converged seeds but ensemble_size=3, drop to `ensemble_size: 2` for now. This is a stopgap — the ensemble is now noisier.

3. **Try different seeds**. If certain seeds consistently get stuck across retrains, replace them in `seeds.list` with new candidates. Stuck-ness is initialization-deterministic, so the same seed will stay stuck across retrains with the same data and hyperparameters.

4. **Inspect what's "stuck" means today**. Run `inspect_ensemble.py` and look at the raw weights for stuck seeds — if they're producing near-uniform 10.8% per asset, they're stuck. If they're producing concentrated-but-bad bets, that's a different problem (could be a regime issue, not a training issue).

### 5.4 Output looks degenerate (single ticker > 60%, NaNs, sum != 1)

**Symptom**: the printed summary is weird.

**Actions**:
- Check today's turbulence value. If extremely high, the gate should have triggered (target weights would all be 0).
- Check Yahoo data integrity for a particular ticker (a missing or zero close in the last 252 days corrupts the covariance).
- Re-run `inspect_turbulence.py` to see the recent turbulence distribution.
- Run `inspect_ensemble.py` to see which seed is producing the degenerate output — if only one seed is bad, the ensemble averaging should mask it; if all three are bad, do not trade today.

### 5.5 Live performance drops below the kill-switch threshold

See section 4.3.

### 5.6 You miss the 4 pm cutoff

**Action**: hold yesterday's portfolio. Do not run `predict_tomorrow.py` after the close intending to execute tomorrow — the prediction is for "yesterday close → today close" and is one day stale by tomorrow morning.

### 5.7 Tax / wash-sale considerations

The strategy has high turnover (often 50–200% annualised on a 9-ETF universe). In a taxable account this can produce significant short-term capital gains. Consider:
- Running the strategy inside a tax-advantaged account (IRA, 401(k)) where churn doesn't trigger immediate tax events.
- Extending the holding period (e.g., rebalance weekly instead of daily) — but this requires re-validating via the walk-forward pipeline.

---

## 6. Paper trading phase (recommended before going live with real money)

Run the daily workflow for 30–90 days without executing real trades. Each day:

1. Run `predict_tomorrow.py`.
2. Save the predicted weights.
3. The next morning, look up the actual close prices that filled.
4. Compute the "what would have happened" portfolio return.
5. Compare to the walk-forward backtest's distribution.

After 60 days of paper trading you'll have ~60 daily returns. Compute live (paper) Sharpe, average daily turnover, worst day return. If these are inside the walk-forward distribution → go live with a small allocation (e.g., 5–10% of investable capital).

---

## 7. Quick reference

### 7.1 Commands

| Action                          | Command                                                            |
|---|---|
| One-time: refresh data          | `python 01_get_data.py --config config_production.json`            |
| One-time: train candidate pool  | `python 02_train.py --config config_production.json`               |
| One-time: rank seed convergence | `python filter_seeds.py`                                           |
| One-time: verify ensemble health| `python inspect_ensemble.py --config config_production.json`       |
| Daily: predict tomorrow         | `python predict_tomorrow.py --config config_production.json`       |
| Inspect today's turbulence      | `python inspect_turbulence.py`                                     |
| Re-validate recipe              | `python 02_train.py --config config.json && python 03_backtest.py --config config.json` |

### 7.2 Files

| File                                              | What it is                                                 |
|---|---|
| `config_production.json`                          | Single-split deployment config (candidate pool + ensemble_size) |
| `config.json`                                     | Walk-forward validation config (keep for revalidation)     |
| `models/agent_ppo_s<seed>.zip`                    | Trained PPO model per seed (production)                    |
| `models/vecnormalize_ppo_s<seed>.pkl`             | Saved VecNormalize stats per seed                          |
| `models/agent_ppo_s<seed>.history.json`           | Validation Sharpe trajectory per seed (input to filter)    |
| `results/target_weights_<YYYYMMDD>.csv`           | The day's predicted target weights                         |
| `results/target_weights_latest.csv`               | Copy of the most recent prediction                         |

### 7.3 Config knobs you might tune

| Knob                              | Default | Effect                                                                          |
|---|---|---|
| `seeds.list`                      | 8 seeds | Candidate pool size. More = more chances to find converged seeds. Costs train time. |
| `seeds.ensemble_size`             | 3       | How many trained candidates participate in the deployed ensemble. No retrain.   |
| `early_stopping.val_fraction`     | 0.1     | Fraction of training data reserved for ES validation Sharpe (last 10% = ~1.7y). |
| `early_stopping.eval_freq`        | 2500    | Steps between validation Sharpe evaluations.                                    |
| `early_stopping.patience`         | 20      | ES gives up after this many evals with no improvement.                          |
| `risk_off.turbulence_threshold`   | 80.0    | Daily turbulence above this → go to cash. Validated on walk-forward.            |

**Do not tune** `env.reward_mode`, `env.transaction_cost_penalty`, PPO model_kwargs, `normalization.*`, or `data.indicators` without re-running the full walk-forward to re-validate.

### 7.4 Healthy run looks like

After `python 02_train.py`:
- Wall-clock 7–15 min for 8 candidates.
- `filter_seeds.py` shows ≥ 3 seeds with `imp ≥ 3` and `best@ ≥ 20000`.
- `inspect_ensemble.py` verdict is `SEEDS AGREE` or `MODERATE DISAGREEMENT`.
- All seeds in the deployed ensemble produce non-uniform weights (top holding > 15%).
- Pairwise L1 distances 0.15–0.35.
- At least 1 ticker appears in all selected seeds' top-3 (some consensus).

If any of these fail, investigate before deploying.

---

## 8. What this guide does NOT cover

- **Broker integration**: you have to wire up order submission to your broker yourself. The script outputs CSV.
- **Position sizing for live capital**: the backtest assumes $1M or $100M; for retail, start much smaller.
- **Tax accounting**: see section 5.7.
- **Regulatory considerations**: depending on your jurisdiction and account size, algorithmic trading may require specific disclosures.
- **Hardware availability and failover**: the script runs on your local machine. If your machine is down at 3:55 pm, you miss the rebalance. Consider a small cloud VM for production reliability.

---

**Bottom line**:
- Train 8 candidates → filter picks top 3 → deploy.
- Run `predict_tomorrow.py` every trading day between 3:30 pm and 3:55 pm ET.
- Submit MOC orders to reach the target weights before 4 pm.
- Retrain monthly (or quarterly). Verify convergence with `filter_seeds.py` and `inspect_ensemble.py` after every retrain.
- Watch the kill-switch metrics. Paper trade first.
