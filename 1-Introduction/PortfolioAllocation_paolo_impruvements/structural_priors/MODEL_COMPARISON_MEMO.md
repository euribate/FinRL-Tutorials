# Model Comparison Workflow — Memo

How to run the 5-algorithm × 3-cadence experiment, modify it, interpret the
two output CSVs, and what to do when it breaks.

This memo is the practical operating manual for `experiments.json` +
`run_experiments.py` + `replay_cadence_sweep.py`. See `EXPERIMENTS_GUIDE.md`
for the generic explanation of how the runner works; see `METHODOLOGY.md`
for the theory.

---

## TL;DR — what this experiment answers

Two questions in one run:

1. **Which RL algorithm performs best on this universe?**
   PPO vs A2C vs DDPG vs TD3 vs SAC, all on the same per-asset encoder,
   same hyperparameters where comparable, same training budget, same prior.
2. **Does rebalancing less often help any of them?**
   Daily / weekly / monthly cadence, applied at backtest (no retraining).

Output: a 5 × 3 matrix of `bt_sharpe` / `bt_max_dd` / `n_trades` plus a
single 5-row algo ranking from the env evaluator.

---

## The moving pieces

| File | Role |
|---|---|
| `config.json` | Base config. `models.<algo>` blocks hold each algorithm's hyperparameters (aligned where possible). `policy_prior`, `risk_off`, etc. The base config is what each experiment is built ON TOP of via overrides. |
| `experiments.json` | List of experiments. Each entry has a `name` and an `overrides` dict (dotted paths into `config.json`). Model-switch experiments flip `models.<algo>.use` between `true` and `false`. |
| `backtrader_config.json` | Stage-4 settings, including the `rebalance.cadence` knob (`daily` / `weekly` / `monthly`). Cadence here applies to the LIVE backtest invoked by the runner. |
| `run_experiments.py` | The training runner. For each experiment: derives a per-experiment config, runs stages 2 (train) / 3 (env backtest) / 4 (backtrader), collects metrics into `experiments_results.csv`. Knows the active algo (`active_algo(cfg)`) so it picks the right column / weights file. |
| `replay_cadence_sweep.py` | Cadence sweep at backtrader-time. For each finished model, runs stage 4 three times (daily / weekly / monthly), writes results into `experiments/<name>/results_backtrader_<cadence>/`, and aggregates everything into `model_cadence_results.csv`. No retraining. |
| `experiments_results.csv` | One row per experiment. Columns: `name`, `algo`, `env_sharpe`, `env_eqw_sharpe`, `env_sharpe_minus_eqw`, `env_cum_return`, `env_maxdd`, `ann_turnover`, `risky_std`, `regime_drift`, plus `bt_sharpe`, `bt_maxdd`, `n_trades` when `--with-backtrader`. |
| `model_cadence_results.csv` | One row per (model, cadence) cell. Columns: `model`, `cadence`, `bt_sharpe`, `bt_cum_return`, `bt_max_dd`, `n_trades`. |

---

## Modifying the experiments

### Adding or removing a model

Each model has a switch entry in `experiments.json` like:

```json
{ "name": "model_ppo",
  "overrides": {
    "models.ppo.use":  true,
    "models.a2c.use":  false,
    "models.ddpg.use": false,
    "models.td3.use":  false,
    "models.sac.use":  false
  }
}
```

- **Remove a model from the comparison**: delete its `model_<algo>` block
  from `experiments.json`. The runner won't train it.
- **Add a new switch experiment** (e.g. a custom PPO variant): copy a
  block, give it a unique `name`, change the relevant `models.<algo>.use`
  flag, and add any other overrides you want for that variant.

### Changing a model's hyperparameters

Edit the relevant `models.<algo>.model_kwargs` block in `config.json`. The
algorithms are aligned on:
- `total_timesteps` (150,000)
- `learning_rate`   (0.0002)
- `gamma`           (0.999)
- `batch_size`      (128, except A2C which has no batch_size knob)
- the per-asset encoder (`hidden=128, emb_dim=1, layers=2`)
- a 64×64 critic head (`vf` for PPO/A2C, `qf` for DDPG/TD3/SAC)

Anything algorithm-specific (clip_range for PPO, target_policy_noise for
TD3, etc.) uses SB3 defaults. **If you change any of the aligned anchors,
do it in `config.json` and the new value applies to every algo
simultaneously** — preserving the apples-to-apples nature of the comparison.

### Per-experiment hyperparameter tweaks

Override the dotted path in the experiment entry. Example: try PPO at
half the learning rate:

```json
{ "name": "model_ppo_lr_half",
  "overrides": {
    "models.ppo.use": true, "models.a2c.use": false,
    "models.ddpg.use": false, "models.td3.use": false, "models.sac.use": false,
    "models.ppo.model_kwargs.learning_rate": 0.0001
  }
}
```

### Adding a grid (alpha sweep on the best model, etc.)

Add a block to the `grid` array in `experiments.json`. Grids auto-name
cells from the swept value, so use scalars (not lists). See
`EXPERIMENTS_GUIDE.md` § "Pattern B — grid" for syntax.

---

## Running the experiments — full workflow

### Step 0 — clean state (optional but recommended)

```bash
pkill -f run_experiments.py ; sleep 2
ps aux | grep -E "run_experiments|02_train" | grep -v grep   # should be empty
rm -rf experiments/model_*
rm -f experiments_results.csv model_cadence_results.csv model_sweep.log nohup.out
```

### Step 1 — verify config (1 command)

```bash
python -c "import json; c=json.load(open('config.json')); bt=json.load(open('backtrader_config.json')); print('reward_mode:',c['env']['reward_mode']); print('prior:',c['policy_prior']['type'],'alpha=',c['policy_prior']['alpha']); print('ent_coef:',c['models']['ppo']['model_kwargs']['ent_coef']); print('risk_off:',c['risk_off']['enabled']); print('bt cadence:',bt['rebalance']['cadence']); print('models use=true:',[a for a,m in c['models'].items() if isinstance(m,dict) and m.get('use')])"
```

Reference baseline (the current best):
```
reward_mode    : log_return
policy_prior   : equal_weight alpha= 0.5
ent_coef       : 0.01
risk_off       : False
bt cadence     : daily
models use=true: ['ppo']
```

### Step 2 — sanity test helpers (~1 second)

```bash
python test_priors.py
```

Must print `SUMMARY: test 1 = PASS  test 2 = PASS  test 3 = PASS`. If
not, stop and investigate.

### Step 3 — launch the sweep (one line)

```bash
caffeinate -i -s nohup python -u run_experiments.py --experiments experiments.json --only model_ppo,model_a2c,model_ddpg,model_td3,model_sac --with-backtrader > model_sweep.log 2>&1 &
```

Note: `-u` flag is important — it forces Python's stdout to be unbuffered
so the log fills incrementally rather than after each experiment finishes.

### Step 4 — verify launch (~5 seconds later)

```bash
ls -la model_sweep.log && head -3 model_sweep.log
```

The first line should be:
```
Experiments: 5  | with_backtrader=True  | already done: 0  | commit=<hash>
```

If yes, sweep is alive. Walk away for 8-15 hours.

### Step 5 — monitor (optional, anytime)

```bash
# (a) live log
tail -f model_sweep.log

# (b) is the sweep alive?
ps aux | grep run_experiments | grep -v grep

# (c) how many walk-forward windows of the CURRENT algo are done (max 12):
ls experiments/model_ppo/models/ 2>/dev/null | grep -c "\.zip$"
```

### Step 6 — cadence sweep (after step 3 finishes, ~5 minutes)

```bash
python replay_cadence_sweep.py
```

This:
1. Discovers every finished `model_<algo>` experiment.
2. For each, runs stage 4 three times (daily / weekly / monthly).
3. Writes results to `experiments/model_<algo>/results_backtrader_<cadence>/`.
4. Aggregates everything into `model_cadence_results.csv`.
5. Prints three pretty 5×3 pivots (`bt_sharpe`, `bt_max_dd`, `n_trades`).

To run a subset: `python replay_cadence_sweep.py --only ppo,a2c`.

### Step 7 — read the two CSVs

```bash
cat experiments_results.csv      # 5-row algo ranking
cat model_cadence_results.csv    # 15-row model × cadence matrix
```

---

## How to interpret the results

### `experiments_results.csv` — the algo ranking

Sorted by `env_sharpe_minus_eqw` (most positive = beats EqualWeight by
most). Headline columns:

| Column | What it means | What to look for |
|---|---|---|
| `algo` | Which algorithm trained this row | Reference for the row's identity |
| `env_sharpe` | Annualised Sharpe on the env-evaluator equity curve | Higher = better risk-adjusted return (GROSS of execution costs) |
| `env_eqw_sharpe` | Sharpe of the EqualWeight baseline (the comparator) | Constant per universe; the bar to beat |
| `env_sharpe_minus_eqw` | The agent's edge over EqualWeight | Positive ↑ = real allocation alpha; negative = the agent underperforms |
| `env_cum_return` | Cumulative return on the test window | Sanity check; doesn't capture risk |
| `env_maxdd` | Maximum drawdown | Lower (less negative) = safer ride |
| `ann_turnover` | Annualised one-way turnover | < 1× = sticky; 1-3× = moderate; > 5× = fidgety; very high suggests noise-chasing |
| `risky_std` | Average per-asset weight std over time | < 0.005 = frozen; > 0.05 = very active |
| `regime_drift` | Max |Δmean| between first and last third of test | Tracks how much the agent adapts to regime shifts |
| `bt_sharpe` | Net-of-cost Sharpe from the backtrader replay | The DEPLOYABLE Sharpe number — usually 0.02-0.05 below env_sharpe if turnover is moderate |
| `bt_maxdd` | Net-of-cost max drawdown | The deployable drawdown |
| `n_trades` | Number of round-trip rebalances backtrader actually executed | Subject to the no-trade band; lower = cleaner |

### `model_cadence_results.csv` — the model × cadence matrix

15 rows = 5 algos × {daily, weekly, monthly}. Use it to read off both
dimensions at once.

| If you see... | It means... |
|---|---|
| One algo dominates across all three cadences | That's the best algorithm on this universe. Pick it. |
| One algo wins at one cadence but loses at another | Algorithm × cadence interaction. Need a second look. |
| All algos give roughly the same bt_sharpe regardless of cadence | The model + universe combination is ceiling-bound. Cadence is irrelevant; deployable is whatever cadence has lowest operational friction (usually weekly or monthly). |
| `bt_sharpe(daily) > bt_sharpe(weekly) > bt_sharpe(monthly)` | The agent's daily decisions carry signal; don't slow them down. |
| `bt_sharpe(monthly) > bt_sharpe(daily)` | Daily decisions are net-negative noise; slow down. |
| `bt_max_dd` improves materially at lower cadence | Less frequent rebalancing damps timing risk. Use lower cadence even if Sharpe is slightly worse. |

### Decision rule for which to deploy

1. Pick the model with the **highest `bt_sharpe` at any cadence** (from
   the 5 × 3 matrix).
2. Within that model, pick the **cadence with the lowest practical
   friction** that's within 0.03 Sharpe of the model's best cadence.
   Usually weekly or monthly is operationally easier than daily.
3. Sanity-check `bt_max_dd` is in your tolerance band (typically < 25 %).
4. Confirm `n_trades` is reasonable (< 50 per year on a 11-year window
   means low operational burden).

### When the matrix tells you "nothing works"

If every `bt_sharpe` is within ±0.03 of EqualWeight's, the conclusion is
that **on this universe, at daily frequency, with this feature set, the
RL allocator adds no measurable value over a static benchmark**. That's
a clean, defensible research result. The deployable answer becomes:

- a static `Prior_equal_weight` portfolio (daily rebalance), OR
- a static risk-parity / inverse-vol portfolio (monthly rebalance),
- plus a turbulence cash gate if you re-enable `risk_off.enabled`.

This is the documented "modal case" from `METHODOLOGY.md` Appendix A.

---

## Failure recovery

### A specific algorithm crashed mid-sweep

The sweep saves results incrementally. The CSV will have rows for the
successful algos; the crashed algo's row will say `status=failed`. To
retry just that algo after fixing the issue:

```bash
python run_experiments.py --experiments experiments.json --only model_<algo> --with-backtrader --force
```

`--force` re-runs even if a stale row exists in the CSV.

### The log shows nothing for >30 minutes

Either:
- Python's parent stdout buffer hasn't flushed (use `-u` flag — already
  in step 3's command).
- The training subprocess is hung. Check with:

  ```bash
  ps aux | grep "02_train" | grep -v grep
  ```
  CPU% should be > 30 %. If 0 %, kill and restart.

### `model_sweep.log` says `appending output to nohup.out`

Harmless nohup default message. Your redirect still works. Verify with
`ls -la model_sweep.log` — if size > 0, the redirect is fine.

### `Experiments: 0 | already done: 0`

The `--only` filter matched no experiment names. Check the spelling of
the algorithm names in `experiments.json` and on the command line.

### `replay_cadence_sweep.py` says "No finished model experiments"

The sweep hasn't completed yet, or `experiments/model_<algo>/results/
weights_<algo>.csv` doesn't exist. The replay script needs the recorded
weights from stage 3, which only appear after `02_train.py` and
`03_backtest.py` both complete successfully.

---

## Quick-reference card

| Goal | Command |
|---|---|
| Clean reset | `pkill -f run_experiments.py ; rm -rf experiments/model_* experiments_results.csv model_cadence_results.csv model_sweep.log nohup.out` |
| Verify config | (one-line python -c from step 1) |
| Sanity test | `python test_priors.py` |
| Launch sweep | `caffeinate -i -s nohup python -u run_experiments.py --experiments experiments.json --only "model_*" --with-backtrader > model_sweep.log 2>&1 &` (`--only` accepts globs; quote the asterisk) |
| Live log | `tail -f model_sweep.log` |
| Is it alive? | `ps aux \| grep run_experiments \| grep -v grep` |
| Progress | `ls experiments/model_*/models/ \| wc -l` (expect 12 per finished algo) |
| Cadence sweep | `python replay_cadence_sweep.py` |
| Subset cadence | `python replay_cadence_sweep.py --only ppo,a2c` |
| Read final tables | `cat experiments_results.csv ; cat model_cadence_results.csv` |
| Re-run one algo | `python run_experiments.py --experiments experiments.json --only model_<algo> --with-backtrader --force` |

---

## Expected runtime

| Phase | Time |
|---|---|
| `test_priors.py` | < 1 second |
| One PPO experiment (11 walk-forward × 150k steps) | 1-2 hours |
| One A2C experiment | 1-1.5 hours (faster than PPO — no surrogate loss / n_epochs) |
| One DDPG experiment | 2-3 hours (per-step replay updates) |
| One TD3 experiment | 2-3.5 hours (twin critics) |
| One SAC experiment | 2-3 hours (entropy term + twin critics) |
| **Full 5-algo sweep** | **8-15 hours** |
| `replay_cadence_sweep.py` (5 algos × 3 cadences × ~30 s) | ~5 minutes |
| **End-to-end clean → final answer** | **~10-15 hours** |

Sweep can run overnight on a laptop with `caffeinate -i -s` + plugged in.
Don't close the lid unless on AC + external display.

---

## Summary

You have a fully-automated 5-algorithm × 3-cadence comparison that runs
unattended in ~10-15 hours and produces two CSVs answering "which RL
algorithm" and "which rebalancing frequency" simultaneously. The
workflow has 7 steps from clean state to interpreted result; only steps
1, 3, and 6 require your attention (total ~10 minutes of typing across
the day). The interpretation rules above are conservative — when the
matrix tells you "no model adds value", that's a legitimate scientific
result, and the deployable answer is the static prior + a cash gate.
