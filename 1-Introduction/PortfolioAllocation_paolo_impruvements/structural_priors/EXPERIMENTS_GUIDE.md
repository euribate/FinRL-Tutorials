# Experiments — How To Use, How To Edit

Quick reference for `experiments.json` + `run_experiments.py`. Memo style;
copy-paste examples below for the most likely changes you'll want to make.

---

## TL;DR — run all experiments

```bash
python run_experiments.py --experiments experiments.json --with-backtrader
```

That runs every experiment in `experiments.json`, skips ones already in
`experiments_results.csv` with `status=ok`, prints a ranked table at the end,
and saves everything to `experiments_results.csv`. Add `--force` to re-run
finished experiments. Use `--only NAME1,NAME2` to run a subset; `--only`
also accepts shell-style **glob patterns** (e.g. `--only "alpha_iv_*"`) so
you don't have to enumerate every grid cell — see "Running, monitoring,
reading results" below.

---

## The model: one base config + a list of overrides

Every experiment is **the base config (`config.json`) with a handful of
dotted-path values replaced**. You don't copy whole configs around. Two
patterns:

### Pattern A — explicit experiments (hand-picked, named)

A list of `{name, overrides}` objects in the `"experiments"` array. Use this
for one-off comparisons, controls, baselines, and anything with non-scalar
overrides (lists, nulls).

```json
"experiments": [
  { "name": "baseline",      "overrides": {} },
  { "name": "prior_off",     "overrides": { "policy_prior.enabled": false } },
  { "name": "log_return",    "overrides": { "env.reward_mode": "log_return" } },
  { "name": "tight_band",    "overrides": { "policy_prior.alpha": 0.25, "models.ppo.model_kwargs.ent_coef": 0.001 } }
]
```

Each entry produces **one** run with that exact override set.

### Pattern B — grid (cartesian product, auto-named)

A list of `{name_prefix, axes}` objects in the `"grid"` array. Use this for
sweeping the same scalar knob(s) across a range. The runner expands the
cartesian product of `axes` into one experiment per combination.

```json
"grid": [
  {
    "name_prefix": "alpha",
    "axes": {
      "policy_prior.alpha": [0.25, 0.5, 1.0, 2.0]
    }
  },
  {
    "name_prefix": "lr",
    "axes": {
      "models.ppo.model_kwargs.learning_rate": [0.0001, 0.0002, 0.0005],
      "models.ppo.model_kwargs.ent_coef":      [0.0, 0.001, 0.01]
    }
  }
]
```

→ The `alpha` block generates 4 experiments named
`alpha_alpha0.25`, `alpha_alpha0.5`, `alpha_alpha1.0`, `alpha_alpha2.0`.
→ The `lr` block generates `3 × 3 = 9` experiments named like
`lr_learning_rate0.0001_ent_coef0.0` (one per combination).

**Use explicit for `net_arch` lists** (`pi: [64,64]`) — grids auto-name from
the value and produce ugly names like `pi[64, 64]`.

---

## How to find the dotted-path key for any setting

The override key is the **dot-joined path into `config.json`**. Open
`config.json`, navigate down the JSON structure to the value you want to
change, and join the keys with `.`.

Worked examples for the keys you're most likely to want:

| What to change in `config.json` | Override key |
|---|---|
| `policy_prior.alpha` | `policy_prior.alpha` |
| `policy_prior.type` | `policy_prior.type` |
| `policy_prior.enabled` | `policy_prior.enabled` |
| `risk_off.enabled` | `risk_off.enabled` |
| `risk_off.turbulence_threshold` | `risk_off.turbulence_threshold` |
| `env.reward_mode` | `env.reward_mode` |
| `env.transaction_cost_penalty` | `env.transaction_cost_penalty` |
| `env.article_reward.lambda_to` | `env.article_reward.lambda_to` |
| `models.ppo.model_kwargs.gamma` | `models.ppo.model_kwargs.gamma` |
| `models.ppo.model_kwargs.ent_coef` | `models.ppo.model_kwargs.ent_coef` |
| `models.ppo.model_kwargs.learning_rate` | `models.ppo.model_kwargs.learning_rate` |
| `models.ppo.total_timesteps` | `models.ppo.total_timesteps` |
| encoder width | `models.ppo.policy_kwargs.features_extractor_kwargs.hidden` |
| encoder embedding dim | `models.ppo.policy_kwargs.features_extractor_kwargs.emb_dim` |
| encoder depth | `models.ppo.policy_kwargs.features_extractor_kwargs.layers` |
| policy head | `models.ppo.policy_kwargs.net_arch.pi` |
| value head | `models.ppo.policy_kwargs.net_arch.vf` |
| disable the encoder (use SB3 default MLP) | `models.ppo.policy_kwargs` → `null` |
| ticker universe | `data.ticker_list` (touches data — see gotcha below) |
| benchmark type | `benchmark.type` |
| cash on/off | `cash.enabled` (touches data) |

If you can't find the dotted path: open `config.json`, copy the JSON path
verbatim, drop the array indices, you're done.

---

## Three worked tweaks you might actually want to do

### 1. Test the cash gate on/off side-by-side

Pure `prior_off` plus the gate-vs-no-gate decomposition. Add to
`experiments`:

```json
{ "name": "gate_on",                       "overrides": { "risk_off.enabled": true } },
{ "name": "gate_off",                      "overrides": { "risk_off.enabled": false } },
{ "name": "gate_on_thresh_60",             "overrides": { "risk_off.enabled": true, "risk_off.turbulence_threshold": 60.0 } },
{ "name": "gate_on_prior_equal_weight",    "overrides": { "risk_off.enabled": true, "policy_prior.type": "equal_weight" } }
```

### 2. Sweep encoder size at the current prior

Add to `grid`:

```json
{
  "name_prefix": "enc",
  "axes": {
    "models.ppo.policy_kwargs.features_extractor_kwargs.hidden":  [32, 64, 128, 192],
    "models.ppo.policy_kwargs.features_extractor_kwargs.emb_dim": [1, 4]
  }
}
```

→ 4 × 2 = 8 experiments named `enc_hidden32_emb_dim1`, etc.

### 3. Compare reward modes with everything else held constant

Add to `experiments`:

```json
{ "name": "rwd_log_return",       "overrides": { "env.reward_mode": "log_return" } },
{ "name": "rwd_diff_sharpe",      "overrides": { "env.reward_mode": "diff_sharpe" } },
{ "name": "rwd_article_absolute", "overrides": { "env.reward_mode": "article_absolute" } }
```

---

## Running, monitoring, reading results

**Run the full set** (env metrics only — fastest):

```bash
python run_experiments.py --experiments experiments.json
```

**Run with the backtrader realism check** too (adds `bt_sharpe` /
`bt_maxdd` / `n_trades` columns):

```bash
python run_experiments.py --experiments experiments.json --with-backtrader
```

**Run a subset** (handy for re-testing one experiment):

```bash
python run_experiments.py --experiments experiments.json --only baseline,gate_on --force
```

**`--only` accepts shell-style glob patterns** (since the `fnmatch` patch):

```bash
# All cells from one grid block:
python run_experiments.py --experiments experiments.json --only "alpha_iv_*" --with-backtrader

# All round-2 prior-type experiments:
python run_experiments.py --experiments experiments.json --only "type_*"

# Mix of patterns and exact names (comma-separated):
python run_experiments.py --experiments experiments.json --only "baseline,prior_off,alpha_*"

# Both alpha-sweep grids together:
python run_experiments.py --experiments experiments.json --only "alpha_*"
```

**Quote any pattern containing `*` or `?`** so the shell doesn't try to
expand it as a file glob before python sees it. Bare names without
metacharacters still match exactly — old `--only baseline,gate_on` usage
is unchanged.

**Background it** so it survives terminal closure:

```bash
nohup caffeinate -i -s python run_experiments.py --experiments experiments.json --with-backtrader > sweep.log 2>&1 &
tail -f sweep.log
```

### What you get

- **`experiments_results.csv`** — one row per experiment, with `env_sharpe`,
  `env_eqw_sharpe`, `env_sharpe_minus_eqw`, `env_cum_return`, `env_maxdd`,
  `ann_turnover`, `risky_std`, `regime_drift`; plus `bt_sharpe`,
  `bt_maxdd`, `n_trades` when `--with-backtrader` is used.
- **Ranked table** printed at the end, sorted by `env_sharpe_minus_eqw`
  (how much PPO beats / loses to EqualWeight).
- **`experiments/<name>/`** subfolders containing each run's per-experiment
  config + the trained model + per-experiment results.

### Resuming

- Already-done experiments (`status=ok` in the CSV) are **skipped** on
  re-runs. Add `--force` to redo them.
- You can stop a long sweep with Ctrl-C and resume by re-running the
  same command — only the unfinished ones will train.

---

## Gotchas

1. **Universe / data changes need a stage-1 re-run.** Overrides under
   `data.*`, `benchmark.*`, `cash.*` cause the runner to rebuild the
   data pickle for that experiment. This is much slower than reusing
   the cached `data/full_data.pkl`. Avoid them unless you genuinely
   want to test a different universe.

2. **`null` overrides are how you "disable" a value.** Setting
   `"models.ppo.policy_kwargs": null` reverts to the SB3 default MLP
   policy. `"policy_prior.cash_share": null` means "default to 1/N".

3. **Grids auto-name from the value.** Scalars are fine; lists produce
   ugly names. Always prefer explicit `experiments` for `net_arch`,
   `policy_kwargs`, or any list-valued setting.

4. **Names must be unique.** The runner raises `ValueError: Duplicate
   experiment name` if you re-use one across explicit + grid blocks.

5. **Edits don't require any other action.** Edit `experiments.json`,
   re-run `run_experiments.py` — the new experiments are picked up
   automatically; finished ones stay finished.

6. **Determinism check.** A grid cell that lands on the same config as
   an existing experiment (e.g., `alpha_alpha1.0` with `policy_prior.alpha=1.0`
   when the base config already has 1.0) should produce **byte-identical**
   results to the baseline. Useful as a sanity check.

7. **`--only` accepts globs but they must be quoted.** Use `--only "alpha_iv_*"`
   (with the quotes) to run every cell whose name starts with `alpha_iv_`.
   Without the quotes, the shell tries to expand `*` against your current
   directory's files first and you get a `no matches found` error or
   silently-wrong selection. Comma-separated mixes of patterns and exact
   names work too: `--only "baseline,prior_off,alpha_iv_*"`.

---

## Workflow I recommend

For exploratory work (1-3 ideas at a time):

1. Edit `experiments.json` — add 2-5 explicit experiments.
2. `python run_experiments.py --experiments experiments.json --with-backtrader`.
3. Look at the ranked table. Decide what to test next.
4. Goto 1.

For a focused sweep (one knob, many values):

1. Add a grid block to `experiments.json` with `name_prefix` + one axis.
2. Run with `--with-backtrader` if cost matters; without if you just
   want shape.
3. Look at `env_sharpe_minus_eqw` for the knob's curve.

For a full investigation (mix of explicit baselines + sweeps):

1. Hand-pick 3-5 controls in `experiments` (baseline, prior_off, etc.).
2. Add 1-2 grids for the axes that matter.
3. Run the whole thing as a background job; come back when it's done.

That's it. Memo over.
