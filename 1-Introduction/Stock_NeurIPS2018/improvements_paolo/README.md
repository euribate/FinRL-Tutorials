# improvements_paolo — Configurable Stock_NeurIPS2018 Pipeline

A script-based reproduction of the three Jupyter notebooks (`Stock_NeurIPS2018_1_Data`, `_2_Train`, `_3_Backtest`) with all knobs exposed through a single JSON config. Lets you swap tickers, indicators, algorithms, hyperparameters, network architectures, and observation normalization without touching code.

## Folder layout

```
improvements_paolo/
├── config.json            ← all personalization lives here
├── utils.py               ← shared helpers (algo registry, env builder, JSON parser)
├── 01_get_data.py         ← download + feature engineering + train/trade split
├── 02_train.py            ← train every model flagged use=true
├── 03_backtest.py         ← backtest each model + MVO + DJIA baselines
├── README.md              ← this file
├── data/                  ← created by 01: train_data.csv, trade_data.csv
├── models/                ← created by 02: agent_<algo>.zip [+ vecnormalize_<algo>.pkl]
├── results/               ← created by 03: equity_curves.csv, equity_plot.png
└── tensorboard/           ← created by 02: TB logs per algorithm
```

## Quick start

From the `improvements_paolo/` directory, activated against the project venv:

```bash
python 01_get_data.py --config config.json
python 02_train.py     --config config.json
python 03_backtest.py  --config config.json
```

Each script takes the same `--config` path. All output paths in the config are relative to this folder.

To watch training in TensorBoard:
```bash
tensorboard --logdir tensorboard
```

## `config.json` — section by section

### `data`
| Field | What it does |
|---|---|
| `ticker_list` | The universe to download. Defaults to the current DJ-30 composition. Replace with any list of Yahoo-Finance symbols. |
| `train_start_date` / `train_end_date` | Training period (inclusive start, exclusive end). |
| `trade_start_date` / `trade_end_date` | Out-of-sample backtest period. |
| `indicators` | List of `stockstats` indicator keys (e.g. `"rsi_30"`, `"close_60_sma"`). Each becomes one feature per ticker in the state vector. |
| `use_vix` / `use_turbulence` | Whether to add the corresponding risk columns during feature engineering. The env reads `risk_indicator_col` (default `"vix"`) for the turbulence override. |

### `env`
Mirrors the `env_kwargs` block from the original notebooks. See §4.5 of `../METHODOLOGY.md` for the meaning of each field.

`turbulence_threshold_train` (commonly `null`) and `turbulence_threshold_trade` (default `70`) are split so you can keep the train env unconstrained while still enforcing the circuit-breaker at backtest time.

### `normalization`
The switch behind the two execution paths.

| `normalize_observations` | Pipeline behavior |
|---|---|
| `false` | Original notebook-style flow via FinRL's `DRLAgent`. No state-vector scaling. |
| `true` | Wraps the env in SB3's `VecNormalize`, which maintains running mean/std for every observation dimension. Stats are saved alongside each model (`vecnormalize_<algo>.pkl`) and reloaded at backtest with `training=False` so the same transform applies to unseen data — no lookahead. |

`normalize_reward` and `clip_obs` are forwarded to `VecNormalize` when normalization is on; ignored otherwise.

### `models`
One entry per algorithm. Each entry has:
- `use` — boolean. If `false`, the algorithm is skipped in both training and backtest.
- `total_timesteps` — env steps for this algorithm. Algorithms can have different budgets.
- `model_kwargs` — passed verbatim to the SB3 constructor (LR, batch size, clip range, etc.). Special string `"action_noise": "normal"` is auto-converted into a `NormalActionNoise` object sized to the action space.
- `policy_kwargs` — passed verbatim to the SB3 policy. Examples:
  ```jsonc
  "policy_kwargs": {
    "net_arch":      {"pi": [256, 256], "vf": [256, 256]},
    "activation_fn": "ReLU"
  }
  ```
  `activation_fn` accepts any class name from `torch.nn` (`"ReLU"`, `"Tanh"`, `"ELU"`, …). `null` falls back to SB3 defaults (64×64 MLP, `Tanh`).

### `training`
| Field | What it does |
|---|---|
| `seed` | Forwarded to every SB3 model for reproducibility. Set to `null` to match the original notebook (non-deterministic runs). |

## Customizing the network (`policy_kwargs.net_arch` / `activation_fn`)

Every model entry exposes `policy_kwargs` so you can resize or restructure the actor/critic networks without editing code. The defaults in `config.json` match Stable-Baselines3 baseline architectures for each algorithm — change them freely.

### Format conventions
- **On-policy algorithms (A2C, PPO)** — use a dict with `pi` (actor) and `vf` (critic):
  ```jsonc
  "net_arch": {"pi": [64, 64], "vf": [64, 64]}
  ```
- **Off-policy algorithms (DDPG, TD3, SAC)** — use a flat list (shared between actor and Q-critic), or a dict with `pi` and `qf`:
  ```jsonc
  "net_arch": [400, 300]
  ```
- **`activation_fn`** — any class name from `torch.nn`: `"Tanh"`, `"ReLU"`, `"ELU"`, `"GELU"`, `"LeakyReLU"`, …

### Examples

**Bigger PPO network (more capacity):**
```jsonc
"ppo": {
  "policy_kwargs": {
    "net_arch":      {"pi": [256, 256, 128], "vf": [256, 256]},
    "activation_fn": "ReLU"
  }
}
```

**Tiny PPO (fast experiments):**
```jsonc
"policy_kwargs": {
  "net_arch":      {"pi": [32, 32], "vf": [32, 32]},
  "activation_fn": "Tanh"
}
```

**Asymmetric — large critic, small actor (often useful when the value function is the bottleneck):**
```jsonc
"policy_kwargs": {
  "net_arch":      {"pi": [64, 64], "vf": [256, 256]},
  "activation_fn": "ReLU"
}
```

**Wider SAC:**
```jsonc
"sac": {
  "policy_kwargs": {
    "net_arch":      [512, 512],
    "activation_fn": "ReLU"
  }
}
```

### Verifying your change

After running `02_train.py`, you can confirm SB3 picked up the architecture by inspecting the saved model:
```python
from stable_baselines3 import PPO
m = PPO.load("models/agent_ppo.zip")
print(m.policy)   # prints the full network with layer sizes
```

### `paths`
Output directories. Relative paths are resolved against the `improvements_paolo/` folder.

## Common recipes

**Compare PPO vs SAC, both at 300k steps with a wider network**
1. In `config.json`, set `models.ppo.use = true`, `models.sac.use = true`, the others `false`.
2. Set `total_timesteps: 300000` for both.
3. Set `policy_kwargs.net_arch = {"pi": [256, 256], "vf": [256, 256]}` and `activation_fn = "ReLU"` for both.
4. Run all three scripts.

**Turn on observation normalization**
1. Set `normalization.normalize_observations = true`.
2. Re-run `02_train.py` (creates `vecnormalize_<algo>.pkl` per model).
3. Re-run `03_backtest.py` (auto-loads the stats files).

**Use a different ticker universe (e.g. five tech names)**
1. Set `data.ticker_list = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]`.
2. Re-run `01_get_data.py` (regenerates CSVs).
3. Re-run `02_train.py` and `03_backtest.py`. `stock_dim` and `state_space` adjust automatically — the env_kwargs are derived from the CSV, not hard-coded.

**Add a new indicator**
1. Append a valid `stockstats` key (e.g. `"wr_14"`) to `data.indicators`.
2. Re-run all three scripts.

## Differences vs. the original notebooks

- **Single source of truth** for tickers, dates, env params, model choices, and hyperparameters — no scattered cell-level constants.
- **Multi-algo orchestration** via `use` flags (the notebook's `if_using_*` booleans, lifted into JSON).
- **MVO bug fixed**: uses `range(stock_dim)` instead of hard-coded `range(29)` (see §8.3 of `../METHODOLOGY.md`).
- **Optional `VecNormalize` path** (§9.6 of the methodology doc).
- **Per-algo TensorBoard runs** under one shared `tensorboard/` directory.
- **Performance metrics printed at end**: cumulative return, annualized Sharpe (252-day), max drawdown — per RL agent and per baseline.

## Things left intentionally out of scope

- Sharpe / DSR reward shaping (kept default `ΔP&L × reward_scaling`).
- Hyperparameter search / sweep tooling.
- Walk-forward retraining (the train-once / alpha-decay setup from §9.5).
- Custom action-space designs (the FinRL `StockTradingEnv` is used as-is).

Each of these can be layered on top later without changing the surface area of `config.json`.
