"""Stage 2 — train every algorithm flagged use=true in config.json.

Three execution paths depending on `normalization.obs_mode`:
  * "off"        -> FinRL DRLAgent path (same as the original notebook).
  * "indicators" -> raw SB3 + SelectiveVecNormalize wrapping (z-score only
                    the indicator dimensions of the state vector).
  * "all"        -> raw SB3 + standard VecNormalize wrapping (z-score every
                    observation dim — matches SB3 defaults).
In the latter two modes the running mean/std stats are persisted alongside
the model (vecnormalize_<algo>.pkl) for use at backtest time.

Usage:
    python 02_train.py --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import (
    ALGO_REGISTRY,
    SelectiveVecNormalize,
    build_env_kwargs,
    enabled_models,
    indicator_indices,
    load_config,
    parse_model_kwargs,
    parse_policy_kwargs,
    resolve_path,
)


def load_train_df(config: dict) -> pd.DataFrame:
    train_path = resolve_path(config, "data_dir") / "train_data.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found — run 01_get_data.py first."
        )
    df = pd.read_csv(train_path)
    df = df.set_index(df.columns[0])
    df.index.names = [""]
    return df


def build_train_env(config: dict, train_df: pd.DataFrame) -> StockTradingEnv:
    stock_dim = len(train_df.tic.unique())
    env_kwargs = build_env_kwargs(config, stock_dim, mode="train")
    return StockTradingEnv(df=train_df, **env_kwargs)


def train_finrl(env: StockTradingEnv, algo: str, model_cfg: dict,
                tb_dir: Path, seed: int):
    """Standard FinRL DRLAgent path (matches the original notebook)."""
    agent = DRLAgent(env=env)
    n_actions = env.action_space.shape[-1]
    model = agent.get_model(
        algo,
        model_kwargs=parse_model_kwargs(model_cfg.get("model_kwargs"), n_actions),
        policy_kwargs=parse_policy_kwargs(model_cfg.get("policy_kwargs")),
        seed=seed,
        tensorboard_log=str(tb_dir),
    )
    return agent.train_model(
        model=model,
        tb_log_name=algo,
        total_timesteps=model_cfg["total_timesteps"],
    )


def train_with_vecnormalize(env: StockTradingEnv, algo: str, model_cfg: dict,
                            norm_cfg: dict, tb_dir: Path, seed: int):
    """Raw SB3 + VecNormalize path. Returns (model, vec_env).

    obs_mode = 'indicators' (recommended) restricts z-score normalization
    to indicator dims only. Prices, cash, and holdings have time-drifting
    or behavior-dependent natural scales that produce off-distribution
    observations at backtest when normalized with training-period stats.

    obs_mode = 'all' z-scores every observation dim (plain VecNormalize).
    """
    venv = DummyVecEnv([lambda: env])
    obs_mode = norm_cfg["obs_mode"]
    vn_kwargs = dict(
        norm_obs=True,
        norm_reward=False,
        clip_obs=norm_cfg.get("clip_obs", 10.0),
    )
    if obs_mode == "indicators":
        sd = env.stock_dim
        ni = len(env.tech_indicator_list)
        idx = indicator_indices(sd, ni)
        print(f"  SelectiveVecNormalize: normalizing {len(idx)} indicator dims "
              f"(of {1 + 2 * sd + ni * sd} total); leaving cash/prices/holdings raw.")
        venv = SelectiveVecNormalize(venv, norm_indices=idx, **vn_kwargs)
    elif obs_mode == "all":
        print("  VecNormalize: normalizing every observation dim (full mode).")
        venv = VecNormalize(venv, **vn_kwargs)
    else:
        raise ValueError(
            f"Unexpected obs_mode={obs_mode!r}; expected 'indicators' or 'all'."
        )
    n_actions = venv.action_space.shape[-1]
    AlgoClass = ALGO_REGISTRY[algo]
    model = AlgoClass(
        "MlpPolicy",
        venv,
        seed=seed,
        verbose=1,
        tensorboard_log=str(tb_dir),
        policy_kwargs=parse_policy_kwargs(model_cfg.get("policy_kwargs")),
        **parse_model_kwargs(model_cfg.get("model_kwargs"), n_actions),
    )
    model.learn(
        total_timesteps=model_cfg["total_timesteps"],
        tb_log_name=algo,
    )
    return model, venv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    train_df = load_train_df(config)

    model_dir = resolve_path(config, "model_dir")
    tb_dir    = resolve_path(config, "tensorboard_dir")
    seed      = config.get("training", {}).get("seed", 42)
    obs_mode  = config["normalization"]["obs_mode"]

    algos = enabled_models(config)
    if not algos:
        print("No algorithms enabled (set models.<name>.use = true).")
        return

    mode_label = {
        "off":        "STANDARD (FinRL DRLAgent, no normalization)",
        "indicators": "NORMALIZED (SelectiveVecNormalize, indicators only)",
        "all":        "NORMALIZED (VecNormalize, all dims)",
    }.get(obs_mode)
    if mode_label is None:
        raise ValueError(
            f"normalization.obs_mode={obs_mode!r}; expected 'off', 'indicators', or 'all'."
        )
    print(f"Mode: {mode_label}")
    print(f"Algorithms: {algos}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()}\n{'='*60}")
        env = build_train_env(config, train_df)
        model_cfg = config["models"][algo]
        model_path = model_dir / f"agent_{algo}.zip"

        if obs_mode == "off":
            model = train_finrl(env, algo, model_cfg, tb_dir, seed)
            model.save(str(model_path))
            print(f"Saved {model_path}")
        else:
            model, venv = train_with_vecnormalize(
                env, algo, model_cfg, config["normalization"], tb_dir, seed
            )
            stats_path = model_dir / f"vecnormalize_{algo}.pkl"
            model.save(str(model_path))
            venv.save(str(stats_path))
            print(f"Saved {model_path}\nSaved {stats_path}")


if __name__ == "__main__":
    main()
