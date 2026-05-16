"""Stage 2 — train every algorithm flagged use=true in config.json.

Two execution paths depending on `normalization.normalize_observations`:
  * False  ->  FinRL DRLAgent path (same as the original notebook).
  * True   ->  raw SB3 + VecNormalize wrapping; running mean/std stats are
               persisted alongside the model for use at backtest time.

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
    build_env_kwargs,
    enabled_models,
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
    """Raw SB3 + VecNormalize path. Returns (model, vec_env)."""
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize(
        venv,
        norm_obs=True,
        norm_reward=norm_cfg.get("normalize_reward", True),
        clip_obs=norm_cfg.get("clip_obs", 10.0),
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
    normalize = config["normalization"]["normalize_observations"]

    algos = enabled_models(config)
    if not algos:
        print("No algorithms enabled (set models.<name>.use = true).")
        return

    print(f"Mode: {'NORMALIZED (VecNormalize)' if normalize else 'STANDARD (FinRL DRLAgent)'}")
    print(f"Algorithms: {algos}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()}\n{'='*60}")
        env = build_train_env(config, train_df)
        model_cfg = config["models"][algo]
        model_path = model_dir / f"agent_{algo}.zip"

        if normalize:
            model, venv = train_with_vecnormalize(
                env, algo, model_cfg, config["normalization"], tb_dir, seed
            )
            stats_path = model_dir / f"vecnormalize_{algo}.pkl"
            model.save(str(model_path))
            venv.save(str(stats_path))
            print(f"Saved {model_path}\nSaved {stats_path}")
        else:
            model = train_finrl(env, algo, model_cfg, tb_dir, seed)
            model.save(str(model_path))
            print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
