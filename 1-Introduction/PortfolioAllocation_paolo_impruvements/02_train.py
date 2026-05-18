"""Stage 2 - train every algorithm flagged use=true in config.json.

Mirrors the training cells of FinRL_PortfolioAllocation_NeurIPS_2020: one
DRLAgent path, no observation normalization (the notebook does not normalize).

Usage:
    python 02_train.py --config config.json
"""
from __future__ import annotations

import argparse

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv

from utils import (
    build_env_kwargs,
    enabled_models,
    load_config,
    parse_model_kwargs,
    parse_policy_kwargs,
    resolve_path,
)


def load_train_df(config: dict) -> pd.DataFrame:
    train_path = resolve_path(config, "data_dir") / "train_data.pkl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found - run 01_get_data.py first."
        )
    return pd.read_pickle(train_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config   = load_config(args.config)
    train_df = load_train_df(config)

    model_dir = resolve_path(config, "model_dir")
    tb_dir    = resolve_path(config, "tensorboard_dir")
    # StockPortfolioEnv.step() hard-codes plt.savefig("results/...") on the
    # terminal step, so ensure a results/ dir exists relative to cwd before training.
    resolve_path(config, "results_dir")
    seed      = config.get("training", {}).get("seed", 42)

    algos = enabled_models(config)
    if not algos:
        print("No algorithms enabled (set models.<name>.use = true).")
        return

    stock_dim  = len(train_df.tic.unique())
    env_kwargs = build_env_kwargs(config, stock_dim)
    print(f"stock_dim={stock_dim}  algorithms={algos}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()}\n{'='*60}")
        e_train_gym  = StockPortfolioEnv(df=train_df, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()

        agent     = DRLAgent(env=env_train)
        model_cfg = config["models"][algo]
        n_actions = env_train.action_space.shape[-1]

        model = agent.get_model(
            algo,
            model_kwargs=parse_model_kwargs(model_cfg.get("model_kwargs"), n_actions),
            policy_kwargs=parse_policy_kwargs(model_cfg.get("policy_kwargs")),
            seed=seed,
            tensorboard_log=str(tb_dir),
        )
        trained = agent.train_model(
            model=model,
            tb_log_name=algo,
            total_timesteps=model_cfg["total_timesteps"],
        )

        model_path = model_dir / f"agent_{algo}.zip"
        trained.save(str(model_path))
        print(f"Saved {model_path}")


if __name__ == "__main__":
    main()
