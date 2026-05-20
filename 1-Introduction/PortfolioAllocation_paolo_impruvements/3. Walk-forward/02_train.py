"""Stage 2 - train every algorithm flagged use=true in config.json.

Two execution paths depending on walk_forward.enabled:

  * Single-split (legacy, walk_forward.enabled=false):
      Train once on data/train_data.pkl, save models/agent_<algo>.zip.
      Identical to ../2.transaction_cost/.

  * Walk-forward (walk_forward.enabled=true, the default in this folder):
      Expand window list via expand_walk_forward_windows(config). For each
      window i: slice data/full_data.pkl to the train range, train a fresh
      model with the same hyperparameters, save as
      models/agent_<algo>_w<i>.zip. After all windows are trained, persist
      a windows.json manifest so stage 3 can iterate the same way.

Usage:
    python 02_train.py --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent

from utils import (
    enabled_models,
    expand_walk_forward_windows,
    load_config,
    make_portfolio_env,
    parse_model_kwargs,
    parse_policy_kwargs,
    resolve_path,
    save_windows_manifest,
    slice_by_dates,
    walk_forward_enabled,
)


def load_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 01_get_data.py first.")
    return pd.read_pickle(path)


def train_one(algo: str, train_df: pd.DataFrame, config: dict,
              tb_dir: Path, tb_log_name: str, seed: int):
    """Train a single algo on a single slice. Returns the trained SB3 model."""
    stock_dim    = len(train_df.tic.unique())
    e_train_gym  = make_portfolio_env(train_df, config, stock_dim)
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
    return agent.train_model(
        model=model,
        tb_log_name=tb_log_name,
        total_timesteps=model_cfg["total_timesteps"],
    )


def train_single_split(config: dict, model_dir: Path, tb_dir: Path,
                       seed: int, algos: list[str]) -> None:
    """Single train/trade split - matches ../2.transaction_cost/ behaviour."""
    data_dir = resolve_path(config, "data_dir")
    train_df = load_pickle(data_dir / "train_data.pkl")

    stock_dim   = len(train_df.tic.unique())
    reward_mode = config["env"].get("reward_mode", "value")
    tc_penalty  = config["env"].get("transaction_cost_penalty", 0.0)
    print(f"stock_dim={stock_dim}  reward_mode={reward_mode}  "
          f"transaction_cost_penalty={tc_penalty}  walk_forward=off  algorithms={algos}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()}\n{'='*60}")
        trained = train_one(algo, train_df, config, tb_dir, algo, seed)
        model_path = model_dir / f"agent_{algo}.zip"
        trained.save(str(model_path))
        print(f"Saved {model_path}")


def train_walk_forward(config: dict, model_dir: Path, tb_dir: Path,
                       seed: int, algos: list[str]) -> None:
    """Per-window training: one fresh model per (train_start, train_end) pair."""
    data_dir = resolve_path(config, "data_dir")
    full_df  = load_pickle(data_dir / "full_data.pkl")

    windows = expand_walk_forward_windows(config)
    reward_mode = config["env"].get("reward_mode", "value")
    tc_penalty  = config["env"].get("transaction_cost_penalty", 0.0)
    print(f"reward_mode={reward_mode}  transaction_cost_penalty={tc_penalty}  "
          f"walk_forward=on  windows={len(windows)}  algorithms={algos}")
    for i, (ts, te, es, ee) in enumerate(windows):
        print(f"  window {i}: train {ts} -> {te}   eval {es} -> {ee}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()} ({len(windows)} windows)\n{'='*60}")
        for i, (ts, te, es, ee) in enumerate(windows):
            print(f"\n--- {algo.upper()} window {i}: train {ts} -> {te} ---")
            train_slice = slice_by_dates(full_df, ts, te)
            if len(train_slice) == 0:
                raise RuntimeError(
                    f"Window {i} train slice {ts} -> {te} is empty. "
                    f"Check data.download_start_date covers the full lookback."
                )
            trained = train_one(
                algo, train_slice, config, tb_dir,
                tb_log_name=f"{algo}_w{i}", seed=seed,
            )
            model_path = model_dir / f"agent_{algo}_w{i}.zip"
            trained.save(str(model_path))
            print(f"Saved {model_path}")

    manifest_path = save_windows_manifest(model_dir, windows)
    print(f"\nSaved {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config    = load_config(args.config)
    model_dir = resolve_path(config, "model_dir")
    tb_dir    = resolve_path(config, "tensorboard_dir")
    # StockPortfolioEnv.step() hard-codes plt.savefig("results/...") on the
    # terminal step, so ensure a results/ dir exists relative to cwd before training.
    resolve_path(config, "results_dir")
    seed = config.get("training", {}).get("seed", 42)

    algos = enabled_models(config)
    if not algos:
        print("No algorithms enabled (set models.<name>.use = true).")
        return

    if walk_forward_enabled(config):
        train_walk_forward(config, model_dir, tb_dir, seed, algos)
    else:
        train_single_split(config, model_dir, tb_dir, seed, algos)


if __name__ == "__main__":
    main()
