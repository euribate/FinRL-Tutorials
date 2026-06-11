"""PPO training loop — slim, single-split, multi-seed, IR-selected.

What changed vs the old 02_train.py:
  * No walk-forward mode. Single train/val/trade split is the only
    supported topology. Walk-forward was a 250-line code path that
    obscured signal at the experimental margin; reintroduce only at
    deployment time, in its own module.
  * No VecNormalize. Per-asset features are already normalised inside
    `data.py` (rank normalisation when Step 1 lands; for Step 0 the
    legacy stockstats indicators are passed through as-is).
  * Single callback (ValidationIRCallback). No sharpe-sel branch.
  * One env per training run — no DummyVecEnv shell. SB3 accepts a
    raw gym.Env at construction; the auto-vec wrapper handles the rest.

Usage:
    python train.py --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from callbacks import ValidationIRCallback
from data import feature_columns, load_config, load_pickles, resolve_path
from env import PortfolioEnv


def _split_train_val(train_df: pd.DataFrame, val_fraction: float
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `val_fraction` of unique dates become the validation slice.

    Order-preserving, no shuffling — the train slice is contiguous and
    earlier than the val slice. This is what the IR callback wants
    (no look-ahead).
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    dates = sorted(train_df["date"].unique())
    cut   = int(len(dates) * (1.0 - val_fraction))
    train_dates = set(dates[:cut])
    val_dates   = set(dates[cut:])
    return (train_df[train_df["date"].isin(train_dates)].copy(),
            train_df[train_df["date"].isin(val_dates)].copy())


def _env_kwargs(config: dict) -> dict:
    env_cfg = config["env"]
    return dict(
        action_logit_scale=float(env_cfg.get("action_logit_scale", 3.0)),
        cost_bps          =float(env_cfg.get("cost_bps", 10.0)),
        cadence           =str(env_cfg.get("cadence", "daily")),
        weekly_day        =str(env_cfg.get("weekly_day", "FRI")),
        reward_scaling    =float(env_cfg.get("reward_scaling", 1.0)),
    )


def train_one_seed(config: dict, seed: int,
                   train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
    """Train one PPO policy with the given seed; ES on val IR.

    Saves agent_ppo_s<seed>.zip and history JSON to model_dir.
    """
    feat_cols  = feature_columns(config)
    env_kwargs = _env_kwargs(config)
    model_dir  = resolve_path(config, "model_dir")
    tb_dir     = resolve_path(config, "tensorboard_dir")
    model_path = model_dir / f"agent_ppo_s{seed}.zip"
    history_path = model_path.with_suffix(".history.json")

    train_env = PortfolioEnv(train_df, feat_cols, **env_kwargs)

    ppo_cfg = config["model"]
    policy_kwargs = ppo_cfg.get("policy_kwargs", None)
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        seed=seed,
        tensorboard_log=str(tb_dir),
        policy_kwargs=policy_kwargs,
        **ppo_cfg.get("model_kwargs", {}),
    )

    es_cfg = config.get("early_stopping", {}) or {}
    callback = ValidationIRCallback(
        val_df=val_df,
        feature_cols=feat_cols,
        env_kwargs=env_kwargs,
        model_save_path=model_path,
        history_path=history_path,
        eval_freq=int(es_cfg.get("eval_freq", 2500)),
        patience=int(es_cfg.get("patience", 20)),
        min_delta=float(es_cfg.get("min_delta", 0.001)),
    )

    print(f"--- PPO seed {seed} ---")
    model.learn(
        total_timesteps=int(ppo_cfg.get("total_timesteps", 150_000)),
        tb_log_name=f"ppo_s{seed}",
        callback=callback,
        reset_num_timesteps=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    _full_df, train_df, _trade_df = load_pickles(config)

    val_fraction = float(config.get("early_stopping", {})
                                 .get("val_fraction", 0.1))
    train_only_df, val_df = _split_train_val(train_df, val_fraction)
    print(f"train_only: {train_only_df['date'].nunique()} dates, "
          f"val: {val_df['date'].nunique()} dates  "
          f"(val_fraction={val_fraction})")

    seeds = list(config.get("seeds", {}).get("list", [42]))
    for s in seeds:
        train_one_seed(config, int(s), train_only_df, val_df)


if __name__ == "__main__":
    main()
