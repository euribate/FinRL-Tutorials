"""Stage 2 - train every algorithm flagged use=true in config.json.

Execution paths:

  * Single-split (walk_forward.enabled=false):
      Train once on data/train_data.pkl, save models/agent_<algo>.zip.

  * Walk-forward (walk_forward.enabled=true, default in this folder):
      For each window i: slice data/full_data.pkl to the train range, train
      a fresh model, save as models/agent_<algo>_w<i>.zip. After all windows
      are trained, persist a windows.json manifest.

  * With early_stopping.enabled=true (default in this folder):
      For each training run, the train slice is further split into
      train_only + validation (last val_fraction of the date range). Every
      eval_freq timesteps the current policy is rolled through the
      validation slice, annualised Sharpe is computed, and the model is
      saved when the metric improves. Training stops after `patience`
      evaluations without improvement. See appendix A.15.

Usage:
    python 02_train.py --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent, TensorboardCallback
from stable_baselines3.common.callbacks import CallbackList

from utils import (
    ValidationSharpeCallback,
    build_vecnormalize,
    enabled_models,
    expand_walk_forward_windows,
    get_seeds,
    load_config,
    make_portfolio_env,
    parse_model_kwargs,
    parse_policy_kwargs,
    resolve_path,
    save_vecnormalize_stats,
    save_windows_manifest,
    slice_by_dates,
    split_train_validation,
    vecnormalize_path_for,
    walk_forward_enabled,
)


def load_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 01_get_data.py first.")
    return pd.read_pickle(path)


def train_one(algo: str, train_df: pd.DataFrame, config: dict,
              tb_dir: Path, tb_log_name: str, seed: int,
              model_path: Path) -> None:
    """Train a single algo on a single slice; save to `model_path`.

    Branches on config.early_stopping.enabled:
      * off: legacy FinRL DRLAgent.train_model path (== ../4.Sharpe Sortino Ratio/).
      * on:  splits train_df into train_only/val, attaches ValidationSharpeCallback
             that saves the best model to `model_path` as training progresses.
    """
    es_cfg     = config.get("early_stopping", {})
    es_enabled = bool(es_cfg.get("enabled", False))
    norm_cfg   = config.get("normalization", {})
    norm_on    = bool(norm_cfg.get("enabled", False))

    if es_enabled:
        val_fraction = float(es_cfg.get("val_fraction", 0.1))
        train_only, val_df = split_train_validation(train_df, val_fraction)
        train_dates  = train_only.date.nunique()
        val_dates    = val_df.date.nunique()
        train_first  = train_only["date"].iloc[0]
        train_last   = train_only["date"].iloc[-1]
        val_first    = val_df["date"].iloc[0]
        val_last     = val_df["date"].iloc[-1]
        print(f"  early_stopping=on  val_fraction={val_fraction:.0%}  "
              f"eval_freq={es_cfg.get('eval_freq', 5000)}  "
              f"patience={es_cfg.get('patience', 5)}  "
              f"min_delta={es_cfg.get('min_delta', 0.01)}")
        print(f"    train_only: {train_first} -> {train_last}  ({train_dates} days)")
        print(f"    validation: {val_first} -> {val_last}  ({val_dates} days)")
        train_slice = train_only
    else:
        train_slice = train_df

    stock_dim    = len(train_slice.tic.unique())
    e_train_gym  = make_portfolio_env(train_slice, config, stock_dim)
    env_train, _ = e_train_gym.get_sb_env()

    # Improvement #6: wrap with VecNormalize when enabled. The wrapper updates
    # obs / reward running stats on every step during training; we snapshot
    # them to disk after (or alongside) the model so stage 3 can reapply the
    # same normalisation at inference.
    if norm_on:
        env_train = build_vecnormalize(env_train, norm_cfg)
        print(f"  normalization=on  norm_obs={norm_cfg.get('norm_obs', True)}  "
              f"norm_reward={norm_cfg.get('norm_reward', True)}  "
              f"clip_obs={norm_cfg.get('clip_obs', 10.0)}")

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

    vn_save_path = vecnormalize_path_for(model_path) if norm_on else None

    if es_enabled:
        history_path = model_path.with_suffix(".history.json")
        es_callback = ValidationSharpeCallback(
            val_df=val_df,
            config=config,
            stock_dim=stock_dim,
            model_save_path=model_path,
            history_path=history_path,
            eval_freq=int(es_cfg.get("eval_freq", 5000)),
            patience=int(es_cfg.get("patience", 5)),
            min_delta=float(es_cfg.get("min_delta", 0.01)),
            ref_vecnormalize=env_train if norm_on else None,
            vn_save_path=vn_save_path,
            selection_metric=str(es_cfg.get("selection_metric", "sharpe")),
        )
        callbacks = CallbackList([TensorboardCallback(), es_callback])
        model.learn(
            total_timesteps=model_cfg["total_timesteps"],
            tb_log_name=tb_log_name,
            callback=callbacks,
        )
        best = es_callback.best_sharpe
        best_str = "-inf (no checkpoint improved)" if best == -float("inf") else f"{best:.4f}"
        print(f"  Saved {model_path}  (best val Sharpe = {best_str})")
        print(f"  History {history_path}")
        if norm_on:
            print(f"  Saved {vn_save_path}")
    else:
        if norm_on:
            # FinRL's DRLAgent.train_model always wraps with TensorboardCallback
            # and doesn't accept user callbacks - call model.learn directly so
            # the VecNormalize wrapping flows through training.
            model.learn(
                total_timesteps=model_cfg["total_timesteps"],
                tb_log_name=tb_log_name,
                callback=TensorboardCallback(),
            )
            model.save(str(model_path))
            save_vecnormalize_stats(env_train, vn_save_path)
            print(f"  Saved {model_path}")
            print(f"  Saved {vn_save_path}")
        else:
            trained = agent.train_model(
                model=model,
                tb_log_name=tb_log_name,
                total_timesteps=model_cfg["total_timesteps"],
            )
            trained.save(str(model_path))
            print(f"  Saved {model_path}")


def train_single_split(config: dict, model_dir: Path, tb_dir: Path,
                       seeds: list[int], algos: list[str]) -> None:
    """Single train/trade split path. Trains one model per (algo, seed) pair."""
    data_dir = resolve_path(config, "data_dir")
    train_df = load_pickle(data_dir / "train_data.pkl")

    stock_dim   = len(train_df.tic.unique())
    reward_mode = config["env"].get("reward_mode", "value")
    tc_penalty  = config["env"].get("transaction_cost_penalty", 0.0)
    eta         = config["env"].get("diff_ratio_eta", 1.0 / 252.0)
    eta_str     = f"  diff_ratio_eta={eta}" if reward_mode in ("diff_sharpe", "diff_sortino") else ""
    es_enabled  = bool(config.get("early_stopping", {}).get("enabled", False))
    norm_on     = bool(config.get("normalization", {}).get("enabled", False))
    ro_on       = bool(config.get("risk_off", {}).get("enabled", False))
    ro_thr      = config.get("risk_off", {}).get("turbulence_threshold", 70.0)
    ro_str      = f"risk_off=on(thr={ro_thr})" if ro_on else "risk_off=off"
    print(f"stock_dim={stock_dim}  reward_mode={reward_mode}  "
          f"transaction_cost_penalty={tc_penalty}{eta_str}  "
          f"walk_forward=off  early_stopping={'on' if es_enabled else 'off'}  "
          f"normalization={'on' if norm_on else 'off'}  {ro_str}  "
          f"seeds={seeds}  algorithms={algos}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()} ({len(seeds)} seeds)\n{'='*60}")
        for s in seeds:
            print(f"\n--- {algo.upper()} seed {s} ---")
            train_one(algo, train_df, config, tb_dir,
                      tb_log_name=f"{algo}_s{s}", seed=s,
                      model_path=model_dir / f"agent_{algo}_s{s}.zip")


def train_walk_forward(config: dict, model_dir: Path, tb_dir: Path,
                       seeds: list[int], algos: list[str]) -> None:
    """Per-window training: one fresh model per (window, seed) pair."""
    data_dir = resolve_path(config, "data_dir")
    full_df  = load_pickle(data_dir / "full_data.pkl")

    windows     = expand_walk_forward_windows(config)
    reward_mode = config["env"].get("reward_mode", "value")
    tc_penalty  = config["env"].get("transaction_cost_penalty", 0.0)
    eta         = config["env"].get("diff_ratio_eta", 1.0 / 252.0)
    eta_str     = f"  diff_ratio_eta={eta}" if reward_mode in ("diff_sharpe", "diff_sortino") else ""
    es_enabled  = bool(config.get("early_stopping", {}).get("enabled", False))
    norm_on     = bool(config.get("normalization", {}).get("enabled", False))
    ro_on       = bool(config.get("risk_off", {}).get("enabled", False))
    ro_thr      = config.get("risk_off", {}).get("turbulence_threshold", 70.0)
    ro_str      = f"risk_off=on(thr={ro_thr})" if ro_on else "risk_off=off"
    print(f"reward_mode={reward_mode}  transaction_cost_penalty={tc_penalty}{eta_str}  "
          f"walk_forward=on  windows={len(windows)}  "
          f"early_stopping={'on' if es_enabled else 'off'}  "
          f"normalization={'on' if norm_on else 'off'}  {ro_str}  "
          f"seeds={seeds}  algorithms={algos}")
    for i, (ts, te, es, ee) in enumerate(windows):
        print(f"  window {i}: train {ts} -> {te}   eval {es} -> {ee}")
    print(f"\nTotal training runs: {len(algos)} algos x {len(windows)} windows "
          f"x {len(seeds)} seeds = {len(algos) * len(windows) * len(seeds)}")

    for algo in algos:
        print(f"\n{'='*60}\nTraining {algo.upper()} "
              f"({len(windows)} windows x {len(seeds)} seeds)\n{'='*60}")
        for i, (ts, te, es, ee) in enumerate(windows):
            train_slice = slice_by_dates(full_df, ts, te)
            if len(train_slice) == 0:
                raise RuntimeError(
                    f"Window {i} train slice {ts} -> {te} is empty. "
                    f"Check data.download_start_date covers the full lookback."
                )
            for s in seeds:
                print(f"\n--- {algo.upper()} window {i} seed {s}: train {ts} -> {te} ---")
                train_one(
                    algo, train_slice, config, tb_dir,
                    tb_log_name=f"{algo}_w{i}_s{s}", seed=s,
                    model_path=model_dir / f"agent_{algo}_w{i}_s{s}.zip",
                )

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
    seeds = get_seeds(config)

    algos = enabled_models(config)
    if not algos:
        print("No algorithms enabled (set models.<name>.use = true).")
        return

    if walk_forward_enabled(config):
        train_walk_forward(config, model_dir, tb_dir, seeds, algos)
    else:
        train_single_split(config, model_dir, tb_dir, seeds, algos)


if __name__ == "__main__":
    main()
