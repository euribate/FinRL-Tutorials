"""Shared helpers for the PortfolioAllocation_paolo_impruvements pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


class LogReturnPortfolioEnv(StockPortfolioEnv):
    """StockPortfolioEnv variant with two improvements layered on upstream:

    1. (Improvement #1) Reward = log(1 + portfolio_return) * reward_scaling
       instead of raw new_portfolio_value.

    2. (Improvement #2) Optional transaction-cost penalty: when `tc_penalty > 0`,
       a per-unit-turnover cost is subtracted from both the reward AND from the
       book equity (portfolio_value / asset_memory / portfolio_return_memory),
       so that downstream consumers (stage 3's DRL_prediction, equity plot,
       QuantStats tearsheet) reflect realistic post-cost performance.

       Math: turnover    = |weights_t - weights_{t-1}|.sum()
             tc_fraction = tc_penalty * turnover
             net_return  = (1 + gross_return) * (1 - tc_fraction) - 1
             reward      = log(1 + net_return) * reward_scaling

       Setting `tc_penalty = 0` short-circuits the cost math entirely; the
       env then behaves identically to the improvement-#1 variant in
       ../1.reward_returns/ - same model, same equity curves, same metrics.
    """

    def __init__(self, *args, tc_penalty: float = 0.0, **kwargs):
        # `tc_penalty` is our own keyword-only arg; consume before super().
        super().__init__(*args, **kwargs)
        self.tc_penalty = float(tc_penalty)

    def step(self, actions):
        obs, _reward, done, truncated, info = super().step(actions)
        if done or not self.portfolio_return_memory:
            return obs, self.reward, done, truncated, info

        gross_return     = float(self.portfolio_return_memory[-1])
        effective_return = gross_return

        # Apply transaction-cost penalty (improvement #2). actions_memory was
        # initialised by the parent with [[1/N]*N] and then appended this step's
        # weights, so [-2] is always available from step 1 onward.
        if self.tc_penalty > 0.0 and len(self.actions_memory) >= 2:
            w_curr = np.asarray(self.actions_memory[-1], dtype=float)
            w_prev = np.asarray(self.actions_memory[-2], dtype=float)
            turnover    = float(np.abs(w_curr - w_prev).sum())
            tc_fraction = self.tc_penalty * turnover
            effective_return = (1.0 + gross_return) * (1.0 - tc_fraction) - 1.0
            # Rewrite the entries the parent just appended so the equity curve
            # and downstream reports reflect the cost too, not just the reward.
            self.portfolio_return_memory[-1] = effective_return
            base = float(self.asset_memory[-2])
            self.portfolio_value = base * (1.0 + effective_return)
            self.asset_memory[-1] = self.portfolio_value

        self.reward = float(np.log(1.0 + effective_return) * self.reward_scaling)
        return obs, self.reward, done, truncated, info


ALGO_REGISTRY = {
    "a2c":  A2C,
    "ppo":  PPO,
    "ddpg": DDPG,
    "td3":  TD3,
    "sac":  SAC,
}

NOISE_REGISTRY = {
    "normal":             NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def resolve_path(config: dict, key: str) -> Path:
    p = Path(config["paths"][key])
    if not p.is_absolute():
        p = project_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def enabled_models(config: dict) -> list[str]:
    return [name for name, cfg in config["models"].items() if cfg.get("use", False)]


# ---------- walk-forward helpers (improvement #3) -----------------------------

def walk_forward_enabled(config: dict) -> bool:
    return bool(config.get("walk_forward", {}).get("enabled", False))


def expand_walk_forward_windows(config: dict) -> list[tuple[str, str, str, str]]:
    """Return the list of [train_start, train_end, eval_start, eval_end] quadruples.

    Priority:
      1. `walk_forward.windows`: explicit list of 4-tuples (ISO date strings).
      2. `walk_forward.auto`: sliding windows generated from train_years /
         eval_years / step_years, anchored on data.train_start_date and capped
         by data.trade_end_date.
    """
    wf = config.get("walk_forward", {})
    explicit = wf.get("windows")
    if explicit:
        return [tuple(w) for w in explicit]

    auto = wf.get("auto", {})
    train_years = int(auto.get("train_years", 6))
    eval_years  = int(auto.get("eval_years", 1))
    step_years  = int(auto.get("step_years", 1))

    start = pd.Timestamp(config["data"]["train_start_date"])
    cap   = pd.Timestamp(config["data"]["trade_end_date"])

    windows: list[tuple[str, str, str, str]] = []
    cur_train_start = start
    while True:
        cur_train_end  = cur_train_start + pd.DateOffset(years=train_years)
        cur_eval_start = cur_train_end
        cur_eval_end   = min(cur_eval_start + pd.DateOffset(years=eval_years), cap)
        if cur_eval_start >= cap:
            break
        windows.append((
            cur_train_start.strftime("%Y-%m-%d"),
            cur_train_end.strftime("%Y-%m-%d"),
            cur_eval_start.strftime("%Y-%m-%d"),
            cur_eval_end.strftime("%Y-%m-%d"),
        ))
        cur_train_start = cur_train_start + pd.DateOffset(years=step_years)
    if not windows:
        raise ValueError(
            "expand_walk_forward_windows() produced no windows. Check "
            "walk_forward.auto vs data.{train_start_date,trade_end_date}."
        )
    return windows


def slice_by_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Return rows where start <= date < end.

    Re-indexes integer factors so the env's `df.loc[day]` lookup works after the
    slice (StockPortfolioEnv requires the day index to start at 0 and be dense).
    """
    sub = df[(df["date"] >= start) & (df["date"] < end)].copy()
    sub = sub.sort_values(["date", "tic"], ignore_index=True)
    sub.index = sub.date.factorize()[0]
    return sub


def save_windows_manifest(model_dir: Path,
                          windows: list[tuple[str, str, str, str]]) -> Path:
    """Persist the window list alongside the trained models so stage 3 can find them."""
    path = model_dir / "windows.json"
    with open(path, "w") as f:
        json.dump([list(w) for w in windows], f, indent=2)
    return path


def load_windows_manifest(model_dir: Path) -> list[tuple[str, str, str, str]]:
    path = model_dir / "windows.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run 02_train.py with walk_forward.enabled=true first."
        )
    with open(path) as f:
        return [tuple(w) for w in json.load(f)]


def resolve_activation(name: str) -> type:
    if not hasattr(nn, name):
        raise ValueError(f"Unknown activation function: {name!r}")
    return getattr(nn, name)


def parse_policy_kwargs(policy_kwargs: dict | None) -> dict | None:
    if not policy_kwargs:
        return None
    parsed = dict(policy_kwargs)
    if isinstance(parsed.get("activation_fn"), str):
        parsed["activation_fn"] = resolve_activation(parsed["activation_fn"])
    return parsed


def parse_model_kwargs(model_kwargs: dict | None, n_actions: int) -> dict:
    if not model_kwargs:
        return {}
    parsed = dict(model_kwargs)
    noise_name = parsed.get("action_noise")
    if isinstance(noise_name, str):
        parsed["action_noise"] = NOISE_REGISTRY[noise_name](
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
    return parsed


def build_env_kwargs(config: dict, stock_dim: int) -> dict:
    """Assemble env_kwargs for StockPortfolioEnv. Matches the notebook's signature."""
    env_cfg = config["env"]
    return {
        "hmax":                 env_cfg["hmax"],
        "initial_amount":       env_cfg["initial_amount"],
        "transaction_cost_pct": env_cfg["transaction_cost_pct"],
        "state_space":          stock_dim,
        "stock_dim":            stock_dim,
        "tech_indicator_list":  config["data"]["indicators"],
        "action_space":         stock_dim,
        "reward_scaling":       env_cfg["reward_scaling"],
    }


def make_portfolio_env(df: pd.DataFrame, config: dict, stock_dim: int):
    """Construct the configured StockPortfolioEnv variant.

    Dispatches on two config knobs in `config.env`:
      * reward_mode:
        - "value"      -> upstream StockPortfolioEnv (reward = portfolio value).
        - "log_return" -> LogReturnPortfolioEnv (reward = log(1 + portfolio_return)
                          * reward_scaling).
        Defaults to "value" so unchanged configs reproduce the notebook.
      * transaction_cost_penalty (float, default 0.0):
        - Per-unit-turnover cost rate. Only applied when reward_mode='log_return'.
        - When 0, behaviour is identical to the no-TC variant in 1.reward_returns/.
        - When > 0, deducted from both the reward signal and the book equity so
          stages 3/4/5 see realistic post-cost performance.

    Warnings:
      * reward_mode='log_return' with tiny reward_scaling (<0.1) - log returns are
        O(0.01)/day, so e.g. 1e-4 scaling shrinks the signal to ~1e-6.
      * reward_mode='value' with transaction_cost_penalty>0 - TC penalty in value
        mode is out of scope for this implementation; the penalty is ignored.
    """
    env_kwargs  = build_env_kwargs(config, stock_dim)
    reward_mode = config["env"].get("reward_mode", "value")
    tc_penalty  = float(config["env"].get("transaction_cost_penalty", 0.0))

    if reward_mode == "value":
        if tc_penalty > 0.0:
            print(
                f"WARNING: env.transaction_cost_penalty={tc_penalty} is set but "
                f"env.reward_mode='value' - the TC penalty is only applied in "
                f"'log_return' mode. The penalty will be ignored. To enable it, "
                f"set env.reward_mode='log_return' in config.json."
            )
        return StockPortfolioEnv(df=df, **env_kwargs)

    if reward_mode == "log_return":
        scaling = float(env_kwargs.get("reward_scaling", 1.0))
        if scaling < 0.1:
            print(
                f"WARNING: reward_mode='log_return' with reward_scaling={scaling}. "
                f"Log returns are O(0.01) per day; this scaling will shrink the "
                f"reward signal to ~{scaling * 0.01:g} - effectively zero gradient. "
                f"Recommended: set env.reward_scaling=1.0 in config.json."
            )
        return LogReturnPortfolioEnv(df=df, tc_penalty=tc_penalty, **env_kwargs)

    raise ValueError(
        f"Unknown env.reward_mode={reward_mode!r}; expected 'value' or 'log_return'."
    )


def compute_cov_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Replicate the notebook's covariance/return loop.

    For every date `t` with at least `lookback` prior trading days, attach:
      * cov_list[t]     - (N, N) sample covariance of the trailing returns.
      * return_list[t]  - (lookback-1, N) DataFrame of the trailing daily returns.

    Both are stored as object cells in the returned DataFrame.
    """
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list: list[np.ndarray] = []
    return_list: list[pd.DataFrame] = []

    for i in range(lookback, len(df.index.unique())):
        data_lookback   = df.loc[i - lookback : i, :]
        price_lookback  = data_lookback.pivot_table(
            index="date", columns="tic", values="close"
        )
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        cov_list.append(return_lookback.cov().values)

    df_cov = pd.DataFrame({
        "date":        df.date.unique()[lookback:],
        "cov_list":    cov_list,
        "return_list": return_list,
    })
    df = df.merge(df_cov, on="date")
    return df.sort_values(["date", "tic"]).reset_index(drop=True)
