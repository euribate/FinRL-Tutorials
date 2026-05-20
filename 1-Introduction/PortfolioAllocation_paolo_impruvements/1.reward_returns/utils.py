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
    """StockPortfolioEnv variant that rewards log-returns instead of dollars.

    Upstream `StockPortfolioEnv.step()` sets `self.reward = new_portfolio_value`,
    which is non-stationary (drifts from $1M to many millions across the 11.5-year
    training horizon) and dominated by late-period samples. Log-returns are
    stationary, bounded in practice, and additive over time - a much cleaner
    gradient signal for actor-critic updates.

    The override calls `super().step()` (which still computes the simple
    portfolio_return and appends it to `self.portfolio_return_memory`) and then
    replaces `self.reward` with `log(1 + portfolio_return) * reward_scaling`.
    """

    def step(self, actions):
        obs, _reward, done, truncated, info = super().step(actions)
        if not done and self.portfolio_return_memory:
            r = float(self.portfolio_return_memory[-1])
            self.reward = float(np.log(1.0 + r) * self.reward_scaling)
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

    Dispatches on `config.env.reward_mode`:
      * "value"      - upstream StockPortfolioEnv (reward = portfolio value).
      * "log_return" - LogReturnPortfolioEnv (reward = log(1 + portfolio_return)
                        * reward_scaling).
    Defaults to "value" so existing configs reproduce the notebook unchanged.

    Emits a warning when reward_mode='log_return' is paired with a tiny
    reward_scaling (the value-mode default of 1e-4). Log returns are O(0.01)
    per day; multiplying by 1e-4 starves the gradient and the agent will not
    learn. Recommended reward_scaling for log_return mode is 1.0.
    """
    env_kwargs  = build_env_kwargs(config, stock_dim)
    reward_mode = config["env"].get("reward_mode", "value")
    if reward_mode == "value":
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
        return LogReturnPortfolioEnv(df=df, **env_kwargs)
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
