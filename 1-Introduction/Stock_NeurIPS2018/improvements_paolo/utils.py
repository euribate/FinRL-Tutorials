"""Shared helpers for the improvements_paolo pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

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
    """Convert JSON-friendly strings into Python objects."""
    if not policy_kwargs:
        return None
    parsed = dict(policy_kwargs)
    if isinstance(parsed.get("activation_fn"), str):
        parsed["activation_fn"] = resolve_activation(parsed["activation_fn"])
    return parsed


def parse_model_kwargs(model_kwargs: dict | None, n_actions: int) -> dict:
    """Convert JSON-friendly strings into SB3 objects (e.g. action_noise)."""
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


def build_env_kwargs(config: dict, stock_dim: int, mode: str) -> dict:
    """Assemble env_kwargs for StockTradingEnv. mode: 'train' or 'trade'."""
    env_cfg = config["env"]
    data_cfg = config["data"]
    if mode == "train":
        turb = env_cfg.get("turbulence_threshold_train")
    elif mode == "trade":
        turb = env_cfg.get("turbulence_threshold_trade")
    else:
        raise ValueError(f"mode must be 'train' or 'trade', got {mode!r}")

    n_indicators = len(data_cfg["indicators"])
    state_space = 1 + 2 * stock_dim + n_indicators * stock_dim

    return {
        "hmax": env_cfg["hmax"],
        "initial_amount": env_cfg["initial_amount"],
        "num_stock_shares": [0] * stock_dim,
        "buy_cost_pct":  [env_cfg["buy_cost_pct"]]  * stock_dim,
        "sell_cost_pct": [env_cfg["sell_cost_pct"]] * stock_dim,
        "reward_scaling": env_cfg["reward_scaling"],
        "state_space": state_space,
        "stock_dim": stock_dim,
        "tech_indicator_list": data_cfg["indicators"],
        "action_space": stock_dim,
        "turbulence_threshold": turb,
        "risk_indicator_col": env_cfg["risk_indicator_col"],
    }
