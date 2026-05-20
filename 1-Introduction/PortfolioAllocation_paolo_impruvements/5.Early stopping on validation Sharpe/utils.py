"""Shared helpers for the PortfolioAllocation_paolo_impruvements pipeline."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch.nn as nn
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


def split_train_validation(train_df: pd.DataFrame,
                           val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training slice into (train_only, validation) by date.

    Reserves the LAST `val_fraction` of unique trading dates for validation.
    Re-factorises the integer index of each slice so the env's `df.loc[day]`
    lookup works (the env expects a dense 0..N-1 day index).
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")
    dates = pd.Index(train_df["date"].unique()).sort_values()
    n_val = max(int(round(len(dates) * val_fraction)), 1)
    cut_date = dates[len(dates) - n_val]
    train_only = train_df[train_df["date"] <  cut_date].copy()
    validation = train_df[train_df["date"] >= cut_date].copy()
    for d in (train_only, validation):
        d.sort_values(["date", "tic"], inplace=True, ignore_index=True)
        d.index = d.date.factorize()[0]
    return train_only, validation


def fetch_yahoo_with_retry(ticker: str, start: str, end: str,
                            retries: int = 3, delay: float = 2.0) -> pd.DataFrame:
    """Download a ticker's OHLCV via FinRL's YahooDownloader, retrying on transient failures.

    Yahoo's cookie/crumb host (`fc.yahoo.com`) intermittently fails DNS
    resolution, which makes a single yfinance call flaky even when the
    network is otherwise fine. We retry a few times with a short delay
    before giving up. Empty DataFrames (Yahoo returned 200 but no rows)
    are also treated as failures so the caller doesn't proceed with an
    empty benchmark series.
    """
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            df = YahooDownloader(start_date=start, end_date=end,
                                 ticker_list=[ticker]).fetch_data()
            if df is None or len(df) == 0:
                raise RuntimeError(f"Empty download for {ticker} ({start} -> {end}).")
            return df
        except Exception as e:
            last_exc = e
            if attempt < retries:
                print(f"  Yahoo download for {ticker} failed "
                      f"(attempt {attempt}/{retries}): {e}")
                print(f"  Retrying in {delay}s...")
                time.sleep(delay)
    raise RuntimeError(
        f"Yahoo download for {ticker} failed after {retries} attempts. "
        f"Last error: {last_exc}"
    )


_REWARD_KINDS = ("log_return", "diff_sharpe", "diff_sortino")


class LogReturnPortfolioEnv(StockPortfolioEnv):
    """StockPortfolioEnv variant with three layered improvements:

    1. (Improvement #1) Reward shaped from the upstream portfolio_return
       instead of raw new_portfolio_value. The `reward_kind` constructor
       argument selects the shape:
         * "log_return"   = log(1 + effective_return) * reward_scaling
         * "diff_sharpe"  = Moody & Saffell differential Sharpe ratio
                            (incremental contribution to running Sharpe)
         * "diff_sortino" = differential Sortino-style ratio that only
                            penalises downside variance

    2. (Improvement #2) Optional transaction-cost penalty: when `tc_penalty > 0`,
       a per-unit-turnover cost is subtracted from both the reward AND from the
       book equity (portfolio_value / asset_memory / portfolio_return_memory),
       so that downstream consumers (stage 3's DRL_prediction, equity plot,
       QuantStats tearsheet) reflect realistic post-cost performance.

       Math: turnover    = |weights_t - weights_{t-1}|.sum()
             tc_fraction = tc_penalty * turnover
             net_return  = (1 + gross_return) * (1 - tc_fraction) - 1

    3. (Improvement #4) Differential Sharpe / Sortino. After computing the
       (possibly cost-adjusted) effective return r_t, the env maintains running
       exponentially-weighted moments and rewards the per-step contribution to
       the running ratio:

           A_t = A_{t-1} + eta * (r_t - A_{t-1})           # EMA of return
           B_t = B_{t-1} + eta * (r_t^2 - B_{t-1})         # EMA of return^2
           D_t = D_{t-1} + eta * (min(r_t,0)^2 - D_{t-1})  # EMA of downside^2

           DSR = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^{1.5}
           DDR = (D_{t-1} * dA - 0.5 * A_{t-1} * dD) / D_{t-1}^{1.5}

       eta ~ 1/252 corresponds to a one-year EMA. The reward is the per-step
       DSR (or DDR) scaled by reward_scaling. Trains the agent to optimise a
       running risk-adjusted ratio rather than raw return.

    Setting `reward_kind="log_return"` and `tc_penalty=0` short-circuits both
    improvements #2 and #4; the env then behaves identically to the variant
    in ../1.reward_returns/.
    """

    def __init__(self, *args,
                 tc_penalty: float = 0.0,
                 reward_kind: str = "log_return",
                 diff_ratio_eta: float = 1.0 / 252.0,
                 **kwargs):
        if reward_kind not in _REWARD_KINDS:
            raise ValueError(
                f"Unknown reward_kind={reward_kind!r}; "
                f"expected one of {_REWARD_KINDS}."
            )
        super().__init__(*args, **kwargs)
        self.tc_penalty     = float(tc_penalty)
        self.reward_kind    = reward_kind
        self.diff_ratio_eta = float(diff_ratio_eta)
        # Running EMA moments used for diff_sharpe / diff_sortino.
        self._A = 0.0  # E[R_t]
        self._B = 0.0  # E[R_t^2]
        self._D = 0.0  # E[min(R_t, 0)^2]  (downside second moment)

    def reset(self, **kwargs):
        result = super().reset(**kwargs)
        # Reset running EMAs between episodes so they don't leak across resets.
        self._A = 0.0
        self._B = 0.0
        self._D = 0.0
        return result

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
            self.portfolio_return_memory[-1] = effective_return
            base = float(self.asset_memory[-2])
            self.portfolio_value = base * (1.0 + effective_return)
            self.asset_memory[-1] = self.portfolio_value

        if self.reward_kind == "log_return":
            self.reward = float(np.log(1.0 + effective_return) * self.reward_scaling)
        elif self.reward_kind == "diff_sharpe":
            self.reward = self._diff_sharpe_reward(effective_return)
        elif self.reward_kind == "diff_sortino":
            self.reward = self._diff_sortino_reward(effective_return)

        return obs, self.reward, done, truncated, info

    def _diff_sharpe_reward(self, r: float) -> float:
        """Moody & Saffell DSR: incremental contribution to running Sharpe ratio."""
        A_prev, B_prev = self._A, self._B
        dA = r - A_prev
        dB = r * r - B_prev
        var = max(B_prev - A_prev * A_prev, 0.0)
        denom = max(var ** 1.5, 1e-8)
        dsr = (B_prev * dA - 0.5 * A_prev * dB) / denom
        # Advance the running EMAs AFTER computing DSR (uses lagged A, B).
        self._A = A_prev + self.diff_ratio_eta * dA
        self._B = B_prev + self.diff_ratio_eta * dB
        return float(dsr * self.reward_scaling)

    def _diff_sortino_reward(self, r: float) -> float:
        """Differential Sortino-style ratio using downside second moment only."""
        A_prev, D_prev = self._A, self._D
        dA = r - A_prev
        downside_sq = (min(r, 0.0)) ** 2
        dD = downside_sq - D_prev
        denom = max(D_prev ** 1.5, 1e-8)
        ddr = (D_prev * dA - 0.5 * A_prev * dD) / denom
        self._A = A_prev + self.diff_ratio_eta * dA
        self._D = D_prev + self.diff_ratio_eta * dD
        return float(ddr * self.reward_scaling)


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

    Dispatches on `config.env` knobs:
      * reward_mode (default "value"):
        - "value"        -> upstream StockPortfolioEnv (reward = portfolio value).
        - "log_return"   -> LogReturnPortfolioEnv with reward_kind="log_return".
        - "diff_sharpe"  -> LogReturnPortfolioEnv with reward_kind="diff_sharpe".
        - "diff_sortino" -> LogReturnPortfolioEnv with reward_kind="diff_sortino".
      * transaction_cost_penalty (float, default 0.0):
        - Per-unit-turnover cost rate. Applied for any non-"value" reward_mode.
        - When 0, no cost deduction (reproduces the no-TC variant).
        - When > 0, deducted from both reward and book equity so stages 3/4/5
          see realistic post-cost performance.
      * diff_ratio_eta (float, default 1/252):
        - EMA decay for the running A / B / D moments used by diff_sharpe and
          diff_sortino. 1/252 ~ one trading year of history.

    Warnings:
      * reward_mode='log_return' with tiny reward_scaling (<0.1) - log returns
        are O(0.01)/day; e.g. 1e-4 scaling starves the gradient.
      * reward_mode='value' with transaction_cost_penalty>0 - TC penalty in
        value mode is out of scope; the penalty is ignored.
    """
    env_kwargs     = build_env_kwargs(config, stock_dim)
    reward_mode    = config["env"].get("reward_mode", "value")
    tc_penalty     = float(config["env"].get("transaction_cost_penalty", 0.0))
    diff_ratio_eta = float(config["env"].get("diff_ratio_eta", 1.0 / 252.0))

    if reward_mode == "value":
        if tc_penalty > 0.0:
            print(
                f"WARNING: env.transaction_cost_penalty={tc_penalty} is set but "
                f"env.reward_mode='value' - the TC penalty is only applied in "
                f"shaped-reward modes (log_return / diff_sharpe / diff_sortino). "
                f"The penalty will be ignored."
            )
        return StockPortfolioEnv(df=df, **env_kwargs)

    if reward_mode in _REWARD_KINDS:
        scaling = float(env_kwargs.get("reward_scaling", 1.0))
        if reward_mode == "log_return" and scaling < 0.1:
            print(
                f"WARNING: reward_mode='log_return' with reward_scaling={scaling}. "
                f"Log returns are O(0.01) per day; this scaling will shrink the "
                f"reward signal to ~{scaling * 0.01:g} - effectively zero gradient. "
                f"Recommended: set env.reward_scaling=1.0 in config.json."
            )
        return LogReturnPortfolioEnv(
            df=df,
            tc_penalty=tc_penalty,
            reward_kind=reward_mode,
            diff_ratio_eta=diff_ratio_eta,
            **env_kwargs,
        )

    raise ValueError(
        f"Unknown env.reward_mode={reward_mode!r}; expected 'value', "
        f"'log_return', 'diff_sharpe', or 'diff_sortino'."
    )


class ValidationSharpeCallback(BaseCallback):
    """Early stopping on annualised Sharpe of the validation slice (improvement #5).

    Every `eval_freq` training timesteps, the current policy is rolled out
    deterministically through a fresh env built from `val_df`, the env's
    `portfolio_return_memory` is extracted, and annualised Sharpe is computed
    as `sqrt(252) * mean(returns) / std(returns)`. The best model so far is
    saved to `model_save_path`; if no improvement larger than `min_delta` is
    seen for `patience` consecutive evaluations, `_on_step` returns False and
    SB3 terminates training early.

    A history JSON (timesteps, sharpe, improved) is written next to the model
    for inspection.
    """

    def __init__(self,
                 val_df: pd.DataFrame,
                 config: dict,
                 stock_dim: int,
                 model_save_path: Path,
                 history_path: Path,
                 eval_freq: int = 5000,
                 patience: int = 5,
                 min_delta: float = 0.01,
                 verbose: int = 1):
        super().__init__(verbose)
        self.val_df          = val_df
        self.config          = config
        self.stock_dim       = stock_dim
        self.model_save_path = Path(model_save_path)
        self.history_path    = Path(history_path)
        self.eval_freq       = int(eval_freq)
        self.patience        = int(patience)
        self.min_delta       = float(min_delta)

        self.best_sharpe          = -np.inf
        self.no_improvement_count = 0
        self.history: list[dict]  = []
        self._next_eval           = self.eval_freq  # first eval at eval_freq steps
        self._saved_any           = False

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_freq

        sharpe = self._evaluate_sharpe()
        improved = sharpe > (self.best_sharpe + self.min_delta)
        if improved:
            prev = self.best_sharpe
            self.best_sharpe = sharpe
            self.no_improvement_count = 0
            self.model.save(str(self.model_save_path))
            self._saved_any = True
            if self.verbose:
                prev_str = "-inf" if prev == -np.inf else f"{prev:.4f}"
                print(f"  [step {self.num_timesteps}] val Sharpe improved: "
                      f"{prev_str} -> {sharpe:.4f}, saved best to {self.model_save_path.name}")
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(f"  [step {self.num_timesteps}] val Sharpe={sharpe:.4f} "
                      f"(best={self.best_sharpe:.4f}, "
                      f"no-improvement {self.no_improvement_count}/{self.patience})")

        self.history.append({
            "timesteps": int(self.num_timesteps),
            "sharpe":    float(sharpe),
            "improved":  bool(improved),
            "best_so_far": float(self.best_sharpe),
        })

        if self.no_improvement_count >= self.patience:
            if self.verbose:
                print(f"  Early stopping: {self.patience} consecutive evaluations "
                      f"without improvement >= {self.min_delta}.")
            self._dump_history()
            return False
        return True

    def _on_training_end(self) -> None:
        self._dump_history()
        if not self._saved_any:
            # No checkpoint was ever saved (no improvement detected). Save the
            # final policy state so stage 3 always has something to load.
            self.model.save(str(self.model_save_path))
            if self.verbose:
                print(f"  No improvement detected during training; "
                      f"saved final policy to {self.model_save_path.name}.")

    def _dump_history(self) -> None:
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"  WARNING: could not write history to {self.history_path}: {e}")

    def _evaluate_sharpe(self) -> float:
        """Roll the current model through the validation env, compute Sharpe."""
        val_gym       = make_portfolio_env(self.val_df, self.config, self.stock_dim)
        sb_env, obs   = val_gym.get_sb_env()
        n_days        = len(val_gym.df.index.unique())
        account_df    = None
        for i in range(n_days):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = sb_env.step(action)
            if i == n_days - 2:
                # Grab the env memory just before the terminal step triggers
                # the auto-reset that DummyVecEnv applies on episode end.
                account_df = sb_env.env_method("save_asset_memory")[0]
            if dones[0]:
                break
        if account_df is None or "daily_return" not in account_df.columns:
            return -np.inf
        returns = np.asarray(account_df["daily_return"].values, dtype=float)
        # Drop the placeholder 0 at index 0 (env initialises portfolio_return_memory=[0]).
        if len(returns) > 1:
            returns = returns[1:]
        if len(returns) == 0:
            return -np.inf
        std = float(returns.std())
        if std < 1e-12:
            return -np.inf
        return float(np.sqrt(252.0) * float(returns.mean()) / std)


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
