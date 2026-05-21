"""Shared helpers for the PortfolioAllocation_paolo_impruvements pipeline."""
from __future__ import annotations

import copy
import json
import os
import pickle
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
from stable_baselines3.common.vec_env import VecNormalize


# ---------- VecNormalize helpers (improvement #6) -----------------------------

def build_vecnormalize(venv, norm_cfg: dict | None = None) -> VecNormalize:
    """Wrap a vectorised env with VecNormalize using config knobs."""
    cfg = norm_cfg or {}
    return VecNormalize(
        venv,
        norm_obs=bool(cfg.get("norm_obs", True)),
        norm_reward=bool(cfg.get("norm_reward", True)),
        clip_obs=float(cfg.get("clip_obs", 10.0)),
        clip_reward=float(cfg.get("clip_reward", 10.0)),
        epsilon=float(cfg.get("epsilon", 1e-8)),
        gamma=float(cfg.get("gamma", 0.99)),
        training=True,
    )


def save_vecnormalize_stats(vn: VecNormalize, path: Path) -> None:
    """Persist only the running statistics + config of a VecNormalize wrapper.

    SB3's default `vn.save()` pickles the entire underlying venv (including
    the env's DataFrame with cov_list / return_list object columns), producing
    files that can be ~200 MB per walk-forward window. The running stats
    themselves (obs_rms.mean / .var, ret_rms.*) are KB-sized, so we save only
    those plus the config flags needed to reconstruct the wrapper at inference.
    """
    stats = {
        "obs_rms":     vn.obs_rms,
        "ret_rms":     vn.ret_rms,
        "clip_obs":    vn.clip_obs,
        "clip_reward": vn.clip_reward,
        "norm_obs":    vn.norm_obs,
        "norm_reward": vn.norm_reward,
        "epsilon":     vn.epsilon,
        "gamma":       vn.gamma,
    }
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_vecnormalize_stats(path: Path, venv) -> VecNormalize:
    """Load saved stats and wrap a fresh venv with them.

    Always returns a VecNormalize in eval mode (training=False, norm_reward=False)
    so running stats stay frozen at inference time and the realised reward
    distribution isn't mutated by the rollout.
    """
    with open(path, "rb") as f:
        stats = pickle.load(f)
    vn = VecNormalize(
        venv,
        norm_obs=stats["norm_obs"],
        norm_reward=False,
        clip_obs=stats["clip_obs"],
        clip_reward=stats["clip_reward"],
        epsilon=stats["epsilon"],
        gamma=stats["gamma"],
        training=False,
    )
    vn.obs_rms = stats["obs_rms"]
    vn.ret_rms = stats["ret_rms"]
    return vn


def vecnormalize_path_for(model_path: Path) -> Path:
    """`models/agent_<...>.zip` -> `models/vecnormalize_<...>.pkl`."""
    name = model_path.name
    if not name.startswith("agent_") or not name.endswith(".zip"):
        raise ValueError(f"Unexpected model filename: {name!r}; "
                         f"expected 'agent_<...>.zip'.")
    new_name = name.replace("agent_", "vecnormalize_", 1).replace(".zip", ".pkl")
    return model_path.parent / new_name


# ---------- Multi-seed ensembling helpers (improvement #8) -------------------

def get_seeds(config: dict) -> list[int]:
    """Return the list of seeds to use for multi-seed training (improvement #8).

    Reads config.seeds.list. Falls back to a single-element list with
    config.training.seed (or 42) for backward compatibility with prior folders
    that don't have a seeds block.
    """
    seeds_cfg = config.get("seeds")
    if seeds_cfg and isinstance(seeds_cfg.get("list"), list) and seeds_cfg["list"]:
        return [int(s) for s in seeds_cfg["list"]]
    legacy = int(config.get("training", {}).get("seed", 42))
    return [legacy]


def average_seed_actions(per_seed_actions: list[pd.DataFrame]) -> pd.DataFrame:
    """Average post-softmax weights across N seeds, row-by-row.

    Each input is a df_actions DataFrame from one seed's
    StockPortfolioEnv.save_action_memory() - index = date, columns = tickers,
    rows sum to 1. Averaging stays on the simplex (non-negative weights
    averaging to a non-negative mean that sums to 1 modulo float drift); we
    re-normalise rows to enforce sum=1 exactly.
    """
    if not per_seed_actions:
        raise ValueError("per_seed_actions is empty")
    base = per_seed_actions[0]
    aligned = [df.reindex(index=base.index, columns=base.columns).fillna(0.0)
               for df in per_seed_actions]
    mean = sum(aligned) / float(len(aligned))
    row_sums = mean.sum(axis=1).replace(0.0, 1.0)
    return mean.div(row_sums, axis=0)


def daily_return_from_weights(weights_df: pd.DataFrame,
                              trade_df: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio daily_return from a sequence of daily weight vectors.

    weights_df: index = date (Timestamp or str), columns = tickers, rows sum to 1.
    trade_df:   long-format with columns at least [date, tic, close].

    Returns a 2-column DataFrame [date, daily_return] following the env's
    convention: first row's daily_return = 0 (no prior price to compare to).

    Used by stage 3 to compute the ensemble's portfolio_return after
    averaging per-seed actions - because you cannot just average the per-seed
    portfolio_return series (the equity dynamics differ per seed).
    """
    prices = (trade_df.sort_values(["date", "tic"])
              .pivot_table(index="date", columns="tic", values="close"))
    prices.index = pd.to_datetime(prices.index)

    w = weights_df.copy()
    w.index = pd.to_datetime(w.index)
    w = w.reindex(index=prices.index).reindex(columns=prices.columns).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    port_return = (rets * w).sum(axis=1)
    port_return.iloc[0] = 0.0  # match env convention

    return pd.DataFrame({
        "date": prices.index.strftime("%Y-%m-%d"),
        "daily_return": port_return.values,
    })


def wrap_eval_env_with_ref_stats(sb_env, ref_vn: VecNormalize) -> VecNormalize:
    """Wrap a fresh eval venv with a snapshot of `ref_vn`'s running stats.

    Used by the early-stopping callback: every evaluation builds a fresh
    validation env, but it needs to apply the SAME normalisation as the
    training env to keep the policy on-distribution. deepcopy obs_rms /
    ret_rms so the eval rollout cannot mutate the training stats.
    """
    eval_vn = VecNormalize(
        sb_env,
        norm_obs=ref_vn.norm_obs,
        norm_reward=False,
        clip_obs=ref_vn.clip_obs,
        clip_reward=ref_vn.clip_reward,
        epsilon=ref_vn.epsilon,
        gamma=ref_vn.gamma,
        training=False,
    )
    eval_vn.obs_rms = copy.deepcopy(ref_vn.obs_rms)
    eval_vn.ret_rms = copy.deepcopy(ref_vn.ret_rms)
    return eval_vn


def split_train_validation(train_df: pd.DataFrame,
                           val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training slice into (train_only, validation) by date.

    Reserves the LAST `val_fraction` of unique trading dates for validation.
    Re-factorises the integer index of each slice so the env's `df.loc[day]`
    lookup works (the env expects a dense 0..N-1 day index).

    Friendly warnings when val_fraction is unusual:
      * < 0.05  -> too small, Sharpe estimate becomes noisy.
      * > 0.30  -> too large, train_only shrinks materially.
    See walk_forward_time_split_and_early_stopping.md at the project root for
    detailed guidance on choosing this value.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")
    if val_fraction < 0.05:
        print(f"  NOTE: val_fraction={val_fraction} is small (<5%). "
              f"Validation Sharpe may be too noisy for reliable ES decisions.")
    elif val_fraction > 0.30:
        print(f"  NOTE: val_fraction={val_fraction} is large (>30%). "
              f"train_only loses a lot of data; PPO may underfit.")
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

    2. (Improvement #2, drift-adjusted in folder 8) Transaction-cost penalty
       on the REAL rebalancing required to maintain target weights, not on
       the change in target weights. Subtracted from both the reward AND
       the book equity (portfolio_value / asset_memory / portfolio_return_memory)
       so downstream consumers see realistic post-cost performance and the
       stage-3 (env) equity curve aligns with stage-4 (backtrader replay).

       Math: w_drift_i  = w_prev_i * (1 + r_i) / (1 + R)
             turnover   = |w_target - w_drift|.sum()
             tc_fraction = tc_penalty * turnover
             net_return  = (1 + gross_return) * (1 - tc_fraction) - 1

       where r_i is the per-ticker return for the bar and R is the portfolio
       gross return. The drift correction makes a buy-and-hold strategy
       (constant target weights) correctly INCUR cost for the daily
       rebalancing required to maintain those weights against price drift -
       matching what a real broker charges. Prior folders (3-7) used the
       naive |w_target - w_prev_target| formula which under-charged cost and
       caused a large stage-3/stage-4 friction gap.

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
                 risk_off_enabled: bool = False,
                 turbulence_threshold: float = 70.0,
                 **kwargs):
        if reward_kind not in _REWARD_KINDS:
            raise ValueError(
                f"Unknown reward_kind={reward_kind!r}; "
                f"expected one of {_REWARD_KINDS}."
            )
        super().__init__(*args, **kwargs)
        self.tc_penalty           = float(tc_penalty)
        self.reward_kind          = reward_kind
        self.diff_ratio_eta       = float(diff_ratio_eta)
        # Improvement #7: turbulence-gated risk-off rule.
        self.risk_off_enabled     = bool(risk_off_enabled)
        self.turbulence_threshold = float(turbulence_threshold)
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

    def _risk_off_active(self) -> bool:
        """True iff the current bar's turbulence exceeds the threshold.

        Reads from self.data["turbulence"], which exists only when stage 1 ran
        FeatureEngineer(use_turbulence=True). When the column is missing, the
        gate is silently inactive - useful so the same env class still works on
        prior-folder data pickles without crashing.
        """
        if not self.risk_off_enabled:
            return False
        if "turbulence" not in self.data.columns:
            return False
        val = float(self.data["turbulence"].values[0])
        return val > self.turbulence_threshold

    def step(self, actions):
        obs, _reward, done, truncated, info = super().step(actions)
        if done or not self.portfolio_return_memory:
            return obs, self.reward, done, truncated, info

        # Improvement #7: turbulence-gated risk-off rule. When turbulence on
        # the new bar exceeds the threshold, override the agent's weights to
        # ZERO (= go to cash) and book a zero gross return for the bar. The TC
        # penalty below then computes turnover against the prior day's weights
        # (so we still pay to liquidate / re-enter realistically).
        if self._risk_off_active():
            n = self.stock_dim
            self.actions_memory[-1] = [0.0] * n
            self.portfolio_return_memory[-1] = 0.0
            self.portfolio_value = float(self.asset_memory[-2])
            self.asset_memory[-1] = self.portfolio_value

        gross_return     = float(self.portfolio_return_memory[-1])
        effective_return = gross_return

        # Improvement #2 (drift-adjusted, folder 8): transaction-cost penalty
        # on the REAL rebalancing required to maintain the agent's target
        # weights, not on the change in target weights.
        #
        # Before this fix, turnover was computed as |w_target - w_prev_target|,
        # which IGNORED the natural weight drift caused by per-ticker price
        # moves. A buy-and-hold strategy showed zero turnover in the env even
        # though backtrader has to rebalance daily to track the constant
        # targets - producing a large stage-3-vs-stage-4 friction gap.
        #
        # The drift-adjusted formula computes the weight vector at the START
        # of the current bar (post per-ticker return, pre-rebalance):
        #     w_drift_i = w_prev_i * (1 + r_i) / (1 + R)
        # where r_i is the per-ticker simple return for the bar and R is the
        # portfolio gross return. Real turnover required to reach the agent's
        # new target is |w_target - w_drift|.sum() - matches backtrader's
        # commission base exactly (modulo integer-share rounding).
        if self.tc_penalty > 0.0 and len(self.actions_memory) >= 2 and self.day > 0:
            w_target = np.asarray(self.actions_memory[-1], dtype=float)
            w_prev   = np.asarray(self.actions_memory[-2], dtype=float)
            prev_close = self.df.loc[self.day - 1, "close"].values.astype(float)
            curr_close = self.data["close"].values.astype(float)
            growth     = curr_close / prev_close
            drifted    = w_prev * growth
            total      = float(drifted.sum())
            if total > 0.0:
                w_drift = drifted / total
            else:
                # Starting from cash (all zeros) - no drift to apply.
                w_drift = w_prev
            turnover    = float(np.abs(w_target - w_drift).sum())
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

    # Improvement #7: turbulence-gated risk-off rule.
    ro_cfg               = config.get("risk_off", {})
    risk_off_enabled     = bool(ro_cfg.get("enabled", False))
    turbulence_threshold = float(ro_cfg.get("turbulence_threshold", 70.0))
    if risk_off_enabled and "turbulence" not in df.columns:
        print(
            f"WARNING: risk_off.enabled=true but the trade/train DataFrame has no "
            f"`turbulence` column. Re-run 01_get_data.py with data.use_turbulence=true "
            f"to compute it. The gate will be silently inactive until then."
        )

    if reward_mode == "value":
        if tc_penalty > 0.0:
            print(
                f"WARNING: env.transaction_cost_penalty={tc_penalty} is set but "
                f"env.reward_mode='value' - the TC penalty is only applied in "
                f"shaped-reward modes (log_return / diff_sharpe / diff_sortino). "
                f"The penalty will be ignored."
            )
        if risk_off_enabled:
            print(
                f"WARNING: risk_off.enabled=true with reward_mode='value' - the "
                f"upstream StockPortfolioEnv does not implement the gate, so it "
                f"will be ignored. Set reward_mode to a shaped mode to activate."
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
            risk_off_enabled=risk_off_enabled,
            turbulence_threshold=turbulence_threshold,
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
                 verbose: int = 1,
                 ref_vecnormalize: VecNormalize | None = None,
                 vn_save_path: Path | None = None):
        super().__init__(verbose)
        self.val_df          = val_df
        self.config          = config
        self.stock_dim       = stock_dim
        self.model_save_path = Path(model_save_path)
        self.history_path    = Path(history_path)
        self.eval_freq       = int(eval_freq)
        self.patience        = int(patience)
        self.min_delta       = float(min_delta)

        # Improvement #6: optional VecNormalize support.
        # ref_vecnormalize is the training-env VecNormalize wrapper. When set,
        # the validation rollout uses a frozen snapshot of its running stats so
        # the policy sees on-distribution observations. vn_save_path is where
        # to persist a stats snapshot whenever the best model is saved, so the
        # checkpoint on disk and the saved stats stay matched.
        self.ref_vn          = ref_vecnormalize
        self.vn_save_path    = Path(vn_save_path) if vn_save_path else None

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
            # Improvement #6: also persist a stats snapshot so the saved model
            # and the saved VecNormalize stats correspond to the same training
            # checkpoint. Without this, the .pkl would lag the .zip by however
            # many steps it took ES to trigger after the best was found.
            if self.ref_vn is not None and self.vn_save_path is not None:
                save_vecnormalize_stats(self.ref_vn, self.vn_save_path)
            self._saved_any = True
            if self.verbose:
                prev_str = "-inf" if prev == -np.inf else f"{prev:.4f}"
                vn_note  = "  +stats" if (self.ref_vn is not None and self.vn_save_path is not None) else ""
                print(f"  [step {self.num_timesteps}] val Sharpe improved: "
                      f"{prev_str} -> {sharpe:.4f}, saved best to {self.model_save_path.name}{vn_note}")
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
            if self.ref_vn is not None and self.vn_save_path is not None:
                save_vecnormalize_stats(self.ref_vn, self.vn_save_path)
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
        """Roll the current model through the validation env, compute Sharpe.

        With improvement #6 enabled (self.ref_vn set), the eval venv is wrapped
        with a snapshot of the training VecNormalize so observations are
        normalised the same way they are during training. Without that, the
        policy would see raw observations at eval time and produce nonsense.
        """
        val_gym     = make_portfolio_env(self.val_df, self.config, self.stock_dim)
        sb_env, _   = val_gym.get_sb_env()

        if self.ref_vn is not None:
            sb_env = wrap_eval_env_with_ref_stats(sb_env, self.ref_vn)

        obs        = sb_env.reset()
        n_days     = len(val_gym.df.index.unique())
        account_df = None
        for i in range(n_days):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, dones, _ = sb_env.step(action)
            if i == n_days - 2:
                account_df = sb_env.env_method("save_asset_memory")[0]
            if dones[0]:
                break
        if account_df is None or "daily_return" not in account_df.columns:
            return -np.inf
        returns = np.asarray(account_df["daily_return"].values, dtype=float)
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
