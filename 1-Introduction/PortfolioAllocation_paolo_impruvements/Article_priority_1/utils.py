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
    """Return the FULL candidate list of seeds for training (improvement #8).

    Reads config.seeds.list. Falls back to a single-element list with
    config.training.seed (or 42) for backward compatibility with prior folders
    that don't have a seeds block.

    Used by 02_train.py - training always processes every candidate so the user
    can later pick the best-converged subset for deployment.
    """
    seeds_cfg = config.get("seeds")
    if seeds_cfg and isinstance(seeds_cfg.get("list"), list) and seeds_cfg["list"]:
        return [int(s) for s in seeds_cfg["list"]]
    legacy = int(config.get("training", {}).get("seed", 42))
    return [legacy]


def pick_ensemble_seeds(config: dict, model_dir, algo: str = "ppo") -> list[int]:
    """Return the seeds to use AT INFERENCE TIME from the trained candidate pool.

    Behaviour:
      * If config.seeds.ensemble_size is unset (or >= len(seeds.list)), return
        the full seeds.list - all trained candidates participate.
      * If config.seeds.ensemble_size = N < len(seeds.list), rank the candidates
        by training convergence quality (more validation Sharpe improvements is
        better; ties broken by best Sharpe), then return the top N.

    Ranking ignores walk-forward histories (those have _w<i> in the filename) -
    only the single-split production histories are considered.

    Missing history files: a candidate with no history file ranks last
    (treated as 0 improvements, -inf best Sharpe). Useful when seeds.list
    contains a seed that has not been trained yet - it won't poison the
    ranking but also won't be selected unless ensemble_size demands it.
    """
    from pathlib import Path  # local import keeps the top of utils.py clean

    all_seeds = get_seeds(config)
    seeds_cfg = config.get("seeds", {}) or {}
    n_req     = seeds_cfg.get("ensemble_size")
    if n_req is None:
        return all_seeds
    try:
        n_req = int(n_req)
    except Exception as e:
        raise ValueError(f"seeds.ensemble_size must be an int; got {n_req!r}.") from e
    if n_req <= 0:
        raise ValueError(f"seeds.ensemble_size must be > 0; got {n_req}.")
    if n_req >= len(all_seeds):
        return all_seeds

    model_dir = Path(model_dir)
    ranked: list[tuple[int, int, float]] = []
    for s in all_seeds:
        h = model_dir / f"agent_{algo}_s{s}.history.json"
        if not h.exists():
            ranked.append((s, 0, float("-inf")))
            continue
        try:
            with open(h) as f:
                data = json.load(f)
            n_imp = sum(1 for d in data if d.get("improved"))
            best  = max((d.get("sharpe", float("-inf")) for d in data),
                        default=float("-inf"))
            ranked.append((s, n_imp, float(best)))
        except Exception:
            ranked.append((s, 0, float("-inf")))

    ranked.sort(key=lambda r: (-r[1], -r[2]))
    return [r[0] for r in ranked[:n_req]]


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
                              trade_df: pd.DataFrame,
                              tc_penalty: float = 0.0,
                              turnover_mode: str = "naive") -> pd.DataFrame:
    """Compute portfolio daily_return from a sequence of daily weight vectors.

    weights_df: index = date (Timestamp or str), columns = tickers, rows sum to 1.
    trade_df:   long-format with columns at least [date, tic, close].
    tc_penalty: per-unit-turnover cost rate (drift-adjusted formula, matches
                LogReturnPortfolioEnv.step()). Set to 0.0 to compute a
                friction-free curve.

    Returns a 2-column DataFrame [date, daily_return] following the env's
    convention: first row's daily_return = 0 (no prior price to compare to).

    Used by stage 3 to compute the ensemble's portfolio_return after
    averaging per-seed actions - because you cannot just average the per-seed
    portfolio_return series (the equity dynamics differ per seed).

    TC formula (matches LogReturnPortfolioEnv.step()):
      growth_i  = price_i[t] / price_i[t-1]
      drifted_i = w_prev_i * growth_i
      w_drift_i = drifted_i / sum(drifted)         (= w_prev if sum == 0)
      turnover  = |w_target - w_drift|.sum()
      net_ret   = (1 + gross_ret) * (1 - tc_penalty * turnover) - 1

    The drift-adjusted turnover correctly charges for the rebalancing required
    to maintain target weights against per-ticker price drift - a buy-and-hold
    strategy still incurs cost, matching what a broker actually charges.
    Without this, the equity curve overstates returns by ~tc_penalty * 252 *
    mean_daily_turnover per year. With the default config (tc_penalty=0.001,
    ~4-5% daily turnover) that is ~1.2 pp/yr drag that would otherwise be missed.
    """
    prices = (trade_df.sort_values(["date", "tic"])
              .pivot_table(index="date", columns="tic", values="close"))
    prices.index = pd.to_datetime(prices.index)

    w = weights_df.copy()
    w.index = pd.to_datetime(w.index)
    w = w.reindex(index=prices.index).reindex(columns=prices.columns).fillna(0.0)

    rets = prices.pct_change().fillna(0.0)
    gross_return = (rets * w).sum(axis=1)
    gross_return.iloc[0] = 0.0  # match env convention - no return on first row

    if tc_penalty <= 0.0:
        port_return = gross_return
    else:
        if turnover_mode == "naive":
            # Article-faithful: change in the agent's TARGET weights.
            turnover = (w - w.shift(1)).abs().sum(axis=1)
        else:
            # Drift-adjusted: compare w_target_t to w_drift_t, where w_drift_t is
            # what w[t-1] becomes after one bar of per-ticker price growth
            # (matches the env's snapshot at the START of bar t, pre-rebalance).
            growth   = 1.0 + rets
            drifted  = w.shift(1).fillna(0.0) * growth
            total    = drifted.sum(axis=1)
            w_drift  = drifted.div(total.where(total > 0.0, 1.0), axis=0)
            turnover = (w - w_drift).abs().sum(axis=1)
        turnover.iloc[0] = 0.0
        tc_drag  = (tc_penalty * turnover).clip(upper=1.0)
        port_return = (1.0 + gross_return) * (1.0 - tc_drag) - 1.0

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


_REWARD_KINDS = ("log_return", "diff_sharpe", "diff_sortino",
                 "article_absolute", "article_benchmark")


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
                 turnover_mode: str = "naive",
                 diff_ratio_eta: float = 1.0 / 252.0,
                 risk_off_enabled: bool = False,
                 turbulence_threshold: float = 70.0,
                 article_return_scale: float = 1000.0,
                 article_lambda_to: float = 0.003,
                 article_lambda_conc: float = 0.1,
                 cash_enabled: bool = False,
                 cash_ticker: str = "CASH",
                 **kwargs):
        if reward_kind not in _REWARD_KINDS:
            raise ValueError(
                f"Unknown reward_kind={reward_kind!r}; "
                f"expected one of {_REWARD_KINDS}."
            )
        if turnover_mode not in ("naive", "drift_adjusted"):
            raise ValueError(f"Unknown turnover_mode={turnover_mode!r}; "
                             f"expected 'naive' or 'drift_adjusted'.")
        super().__init__(*args, **kwargs)
        self.tc_penalty           = float(tc_penalty)
        self.reward_kind          = reward_kind
        self.turnover_mode        = turnover_mode
        self.diff_ratio_eta       = float(diff_ratio_eta)
        # Improvement #7: turbulence-gated risk-off rule.
        self.risk_off_enabled     = bool(risk_off_enabled)
        self.turbulence_threshold = float(turbulence_threshold)
        # Article reward (priority-1): coefficients for article_absolute /
        # article_benchmark. HHI concentration + turnover shaping penalties.
        self.article_return_scale = float(article_return_scale)
        self.article_lambda_to    = float(article_lambda_to)
        self.article_lambda_conc  = float(article_lambda_conc)
        # Cash-as-synthetic-asset: when true, one asset is CASH; the risk-off
        # gate routes to 100% cash instead of zeroing all weights. The weight
        # vector follows alphabetical ticker order, so locate CASH by name.
        self.cash_enabled         = bool(cash_enabled)
        self.cash_ticker          = str(cash_ticker)
        self.cash_idx             = -1
        if self.cash_enabled:
            tics = list(self.data.tic.values)
            if self.cash_ticker in tics:
                self.cash_idx = tics.index(self.cash_ticker)
            else:
                print(f"WARNING: cash_enabled=true but ticker {self.cash_ticker!r} "
                      f"not found in the data ({tics}). Risk-off gate will fall "
                      f"back to zero-weight behaviour.")
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

        # Improvement #7: turbulence-gated risk-off rule. When turbulence on the
        # new bar exceeds the threshold, override the agent's weights to go to
        # cash. With an explicit CASH asset (cash_enabled) route 100% into it and
        # book the cash bar return; otherwise zero all weights and book 0%. The
        # TC penalty below then charges turnover for the liquidation / re-entry.
        if self._risk_off_active():
            n = self.stock_dim
            if self.cash_enabled and self.cash_idx >= 0:
                w = [0.0] * n
                w[self.cash_idx] = 1.0
                self.actions_memory[-1] = w
                if self.day > 0:
                    prev_close = self.df.loc[self.day - 1, "close"].values.astype(float)
                    curr_close = self.data["close"].values.astype(float)
                    cash_ret = float(curr_close[self.cash_idx] / prev_close[self.cash_idx] - 1.0)
                else:
                    cash_ret = 0.0
                self.portfolio_return_memory[-1] = cash_ret
                base = float(self.asset_memory[-2])
                self.portfolio_value = base * (1.0 + cash_ret)
                self.asset_memory[-1] = self.portfolio_value
            else:
                self.actions_memory[-1] = [0.0] * n
                self.portfolio_return_memory[-1] = 0.0
                self.portfolio_value = float(self.asset_memory[-2])
                self.asset_memory[-1] = self.portfolio_value

        gross_return     = float(self.portfolio_return_memory[-1])
        effective_return = gross_return

        # Always compute the drift-adjusted turnover and the HHI concentration of
        # the target weights. Both are needed by the article reward EVEN WHEN
        # tc_penalty == 0; the real transaction cost (below) is applied only when
        # tc_penalty > 0. See folder-8 README A.12/A.20 for the drift-adjusted
        # turnover rationale.
        turnover = 0.0
        hhi      = 0.0
        if self.actions_memory:
            w_now = np.asarray(self.actions_memory[-1], dtype=float)
            hhi = float(np.sum(w_now * w_now))
        if len(self.actions_memory) >= 2 and self.day > 0:
            w_target = np.asarray(self.actions_memory[-1], dtype=float)
            w_prev   = np.asarray(self.actions_memory[-2], dtype=float)
            if self.turnover_mode == "naive":
                # Article-faithful: change in the agent's TARGET weights.
                turnover = float(np.abs(w_target - w_prev).sum())
            else:
                # drift_adjusted: charge the real rebalancing trade (matches a
                # broker). w_drift = previous weights after one bar of price drift.
                prev_close = self.df.loc[self.day - 1, "close"].values.astype(float)
                curr_close = self.data["close"].values.astype(float)
                growth     = curr_close / prev_close
                drifted    = w_prev * growth
                total      = float(drifted.sum())
                w_drift    = drifted / total if total > 0.0 else w_prev
                turnover   = float(np.abs(w_target - w_drift).sum())

        # Real transaction cost: deduct from the booked return AND the equity so
        # downstream consumers see post-cost performance.
        if self.tc_penalty > 0.0 and turnover > 0.0:
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
        elif self.reward_kind in ("article_absolute", "article_benchmark"):
            self.reward = self._article_reward(effective_return, turnover, hhi)

        return obs, self.reward, done, truncated, info

    @staticmethod
    def _safe_log1p(r: float) -> float:
        """log(1+r) guarded against r <= -1 (total wipeout)."""
        return float(np.log(max(1.0 + r, 1e-8)))

    def _benchmark_return(self) -> float:
        """Per-bar benchmark return from the broadcast 'benchmark_return' column."""
        if "benchmark_return" in self.data.columns:
            return float(self.data["benchmark_return"].values[0])
        return 0.0

    def _article_reward(self, r_net: float, turnover: float, hhi: float) -> float:
        """Article reward (Kashif & Slepaczuk 2026).

        absolute:  scale*log(1+r_net) - lambda_to*TO*100 - lambda_conc*(HHI-1/N)*100
        benchmark: scale*(log(1+r_net) - log(1+r_bench)) - same penalties

        N is the number of weights (includes CASH when enabled). Turnover is
        charged here on TOP of the real TC already baked into r_net, exactly as
        in the paper (turnover penalised twice on purpose).
        """
        n = len(self.actions_memory[-1]) if self.actions_memory else self.stock_dim
        hhi_min = 1.0 / max(n, 1)
        term = self.article_return_scale * self._safe_log1p(r_net)
        if self.reward_kind == "article_benchmark":
            term -= self.article_return_scale * self._safe_log1p(self._benchmark_return())
        pen_to   = self.article_lambda_to * turnover * 100.0
        pen_conc = self.article_lambda_conc * max(hhi - hhi_min, 0.0) * 100.0
        return float(term - pen_to - pen_conc)

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


def benchmark_label(config: dict) -> str:
    """Human-readable label for the configured benchmark (for plots / columns)."""
    b = config.get("benchmark", {}) or {}
    if b.get("type", "equal_weight") == "equal_weight":
        return "EqualWeight"
    return str(b.get("ticker", "^DJI"))


def load_benchmark_returns(config: dict) -> pd.Series:
    """Per-date benchmark daily-return series stored by stage 1.

    Returns the `benchmark_return` column from data/full_data.pkl as a
    DatetimeIndex-ed Series. This is the SAME series the reward / beta /
    market-proxy features use, so every consumer (stages 3/4/5 baselines and
    the QuantStats benchmark) agrees with config.benchmark exactly - whether
    that's equal_weight or a ticker. Excludes the synthetic CASH asset because
    stage 1 computes benchmark_return BEFORE injecting cash.
    """
    data_dir = resolve_path(config, "data_dir")
    full = pd.read_pickle(data_dir / "full_data.pkl")
    if "benchmark_return" not in full.columns:
        raise KeyError("full_data.pkl has no 'benchmark_return' column. Re-run "
                       "01_get_data.py so stage 1 writes it.")
    s = full.drop_duplicates("date").set_index("date")["benchmark_return"]
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


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


def resolve_indicator_list(config: dict) -> list[str]:
    """Full ordered indicator list the env state is built from.

    = config.data.indicators (stockstats) + custom per-asset feature columns +
    custom global feature columns. The custom column names are resolved from the
    feature registry (features.expected_columns) so config lists registry KEYS
    (e.g. 'mom') while the env state uses the produced COLUMN names (mom_1..mom_60).
    Stage 1 must have written every one of these columns to the data.
    """
    indicators = list(config["data"].get("indicators", []))
    cf = config["data"].get("custom_features", {}) or {}
    per_asset = cf.get("per_asset", []) or []
    global_   = cf.get("global", []) or []
    if per_asset or global_:
        import features  # local import to avoid a hard dependency when unused
        indicators = indicators + features.expected_columns(per_asset, global_)
    return indicators


def build_env_kwargs(config: dict, stock_dim: int) -> dict:
    """Assemble env_kwargs for StockPortfolioEnv. Matches the notebook's signature.

    tech_indicator_list is the COMBINED list (stockstats indicators + custom
    per-asset + custom global feature columns). stock_dim already includes the
    synthetic CASH asset when cash.enabled=true, because callers derive it from
    len(df.tic.unique()) and stage 1 injects CASH into the data.
    """
    env_cfg = config["env"]
    return {
        "hmax":                 env_cfg["hmax"],
        "initial_amount":       env_cfg["initial_amount"],
        "transaction_cost_pct": env_cfg["transaction_cost_pct"],
        "state_space":          stock_dim,
        "stock_dim":            stock_dim,
        "tech_indicator_list":  resolve_indicator_list(config),
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
    turnover_mode  = config["env"].get("turnover_mode", "naive")
    diff_ratio_eta = float(config["env"].get("diff_ratio_eta", 1.0 / 252.0))

    # Article reward coefficients (priority-1).
    ar_cfg               = config["env"].get("article_reward", {}) or {}
    article_return_scale = float(ar_cfg.get("return_scale", 1000.0))
    article_lambda_to    = float(ar_cfg.get("lambda_to", 0.003))
    article_lambda_conc  = float(ar_cfg.get("lambda_conc", 0.1))

    # Cash-as-synthetic-asset (priority-1).
    cash_cfg     = config.get("cash", {}) or {}
    cash_enabled = bool(cash_cfg.get("enabled", False))
    cash_ticker  = str(cash_cfg.get("ticker", "CASH"))
    if cash_enabled and cash_ticker not in set(df.tic.unique()):
        print(
            f"WARNING: cash.enabled=true but ticker {cash_ticker!r} is not in the "
            f"data. Re-run 01_get_data.py with cash.enabled=true so stage 1 injects "
            f"the synthetic CASH asset. The risk-off gate will fall back to "
            f"zero-weight behaviour."
        )

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
    if reward_mode == "article_benchmark" and "benchmark_return" not in df.columns:
        print(
            f"WARNING: reward_mode='article_benchmark' but the DataFrame has no "
            f"`benchmark_return` column. Re-run 01_get_data.py so stage 1 writes it. "
            f"The benchmark term will be treated as 0 (== article_absolute)."
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
            turnover_mode=turnover_mode,
            diff_ratio_eta=diff_ratio_eta,
            risk_off_enabled=risk_off_enabled,
            turbulence_threshold=turbulence_threshold,
            article_return_scale=article_return_scale,
            article_lambda_to=article_lambda_to,
            article_lambda_conc=article_lambda_conc,
            cash_enabled=cash_enabled,
            cash_ticker=cash_ticker,
            **env_kwargs,
        )

    raise ValueError(
        f"Unknown env.reward_mode={reward_mode!r}; expected 'value', 'log_return', "
        f"'diff_sharpe', 'diff_sortino', 'article_absolute', or 'article_benchmark'."
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
