"""Vendored PortfolioEnv — replaces FinRL's StockPortfolioEnv + the
LogReturnPortfolioEnv subclass + every patch we ever applied to them.

Design choices made here are LOAD-BEARING and address every silent
convention bug the structural_priors triage uncovered:

  1. SYMMETRIC ACTION BOX DEFAULT. action_space = Box(-s, +s, (N,)).
     The legacy upstream Box(0, 1) caps single-asset weight at e/(e+N-1).
     With N=14 that is ~17%; the policy could not learn to concentrate
     even when it wanted to. The default here is s=3.0 (cap ~97% for
     N=14). Setting s=1.0 explicitly reproduces a conservative box;
     setting s tiny reproduces the upstream cage.

  2. DECISION-DATE INDEXED WEIGHTS. The row for date d holds the
     allocation the agent CHOSE at the close of d, to be earned over
     d -> d+1. The upstream env stored row d with the weights HELD
     OVER d (i.e. chosen at d-1). That convention silently shifted
     downstream backtrader replays by one bar. Killing it at the env
     level kills the whole class of alignment bugs.

  3. SINGLE ALL-IN COST PARAMETER IN BPS. cost_bps applied per unit
     of L1 turnover. No drift_adjusted vs naive split, no tc_penalty
     vs transaction_cost_pct duplication: one knob, one accounting
     path. Default 10 bps round-trip (5 each way).

  4. NATIVE WEEKLY ACTION-REPEAT WITH PER-DECISION REWARD. cadence
     in {'daily', 'weekly'} is set at construction; on non-rebalance
     bars the env discards the agent's fresh action and replays the
     last rebalance-day action. Reward is accumulated over the
     holding window and only delivered on the next rebalance day.
     Non-rebalance bars return reward=0. This matches the agent's
     decision frequency to its reward signal — the bug in METHODOLOGY
     Appendix A.4.2.

  5. BENCHMARK-RELATIVE REWARD AS THE ONLY MODE. reward =
     log(1 + r_net) - log(1 + r_bench) per decision-event, summed
     over the holding window. No diff_sharpe, no article_absolute, no
     value modes — those were either redundant (log_return is
     diff_sharpe's myopic limit) or systematically attractor-prone
     (EW). The benchmark is whatever column `benchmark_return` holds
     in the dataframe — set in data.py.

No FinRL imports. Pure gymnasium.Env subclass, ~250 lines of code +
docstrings.
"""
from __future__ import annotations

import math
from typing import Optional

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Rebalance-date helper (kept inline so the env is single-file)
# ---------------------------------------------------------------------------

_WEEKLY_DAY_INT = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4}


def compute_rebalance_dates(unique_dates, cadence: str = "daily",
                            weekly_day: str = "FRI") -> set:
    """Return the set of dates on which the env should rebalance.

    'daily': every date in `unique_dates`.
    'weekly': the latest Mon-<weekly_day> trading day in each ISO week.
              Default FRI; if Friday is missing (holiday), the latest
              Mon-Thu in that week is used (the practitioner convention).

    Returns a set of pd.Timestamp.normalize() values.
    """
    if cadence not in ("daily", "weekly"):
        raise ValueError(f"cadence must be 'daily' or 'weekly', got {cadence!r}")
    dts = sorted({pd.Timestamp(d).normalize() for d in unique_dates})
    if cadence == "daily":
        return set(dts)
    if weekly_day not in _WEEKLY_DAY_INT:
        raise ValueError(f"weekly_day must be one of {tuple(_WEEKLY_DAY_INT)}, "
                         f"got {weekly_day!r}")
    wd_int = _WEEKLY_DAY_INT[weekly_day]
    by_iso_week: dict[tuple[int, int], list[pd.Timestamp]] = {}
    for d in dts:
        iso = d.isocalendar()
        key = (int(iso.year), int(iso.week))
        if d.weekday() <= wd_int:
            by_iso_week.setdefault(key, []).append(d)
    return {max(group) for group in by_iso_week.values()}


# ---------------------------------------------------------------------------
# PortfolioEnv
# ---------------------------------------------------------------------------

class PortfolioEnv(gym.Env):
    """Vendored long-only portfolio allocation environment.

    Args
    ----
    df:
        Long-format DataFrame with columns:
          date (str YYYY-MM-DD), tic (str), close (float),
          benchmark_return (float, per-date scalar broadcast to all tics),
          plus any number of FEATURE columns (per-asset rows) that will
          form the state.
        Per-asset features must already be normalised in `data.py`
        (rank normalisation is the recommended default).
    feature_cols:
        Column names from df to include in the state vector. Order
        matters and must match what the policy expects.
    action_logit_scale:
        Half-width of the symmetric action box. Default 3.0 (cap ~97%
        for N=14). Pass 1.0 for a conservative [-1, +1] box.
    cost_bps:
        All-in round-trip transaction cost in basis points, applied
        to the L1 change in TARGET weights between consecutive
        rebalance days. Default 10 bps. 0.0 disables.
    cadence:
        'daily' or 'weekly'.
    weekly_day:
        Weekday key when cadence='weekly'. Default 'FRI'.
    reward_scaling:
        Multiplier on the per-decision reward delivered at rebalance bars.
        Default 1.0; the article uses 1000 to lift the reward into a
        gradient-friendly range.
    """

    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, feature_cols: list[str],
                 action_logit_scale: float = 3.0,
                 cost_bps: float = 10.0,
                 cadence: str = "daily",
                 weekly_day: str = "FRI",
                 reward_scaling: float = 1.0):
        super().__init__()
        if action_logit_scale <= 0.0:
            raise ValueError(f"action_logit_scale must be > 0, got {action_logit_scale}")
        if "benchmark_return" not in df.columns:
            raise ValueError("df must contain a 'benchmark_return' column "
                             "(per-date scalar). Set it up in data.py.")
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols missing from df: {missing}")

        # Sort + factorize by date so day index is contiguous.
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        df.index = df.date.factorize()[0]
        self.df          = df
        self.feature_cols = list(feature_cols)

        self.tickers   = sorted(df.tic.unique())
        self.n_assets  = len(self.tickers)
        self.n_dates   = int(df.index.max()) + 1
        self.n_features = len(self.feature_cols)

        self.action_logit_scale = float(action_logit_scale)
        self.cost_bps   = float(cost_bps)
        self.cadence    = cadence
        self.weekly_day = weekly_day
        self.reward_scaling = float(reward_scaling)

        # Precompute the set of rebalance dates as integer day indices.
        date_to_day = {pd.Timestamp(d).normalize(): i
                       for i, d in enumerate(df.date.unique())}
        rb_set = compute_rebalance_dates(df.date.unique(),
                                         cadence=cadence, weekly_day=weekly_day)
        self._rebalance_days = {date_to_day[d] for d in rb_set
                                if d in date_to_day}

        # gym spaces.
        self.action_space      = spaces.Box(low=-self.action_logit_scale,
                                            high=+self.action_logit_scale,
                                            shape=(self.n_assets,),
                                            dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=+np.inf,
                                            shape=(self.n_assets, self.n_features),
                                            dtype=np.float32)

        # Per-episode state (initialised in reset).
        self._day: int = 0
        self._last_rebalance_action: Optional[np.ndarray] = None
        self._last_target_weights:   Optional[np.ndarray] = None  # for TC
        self._reward_accum:          float = 0.0
        self._actions_log:   list[np.ndarray] = []
        self._weights_log:   list[np.ndarray] = []
        self._returns_log:   list[float]      = []
        self._dates_log:     list[str]        = []

    # ------------------------------------------------------------------ gym

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None):
        super().reset(seed=seed)
        self._day = 0
        self._last_rebalance_action = None
        self._last_target_weights   = None
        self._reward_accum = 0.0
        self._actions_log.clear()
        self._weights_log.clear()
        self._returns_log.clear()
        self._dates_log.clear()
        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # Action gate: replace agent's fresh action with the last
        # rebalance-day action on non-rebalance bars.
        is_rebalance = self._day in self._rebalance_days
        if is_rebalance or self._last_rebalance_action is None:
            self._last_rebalance_action = action.copy()
        applied_action = self._last_rebalance_action

        # Symmetric box -> softmax weights. The shift-invariance of
        # softmax means subtracting the max for numerical stability
        # does not change the result.
        a = applied_action.astype(np.float64)
        a = a - a.max()
        w = np.exp(a)
        w = w / w.sum()
        w = w.astype(np.float64)

        # Compute return earned over self._day -> self._day+1.
        # Per design choice #2, this row's date is self._day's date
        # (the DECISION date), and the return earned is r_{d -> d+1}.
        if self._day + 1 >= self.n_dates:
            # Terminal — no next bar to earn over.
            done = True
            terminated = True
            truncated  = False
            self._actions_log.append(applied_action.copy())
            self._weights_log.append(w.copy())
            self._returns_log.append(0.0)
            self._dates_log.append(self._date_str(self._day))
            return self._obs(), 0.0, terminated, truncated, {}

        date_today = self._date_str(self._day)
        prices_today = self._prices(self._day)
        prices_next  = self._prices(self._day + 1)
        per_asset_ret = prices_next / np.maximum(prices_today, 1e-12) - 1.0
        gross_return  = float(np.dot(w, per_asset_ret))

        # Transaction cost: cost_bps * L1(w_target - w_prev) / 2 / 1e4
        # The /2 converts round-trip bps to one-side; we apply to the
        # full L1 to capture both sides of the rebalance.
        turnover = 0.0
        if self._last_target_weights is not None:
            turnover = float(np.abs(w - self._last_target_weights).sum())
        tc_fraction = (self.cost_bps / 1e4) * turnover
        net_return  = (1.0 + gross_return) * (1.0 - tc_fraction) - 1.0
        self._last_target_weights = w.copy()

        # Reward: benchmark-relative log-return, accumulated over the
        # holding window and only delivered on the NEXT rebalance day.
        bench_return = float(self._benchmark_return(self._day + 1))
        per_event_reward = (math.log(max(1.0 + net_return, 1e-8))
                            - math.log(max(1.0 + bench_return, 1e-8)))
        self._reward_accum += per_event_reward * self.reward_scaling

        next_is_rebalance = (self._day + 1) in self._rebalance_days
        if next_is_rebalance or self._day + 1 == self.n_dates - 1:
            reward = self._reward_accum
            self._reward_accum = 0.0
        else:
            reward = 0.0

        # Log book.
        self._actions_log.append(applied_action.copy())
        self._weights_log.append(w.copy())
        self._returns_log.append(net_return)
        self._dates_log.append(date_today)

        self._day += 1
        terminated = self._day >= self.n_dates - 1
        truncated  = False
        return self._obs(), float(reward), terminated, truncated, {}

    # --------------------------------------------------------------- helpers

    def _obs(self) -> np.ndarray:
        """Return state vector for the current day."""
        d = min(self._day, self.n_dates - 1)
        rows = self.df.loc[d]
        if isinstance(rows, pd.Series):  # single-asset edge case
            rows = rows.to_frame().T
        rows = rows.sort_values("tic")
        out = rows[self.feature_cols].to_numpy(dtype=np.float32, copy=True)
        return out

    def _prices(self, day: int) -> np.ndarray:
        rows = self.df.loc[day]
        if isinstance(rows, pd.Series):
            rows = rows.to_frame().T
        rows = rows.sort_values("tic")
        return rows["close"].to_numpy(dtype=np.float64, copy=True)

    def _benchmark_return(self, day: int) -> float:
        rows = self.df.loc[day]
        if isinstance(rows, pd.Series):
            return float(rows["benchmark_return"])
        return float(rows["benchmark_return"].iloc[0])

    def _date_str(self, day: int) -> str:
        rows = self.df.loc[day]
        if isinstance(rows, pd.Series):
            return str(rows["date"])
        return str(rows["date"].iloc[0])

    # ------------------------------------------------------------ saved logs

    def save_action_memory(self) -> pd.DataFrame:
        """Per-day post-softmax weights indexed by DECISION DATE.

        Row d holds the weights chosen at the close of day d, to be
        earned over d -> d+1. Compare to the FinRL convention where
        row d held the weights HELD OVER d.
        """
        df = pd.DataFrame(self._weights_log, columns=self.tickers)
        df.insert(0, "date", self._dates_log)
        return df

    def save_asset_memory(self) -> pd.DataFrame:
        """Per-day net portfolio return (decision-date indexed)."""
        return pd.DataFrame({
            "date":         self._dates_log,
            "daily_return": self._returns_log,
        })


# ---------------------------------------------------------------------------
# Convenience: action -> softmax weights (used in tests, also handy in
# diagnostics and gen_synthetic_data when reconstructing weights from logs).
# ---------------------------------------------------------------------------

def softmax_weights(action: np.ndarray) -> np.ndarray:
    """Numerically-stable softmax on an action vector.

    Identical to what the env uses internally; exposed for tests.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()
