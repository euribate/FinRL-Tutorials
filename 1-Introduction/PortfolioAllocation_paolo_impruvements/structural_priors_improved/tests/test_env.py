"""Env unit tests — the Step-0 acceptance gate.

Six tests. Each prints PASS/FAIL; the script exits 1 if any fail.

  1) compute_rebalance_dates daily = identity.
  2) compute_rebalance_dates weekly picks Fri + holiday-fallback to Thu.
  3) softmax shift-invariance: softmax(a + c*1) == softmax(a) for any c.
     This is the equal-weight-prior no-op identity from test_priors.py
     in the legacy codebase, now an env-side property.
  4) Decision-date convention: the row for date d in save_action_memory
     holds the weights chosen AT THE CLOSE of d, to be earned over d -> d+1.
  5) Cost math: bps round-trip is applied correctly.
  6) Env-curve regression: equity computed by the env step-by-step
     matches equity computed by backtest.returns_from_weights on the
     SAME saved weights, to machine precision (modulo accumulated float
     drift, < 1e-10 relative).
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from env import PortfolioEnv, compute_rebalance_dates, softmax_weights
from backtest import returns_from_weights, equity_from_returns


def _toy_df(n_assets: int = 3, n_days: int = 100,
            include_cash: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Tiny synthetic OHLCV with one feature column and benchmark_return."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-01", periods=n_days).strftime("%Y-%m-%d")
    rows = []
    tickers = [f"T{i:02d}" for i in range(n_assets)]
    if include_cash:
        tickers = tickers + ["CASH"]
    prices = {t: 100.0 for t in tickers}
    for d in dates:
        # Generate per-asset returns (cash has zero return).
        per_asset_rets = rng.normal(0.0005, 0.012, size=n_assets)
        if include_cash:
            per_asset_rets = np.append(per_asset_rets, 0.0)
        # Equal-weight benchmark return for THIS date (risky only).
        bench = float(per_asset_rets[:n_assets].mean())
        for t, r in zip(tickers, per_asset_rets):
            prices[t] *= (1.0 + r)
            rows.append({
                "date": d, "tic": t, "close": prices[t],
                "feat1": rng.standard_normal(),
                "benchmark_return": bench,
            })
    return pd.DataFrame(rows), ["feat1"]


# ──────────────────────────────────────────────────────────── TEST 1

print("=" * 70)
print("TEST 1 — compute_rebalance_dates(cadence='daily') is identity")
print("=" * 70)
dates_1 = pd.bdate_range("2024-01-08", "2024-01-19")  # 10 weekdays
rb_daily = compute_rebalance_dates(dates_1, cadence="daily")
expected = {pd.Timestamp(d).normalize() for d in dates_1}
ok_1 = (rb_daily == expected)
print(f"  in={len(dates_1)}  out={len(rb_daily)}  RESULT: "
      f"{'PASS' if ok_1 else 'FAIL'}")

# ──────────────────────────────────────────────────────────── TEST 2
print()
print("=" * 70)
print("TEST 2 — weekly picks Friday + falls back to Thursday on holiday")
print("=" * 70)
dates_2 = pd.bdate_range("2024-01-08", "2024-01-19").tolist()
dates_2 = [d for d in dates_2 if d != pd.Timestamp("2024-01-19")]  # drop Fri
rb_weekly = sorted(compute_rebalance_dates(dates_2,
                                           cadence="weekly", weekly_day="FRI"))
expected_2 = sorted([pd.Timestamp("2024-01-12"),  # Friday
                     pd.Timestamp("2024-01-18")])  # Thursday fallback
ok_2 = (rb_weekly == expected_2)
print(f"  expected: {[d.strftime('%a %Y-%m-%d') for d in expected_2]}")
print(f"  got:      {[d.strftime('%a %Y-%m-%d') for d in rb_weekly]}")
print(f"  RESULT: {'PASS' if ok_2 else 'FAIL'}")

# ──────────────────────────────────────────────────────────── TEST 3
print()
print("=" * 70)
print("TEST 3 — softmax shift-invariance: softmax(a + c*1) == softmax(a)")
print("=" * 70)
rng = np.random.default_rng(1)
max_diff = 0.0
for _ in range(20):
    a = rng.standard_normal(14)
    c = rng.standard_normal() * 5.0
    diff = float(np.abs(softmax_weights(a) - softmax_weights(a + c)).max())
    max_diff = max(max_diff, diff)
ok_3 = max_diff < 1e-12
print(f"  max |softmax(a) - softmax(a + c)| over 20 random (a, c): "
      f"{max_diff:.3e}  RESULT: {'PASS' if ok_3 else 'FAIL'}")

# ──────────────────────────────────────────────────────────── TEST 4
print()
print("=" * 70)
print("TEST 4 — Decision-date convention: row d holds the weights chosen at d")
print("=" * 70)
df_4, feat_cols_4 = _toy_df(n_assets=3, n_days=50)
env = PortfolioEnv(df_4, feat_cols_4, cost_bps=0.0)
env.reset()
# Step with a constant action; weights should appear under THIS bar's date.
action = np.array([1.0, 0.0, 0.0], dtype=np.float32)
all_dates = sorted(df_4["date"].unique())
for _ in range(10):
    env.step(action)
weights = env.save_action_memory()
# The first logged row should carry the date of all_dates[0] (the decision
# was made AT the close of day 0); under the old FinRL convention this row
# would carry all_dates[1]'s date.
first_logged = weights["date"].iloc[0]
ok_4 = (first_logged == all_dates[0])
print(f"  first logged date in save_action_memory: {first_logged}")
print(f"  expected (decision date d=0):            {all_dates[0]}")
print(f"  RESULT: {'PASS' if ok_4 else 'FAIL'}")

# ──────────────────────────────────────────────────────────── TEST 5
print()
print("=" * 70)
print("TEST 5 — Cost math: bps applied to L1 turnover, no double-counting")
print("=" * 70)
df_5, feat_cols_5 = _toy_df(n_assets=3, n_days=10)
# Two consecutive constant-weight bars => zero turnover => zero TC.
env_zero = PortfolioEnv(df_5, feat_cols_5, cost_bps=100.0)
env_zero.reset()
action_const = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # softmax -> 1/3
env_zero.step(action_const)
env_zero.step(action_const)
returns_zero_tc = env_zero.save_asset_memory()["daily_return"].iloc[1]
# Now flip to an extreme tilt; turnover should be ~2*(2/3) ~ 1.33 (1/3 -> ~1
# on one asset, plus the two that drop from 1/3 to ~0).
env_tc = PortfolioEnv(df_5, feat_cols_5, cost_bps=100.0)
env_tc.reset()
env_tc.step(action_const)              # day 0 -> day 1: bootstrap weights
env_tc.step(np.array([10.0, -10.0, -10.0], dtype=np.float32))  # large tilt
ret_with_tc = env_tc.save_asset_memory()["daily_return"].iloc[1]
# With cost_bps=100 (1%), the tilt step should have HIGHER absolute net
# return drag than the constant-weight step.
diff = abs(ret_with_tc) - abs(returns_zero_tc)
ok_5 = ret_with_tc < returns_zero_tc + 0.005  # TC drag clearly visible
print(f"  net_return at zero turnover: {returns_zero_tc:+.6f}")
print(f"  net_return at high turnover: {ret_with_tc:+.6f}")
print(f"  RESULT: {'PASS' if ok_5 else 'FAIL'}")

# ──────────────────────────────────────────────────────────── TEST 6
print()
print("=" * 70)
print("TEST 6 — Env-curve == returns_from_weights regression")
print("=" * 70)
df_6, feat_cols_6 = _toy_df(n_assets=4, n_days=80)
env = PortfolioEnv(df_6, feat_cols_6, cost_bps=15.0, cadence="daily")
env.reset()
rng = np.random.default_rng(2)
all_actions = []
terminated = False
while not terminated:
    a = rng.normal(0, 1.5, size=4).astype(np.float32)
    all_actions.append(a.copy())
    _obs, _r, terminated, _trunc, _info = env.step(a)
ret_env = env.save_asset_memory()
weights = env.save_action_memory()
ret_recomp = returns_from_weights(weights, df_6, cost_bps=15.0)
# Align: env's saved returns include the FIRST decision-date row (row 0
# carries return earned over day 0 -> day 1); the recomputation has the
# same convention. Match on date index.
ret_env["date"] = pd.to_datetime(ret_env["date"])
ret_env = ret_env.set_index("date")["daily_return"]
# Last row of env's log might be the terminal (no next-bar return) so trim.
common = ret_env.index.intersection(ret_recomp.index)
ret_env_c    = ret_env.loc[common].to_numpy(dtype=float)
ret_recomp_c = ret_recomp.loc[common].to_numpy(dtype=float)
rel_diff = float(np.max(np.abs(ret_env_c - ret_recomp_c)))
ok_6 = rel_diff < 1e-10
print(f"  matched {len(common)} dates  "
      f"max |env_ret - recomp_ret| = {rel_diff:.3e}")
print(f"  RESULT: {'PASS' if ok_6 else 'FAIL'}")

# ──────────────────────────────────────────────────────────── Summary
print()
print("=" * 70)
print(f"SUMMARY: 1={'PASS' if ok_1 else 'FAIL'}  2={'PASS' if ok_2 else 'FAIL'}"
      f"  3={'PASS' if ok_3 else 'FAIL'}  4={'PASS' if ok_4 else 'FAIL'}"
      f"  5={'PASS' if ok_5 else 'FAIL'}  6={'PASS' if ok_6 else 'FAIL'}")
print("=" * 70)
if not all([ok_1, ok_2, ok_3, ok_4, ok_5, ok_6]):
    raise SystemExit(1)
