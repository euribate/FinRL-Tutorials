"""Diagnostic - is the trained policy actively allocating, or frozen?

Reads the stitched ensemble weights stage 3 wrote (results/weights_<algo>.csv)
and reports the three things that matter when tuning the turnover penalty
(env.article_reward.lambda_to):

  1. PER-ASSET WEIGHT STD - how much each asset's weight moves over time.
     Near 0 for the risky assets => the policy has collapsed to a static
     (usually equal-weight) allocation; the lambda_to penalty is too high.
  2. TURNOVER - mean/median/max daily L1 turnover + how many days actually
     rebalance. Median ~0 with rare big spikes => almost all turnover is the
     risk-off gate flipping to cash, not learned rebalancing.
  3. REGIME DRIFT - average weights in the first third vs the last third of
     the period. No drift => the policy doesn't adapt across regimes.

Plus CASH usage per year (is cash rising in stress years?).

Usage:
    python inspect_policy.py                       # results/weights_ppo.csv
    python inspect_policy.py --weights results/weights_ppo.csv --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="results/weights_ppo.csv")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--turnover-threshold", type=float, default=0.02,
                        help="Daily L1 turnover above which a day counts as an active rebalance.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cash_ticker = str((cfg.get("cash", {}) or {}).get("ticker", "CASH"))
    lam_to = (cfg.get("env", {}).get("article_reward", {}) or {}).get("lambda_to")

    wpath = Path(args.weights)
    if not wpath.exists():
        raise FileNotFoundError(f"{wpath} not found - run 03_backtest.py first.")

    # Staleness guard: weights_ppo.csv must be NEWER than the trained models,
    # otherwise you're inspecting weights from an earlier lambda_to. This catches
    # the common trap of retraining (stage 2) but forgetting to re-run stage 3.
    import datetime
    models = list(Path("models").glob("agent_ppo*.zip"))
    if models:
        newest_model = max(m.stat().st_mtime for m in models)
        if wpath.stat().st_mtime < newest_model:
            wt = datetime.datetime.fromtimestamp(wpath.stat().st_mtime)
            mt_ = datetime.datetime.fromtimestamp(newest_model)
            print("!" * 70)
            print(f"STALE WEIGHTS: {wpath.name} ({wt:%Y-%m-%d %H:%M}) is OLDER than the "
                  f"newest model ({mt_:%Y-%m-%d %H:%M}).")
            print("You retrained (stage 2) but did NOT re-run stage 3. The numbers below "
                  "reflect the PREVIOUS lambda_to, not the current one.")
            print("Fix: python 03_backtest.py --config config.json   then re-run this.")
            print("!" * 70 + "\n")

    w = pd.read_csv(wpath, index_col=0)
    w.index = pd.to_datetime(w.index)
    w = w.sort_index()
    risky = [c for c in w.columns if c != cash_ticker]

    print(f"Weights file: {wpath}   rows={len(w)}   "
          f"{w.index.min().date()} -> {w.index.max().date()}")
    print(f"lambda_to in {args.config}: {lam_to}\n")

    # ---- 1. per-asset weight variation ----
    print("=== 1. Per-asset weight variation (std near 0 = frozen) ===")
    stats = pd.DataFrame({"mean": w.mean(), "std": w.std(),
                          "min": w.min(), "max": w.max()}).round(3)
    print(stats.to_string())
    risky_std = w[risky].std().mean()
    print(f"\n  mean std across RISKY assets: {risky_std:.4f}  "
          f"(combined verdict printed at the end)")

    # ---- 2. turnover ----
    turn = w.diff().abs().sum(axis=1).dropna()
    thr = args.turnover_threshold
    active = turn > thr
    flips  = turn > 0.5
    print(f"\n=== 2. Turnover (naive L1) ===")
    print(f"  mean daily {turn.mean():.4f}   median {turn.median():.4f}   max {turn.max():.4f}")
    print(f"  annualised one-way: {turn.mean()*252/2*100:.0f}%")
    print(f"  active days (turnover > {thr}): {active.sum()} of {len(turn)} ({100*active.mean():.0f}%)")
    print(f"  near-flip days (turnover > 0.5): {flips.sum()}")
    if flips.sum() and active.sum():
        share = 100 * turn[flips].sum() / turn.sum()
        print(f"  share of all turnover from near-flip days: {share:.0f}% "
              f"(high => dominated by the risk-off gate, not learned rebalancing)")

    # ---- 3. regime drift ----
    print(f"\n=== 3. Regime drift (first third vs last third) ===")
    n = len(w)
    early = w.iloc[: n // 3].mean()
    late  = w.iloc[2 * n // 3:].mean()
    drift = (late - early).abs().sort_values(ascending=False)
    print(f"  period split: early={w.index[0].date()}..{w.index[n//3-1].date()}  "
          f"late={w.index[2*n//3].date()}..{w.index[-1].date()}")
    for t in drift.head(6).index:
        print(f"    {t:6s}: early {early[t]:.3f} -> late {late[t]:.3f}   (delta {late[t]-early[t]:+.3f})")
    print(f"  max |drift| across assets: {drift.iloc[0]:.4f} "
          f"({'static - no regime adaptation' if drift.iloc[0] < 0.02 else 'adapts across regimes'})")

    # ---- cash usage by year ----
    if cash_ticker in w.columns:
        print(f"\n=== {cash_ticker} weight by year (should rise in stress years) ===")
        by_year = w[cash_ticker].groupby(w.index.year).agg(["mean", "max"]).round(3)
        print(by_year.to_string())

    # ---- combined verdict (std + regime drift, with a turnover note) ----
    max_drift = float(drift.iloc[0])
    ann_turn  = turn.mean() * 252 / 2
    print(f"\n=== VERDICT ===")
    print(f"  risky std={risky_std:.4f}   max regime drift={max_drift:.4f}   "
          f"annualised turnover={ann_turn*100:.0f}%")
    if risky_std < 0.012 and max_drift < 0.008:
        print("  FROZEN - collapsed to ~equal-weight; only the risk-off gate moves cash. "
              "lambda_to too high. Lower it.")
    elif risky_std > 0.020 or max_drift > 0.015:
        print("  ACTIVE - the policy genuinely allocates (real per-asset variation and/or "
              "regime drift).")
    else:
        print("  LIGHTLY ACTIVE - some differentiation but muted.")
    # Turnover caution: separate axis from frozen/active.
    if ann_turn > 3.0:
        print(f"  NOTE: turnover is HIGH ({ann_turn*100:.0f}%/yr) - this lambda_to is too low "
              f"for 'cautious'; the policy fidgets with many small trades.")
    elif ann_turn < 0.5:
        print(f"  NOTE: turnover is very low ({ann_turn*100:.0f}%/yr) - cautious, but check it "
              f"isn't frozen above.")
    # Conviction note - depends on whether the concentration penalty is binding.
    # Measure persistent tilt among the RISKY assets only: renormalise their
    # time-averaged weights to sum to 1 (removing the cash allocation, which
    # otherwise makes every risky asset look ~1pp below 1/N), then compare to
    # equal weight. ~0 => the agent holds equal weight among risky assets.
    lam_c = (cfg.get("env", {}).get("article_reward", {}) or {}).get("lambda_conc")
    risky_mean = w[risky].mean()
    risky_mean = risky_mean / risky_mean.sum()
    mean_dev = float((risky_mean - 1.0 / len(risky)).abs().mean())
    if lam_c is not None and lam_c <= 0.001:
        if mean_dev < 0.01:
            print(f"  NOTE: lambda_conc={lam_c} (no concentration penalty) yet average weights "
                  f"still sit ~1/N (mean deviation {mean_dev:.3f}). The equal-weight clustering "
                  f"is the AGENT'S OWN CHOICE, not the penalty - i.e. the allocation signal is "
                  f"weak; given freedom to concentrate, it declines to.")
        else:
            print(f"  NOTE: lambda_conc={lam_c} (off) and average weights deviate {mean_dev:.3f} "
                  f"from 1/N - the agent IS taking persistent positions.")
    else:
        print(f"  NOTE: per-asset weights cluster near 1/N partly because lambda_conc={lam_c} "
              f"(concentration penalty) caps conviction. Set lambda_conc=0 to test whether the "
              f"agent has real views or just defaults to 1/N.")


if __name__ == "__main__":
    main()
