"""Diagnostic - is the action-space geometry capping the portfolio weights?

Background. The upstream FinRL StockPortfolioEnv declares
    action_space = Box(low=0, high=1)
and softmaxes the actions. SB3 clips actions to the box, so the softmax
logits are confined to [0, 1] - a maximum logit spread of 1.0. With N assets
the most concentrated portfolio the policy can EVER express is

    w_max = e / (e + (N-1))        w_min = 1 / (e + (N-1))

For N=6 (5 ETFs + CASH): w_max ~ 35.2%, w_min ~ 13.0%. The agent cannot
deviate far from equal weight no matter what it learns.

This script answers, from the recorded weights CSV alone:
  1. What is the realized max/min weight and implied logit spread per day?
  2. Does the realized spread hug the legacy cap (=> the cap is BINDING)?
  3. How far do the weights ever travel from equal weight?

Run it BEFORE changing anything (baseline evidence), and again AFTER
retraining with env.action_logit_scale > 1 (treatment evidence).

Usage:
    python inspect_action_range.py                          # default CSV
    python inspect_action_range.py --weights results/weights_ppo.csv
    python inspect_action_range.py --env-check --config config.json
        # builds the live env via make_portfolio_env and verifies the
        # action_space bounds + that an extreme action concentrates weight.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- theory ----------

def legacy_caps(n_assets: int, spread: float = 1.0) -> tuple[float, float]:
    """Max/min reachable single-asset weight for a given max logit spread.

    One logit at +spread/... actually the extreme is one logit at the top of
    the box and all others at the bottom: w_max = e^s / (e^s + (N-1)),
    w_min for the others = 1 / (e^s + (N-1)).
    """
    es = math.exp(spread)
    w_max = es / (es + (n_assets - 1))
    w_min = 1.0 / (es + (n_assets - 1))
    return w_max, w_min


def print_cap_table(n_assets: int) -> None:
    print(f"\nTheoretical single-asset weight caps for N={n_assets} assets")
    print(f"{'logit box':>14} {'spread':>7} {'w_max':>8} {'w_min':>8}")
    for label, spread in [("[0, 1] legacy", 1.0),
                          ("[-1, +1]",      2.0),
                          ("[-2, +2]",      4.0),
                          ("[-3, +3]",      6.0),
                          ("[-5, +5]",     10.0)]:
        w_max, w_min = legacy_caps(n_assets, spread)
        print(f"{label:>14} {spread:>7.1f} {w_max*100:>7.1f}% {w_min*100:>7.2f}%")


# ---------- empirics ----------

def analyze_weights(csv_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found - run 03_backtest.py first, or pass "
            f"--weights with the correct path."
        )
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    n = df.shape[1]
    eq = 1.0 / n

    # Drop rows that are exactly the dummy equal-weight placeholder FinRL
    # injects at each episode start (all entries == 1/N to float precision):
    # they are artifacts, not decisions, and would bias the stats toward eq.
    is_dummy = (df.sub(eq).abs() < 1e-12).all(axis=1)
    n_dummy = int(is_dummy.sum())
    body = df[~is_dummy]

    row_max = body.max(axis=1)
    row_min = body.min(axis=1)
    # Implied logit spread: w_i ~ e^{a_i}  =>  max(a) - min(a) = ln(wmax/wmin)
    spread = np.log(row_max / row_min.clip(lower=1e-12))
    l1_from_eq = body.sub(eq).abs().sum(axis=1)
    daily_turnover = body.diff().abs().sum(axis=1).iloc[1:]

    legacy_wmax, legacy_wmin = legacy_caps(n, 1.0)

    print(f"\nWeights file : {csv_path}")
    print(f"Assets (N)   : {n}   equal weight = {eq*100:.2f}%")
    print(f"Rows         : {len(df)}  ({n_dummy} dummy equal-weight rows excluded)")
    print(f"Date range   : {df.index.min().date()} -> {df.index.max().date()}")

    def pct_line(name, s):
        q = s.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
        print(f"  {name:<22} p01={q.iloc[0]:.4f}  p25={q.iloc[1]:.4f}  "
              f"med={q.iloc[2]:.4f}  p75={q.iloc[3]:.4f}  p99={q.iloc[4]:.4f}  "
              f"max={s.max():.4f}")

    print("\nPer-day realized statistics:")
    pct_line("max weight", row_max)
    pct_line("min weight", row_min)
    pct_line("implied logit spread", spread)
    pct_line("L1 dist from equal-w", l1_from_eq)
    pct_line("turnover |dw|", daily_turnover)

    print(f"\nLegacy [0,1]-box caps for N={n}: "
          f"w_max <= {legacy_wmax*100:.1f}%, w_min >= {legacy_wmin*100:.2f}%, "
          f"spread <= 1.0")

    # ---------- verdict ----------
    p99_spread = float(spread.quantile(0.99))
    max_spread = float(spread.max())
    near_eq_days = float((l1_from_eq < 0.05).mean())

    print("\nVERDICT")
    if max_spread <= 1.0 + 1e-6:
        print(f"  * Logit spread NEVER exceeds 1.0 (max={max_spread:.3f}): the "
              f"legacy Box(0,1) cap is consistent with every row. The policy "
              f"is operating inside (or pinned against) the structural cage.")
        if p99_spread > 0.85:
            print(f"  * p99 spread = {p99_spread:.3f} hugs the 1.0 ceiling: the "
                  f"cap is BINDING - the policy is pushing against the wall. "
                  f"Widening the box (env.action_logit_scale) should be the "
                  f"first intervention; the agent wants to concentrate more "
                  f"than the geometry allows.")
        else:
            print(f"  * p99 spread = {p99_spread:.3f} sits well below 1.0: the "
                  f"policy is not even using the range it has. Geometry alone "
                  f"is not the whole story - expect incentive-layer fixes "
                  f"(reward baseline, ent_coef, gamma) to be needed too. Widen "
                  f"the box anyway so later experiments are interpretable.")
    else:
        print(f"  * Logit spread exceeds 1.0 (max={max_spread:.3f}): the box "
              f"has already been widened (or the CSV predates this analysis).")
    print(f"  * Fraction of days within L1<0.05 of equal weight: "
          f"{near_eq_days*100:.1f}%")

    print_cap_table(n)


# ---------- live env integration check ----------

def env_check(config_path: str) -> None:
    """Build the real env via make_portfolio_env and verify the geometry.

    Three assertions:
      1. action_space bounds match env.action_logit_scale from the config.
      2. An extreme action (one logit at +high, rest at low) produces a
         weight above the legacy 35%-style cap.
      3. A zero action still produces exactly equal weight (sanity).
    """
    from data import load_config, feature_columns, load_pickles
    from env import PortfolioEnv

    cfg = load_config(config_path)
    _full, _train, trade = load_pickles(cfg)
    feat_cols = feature_columns(cfg)
    env_cfg = cfg["env"]
    env = PortfolioEnv(
        trade, feat_cols,
        action_logit_scale=float(env_cfg.get("action_logit_scale", 3.0)),
        cost_bps=float(env_cfg.get("cost_bps", 10.0)),
        cadence=str(env_cfg.get("cadence", "daily")),
        weekly_day=str(env_cfg.get("weekly_day", "FRI")),
        reward_scaling=float(env_cfg.get("reward_scaling", 1.0)),
    )

    n = env.n_assets
    lo = float(env.action_space.low.min())
    hi = float(env.action_space.high.max())
    scale = float(env_cfg.get("action_logit_scale", 3.0))
    print(f"\n--env-check (N={n})")
    print(f"  action_space: low={lo:+.2f} high={hi:+.2f}  "
          f"(config action_logit_scale={scale})")

    if abs(lo + scale) < 1e-6 and abs(hi - scale) < 1e-6:
        print("  [PASS] symmetric box bounds match config.")
    else:
        print(f"  [FAIL] expected [{-scale:+.2f}, {+scale:+.2f}]")
        return

    # Extreme action: concentrates on asset 0; should exceed legacy cap.
    env.reset()
    extreme = np.full(n, lo, dtype=np.float32)
    extreme[0] = hi
    env.step(extreme)
    w = env._weights_log[-1]
    legacy_wmax, _ = legacy_caps(n, 1.0)
    print(f"  extreme action -> max weight {w.max()*100:.1f}%  "
          f"min weight {w.min()*100:.2f}%")
    if w.max() > legacy_wmax + 0.02:
        print(f"  [PASS] concentration {w.max()*100:.1f}% > legacy cap "
              f"{legacy_wmax*100:.1f}% — symmetric box is open.")
    else:
        print(f"  [FAIL] still capped near {legacy_wmax*100:.1f}%.")

    # Zero action: should map to exact 1/N (softmax shift-invariance).
    env.reset()
    env.step(np.zeros(n, dtype=np.float32))
    w0 = env._weights_log[-1]
    if np.allclose(w0, 1.0 / n, atol=1e-9):
        print(f"  [PASS] zero action -> exact equal weight (1/{n}).")
    else:
        print(f"  [WARN] zero action gave weights {w0}.")


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="results/weights_ppo.csv",
                        help="Path to the stage-3 weights CSV.")
    parser.add_argument("--env-check", action="store_true",
                        help="Also build the live env and verify the bounds.")
    parser.add_argument("--config", default="config.json",
                        help="Config for --env-check.")
    args = parser.parse_args()

    analyze_weights(Path(args.weights))
    if args.env_check:
        env_check(args.config)


if __name__ == "__main__":
    main()
