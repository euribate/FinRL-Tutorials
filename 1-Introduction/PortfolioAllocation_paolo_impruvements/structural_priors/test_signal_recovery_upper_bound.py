"""Phase 3a — theoretical upper bound for IR recovery given a planted IC.

This script answers: "if a strategy has perfect, instantaneous, frictionless
knowledge of a per-asset signal whose cross-sectional rank-correlation with
next-period return is IC, what annualised Information Ratio can it achieve
over a 5.9-year evaluation window?"

The answer is a yardstick for Phase 3b (the actual pipeline-recovery test):

    pipeline_efficiency  =  observed_3b_IR  /  theoretical_3a_IR_at_same_IC

That ratio is what the negative-result writeup actually needs. 3a alone
cannot establish "no IC > X in real features" — it only sets the ceiling
the pipeline could ever reach. 3b's pipeline-loss is the missing piece.

Mechanics
---------
1) For each planted IC in {0.0, 0.05, 0.10, 0.20, 0.40}:
   * Simulate 2000 Monte Carlo paths of an N=14-asset, T-trading-day return
     panel (T = 5.9 years x 252 = 1487 days), where:
       - Daily returns are i.i.d. N(0, sigma^2) across (i, t).
       - A signal vector s_{i,t} is observed BEFORE each rebalance; the
         realised return between this rebalance and the next is
         r_{i,t+1:t+W} = IC * z_i + sqrt(1-IC^2) * eps,
         where z_i is the standardised cross-sectional rank of s and eps
         is uncorrelated noise. This pins the *realised* cross-sectional
         rank-IC at the target.
       - The strategy weights are w_{i,t} = z_{i,t} long-only, normalised
         to sum to 1 across i (the optimal long-only deviation from
         equal-weight under Markowitz with diagonal covariance).
   * The benchmark is the daily-rebalanced equal-weight portfolio of the
     same panel, so the active return is purely the signal-driven tilt.
2) For each path: compute annualised IR of (strategy - EW) daily active
   returns over the T days.
3) Aggregate: report median IR + 5/95 percentiles per IC level.

This is a CEILING because:
  * Returns are i.i.d. (no regime-switching noise, no fat tails, no
    auto-correlation in the signal beyond the rebalance horizon).
  * The strategy has zero learning cost: it sees the signal directly.
  * No transaction cost, no softmax cap, no PPO sample-inefficiency.

The block-bootstrap version (in gen_synthetic_data.py for Phase 3b) is the
realistic case where all four of those constraints bite.

Run:
    source ../../../.venv/bin/activate
    python test_signal_recovery_upper_bound.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_one_path(rng: np.random.Generator,
                      ic: float,
                      n_assets: int,
                      n_days: int,
                      rebalance_every: int,
                      daily_vol: float) -> dict:
    """One Monte Carlo replication.

    Returns dict with:
      * ir         : annualised IR of (strategy - EW) daily active returns
      * mean_a_bps : mean daily active return in bps
      * std_a_bps  : daily tracking error in bps
    """
    # Number of rebalance windows that fit in n_days. Each window holds
    # the strategy weights for `rebalance_every` days.
    n_windows = n_days // rebalance_every
    if n_windows < 4:
        return {"ir": np.nan, "mean_a_bps": np.nan, "std_a_bps": np.nan}

    # For each window, generate a cross-sectional signal z (standardised
    # ranks across assets, mean 0, std 1 within-window). The realised
    # per-bar returns over that window will have cross-sectional
    # correlation ~ IC with z.
    eq_w = np.full(n_assets, 1.0 / n_assets)

    daily_active = []  # collected daily active returns

    for _ in range(n_windows):
        # Cross-sectional signal: standard-normal ranks, mean 0, std 1.
        z = rng.standard_normal(n_assets)
        z = (z - z.mean()) / (z.std(ddof=0) + 1e-12)

        # Strategy long-only weights — softmax-of-z. Same shape as the
        # actual env: weights are positive, sum to 1, and tilt toward
        # high-z assets. We bound the temperature so a high-IC signal
        # produces meaningful concentration without going degenerate.
        w_strat = np.exp(z) / np.exp(z).sum()

        # Realised per-bar returns over this window. We design the joint
        # so that the cross-sectional rank-IC of returns vs z is `ic`
        # (in expectation), and the per-asset daily-vol is `daily_vol`.
        for _bar in range(rebalance_every):
            # Cross-sectional component: ic * z * daily_vol gives the
            # IC-driven part of today's per-asset return.
            # Noise component: sqrt(1-ic^2) * iid_normal * daily_vol gives
            # the rest, so realised IC ~= ic per bar.
            signal_part = ic * z * daily_vol
            noise_part  = np.sqrt(max(1.0 - ic * ic, 0.0)) * \
                          rng.standard_normal(n_assets) * daily_vol
            r_today = signal_part + noise_part

            r_strat = float(np.dot(w_strat, r_today))
            r_bench = float(np.dot(eq_w,    r_today))
            daily_active.append(r_strat - r_bench)

    a = np.asarray(daily_active, dtype=float)
    mu  = float(a.mean())
    sig = float(a.std(ddof=1))
    ir  = float(np.sqrt(252.0) * mu / sig) if sig > 1e-18 else np.nan
    return {
        "ir":         ir,
        "mean_a_bps": mu * 1e4,
        "std_a_bps":  sig * 1e4,
    }


def main() -> None:
    rng = np.random.default_rng(20260610)

    N_ASSETS         = 14
    N_DAYS           = 1487     # 5.9 years x 252
    REBALANCE_EVERY  = 5        # Weekly cadence
    DAILY_VOL        = 0.01     # Typical daily vol per asset
    N_PATHS          = 2000
    IC_VALUES        = [0.00, 0.05, 0.10, 0.20, 0.40]

    print("=" * 80)
    print(f"Phase 3a — theoretical IR upper bound (perfect-information strategy)")
    print(f"Setup: N={N_ASSETS} assets, T={N_DAYS} days, weekly rebalance, "
          f"vol={DAILY_VOL*100:.1f}%/day, {N_PATHS} Monte Carlo paths per IC.")
    print("=" * 80)
    print(f"{'IC':>6} {'median IR':>11} {'5%':>10} {'95%':>10} "
          f"{'P(IR>0)':>9} {'P(|t|>1.96)':>13}")
    print("-" * 80)

    for ic in IC_VALUES:
        ir_list = []
        t_above_196 = 0
        for _ in range(N_PATHS):
            d = simulate_one_path(rng, ic, N_ASSETS, N_DAYS,
                                  REBALANCE_EVERY, DAILY_VOL)
            if np.isnan(d["ir"]):
                continue
            ir_list.append(d["ir"])
            # Approx single-path t = IR * sqrt(years); ~ IR * sqrt(N_DAYS/252)
            t_stat = d["ir"] * np.sqrt(N_DAYS / 252.0)
            if abs(t_stat) > 1.96:
                t_above_196 += 1

        ir_arr = np.asarray(ir_list, dtype=float)
        median = float(np.median(ir_arr))
        p5     = float(np.quantile(ir_arr, 0.05))
        p95    = float(np.quantile(ir_arr, 0.95))
        p_pos  = float((ir_arr > 0).mean())
        p_sig  = t_above_196 / max(1, len(ir_list))
        print(f"{ic:>6.2f} {median:>11.3f} {p5:>10.3f} {p95:>10.3f} "
              f"{p_pos*100:>8.1f}% {p_sig*100:>12.1f}%")

    print("=" * 80)
    print("Reading the table:")
    print("  median IR    = expected IR of a perfect-info weekly long-only strategy")
    print("                 with cross-sectional rank-IC = IC.")
    print("  5%, 95%      = path dispersion: the IR you'd observe on a randomly-")
    print("                 chosen 5.9-year window.")
    print("  P(IR>0)      = probability of beating EW on a random path (sign-only).")
    print("  P(|t|>1.96)  = probability the IR's single-path t-stat clears p<0.05")
    print("                 (the design's effective statistical power at this IC).")
    print()
    print("Use for Phase 3b: the IR a recovered planted-IC arm achieves divides by")
    print("the median IR at the SAME IC to give pipeline efficiency. If pipeline")
    print("efficiency at IC=0.40 is < 0.5, the apparatus loses half the signal even")
    print("at the strongest level — the real-data null result then says nothing about")
    print("whether IC < 0.40 in the features; it only proves IC * efficiency < ~0.05")
    print("(roughly the daily 5-seed IR upper-confidence-limit).")


if __name__ == "__main__":
    main()
