"""Diagnostic - show per-seed weight vectors at inference and quantify agreement.

Use this to decide whether the multi-seed ensemble is healthy:
  * If the 3 seeds roughly agree (same top holdings, low pairwise distances),
    the ensemble averaging is doing what it's supposed to and you can deploy.
  * If they wildly disagree (different top holdings, large pairwise distances),
    the per-seed policies have not converged on a common solution - retrain
    with tighter early stopping (lower min_delta, higher patience, finer
    eval_freq) so PPO has more chance to find a stable optimum across seeds.

The script reuses predict_tomorrow.py's data pipeline (fresh Yahoo pull, full
feature recomputation, per-seed deterministic rollout) so the weights printed
here are EXACTLY what predict_tomorrow.py would produce - just shown side by
side instead of averaged.

Usage:
    python inspect_ensemble.py --config config_production.json
    python inspect_ensemble.py --config config_production.json --asof 2026-05-21
"""
from __future__ import annotations

import argparse
import datetime as dt
from itertools import combinations

import numpy as np
import pandas as pd

from predict_tomorrow import (
    LOOKBACK_PAD_DAYS,
    build_features,
    fetch_recent_data,
    predict_last_action,
    softmax,
)
from utils import (
    enabled_models,
    get_seeds,
    load_config,
    pick_ensemble_seeds,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_production.json")
    parser.add_argument("--asof", default=None,
                        help="Prediction date YYYY-MM-DD; defaults to today.")
    parser.add_argument("--algo", default=None,
                        help="Override the algo; defaults to first models.<name>.use=true.")
    parser.add_argument("--topk", type=int, default=3,
                        help="How many top holdings to print per seed (default 3).")
    return parser.parse_args()


def jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """JS distance between two probability vectors. Returns 0 (identical) to 1 (disjoint).

    Symmetric, bounded, and well-behaved on the simplex - more interpretable
    than KL divergence for portfolio-weight comparisons.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = (a > 0) & (b > 0)
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    js_div = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(js_div / np.log(2.0), 0.0)))


def main() -> None:
    args = parse_args()

    config    = load_config(args.config)
    model_dir = resolve_path(config, "model_dir")

    algos = enabled_models(config)
    if args.algo:
        algos = [args.algo]
    algo = algos[0]

    # Honour ensemble_size if set, otherwise show all candidates.
    seeds = pick_ensemble_seeds(config, model_dir, algo=algo)
    all_candidates = get_seeds(config)
    if seeds != all_candidates:
        skipped = [s for s in all_candidates if s not in seeds]
        print(f"ensemble_size={config['seeds'].get('ensemble_size')}: "
              f"inspecting {len(seeds)} selected of {len(all_candidates)} candidates.")
        print(f"  selected: {seeds}")
        print(f"  skipped:  {skipped}")

    asof = (dt.date.fromisoformat(args.asof) if args.asof
            else dt.date.today())
    print(f"asof: {asof}   algo: {algo}   seeds: {seeds}")
    print(f"Lookback fetch: {LOOKBACK_PAD_DAYS} calendar days")

    df_raw = fetch_recent_data(config, asof)
    df_raw = df_raw[df_raw["date"] <= asof.strftime("%Y-%m-%d")].copy()
    actual_asof = df_raw["date"].max()
    print(f"Using bars up to {actual_asof}")

    print("Computing indicators + turbulence + rolling covariance...")
    df_full = build_features(df_raw, config)

    tickers = list(pd.Index(df_full.tic.unique()).sort_values())
    n = len(tickers)

    print(f"\nRunning {len(seeds)} seeds...")
    per_seed: dict[int, np.ndarray] = {}
    for s in seeds:
        raw = predict_last_action(algo, s, df_full, config, model_dir)
        per_seed[s] = softmax(raw)

    # Ensemble average (this is what predict_tomorrow.py would deploy)
    stacked = np.vstack([per_seed[s] for s in seeds])
    ensemble = stacked.mean(axis=0)
    ensemble = ensemble / ensemble.sum() if ensemble.sum() > 0 else ensemble

    # ---------- Side-by-side weights ----------
    print(f"\n=== Per-seed weights for {actual_asof} ===\n")
    header = f"  {'ticker':<8}"
    for s in seeds:
        header += f"  {'seed_' + str(s):>9}"
    header += f"  {'ensemble':>9}  {'mean':>7}  {'std':>7}  {'range':>7}"
    print(header)
    print(f"  {'-'*8}" + ("  " + "-"*9) * (len(seeds) + 1) + f"  {'-'*7}  {'-'*7}  {'-'*7}")

    for i, t in enumerate(tickers):
        row = f"  {t:<8}"
        seed_vals = [per_seed[s][i] for s in seeds]
        for v in seed_vals:
            row += f"  {v:>8.1%}"
        mean = float(np.mean(seed_vals))
        std  = float(np.std(seed_vals))
        rng  = float(max(seed_vals) - min(seed_vals))
        row += f"  {ensemble[i]:>8.1%}  {mean:>6.1%}  {std:>6.1%}  {rng:>6.1%}"
        print(row)

    sum_row = f"  {'SUM':<8}"
    for s in seeds:
        sum_row += f"  {per_seed[s].sum():>8.1%}"
    sum_row += f"  {ensemble.sum():>8.1%}"
    print(f"  {'-'*8}" + ("  " + "-"*9) * (len(seeds) + 1))
    print(sum_row)

    # ---------- Top-K holdings per seed ----------
    K = args.topk
    print(f"\n=== Top-{K} holdings per seed ===\n")
    top_sets: dict[int, set[str]] = {}
    for s in seeds:
        ranked = sorted(zip(tickers, per_seed[s]), key=lambda kv: -kv[1])[:K]
        top_sets[s] = {t for t, _ in ranked}
        descr = ", ".join(f"{t}({v:.1%})" for t, v in ranked)
        print(f"  seed={s:>5}:  {descr}")

    # Ensemble top-K
    ranked_ens = sorted(zip(tickers, ensemble), key=lambda kv: -kv[1])[:K]
    ens_top = {t for t, _ in ranked_ens}
    descr = ", ".join(f"{t}({v:.1%})" for t, v in ranked_ens)
    print(f"  {'ensemble':>10}:  {descr}")

    # Overlap analysis
    if len(seeds) >= 2:
        all_three = set.intersection(*top_sets.values())
        union     = set.union(*top_sets.values())
        print(f"\n  Tickers in ALL seeds' top-{K}: {sorted(all_three) or '(none)'} "
              f"({len(all_three)}/{K})")
        print(f"  Tickers in ANY seed's top-{K}: {sorted(union)} ({len(union)} unique)")

    # ---------- Pairwise disagreement ----------
    print(f"\n=== Pairwise disagreement ===\n")
    print(f"  {'pair':<22} {'L1':>8} {'L_inf':>8} {'JSdist':>8}  interpretation")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}  {'-'*30}")
    pairwise_l1 = []
    for s1, s2 in combinations(seeds, 2):
        w1, w2 = per_seed[s1], per_seed[s2]
        l1  = float(np.abs(w1 - w2).sum())
        linf = float(np.abs(w1 - w2).max())
        js  = jensen_shannon_distance(w1, w2)
        pairwise_l1.append(l1)
        if l1 < 0.10:
            label = "very close"
        elif l1 < 0.25:
            label = "close"
        elif l1 < 0.50:
            label = "diverging"
        else:
            label = "WILDLY DIFFERENT"
        print(f"  seed_{s1:<5} vs seed_{s2:<5}  {l1:>7.3f}  {linf:>7.3f}  {js:>7.3f}  {label}")

    # ---------- Overall verdict ----------
    print(f"\n=== Verdict ===\n")
    avg_l1 = float(np.mean(pairwise_l1)) if pairwise_l1 else 0.0
    overlap_frac = (len(set.intersection(*top_sets.values())) / K) if (top_sets and K > 0) else 1.0
    print(f"  Average pairwise L1 distance: {avg_l1:.3f}")
    print(f"  Top-{K} overlap across all seeds: {overlap_frac:.0%}")
    if avg_l1 < 0.15 and overlap_frac >= 2 / K:
        print(f"\n  -> SEEDS AGREE. Ensemble averaging is doing its job; deploy as-is.")
    elif avg_l1 < 0.30:
        print(f"\n  -> MODERATE DISAGREEMENT. Ensemble averaging is meaningful but watch")
        print(f"     for the seeds drifting further apart over time. Consider tighter ES.")
    else:
        print(f"\n  -> SEEDS DISAGREE STRONGLY. The per-seed policies have not converged")
        print(f"     on a common solution. Consider retraining with:")
        print(f"       early_stopping.min_delta:    0.001  (was {config['early_stopping']['min_delta']})")
        print(f"       early_stopping.patience:     20     (was {config['early_stopping']['patience']})")
        print(f"       early_stopping.eval_freq:    2500   (was {config['early_stopping']['eval_freq']})")

    # ---------- Notes for context ----------
    last_row = df_full[df_full["date"] == actual_asof]
    if "turbulence" in df_full.columns:
        turb = float(last_row["turbulence"].iloc[0])
        ro_cfg = config.get("risk_off", {})
        ro_thr = float(ro_cfg.get("turbulence_threshold", 70.0))
        ro_on = bool(ro_cfg.get("enabled", False))
        gate_active = ro_on and turb > ro_thr
        print(f"\n  Turbulence on {actual_asof}: {turb:.2f}   threshold: {ro_thr:.2f}   "
              f"risk_off: {'YES (ensemble overridden to cash)' if gate_active else 'no'}")
        if gate_active:
            print("  -> Note: the printed weights are MODEL weights (pre-gate). "
                  "predict_tomorrow.py would zero them out.")


if __name__ == "__main__":
    main()
