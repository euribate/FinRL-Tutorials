"""Stage 5 - daily live-trading inference.

Pulls fresh OHLC for the configured tickers, recomputes today's observation
(indicators + rolling covariance + turbulence), runs each seed in the
deployed ensemble, averages the post-softmax weight vectors, and writes the
target portfolio for the NEXT trading day.

Designed to run BEFORE the 4 pm US market close: yfinance's intraday quote
on the current trading day is treated as today's close, and the prediction
is the portfolio to hold from today's close to tomorrow's close. Execute
orders (MOC or VWAP into the close) to reach the target before 4 pm.

Usage:
    python predict_tomorrow.py --config config_production.json
    python predict_tomorrow.py --config config_production.json --asof 2026-05-21
    python predict_tomorrow.py --config config_production.json --algo ppo

Output:
    results/target_weights_<asof>.csv        per-ticker target & model weights
    results/target_weights_latest.csv        a copy of the most recent file
    stdout                                   human-readable summary
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from utils import (
    ALGO_REGISTRY,
    compute_cov_features,
    enabled_models,
    get_seeds,
    load_config,
    load_vecnormalize_stats,
    make_portfolio_env,
    pick_ensemble_seeds,
    resolve_path,
    vecnormalize_path_for,
)


# ~252 trading-day lookback + buffer for weekends/holidays and indicator warmup.
# 500 calendar days comfortably covers the cov lookback and the longest SMA (60d).
LOOKBACK_PAD_DAYS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_production.json")
    parser.add_argument("--asof", default=None,
                        help="Prediction date YYYY-MM-DD; defaults to today's date. "
                             "The model produces the portfolio to hold from --asof's close "
                             "to the next trading day's close.")
    parser.add_argument("--algo", default=None,
                        help="Override the algo; defaults to the single algo in config.models with use=true.")
    return parser.parse_args()


def fetch_recent_data(config: dict, asof: dt.date) -> pd.DataFrame:
    tickers = config["data"]["ticker_list"]
    start = (asof - dt.timedelta(days=LOOKBACK_PAD_DAYS)).strftime("%Y-%m-%d")
    end   = (asof + dt.timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Fetching {len(tickers)} tickers from {start} to {end}...")
    df_raw = YahooDownloader(start_date=start, end_date=end,
                             ticker_list=tickers).fetch_data()
    if df_raw is None or df_raw.empty:
        raise RuntimeError("YahooDownloader returned no data.")
    last_date = df_raw["date"].max()
    print(f"  Raw shape: {df_raw.shape}   tickers: {df_raw.tic.nunique()}   "
          f"latest bar: {last_date}")
    return df_raw


def build_features(df_raw: pd.DataFrame, config: dict) -> pd.DataFrame:
    data_cfg = config["data"]
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=data_cfg["indicators"],
        use_turbulence=data_cfg["use_turbulence"],
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df_raw)
    df_full = compute_cov_features(df_processed, lookback=data_cfg["lookback"])
    # compute_cov_features ends with reset_index(drop=True) -> sequential 0..N
    # row index. StockPortfolioEnv expects a day-factorised index (df.loc[day]
    # returns all tickers for that day). In the normal pipeline, data_split /
    # slice_by_dates re-factorises before the env sees the frame; here we
    # build the frame ourselves, so apply the same factorisation explicitly.
    df_full = df_full.sort_values(["date", "tic"], ignore_index=True)
    df_full.index = df_full.date.factorize()[0]
    return df_full


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def predict_last_action(algo: str, seed: int, df: pd.DataFrame, config: dict,
                        model_dir: Path) -> np.ndarray:
    """Roll the (algo, seed) model through df, return the LAST raw action vector.

    The env's softmax is applied externally because the env never records the
    action on its terminal step (the last day = `asof` = today). We re-roll
    deterministically through the full slice so any internal env state
    (per-step EMAs for diff_sharpe etc.) is in the same shape as during the
    walk-forward eval.
    """
    AlgoClass  = ALGO_REGISTRY[algo]
    model_path = model_dir / f"agent_{algo}_s{seed}.zip"
    vn_path    = vecnormalize_path_for(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - retrain stage 2 with config_production.json."
        )
    if not vn_path.exists():
        raise FileNotFoundError(
            f"{vn_path} not found - retrain stage 2 with normalization.enabled=true."
        )

    model = AlgoClass.load(str(model_path))

    stock_dim = len(df.tic.unique())
    env       = make_portfolio_env(df, config, stock_dim)
    sb_env, _ = env.get_sb_env()
    venv      = load_vecnormalize_stats(vn_path, sb_env)

    obs    = venv.reset()
    n_days = len(env.df.index.unique())
    last_raw_action: np.ndarray | None = None
    for i in range(n_days):
        action, _ = model.predict(obs, deterministic=True)
        last_raw_action = np.asarray(action[0], dtype=float).copy()
        obs, _, dones, _ = venv.step(action)
        if dones[0]:
            break

    if last_raw_action is None:
        raise RuntimeError(f"No action captured for {algo} seed {seed}.")
    return last_raw_action


def average_weights(per_seed_weights: list[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(per_seed_weights)
    mean    = stacked.mean(axis=0)
    s       = mean.sum()
    return mean / s if s > 0 else mean


def main() -> None:
    args = parse_args()

    config    = load_config(args.config)
    model_dir = resolve_path(config, "model_dir")
    results_dir = resolve_path(config, "results_dir")

    algos = enabled_models(config)
    if args.algo:
        algos = [args.algo]
    if not algos:
        print("No algorithms enabled (set models.<name>.use = true in the config).")
        sys.exit(1)
    if len(algos) > 1:
        print(f"Multiple algos enabled ({algos}); predict_tomorrow.py uses the first: {algos[0]}")
    algo = algos[0]

    # Production: use the top-N best-converged seeds from the candidate pool.
    # Controlled by config.seeds.ensemble_size; falls back to the full
    # seeds.list when unset.
    candidates = get_seeds(config)
    seeds      = pick_ensemble_seeds(config, model_dir, algo=algo)
    if seeds != candidates:
        skipped = [s for s in candidates if s not in seeds]
        print(f"ensemble_size={config['seeds'].get('ensemble_size')}: "
              f"using {len(seeds)} of {len(candidates)} candidate seeds.")
        print(f"  selected: {seeds}")
        print(f"  skipped:  {skipped}  (ranked lower by training convergence)")

    asof = (dt.date.fromisoformat(args.asof) if args.asof
            else dt.date.today())
    print(f"asof: {asof}   algo: {algo}   seeds: {seeds}")

    df_raw = fetch_recent_data(config, asof)

    # Restrict to rows on or before asof; treat the asof row's price as today's close.
    df_raw = df_raw[df_raw["date"] <= asof.strftime("%Y-%m-%d")].copy()
    if df_raw.empty:
        raise RuntimeError(f"No data on or before {asof}. Wait for the market to open or "
                           f"pass a more recent --asof.")
    actual_asof_row = df_raw["date"].max()
    if actual_asof_row != asof.strftime("%Y-%m-%d"):
        print(f"  WARNING: requested asof={asof} but latest available bar is {actual_asof_row}. "
              f"Yahoo may not have today's intraday row yet; prediction uses {actual_asof_row}.")
    print(f"  Using bars up to {actual_asof_row} ({df_raw.tic.nunique()} tickers).")

    print("Computing indicators + turbulence + rolling covariance...")
    df_full = build_features(df_raw, config)
    print(f"  Featured shape: {df_full.shape}   dates: {df_full.date.nunique()}")
    if df_full["date"].max() != actual_asof_row:
        raise RuntimeError(f"Feature engineering trimmed the last row "
                           f"(have {df_full['date'].max()}, expected {actual_asof_row}). "
                           f"Likely an indicator warmup issue - increase LOOKBACK_PAD_DAYS.")

    tickers = list(pd.Index(df_full.tic.unique()).sort_values())
    n       = len(tickers)

    print(f"\nRunning {len(seeds)} seeds...")
    per_seed_weights: list[np.ndarray] = []
    for s in seeds:
        raw = predict_last_action(algo, s, df_full, config, model_dir)
        w   = softmax(raw)
        per_seed_weights.append(w)
        top3 = sorted(zip(tickers, w), key=lambda kv: -kv[1])[:3]
        top3_str = ", ".join(f"{t}={v:.1%}" for t, v in top3)
        print(f"  seed={s:>5}   model weights top3: {top3_str}")

    model_weights = average_weights(per_seed_weights)

    last_row = df_full[df_full["date"] == actual_asof_row]
    turb     = float(last_row["turbulence"].iloc[0]) if "turbulence" in df_full.columns else float("nan")
    ro_cfg   = config.get("risk_off", {})
    ro_on    = bool(ro_cfg.get("enabled", False))
    ro_thr   = float(ro_cfg.get("turbulence_threshold", 70.0))
    risk_off_triggered = ro_on and not np.isnan(turb) and turb > ro_thr

    target_weights = np.zeros(n) if risk_off_triggered else model_weights.copy()

    out_df = pd.DataFrame({
        "date":            [actual_asof_row] * n,
        "ticker":          tickers,
        "model_weight":    model_weights,
        "target_weight":   target_weights,
        "turbulence":      [turb] * n,
        "risk_off_active": [risk_off_triggered] * n,
        "ensemble_size":   [len(seeds)] * n,
    })

    fn_asof   = actual_asof_row.replace("-", "")
    out_path  = results_dir / f"target_weights_{fn_asof}.csv"
    latest    = results_dir / "target_weights_latest.csv"
    out_df.to_csv(out_path, index=False)
    out_df.to_csv(latest, index=False)
    print(f"\nWrote {out_path}")
    print(f"Wrote {latest}")

    print(f"\n=== Target portfolio for execution at the {actual_asof_row} close ===")
    print(f"  Ensemble: {len(seeds)} seeds ({seeds})    algo: {algo.upper()}")
    print(f"  Turbulence on {actual_asof_row}: {turb:.2f}   threshold: {ro_thr:.2f}   "
          f"risk_off: {'YES (GO TO CASH)' if risk_off_triggered else 'no'}")
    print(f"  Holding period: {actual_asof_row} close -> next trading day close")
    print()
    print(f"  {'ticker':<8}  {'target':>8}  {'model':>8}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    for t, tgt, mdl in zip(tickers, target_weights, model_weights):
        print(f"  {t:<8}  {tgt:>7.1%}  {mdl:>7.1%}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'SUM':<8}  {target_weights.sum():>7.1%}  {model_weights.sum():>7.1%}")
    if risk_off_triggered:
        print("\n  >>> RISK-OFF: liquidate to cash and pay the turnover cost. <<<")


if __name__ == "__main__":
    main()
