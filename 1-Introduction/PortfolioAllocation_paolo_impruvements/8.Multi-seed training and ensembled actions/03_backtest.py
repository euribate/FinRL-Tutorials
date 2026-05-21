"""Stage 3 - backtest enabled algorithms + Min-Variance + DJIA baselines.

Two execution paths depending on walk_forward.enabled:

  * Single-split (walk_forward.enabled=false): load models/agent_<algo>.zip,
    run prediction on data/trade_data.pkl, plot vs baselines. Identical to
    ../2.transaction_cost/.

  * Walk-forward (walk_forward.enabled=true, the default in this folder):
    read models/windows.json, for each window load models/agent_<algo>_w<i>.zip,
    slice data/full_data.pkl to the eval range, run prediction, capture the
    per-day returns. Concatenate per-window returns into a single continuous
    out-of-sample series and treat that as the strategy's equity curve. The
    Min-Variance and DJIA baselines cover the same union of eval windows.

Outputs:
    results/equity_curves.csv    daily portfolio value per strategy
    results/equity_plot.png      overlaid equity curves
    stdout                       cumulative return / Sharpe / max drawdown

Usage:
    python 03_backtest.py --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from finrl.agents.stablebaselines3.models import DRLAgent
from pypfopt.efficient_frontier import EfficientFrontier

from utils import (
    ALGO_REGISTRY,
    average_seed_actions,
    daily_return_from_weights,
    enabled_models,
    fetch_yahoo_with_retry,
    get_seeds,
    load_config,
    load_vecnormalize_stats,
    load_windows_manifest,
    make_portfolio_env,
    resolve_path,
    slice_by_dates,
    vecnormalize_path_for,
    walk_forward_enabled,
)


def load_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 01_get_data.py first.")
    return pd.read_pickle(path)


def predict_one(algo: str, trade_df: pd.DataFrame, config: dict,
                model_dir: Path, model_filename: str | None = None
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run an SB3 agent against a fresh StockPortfolioEnv variant.

    Returns (daily_return, df_actions) where df_actions is the per-day
    post-softmax weight vector exported by StockPortfolioEnv.save_action_memory().
    Stage 4 consumes df_actions to replay the strategy through backtrader.

    Branches on config.normalization.enabled:
      * off (legacy): DRLAgent.DRL_prediction (matches ../5.Early stopping/).
      * on:  manual rollout against a VecNormalize-wrapped env loaded with the
             frozen training stats. Required because DRLAgent.DRL_prediction
             does not know about VecNormalize wrappers and would feed the
             policy un-normalised observations - off-distribution from training.
    """
    AlgoClass = ALGO_REGISTRY[algo]
    fname = model_filename or f"agent_{algo}.zip"
    model_path = model_dir / fname
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - run 02_train.py first."
        )
    model = AlgoClass.load(str(model_path))

    stock_dim = len(trade_df.tic.unique())
    env       = make_portfolio_env(trade_df, config, stock_dim)

    norm_on   = bool(config.get("normalization", {}).get("enabled", False))
    if not norm_on:
        daily_return, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)
        return daily_return, df_actions

    # ----- VecNormalize path: manual rollout with frozen training stats -----
    vn_path = vecnormalize_path_for(model_path)
    if not vn_path.exists():
        raise FileNotFoundError(
            f"{vn_path} not found. With normalization.enabled=true, stage 2 "
            f"must have saved per-model VecNormalize stats. Re-run 02_train.py "
            f"in this folder to regenerate them."
        )

    sb_env, _ = env.get_sb_env()
    venv      = load_vecnormalize_stats(vn_path, sb_env)

    obs       = venv.reset()
    n_days    = len(env.df.index.unique())
    daily_return = None
    df_actions   = None
    for i in range(n_days):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        if i == n_days - 2:
            # Snapshot the env memory just before the terminal step triggers
            # DummyVecEnv's auto-reset (which would wipe portfolio_return_memory).
            daily_return = venv.env_method("save_asset_memory")[0]
            df_actions   = venv.env_method("save_action_memory")[0]
        if dones[0]:
            break
    if daily_return is None or df_actions is None:
        raise RuntimeError(
            f"Manual rollout for {algo} ended before the snapshot was taken; "
            f"this should not happen with a non-empty trade dataset."
        )
    return daily_return, df_actions


def predict_walk_forward(algo: str, full_df: pd.DataFrame, config: dict,
                         model_dir: Path,
                         windows: list[tuple[str, str, str, str]],
                         seeds: list[int]
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stitch per-window predictions into single continuous (daily_return, df_actions).

    Multi-seed ensembling (improvement #8): for each window, run ALL seeds'
    trained models through fresh envs, average the post-softmax weights
    row-by-row, then compute portfolio_return from the averaged weights.
    Returning the per-seed equity curves and averaging those would be wrong
    because each seed's portfolio_value drifts independently.

    When seeds=[s] (singleton), this is identical to the single-seed
    behaviour of folder 7 modulo the model filename suffix `_s<seed>`.
    """
    return_pieces: list[pd.DataFrame] = []
    action_pieces: list[pd.DataFrame] = []
    for i, (_ts, _te, es, ee) in enumerate(windows):
        eval_slice = slice_by_dates(full_df, es, ee)
        if len(eval_slice) == 0:
            print(f"  window {i}: empty eval slice {es} -> {ee}, skipping.")
            continue
        print(f"  window {i}: predicting {es} -> {ee} "
              f"({eval_slice.date.nunique()} days)  "
              f"x {len(seeds)} seeds")
        per_seed_actions: list[pd.DataFrame] = []
        for s in seeds:
            _ret, acts = predict_one(
                algo, eval_slice, config, model_dir,
                model_filename=f"agent_{algo}_w{i}_s{s}.zip",
            )
            per_seed_actions.append(acts)
        # Ensemble: average post-softmax weights across seeds; recompute
        # portfolio_return from the averaged weights against the eval slice.
        ensemble_actions = average_seed_actions(per_seed_actions)
        ensemble_returns = daily_return_from_weights(ensemble_actions, eval_slice)
        return_pieces.append(ensemble_returns)
        action_pieces.append(ensemble_actions)

    if not return_pieces:
        raise RuntimeError("No window produced any predictions.")

    stitched_returns = pd.concat(return_pieces, ignore_index=True)
    stitched_returns["date"] = pd.to_datetime(stitched_returns["date"])
    stitched_returns = stitched_returns.sort_values("date").drop_duplicates(
        subset=["date"], keep="first"
    ).reset_index(drop=True)

    stitched_actions = pd.concat(action_pieces)
    stitched_actions = stitched_actions[~stitched_actions.index.duplicated(keep="first")]
    stitched_actions = stitched_actions.sort_index()
    return stitched_returns, stitched_actions


def daily_return_to_equity(daily_return: pd.DataFrame, initial: float) -> pd.Series:
    df = daily_return.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return (1.0 + df["daily_return"]).cumprod() * initial


def compute_min_variance_baseline(trade_df: pd.DataFrame, initial: float,
                                  weight_bounds: tuple[float, float]) -> pd.Series:
    """Daily-rebalanced minimum-volatility portfolio (notebook cell 78)."""
    unique_trade_date = trade_df.date.unique()
    portfolio = pd.Series(index=unique_trade_date, dtype=float)
    portfolio.iloc[0] = initial

    for i in range(len(unique_trade_date) - 1):
        df_temp      = trade_df[trade_df.date == unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = trade_df[trade_df.date == unique_trade_date[i + 1]].reset_index(drop=True)

        Sigma = df_temp.return_list[0].cov()
        ef = EfficientFrontier(None, Sigma, weight_bounds=tuple(weight_bounds))
        ef.min_volatility()
        weights = ef.clean_weights()

        cap = float(portfolio.iloc[i])
        cash_per_stock = np.array([cap * w for w in weights.values()])
        shares = cash_per_stock / np.array(df_temp.close, dtype=float)
        next_prices = np.array(df_temp_next.close, dtype=float)
        portfolio.iloc[i + 1] = float(np.dot(shares, next_prices))

    portfolio.index = pd.to_datetime(portfolio.index)
    portfolio.name = "MinVariance"
    return portfolio


def compute_dji_baseline(start: str, end: str, ticker: str, initial: float) -> pd.Series:
    df_dji = fetch_yahoo_with_retry(ticker, start, end)[["date", "close"]]
    first = df_dji["close"].iloc[0]
    scaled = df_dji["close"] / first * initial
    s = pd.Series(scaled.values, index=pd.to_datetime(df_dji["date"].values), name="DJIA")
    return s


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {"cum_return": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    rets = equity.pct_change().dropna()
    cum_return = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe = (rets.mean() / (rets.std() + 1e-12)) * np.sqrt(252)
    drawdown = (equity - equity.cummax()) / equity.cummax()
    return {
        "cum_return": float(cum_return),
        "sharpe":     float(sharpe),
        "max_dd":     float(drawdown.min()),
    }


def trade_period_dates(config: dict, walk_forward: bool,
                       windows: list[tuple[str, str, str, str]] | None,
                       trade_df: pd.DataFrame | None) -> tuple[str, str]:
    """Determine the start/end of the out-of-sample period for baselines."""
    if walk_forward and windows:
        return windows[0][2], windows[-1][3]
    if trade_df is not None and len(trade_df):
        d = pd.to_datetime(trade_df["date"])
        return d.min().strftime("%Y-%m-%d"), d.max().strftime("%Y-%m-%d")
    return config["data"]["trade_start_date"], config["data"]["trade_end_date"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config      = load_config(args.config)
    data_dir    = resolve_path(config, "data_dir")
    model_dir   = resolve_path(config, "model_dir")
    results_dir = resolve_path(config, "results_dir")

    initial      = float(config["env"]["initial_amount"])
    walk_forward = walk_forward_enabled(config)
    seeds        = get_seeds(config)
    equity: dict[str, pd.Series] = {}
    print(f"seeds={seeds}  (ensemble size = {len(seeds)})")

    if walk_forward:
        full_df = load_pickle(data_dir / "full_data.pkl")
        windows = load_windows_manifest(model_dir)
        print(f"walk_forward=on  windows={len(windows)}")

        for algo in enabled_models(config):
            print(f"\nBacktesting {algo.upper()} "
                  f"(walk-forward, {len(windows)} windows, ensemble of {len(seeds)} seeds)...")
            stitched_returns, stitched_actions = predict_walk_forward(
                algo, full_df, config, model_dir, windows, seeds
            )
            equity[algo.upper()] = daily_return_to_equity(stitched_returns, initial)
            # Persist the stitched (ensemble-averaged) weights so stage 4 can replay them.
            weights_path = results_dir / f"weights_{algo}.csv"
            stitched_actions.to_csv(weights_path)
            print(f"Saved {weights_path}")

        # Build the baselines over the union of eval slices.
        eval_start = windows[0][2]
        eval_end   = windows[-1][3]
        eval_df    = slice_by_dates(full_df, eval_start, eval_end)
        baseline_start, baseline_end = eval_start, eval_end
    else:
        print("walk_forward=off")
        eval_df = load_pickle(data_dir / "trade_data.pkl")
        for algo in enabled_models(config):
            print(f"\nBacktesting {algo.upper()} (ensemble of {len(seeds)} seeds)...")
            # Multi-seed in single-split mode: run each seed against the same
            # trade env, average post-softmax weights, recompute portfolio_return.
            per_seed_actions: list[pd.DataFrame] = []
            for s in seeds:
                _ret, acts = predict_one(
                    algo, eval_df, config, model_dir,
                    model_filename=f"agent_{algo}_s{s}.zip",
                )
                per_seed_actions.append(acts)
            ensemble_actions = average_seed_actions(per_seed_actions)
            ensemble_returns = daily_return_from_weights(ensemble_actions, eval_df)
            equity[algo.upper()] = daily_return_to_equity(ensemble_returns, initial)
            weights_path = results_dir / f"weights_{algo}.csv"
            ensemble_actions.to_csv(weights_path)
            print(f"Saved {weights_path}")
        baseline_start, baseline_end = trade_period_dates(config, False, None, eval_df)

    print("\nComputing Min-Variance baseline...")
    weight_bounds = config["baselines"]["min_variance"]["weight_bounds"]
    equity["MinVariance"] = compute_min_variance_baseline(eval_df, initial, weight_bounds)

    print("Computing DJIA baseline...")
    try:
        equity["DJIA"] = compute_dji_baseline(
            start=baseline_start,
            end=baseline_end,
            ticker=config["baselines"]["dji_ticker"],
            initial=initial,
        )
    except Exception as e:
        # DJIA download from Yahoo can fail transiently on DNS resolution.
        # Keep the DRL agent + MinVariance curves we already have rather
        # than crashing the whole script.
        print(f"WARNING: DJIA baseline unavailable ({e}). "
              f"Skipping DJIA column - re-run later for the full comparison.")

    result = pd.concat(equity, axis=1).sort_index().ffill()

    csv_path = results_dir / "equity_curves.csv"
    result.to_csv(csv_path)
    print(f"\nSaved {csv_path}")

    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    title = ("Portfolio Allocation (Walk-Forward): RL Agents vs Min-Variance vs DJIA"
             if walk_forward else
             "Portfolio Allocation: RL Agents vs Min-Variance vs DJIA")
    plt.title(title)
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path = results_dir / "equity_plot.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved {plot_path}")

    print("\n" + "=" * 60)
    print(f"{'Strategy':<14} {'CumReturn':>12} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 60)
    for name in result.columns:
        m = compute_metrics(result[name])
        print(f"{name:<14} {m['cum_return']*100:>11.2f}% "
              f"{m['sharpe']:>10.3f} {m['max_dd']*100:>9.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
