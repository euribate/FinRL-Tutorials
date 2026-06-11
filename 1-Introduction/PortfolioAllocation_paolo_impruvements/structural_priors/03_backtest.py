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
        # Pass tc_penalty so the drift-adjusted TC drag the env applies inside
        # step() is also applied to the ensembled curve - otherwise the equity
        # plot reports a friction-free backtest (the per-seed env runs already
        # paid TC inside their own step() but those returns are discarded once
        # we average the WEIGHTS rather than the per-seed return series).
        tc_penalty       = float(config["env"].get("transaction_cost_penalty", 0.0))
        turnover_mode    = config["env"].get("turnover_mode", "naive")
        ensemble_actions = average_seed_actions(per_seed_actions)
        ensemble_returns = daily_return_from_weights(ensemble_actions, eval_slice,
                                                    tc_penalty=tc_penalty,
                                                    turnover_mode=turnover_mode)
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
                                  weight_bounds: tuple[float, float],
                                  exclude_tickers: set[str] | None = None) -> pd.Series:
    """Daily-rebalanced minimum-volatility portfolio (notebook cell 78).

    exclude_tickers: tickers to drop before optimising. The synthetic CASH
    asset MUST be excluded - its zero variance makes the covariance singular and
    the min-vol optimiser would otherwise allocate 100% to cash (a flat line, a
    meaningless benchmark). Min-Variance is a risky-asset baseline.
    """
    exclude = set(exclude_tickers or [])
    unique_trade_date = trade_df.date.unique()
    portfolio = pd.Series(index=unique_trade_date, dtype=float)
    portfolio.iloc[0] = initial

    for i in range(len(unique_trade_date) - 1):
        df_temp      = trade_df[trade_df.date == unique_trade_date[i]].reset_index(drop=True)
        df_temp_next = trade_df[trade_df.date == unique_trade_date[i + 1]].reset_index(drop=True)

        Sigma = df_temp.return_list[0].cov()
        if exclude:
            df_temp = df_temp[~df_temp.tic.isin(exclude)].reset_index(drop=True)
            keep = [c for c in Sigma.columns if c not in exclude]
            Sigma = Sigma.loc[keep, keep]

        ef = EfficientFrontier(None, Sigma, weight_bounds=tuple(weight_bounds))
        ef.min_volatility()
        weights = ef.clean_weights()  # dict ticker -> weight

        cap = float(portfolio.iloc[i])
        # Align weights / prices by ticker (robust to ordering).
        w_arr       = np.array([weights[t] for t in df_temp.tic], dtype=float)
        prices_now  = df_temp.close.values.astype(float)
        shares      = cap * w_arr / prices_now
        next_price  = dict(zip(df_temp_next.tic, df_temp_next.close.astype(float)))
        next_prices = np.array([next_price[t] for t in df_temp.tic], dtype=float)
        portfolio.iloc[i + 1] = float(np.dot(shares, next_prices))

    portfolio.index = pd.to_datetime(portfolio.index)
    portfolio.name = "MinVariance"
    return portfolio


def compute_dji_baseline(start: str, end: str, ticker: str, initial: float,
                         name: str = "DJIA") -> pd.Series:
    df_dji = fetch_yahoo_with_retry(ticker, start, end)[["date", "close"]]
    first = df_dji["close"].iloc[0]
    scaled = df_dji["close"] / first * initial
    s = pd.Series(scaled.values, index=pd.to_datetime(df_dji["date"].values), name=name)
    return s


def compute_equal_weight_baseline(trade_df: pd.DataFrame, initial: float,
                                  exclude_tickers: set[str] | None = None) -> pd.Series:
    """Daily-rebalanced equal-weight portfolio of the (risky) tickers.

    Matches the benchmark.type='equal_weight' definition used for the reward /
    beta / market-proxy features: each day allocate 1/M to each of the M tickers
    and hold to the next day. The synthetic CASH asset is excluded (it's a
    risky-asset benchmark, parallel to Min-Variance).
    """
    exclude = set(exclude_tickers or [])
    dates = trade_df.date.unique()
    portfolio = pd.Series(index=dates, dtype=float)
    portfolio.iloc[0] = initial

    for i in range(len(dates) - 1):
        df_t = trade_df[trade_df.date == dates[i]]
        df_n = trade_df[trade_df.date == dates[i + 1]]
        if exclude:
            df_t = df_t[~df_t.tic.isin(exclude)]
            df_n = df_n[~df_n.tic.isin(exclude)]
        df_t = df_t.reset_index(drop=True)
        m = len(df_t)
        cap = float(portfolio.iloc[i])
        prices_now  = df_t.close.values.astype(float)
        shares      = (cap / m) / prices_now
        next_price  = dict(zip(df_n.tic, df_n.close.astype(float)))
        next_prices = np.array([next_price[t] for t in df_t.tic], dtype=float)
        portfolio.iloc[i + 1] = float(np.dot(shares, next_prices))

    portfolio.index = pd.to_datetime(portfolio.index)
    portfolio.name = "EqualWeight"
    return portfolio


def compute_prior_baseline(trade_df: pd.DataFrame, initial: float,
                           prior_type: str,
                           cash_ticker: str = "CASH",
                           cash_enabled: bool = False,
                           cash_share: float = None) -> pd.Series:
    """Daily-rebalanced *static* structural-prior portfolio.

    For each trading day, recompute w_prior from that day's cov_list (the
    same 252-day trailing covariance the env sees) using the SAME math as
    LogReturnPortfolioEnv._compute_prior_weights -> compute_prior_weights_
    from_cov, then hold to the next bar. This is the right comparator for
    a residual-policy PPO: it answers "what would running the prior with
    NO learning have produced?" Any Sharpe lift PPO shows over this curve
    is attributable to the agent's learned tilts; any lift over EqualWeight
    that this curve also shows is the prior alone.
    """
    from utils import compute_prior_weights_from_cov

    dates = trade_df.date.unique()
    portfolio = pd.Series(index=dates, dtype=float)
    portfolio.iloc[0] = initial

    for i in range(len(dates) - 1):
        df_t = trade_df[trade_df.date == dates[i]].reset_index(drop=True)
        df_n = trade_df[trade_df.date == dates[i + 1]]

        tics = list(df_t.tic.values)
        if cash_enabled and cash_ticker in tics:
            cash_idx = tics.index(cash_ticker)
        else:
            cash_idx = -1

        cov = np.asarray(df_t["cov_list"].values[0], dtype=float)
        w = compute_prior_weights_from_cov(
            cov=cov, prior_type=prior_type,
            cash_idx=cash_idx, cash_share=cash_share,
        )

        cap        = float(portfolio.iloc[i])
        prices_now = df_t.close.values.astype(float)
        # Cash is synthetic and held as-is; its price grows at the configured
        # risk_free_rate (constant per-day return). The standard portfolio
        # accounting handles this transparently because CASH is just another
        # row in the dataframe.
        shares     = (cap * w) / np.maximum(prices_now, 1e-12)
        next_price = dict(zip(df_n.tic, df_n.close.astype(float)))
        next_prices = np.array([next_price[t] for t in df_t.tic], dtype=float)
        portfolio.iloc[i + 1] = float(np.dot(shares, next_prices))

    portfolio.index = pd.to_datetime(portfolio.index)
    portfolio.name = f"Prior_{prior_type}"
    return portfolio


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


def compute_active_stats(strategy_equity: pd.Series,
                         benchmark_equity: pd.Series) -> dict:
    """Active-return statistics of `strategy` vs `benchmark`.

    Aligns the two equity curves on common dates, computes daily active
    returns r_active = r_strategy - r_benchmark, then reports:
      * mean_active_bps : mean daily active return in basis points
      * std_active_bps  : daily tracking error in bps
      * ir              : annualised information ratio = sqrt(252) * mean / std
      * t_iid           : paired t-stat assuming iid daily active returns
      * p_iid           : two-sided p-value for t_iid
      * t_nw            : Newey-West HAC-adjusted t-stat (lag = floor(T^(1/4)))
      * p_nw            : two-sided p-value for t_nw
      * n_days          : sample size used

    The NW adjustment matters because daily active returns of a portfolio
    strategy carry serial correlation (regime stickiness, position drift).
    Reporting BOTH t_iid and t_nw lets the reader see how much the iid
    assumption inflates significance. Both p-values use the normal
    approximation (T >> 30 in practice).
    """
    s = strategy_equity.dropna()
    b = benchmark_equity.dropna()
    common = s.index.intersection(b.index)
    if len(common) < 30:
        return {k: np.nan for k in ["mean_active_bps", "std_active_bps", "ir",
                                    "t_iid", "p_iid", "t_nw", "p_nw", "n_days"]}
    r_s = s.loc[common].pct_change().dropna()
    r_b = b.loc[common].pct_change().dropna()
    common2 = r_s.index.intersection(r_b.index)
    active = (r_s.loc[common2] - r_b.loc[common2]).values.astype(float)
    n = len(active)
    if n < 30:
        return {k: np.nan for k in ["mean_active_bps", "std_active_bps", "ir",
                                    "t_iid", "p_iid", "t_nw", "p_nw", "n_days"]}

    mu  = float(active.mean())
    sig = float(active.std(ddof=1))
    ir  = float(np.sqrt(252.0) * mu / sig) if sig > 1e-18 else np.nan

    # iid paired t-stat: t = mean / (std / sqrt(N))
    t_iid = float(mu / (sig / np.sqrt(n))) if sig > 1e-18 else np.nan

    # Newey-West HAC variance of the mean: sum of weighted autocovariances up
    # to lag L = floor(T^(1/4)) (Bartlett kernel). For daily returns over a
    # ~6-year sample, T ~ 1500 -> L = 6. Standard choice in finance.
    L = int(np.floor(n ** 0.25))
    centered = active - mu
    gamma0 = float(np.dot(centered, centered) / n)
    nw_var = gamma0
    for lag in range(1, L + 1):
        w_l = 1.0 - lag / (L + 1.0)
        gam = float(np.dot(centered[lag:], centered[:-lag]) / n)
        nw_var += 2.0 * w_l * gam
    if nw_var > 1e-18:
        nw_se = float(np.sqrt(nw_var / n))
        t_nw  = float(mu / nw_se)
    else:
        t_nw = np.nan

    from math import erf, sqrt
    def two_sided_p(t):
        if not np.isfinite(t):
            return np.nan
        return float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))))

    return {
        "mean_active_bps": mu * 1e4,
        "std_active_bps":  sig * 1e4,
        "ir":              ir,
        "t_iid":           t_iid,
        "p_iid":           two_sided_p(t_iid),
        "t_nw":            t_nw,
        "p_nw":            two_sided_p(t_nw),
        "n_days":          int(n),
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
        # Per-seed equity curves, captured for the multi-seed dispersion
        # block printed at end-of-run alongside the ensemble active-stats.
        per_seed_equity_by_algo: dict[str, dict[int, pd.Series]] = {}
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
            tc_penalty       = float(config["env"].get("transaction_cost_penalty", 0.0))
            turnover_mode    = config["env"].get("turnover_mode", "naive")
            ensemble_returns = daily_return_from_weights(ensemble_actions, eval_df,
                                                        tc_penalty=tc_penalty,
                                                        turnover_mode=turnover_mode)
            equity[algo.upper()] = daily_return_to_equity(ensemble_returns, initial)
            weights_path = results_dir / f"weights_{algo}.csv"
            ensemble_actions.to_csv(weights_path)
            print(f"Saved {weights_path}")
            # Capture per-seed equity for the dispersion report. Each seed's
            # return series is computed via the same daily_return_from_weights
            # helper so per-seed and ensemble use the same TC accounting.
            if len(seeds) > 1:
                per_seed_equity_by_algo[algo.upper()] = {}
                for s, acts in zip(seeds, per_seed_actions):
                    per_seed_returns = daily_return_from_weights(
                        acts, eval_df,
                        tc_penalty=tc_penalty,
                        turnover_mode=turnover_mode,
                    )
                    per_seed_equity_by_algo[algo.upper()][s] = \
                        daily_return_to_equity(per_seed_returns, initial)
        baseline_start, baseline_end = trade_period_dates(config, False, None, eval_df)

    print("\nComputing Min-Variance baseline...")
    weight_bounds = config["baselines"]["min_variance"]["weight_bounds"]
    # Exclude the synthetic CASH asset: its zero variance would otherwise make
    # the optimiser allocate 100% to cash (a flat, meaningless baseline).
    cash_cfg = config.get("cash", {}) or {}
    exclude = {str(cash_cfg.get("ticker", "CASH"))} if cash_cfg.get("enabled", False) else None
    equity["MinVariance"] = compute_min_variance_baseline(eval_df, initial, weight_bounds,
                                                          exclude_tickers=exclude)

    # Third baseline follows the benchmark block: equal_weight -> a daily-
    # rebalanced equal-weight portfolio of the risky ETFs (replaces DJIA);
    # ticker -> a buy-and-hold line of that benchmark ticker.
    bench_cfg = config.get("benchmark", {}) or {}
    btype = bench_cfg.get("type", "equal_weight")
    if btype == "equal_weight":
        print("Computing Equal-Weight baseline...")
        # Standard EqualWeight (fully invested, excludes CASH).
        equity["EqualWeight"] = compute_equal_weight_baseline(
            eval_df, initial, exclude_tickers=exclude)
        # Cash-inclusive EqualWeight (1/(M+1) across the M risky tickers + CASH).
        # Isolates the agent's pure allocation skill from the cash drag. Controlled
        # by benchmark.show_cash_inclusive (default true), and only emitted when
        # cash.enabled=true - otherwise it would be identical to EqualWeight.
        if bench_cfg.get("show_cash_inclusive", True) and cash_cfg.get("enabled", False):
            equity["EqualWeight_w_Cash"] = compute_equal_weight_baseline(
                eval_df, initial, exclude_tickers=None)
        # Structural-prior baseline: a daily-rebalanced STATIC portfolio of
        # the same w_prior the env uses at training time. The right comparator
        # for a residual-policy PPO - lift over this curve is the agent's
        # learned tilt, lift over EqualWeight that this curve ALSO shows is
        # the prior alone. Only emitted when policy_prior is enabled (otherwise
        # the residual mechanism is off and the curve has no interpretation).
        pp_cfg = config.get("policy_prior", {}) or {}
        if pp_cfg.get("enabled", False) and pp_cfg.get("type", "none") != "none":
            ptype = str(pp_cfg.get("type"))
            print(f"Computing static structural-prior baseline (type={ptype})...")
            equity[f"Prior_{ptype}"] = compute_prior_baseline(
                eval_df, initial,
                prior_type=ptype,
                cash_ticker=str(cash_cfg.get("ticker", "CASH")),
                cash_enabled=bool(cash_cfg.get("enabled", False)),
                cash_share=pp_cfg.get("cash_share", None),
            )
    else:
        bt_ticker = bench_cfg.get("ticker", config["baselines"]["dji_ticker"])
        print(f"Computing benchmark-ticker baseline ({bt_ticker})...")
        try:
            equity[bt_ticker] = compute_dji_baseline(
                start=baseline_start,
                end=baseline_end,
                ticker=bt_ticker,
                initial=initial,
                name=bt_ticker,
            )
        except Exception as e:
            # Yahoo download can fail transiently on DNS resolution. Keep the
            # DRL agent + MinVariance curves rather than crashing.
            print(f"WARNING: benchmark baseline {bt_ticker} unavailable ({e}). "
                  f"Skipping it - re-run later for the full comparison.")

    result = pd.concat(equity, axis=1).sort_index().ffill()

    csv_path = results_dir / "equity_curves.csv"
    result.to_csv(csv_path)
    print(f"\nSaved {csv_path}")

    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    third = "EqualWeight" if btype == "equal_weight" else bench_cfg.get("ticker", "DJIA")
    prefix = "Portfolio Allocation (Walk-Forward)" if walk_forward else "Portfolio Allocation"
    plt.title(f"{prefix}: RL Agents vs Min-Variance vs {third}")
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

    # ---- Active-return statistics vs benchmark ----
    # Pick the benchmark column to compare every strategy against.
    #
    # IMPORTANT: when cash.enabled=true, the PPO agent has CASH as one of
    # its allocatable assets and the trained policy structurally holds some
    # fraction in cash on most bars (typical 5-15%). Comparing such a
    # cash-holding strategy to the FULLY-INVESTED EqualWeight benchmark
    # creates a systematic drag during periods where risk assets compounded
    # significantly: avg_cash_weight * (EW_risky_return - cash_return) per
    # bar, which can produce |IR| ~= 1 with significant t-stat purely from
    # the cash exposure, with zero correlation between the strategy's tilt
    # and any actual signal in the features. To control for this and isolate
    # the agent's tilting skill, the active-stats benchmark is
    # EqualWeight_w_Cash (also fully invested, but holds 1/(M+1) in CASH
    # alongside 1/M in each risky asset) — the apples-to-apples comparator.
    if btype == "equal_weight":
        # Prefer the cash-inclusive EW when available; it controls out the
        # structural cash drag of the agent's allocations. Fall back to the
        # cash-free EW only if the cash-inclusive column wasn't produced.
        bench_col = ("EqualWeight_w_Cash"
                     if "EqualWeight_w_Cash" in result.columns
                     else "EqualWeight")
    else:
        bench_col = bench_cfg.get("ticker", "DJIA")
    if bench_col in result.columns:
        print(f"\nActive-return statistics vs {bench_col}")
        print("=" * 92)
        print(f"{'Strategy':<14} {'Active_bps':>11} {'TE_bps':>9} "
              f"{'IR':>7} {'t_iid':>7} {'p_iid':>7} {'t_NW':>7} {'p_NW':>7} {'N':>5}")
        print("-" * 92)
        for name in result.columns:
            if name == bench_col:
                continue
            a = compute_active_stats(result[name], result[bench_col])
            if not np.isfinite(a.get("ir", np.nan)):
                continue
            print(f"{name:<14} {a['mean_active_bps']:>10.2f}  "
                  f"{a['std_active_bps']:>8.2f} "
                  f"{a['ir']:>7.3f} "
                  f"{a['t_iid']:>7.2f} {a['p_iid']:>7.3f} "
                  f"{a['t_nw']:>7.2f} {a['p_nw']:>7.3f} "
                  f"{a['n_days']:>5d}")
        print("=" * 92)
        print(f"Notes: Active_bps = mean daily active return (strategy - {bench_col}) in bps. "
              f"TE_bps = daily tracking-error stdev in bps. IR = sqrt(252)*mean/std (annualised). "
              f"t_iid/p_iid assume iid daily active returns; t_NW/p_NW use Newey-West HAC "
              f"with lag = floor(N^(1/4)) to correct for daily serial correlation. "
              f"Two-sided p-values; p < 0.05 ~= |t| > 1.96.")

        # ---- Cash-drag decomposition: side-by-side IRs vs both EW variants ----
        # When BOTH EqualWeight (cash-free) and EqualWeight_w_Cash are emitted,
        # show each strategy's IR vs each anchor side-by-side. The difference is
        # the deterministic cash-drag effect: a strategy holding average cash
        # weight c, against a cash-free EW that earned excess equity premium e
        # per bar, has mechanical active return of -c*e per bar with ~zero
        # tracking error contribution, producing |IR| up to ~1 purely from cash
        # exposure with no correlation to any signal. Comparing to EW_w_Cash
        # removes this drag.
        if "EqualWeight" in result.columns and "EqualWeight_w_Cash" in result.columns:
            print(f"\nCash-drag decomposition: IR vs both EW variants")
            print("=" * 92)
            print(f"{'Strategy':<20} {'IR vs EW (no cash)':>22} {'IR vs EW_w_Cash':>22} "
                  f"{'Delta (= cash drag)':>26}")
            print("-" * 92)
            for name in result.columns:
                if name in ("EqualWeight", "EqualWeight_w_Cash"):
                    continue
                a_no = compute_active_stats(result[name], result["EqualWeight"])
                a_wc = compute_active_stats(result[name], result["EqualWeight_w_Cash"])
                if not np.isfinite(a_no.get("ir", np.nan)) or \
                   not np.isfinite(a_wc.get("ir", np.nan)):
                    continue
                delta = a_no["ir"] - a_wc["ir"]
                print(f"{name:<20} {a_no['ir']:>22.3f} {a_wc['ir']:>22.3f} "
                      f"{delta:>26.3f}")
            print("=" * 92)
            print(f"Reading: Delta is the deterministic effect of comparing a "
                  f"cash-holding strategy to a cash-free benchmark. The "
                  f"apples-to-apples IR is the EW_w_Cash column.")

        # ---- Per-seed dispersion report (single-split, multi-seed only) ----
        # When > 1 seeds are trained, show the per-seed IR distribution vs
        # the benchmark so the reader sees the seed-to-seed dispersion that
        # the ensemble averages out. With ~6 years of evaluation, the
        # minimum-detectable annualised IR at p<0.05 is roughly 0.8 (since
        # t = IR * sqrt(years)); realistic alpha edges of 0.2-0.4 are
        # therefore unreachable on a single path. The dispersion of per-seed
        # IRs around zero is the headline robustness check: a tight cluster
        # straddling zero is the expected null outcome.
        if not walk_forward and 'per_seed_equity_by_algo' in dir() \
           and per_seed_equity_by_algo and bench_col in result.columns:
            print(f"\nPer-seed IR distribution vs {bench_col} (single-split, "
                  f"5.9-year MDE @ p<0.05 ~ IR 0.8)")
            print("=" * 92)
            print(f"{'Algo / seed':<18} {'Active_bps':>11} {'TE_bps':>9} "
                  f"{'IR':>7} {'t_iid':>7} {'p_iid':>7} {'t_NW':>7} {'p_NW':>7} {'N':>5}")
            print("-" * 92)
            for algo_up, seed_to_eq in per_seed_equity_by_algo.items():
                ir_list = []
                for s, eq in seed_to_eq.items():
                    a = compute_active_stats(eq, result[bench_col])
                    if not np.isfinite(a.get("ir", np.nan)):
                        continue
                    print(f"  {algo_up:<10}s={s:<5} {a['mean_active_bps']:>10.2f}  "
                          f"{a['std_active_bps']:>8.2f} "
                          f"{a['ir']:>7.3f} "
                          f"{a['t_iid']:>7.2f} {a['p_iid']:>7.3f} "
                          f"{a['t_nw']:>7.2f} {a['p_nw']:>7.3f} "
                          f"{a['n_days']:>5d}")
                    ir_list.append(a["ir"])
                if len(ir_list) >= 2:
                    ir_arr = np.asarray(ir_list, dtype=float)
                    print(f"  {algo_up:<10}MEAN  {' ' * 21}"
                          f"{ir_arr.mean():>7.3f}  (sd={ir_arr.std(ddof=1):.3f}, "
                          f"min={ir_arr.min():.3f}, max={ir_arr.max():.3f}, "
                          f"signs +/- = {int((ir_arr>0).sum())}/{int((ir_arr<0).sum())})")
            print("=" * 92)
            print(f"Notes: Cross-seed mean and sd are descriptive only; with N_seeds=5 the "
                  f"sd is not a power calculation. Use the IR ranges to gauge robustness: "
                  f"a tight cluster straddling zero confirms the ensemble's null result.")


if __name__ == "__main__":
    main()
