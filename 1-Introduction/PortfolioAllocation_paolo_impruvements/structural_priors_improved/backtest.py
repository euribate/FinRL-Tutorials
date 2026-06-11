"""Backtest: predict on trade slice, ensemble across seeds, compute
metrics + active-return stats (vs EW and vs EW_w_Cash) + per-seed
dispersion.

What survived verbatim from 03_backtest.py (these were validated to
four decimals during the triage and MUST NOT be modified):

  * compute_metrics       - sharpe / cum return / max drawdown
  * compute_active_stats  - mean/std active bps, IR, iid + Newey-West
                            HAC t-stat and p-value
  * Cash-drag decomposition block - prints IR vs EW (no cash) AND
                            IR vs EW_w_Cash with a Delta column

The walk-forward branch is gone. The DJIA-ticker benchmark branch is
gone. The Min-Variance baseline is gone (it was a relic from FinRL and
contributed nothing once the cash-drag-corrected EW was the right
benchmark). What's left is the experiment-relevant comparison: PPO vs
EqualWeight vs EqualWeight_w_Cash, with active stats and per-seed IR
dispersion.

Usage:
    python backtest.py --config config.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from data import feature_columns, load_config, load_pickles, resolve_path
from env import PortfolioEnv


# ─────────────────────────────────────────────────────────────────────────
# Verbatim ports — DO NOT MODIFY
# ─────────────────────────────────────────────────────────────────────────

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

    t_iid = float(mu / (sig / np.sqrt(n))) if sig > 1e-18 else np.nan

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


# ─────────────────────────────────────────────────────────────────────────
# Predict + ensemble
# ─────────────────────────────────────────────────────────────────────────

def _env_kwargs(config: dict) -> dict:
    env_cfg = config["env"]
    return dict(
        action_logit_scale=float(env_cfg.get("action_logit_scale", 3.0)),
        cost_bps          =float(env_cfg.get("cost_bps", 10.0)),
        cadence           =str(env_cfg.get("cadence", "daily")),
        weekly_day        =str(env_cfg.get("weekly_day", "FRI")),
        reward_scaling    =float(env_cfg.get("reward_scaling", 1.0)),
    )


def predict_one(config: dict, seed: int,
                trade_df: pd.DataFrame
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll one trained model through a trade env.

    Returns (daily_return_df, weights_df) where weights_df is indexed by
    DECISION date (the new env's convention).
    """
    feat_cols  = feature_columns(config)
    env_kwargs = _env_kwargs(config)
    model_dir  = resolve_path(config, "model_dir")
    model_path = model_dir / f"agent_ppo_s{seed}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found — train first.")
    model = PPO.load(str(model_path))

    env = PortfolioEnv(trade_df, feat_cols, **env_kwargs)
    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, _truncated, _ = env.step(action)
    return env.save_asset_memory(), env.save_action_memory()


def average_seed_weights(per_seed: list[pd.DataFrame]) -> pd.DataFrame:
    """Row-wise mean of post-softmax weights across seeds.

    Assumes per-seed DataFrames share the same `date` column ordering
    (true when they are produced from the same trade slice).
    """
    base = per_seed[0].copy()
    base.set_index("date", inplace=True)
    cols = list(base.columns)
    stack = np.stack([df.set_index("date")[cols].to_numpy(dtype=float)
                      for df in per_seed], axis=0)
    mean = stack.mean(axis=0)
    out = pd.DataFrame(mean, index=base.index, columns=cols)
    out.reset_index(inplace=True)
    return out


def returns_from_weights(weights: pd.DataFrame,
                         trade_df: pd.DataFrame,
                         cost_bps: float) -> pd.Series:
    """Compute daily portfolio returns from a weights matrix.

    weights: row d = target weights chosen at close of d (decision-date
             indexed, per the new env convention).
    The return earned on row d's weights is over d -> d+1, applied to the
    NEXT row of the equity curve.

    The cost on day d is cost_bps * L1(w_d - w_{d-1}) / 1e4, applied at
    rebalance — same accounting as the env.
    """
    dates = list(weights["date"])
    tickers = [c for c in weights.columns if c != "date"]
    w = weights[tickers].to_numpy(dtype=float)
    n = len(dates)
    out_dates = []
    out_rets  = []
    for i in range(n - 1):
        date_today = dates[i]
        rows_today = trade_df[trade_df["date"] == date_today]
        rows_next  = trade_df[trade_df["date"] == dates[i + 1]]
        rows_today = rows_today[rows_today["tic"].isin(tickers)].sort_values("tic")
        rows_next  = rows_next [rows_next ["tic"].isin(tickers)].sort_values("tic")
        if len(rows_today) != len(tickers) or len(rows_next) != len(tickers):
            continue
        p_t = rows_today["close"].to_numpy(dtype=float)
        p_n = rows_next ["close"].to_numpy(dtype=float)
        per_asset_ret = p_n / np.maximum(p_t, 1e-12) - 1.0
        gross = float(np.dot(w[i], per_asset_ret))
        if i == 0:
            turnover = 0.0
        else:
            turnover = float(np.abs(w[i] - w[i - 1]).sum())
        tc_frac = (cost_bps / 1e4) * turnover
        net = (1.0 + gross) * (1.0 - tc_frac) - 1.0
        out_dates.append(date_today)
        out_rets.append(net)
    return pd.Series(out_rets, index=pd.to_datetime(out_dates))


def equity_from_returns(returns: pd.Series, initial: float = 1e8) -> pd.Series:
    return (1.0 + returns).cumprod() * initial


# ─────────────────────────────────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────────────────────────────────

def equal_weight_baseline(trade_df: pd.DataFrame, initial: float,
                          exclude_tickers: set[str] | None = None,
                          name: str = "EqualWeight") -> pd.Series:
    """Daily-rebalanced equal weight over (risky | risky+cash) assets.

    exclude=set() to include CASH (gives EqualWeight_w_Cash).
    exclude={'CASH'} for the cash-free EqualWeight (the FinRL convention).
    """
    exclude = set(exclude_tickers or [])
    dates = sorted(trade_df["date"].unique())
    portfolio = pd.Series(index=pd.to_datetime(dates), dtype=float)
    portfolio.iloc[0] = initial
    for i in range(len(dates) - 1):
        df_t = trade_df[trade_df["date"] == dates[i]]
        df_n = trade_df[trade_df["date"] == dates[i + 1]]
        if exclude:
            df_t = df_t[~df_t["tic"].isin(exclude)]
            df_n = df_n[~df_n["tic"].isin(exclude)]
        df_t = df_t.sort_values("tic").reset_index(drop=True)
        df_n = df_n.set_index("tic")
        m = len(df_t)
        cap = float(portfolio.iloc[i])
        shares = (cap / m) / df_t["close"].to_numpy(dtype=float)
        next_prices = df_n.loc[df_t["tic"]]["close"].to_numpy(dtype=float)
        portfolio.iloc[i + 1] = float(np.dot(shares, next_prices))
    portfolio.name = name
    return portfolio


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config       = load_config(args.config)
    _full, _train, trade_df = load_pickles(config)
    initial      = float(config["env"].get("initial_amount", 1e8))
    cost_bps     = float(config["env"].get("cost_bps", 10.0))
    results_dir  = resolve_path(config, "results_dir")

    seeds = list(config.get("seeds", {}).get("list", [42]))
    cash_ticker = str(config.get("cash", {}).get("ticker", "CASH"))

    # --- Predict per seed, ensemble weights ---
    per_seed_weights: dict[int, pd.DataFrame] = {}
    for s in seeds:
        _ret, weights = predict_one(config, s, trade_df)
        per_seed_weights[s] = weights
    ensemble_weights = average_seed_weights(list(per_seed_weights.values()))
    ensemble_weights.to_csv(results_dir / "weights_ppo.csv", index=False)

    ensemble_returns = returns_from_weights(ensemble_weights, trade_df,
                                            cost_bps=cost_bps)
    equity = {
        "PPO":                 equity_from_returns(ensemble_returns, initial),
        "EqualWeight":         equal_weight_baseline(trade_df, initial,
                                                     exclude_tickers={cash_ticker},
                                                     name="EqualWeight"),
        "EqualWeight_w_Cash":  equal_weight_baseline(trade_df, initial,
                                                     exclude_tickers=set(),
                                                     name="EqualWeight_w_Cash"),
    }
    result = pd.concat(equity, axis=1).sort_index().ffill()
    result.to_csv(results_dir / "equity_curves.csv")

    # Plot
    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    plt.title("Portfolio Allocation: PPO vs EqualWeight vs EqualWeight_w_Cash")
    plt.ylabel("Portfolio Value")
    plt.xlabel("Date")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(results_dir / "equity_plot.png", dpi=120)
    plt.close()

    # Metrics table
    print("\n" + "=" * 60)
    print(f"{'Strategy':<22} {'CumReturn':>12} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 60)
    for name in result.columns:
        m = compute_metrics(result[name])
        print(f"{name:<22} {m['cum_return']*100:>11.2f}% "
              f"{m['sharpe']:>10.3f} {m['max_dd']*100:>9.2f}%")
    print("=" * 60)

    # Active stats vs EW_w_Cash (the correct benchmark for a cash-holding agent)
    bench_col = "EqualWeight_w_Cash"
    print(f"\nActive-return statistics vs {bench_col}")
    print("=" * 92)
    print(f"{'Strategy':<22} {'Active_bps':>11} {'TE_bps':>9} "
          f"{'IR':>7} {'t_iid':>7} {'p_iid':>7} {'t_NW':>7} {'p_NW':>7} {'N':>5}")
    print("-" * 92)
    for name in result.columns:
        if name == bench_col:
            continue
        a = compute_active_stats(result[name], result[bench_col])
        if not np.isfinite(a.get("ir", np.nan)):
            continue
        print(f"{name:<22} {a['mean_active_bps']:>10.2f}  "
              f"{a['std_active_bps']:>8.2f} {a['ir']:>7.3f} "
              f"{a['t_iid']:>7.2f} {a['p_iid']:>7.3f} "
              f"{a['t_nw']:>7.2f} {a['p_nw']:>7.3f} "
              f"{a['n_days']:>5d}")
    print("=" * 92)

    # Cash-drag decomposition (the analyst's catch — keep it forever)
    print(f"\nCash-drag decomposition: IR vs both EW variants")
    print("=" * 92)
    print(f"{'Strategy':<22} {'IR vs EW (no cash)':>22} {'IR vs EW_w_Cash':>22} "
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
        print(f"{name:<22} {a_no['ir']:>22.3f} {a_wc['ir']:>22.3f} "
              f"{delta:>26.3f}")
    print("=" * 92)

    # Per-seed dispersion
    if len(seeds) >= 2:
        print(f"\nPer-seed IR distribution vs {bench_col}")
        print("=" * 92)
        ir_list = []
        for s in seeds:
            seed_returns = returns_from_weights(per_seed_weights[s],
                                                trade_df, cost_bps=cost_bps)
            seed_eq = equity_from_returns(seed_returns, initial)
            a = compute_active_stats(seed_eq, result[bench_col])
            if not np.isfinite(a.get("ir", np.nan)):
                continue
            print(f"  PPO  s={s:<5} IR={a['ir']:+.3f}  t_NW={a['t_nw']:+.2f}  "
                  f"p_NW={a['p_nw']:.3f}")
            ir_list.append(a["ir"])
        if len(ir_list) >= 2:
            ir_arr = np.asarray(ir_list)
            print(f"  MEAN={ir_arr.mean():+.3f}  sd={ir_arr.std(ddof=1):.3f}  "
                  f"signs +/- = {int((ir_arr>0).sum())}/{int((ir_arr<0).sum())}")


if __name__ == "__main__":
    main()
