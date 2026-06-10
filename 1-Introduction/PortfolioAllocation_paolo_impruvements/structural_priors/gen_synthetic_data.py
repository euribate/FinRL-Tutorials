"""Phase 3b — block-bootstrap synthetic data generator with planted alpha_signal.

Generates a synthetic 14-asset return panel that preserves the cross-sectional
correlation and volatility-clustering structure of the real data (via
moving-block bootstrap on the *date* index), with an `alpha_signal` per-asset
column whose cross-sectional rank-correlation with next-week's cumulative
return is set to a target IC.

The synthetic pickles are written to the same schema as 01_get_data.py's
output, so 02_train.py / 03_backtest.py can run on them with only a config
swap (paths point at the synthetic dir, custom_features.per_asset includes
'alpha_signal').

Per the analyst's framing: 3a alone proves only an upper bound on what's
theoretically extractable; the pipeline-loss measurement requires running
the actual PPO pipeline on the bootstrapped panel and comparing to the
matched-IC theoretical ceiling. This script makes that comparison runnable.

Usage:
    python gen_synthetic_data.py --config config_phase3_placebo_IC0.json --ic 0.0  --seed 42
    python gen_synthetic_data.py --config config_phase3_synth_IC040.json --ic 0.40 --seed 42

Validation:
    The end of generation prints a "Realised cross-sectional rank-IC" line.
    If it deviates from the target IC by more than 0.02, the injection math
    is broken and the run should not be trusted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from scipy.stats import rankdata, spearmanr

import features as F
from utils import (
    compute_cov_features,
    compute_rebalance_dates,
    load_config,
    resolve_indicator_list,
    resolve_path,
)


# ─────────────────────────────────────────────────────────────────────────
# Block bootstrap of the per-asset return panel
# ─────────────────────────────────────────────────────────────────────────

def block_bootstrap_returns(returns_panel: pd.DataFrame,
                            block_len: int,
                            rng: np.random.Generator) -> pd.DataFrame:
    """Moving-block bootstrap on the DATE index of a (T x N) returns panel.

    Sampling whole date-blocks (rather than per-asset) preserves the
    cross-sectional correlation structure within each block: every asset
    is resampled to the same bootstrap date, so the market factor,
    sector co-movements, and pairwise correlations are kept intact within
    block_len days. Across blocks the joint structure is destroyed in
    proportion to 1/block_len, which is the standard bootstrap trade-off.

    The output has the same row count, the same column set, and the
    same index as the input.
    """
    T = len(returns_panel)
    if T < block_len + 1:
        raise ValueError(f"block_len={block_len} too long for T={T} dates")

    # Number of blocks to draw, with ceiling so we always cover T rows.
    n_blocks = (T + block_len - 1) // block_len
    starts   = rng.integers(0, T - block_len + 1, size=n_blocks)
    rows     = []
    for s in starts:
        rows.append(returns_panel.iloc[s : s + block_len].values)
    bootstrap_panel = np.concatenate(rows, axis=0)[:T]

    return pd.DataFrame(
        bootstrap_panel,
        index=returns_panel.index,
        columns=returns_panel.columns,
    )


# ─────────────────────────────────────────────────────────────────────────
# Synthetic OHLCV from bootstrapped per-asset returns
# ─────────────────────────────────────────────────────────────────────────

def returns_to_synthetic_ohlcv(returns_panel: pd.DataFrame,
                               dates: pd.DatetimeIndex,
                               start_price: float = 100.0,
                               daily_range_pct: float = 0.005,
                               volume_default: float = 1e6) -> pd.DataFrame:
    """Convert (T x N) simple-returns to a FinRL-style long-format OHLCV df.

    Output columns: date, tic, open, high, low, close, volume, day. Open is
    the previous close; high/low bracket by +/- daily_range_pct. Volume is
    a flat constant — the env never reads it, but FeatureEngineer's
    stockstats path may.
    """
    if len(returns_panel) != len(dates):
        raise ValueError(
            f"returns_panel rows ({len(returns_panel)}) != dates ({len(dates)})"
        )
    rows = []
    for tic in returns_panel.columns:
        r = returns_panel[tic].to_numpy(dtype=float)
        close = np.empty(len(r), dtype=float)
        close[0] = start_price
        for t in range(1, len(r)):
            close[t] = close[t - 1] * (1.0 + r[t])
        open_  = np.empty_like(close)
        open_[0] = start_price
        open_[1:] = close[:-1]
        high = np.maximum(open_, close) * (1.0 + daily_range_pct)
        low  = np.minimum(open_, close) * (1.0 - daily_range_pct)
        for i, d in enumerate(dates):
            rows.append({
                "date":   d.strftime("%Y-%m-%d"),
                "tic":    tic,
                "open":   open_[i],
                "high":   high[i],
                "low":    low[i],
                "close":  close[i],
                "volume": volume_default,
                "day":    d.weekday(),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# Plant alpha_signal column with a target cross-sectional rank-IC
# ─────────────────────────────────────────────────────────────────────────

def plant_alpha_signal(df: pd.DataFrame,
                      rebalance_dates: set,
                      target_ic: float,
                      window: int,
                      rng: np.random.Generator) -> pd.DataFrame:
    """Inject an `alpha_signal` column into df with planted rank-IC.

    For each rebalance day t in `rebalance_dates`:
      1. Compute the cumulative cross-sectional return over the next `window`
         trading days for each asset.
      2. Cross-sectionally standardise: z_i = (rank(r_i) - mean) / std.
      3. Generate alpha_i = target_ic * z_i + sqrt(1 - target_ic^2) * eps_i,
         where eps_i is iid N(0,1) cross-sectionally.
      4. Broadcast alpha across the holding window (t to next rebalance day),
         so the agent sees a constant alpha within each rebalance week.

    For target_ic = 0 this reduces to pure noise (alpha is independent of z).
    The injection is forward-looking by construction — that is the point of
    a planted-signal test: the feature actually predicts the next-window
    return so the pipeline's ability to use it can be measured.
    """
    df = df.copy()
    df["date_d"] = pd.to_datetime(df["date"])

    # Per-asset cumulative-return-over-next-W-days series. Compute per ticker
    # then stack so we can index by (date, tic).
    df_sorted = df.sort_values(["tic", "date_d"]).reset_index(drop=True)
    df_sorted["log_ret"] = np.log(df_sorted.groupby("tic")["close"].shift(0) /
                                   df_sorted.groupby("tic")["close"].shift(1))
    # Forward cumulative log-return over W bars (t+1 ... t+W). Implemented as
    # rolling forward sum of log_ret shifted by -W applied to a reversed
    # series — equivalent to roll(W).sum().shift(-W).
    df_sorted["fwd_logret_W"] = (df_sorted.groupby("tic")["log_ret"]
                                  .shift(-1)
                                  .groupby(df_sorted["tic"])
                                  .rolling(window, min_periods=window).sum()
                                  .reset_index(level=0, drop=True))
    fwd_pivot = df_sorted.pivot(index="date_d", columns="tic",
                                values="fwd_logret_W").sort_index()

    # alpha_signal series, initialised to NaN; filled on rebalance days then
    # forward-filled across the holding window.
    alpha_pivot = pd.DataFrame(np.nan,
                               index=fwd_pivot.index,
                               columns=fwd_pivot.columns)
    rb_sorted = sorted(rebalance_dates)
    last_holding = None
    for t in rb_sorted:
        if t not in fwd_pivot.index:
            continue
        fwd = fwd_pivot.loc[t].to_numpy(dtype=float)
        if np.isnan(fwd).any():
            # Near the end of the panel where the W-step forward window
            # leaks past the last date. Plant noise (IC has no defined
            # value here; safer than dropping the row).
            alpha = rng.standard_normal(len(fwd))
        else:
            # Standardise the cross-sectional rank to mean 0, std 1.
            ranks = rankdata(fwd) - (len(fwd) + 1) / 2.0
            z     = ranks / (ranks.std(ddof=0) + 1e-12)
            eps   = rng.standard_normal(len(fwd))
            eps   = (eps - eps.mean()) / (eps.std(ddof=0) + 1e-12)
            alpha = target_ic * z + np.sqrt(max(1.0 - target_ic ** 2, 0.0)) * eps
        alpha_pivot.loc[t] = alpha
        last_holding = alpha

    # Forward-fill across non-rebalance days so the holding-week constant
    # alpha is the value the agent sees on every bar between rebalances.
    alpha_pivot = alpha_pivot.ffill()
    # The first few rows before any rebalance day are still NaN. Fill with
    # pure noise so the env always has a finite feature value.
    if alpha_pivot.iloc[0].isna().any():
        first_filled = alpha_pivot.bfill().iloc[0]
        alpha_pivot.iloc[0] = first_filled
        alpha_pivot = alpha_pivot.ffill()

    # Stack back to long format and merge into df by (date, tic).
    alpha_long = (alpha_pivot.stack().rename("alpha_signal").reset_index()
                  .rename(columns={"date_d": "date"}))
    alpha_long["date"] = alpha_long["date"].dt.strftime("%Y-%m-%d")
    df = df.drop(columns=["date_d"]).merge(alpha_long, on=["date", "tic"], how="left")

    # Realised IC validation (on rebalance days only, where alpha was
    # actually fresh — non-rebalance days carry the same value forward).
    rb_dates_str = {pd.Timestamp(d).strftime("%Y-%m-%d") for d in rb_sorted}
    rb_rows = df[df["date"].isin(rb_dates_str)].copy()
    fwd_long = df_sorted[["date_d", "tic", "fwd_logret_W"]].copy()
    fwd_long["date"] = fwd_long["date_d"].dt.strftime("%Y-%m-%d")
    fwd_long = fwd_long.drop(columns=["date_d"])
    rb_rows = rb_rows.merge(fwd_long, on=["date", "tic"], how="left")

    # Per-date cross-sectional rank-IC, then average.
    per_date_ics = []
    for d, sub in rb_rows.groupby("date"):
        if sub["fwd_logret_W"].isna().any() or sub["alpha_signal"].isna().any():
            continue
        if len(sub) < 3:
            continue
        rho, _ = spearmanr(sub["alpha_signal"], sub["fwd_logret_W"])
        if np.isfinite(rho):
            per_date_ics.append(rho)
    realised_ic = float(np.mean(per_date_ics)) if per_date_ics else np.nan
    print(f"  Realised cross-sectional rank-IC: {realised_ic:+.4f}  "
          f"(target {target_ic:+.4f},  averaged over "
          f"{len(per_date_ics)} rebalance days)")
    print(f"  IC injection {'OK' if abs(realised_ic - target_ic) < 0.02 else 'OFF >0.02'}")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Pipeline driver — mirrors 01_get_data.py without the Yahoo download
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   required=True)
    parser.add_argument("--ic",       type=float, required=True)
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--block-len", type=int,  default=20,
                        help="Moving-block bootstrap block length (days).")
    args = parser.parse_args()

    config   = load_config(args.config)
    data_cfg = config["data"]
    # OUTPUT dir = config.paths.data_dir (where 02_train.py reads from)
    data_dir = resolve_path(config, "data_dir")
    data_dir.mkdir(parents=True, exist_ok=True)
    # SOURCE dir = config.paths.source_data_dir, or default to ../<real data> sibling.
    # The synthetic configs should set source_data_dir to wherever the REAL pipeline
    # has already produced full_data.pkl (typically the base config's data_dir 'data').
    source_data_dir = Path(config.get("paths", {})
                                  .get("source_data_dir", "data"))
    if not source_data_dir.is_absolute():
        source_data_dir = Path(args.config).resolve().parent / source_data_dir

    rng = np.random.default_rng(args.seed)

    # ---- 1. Load REAL panel (use full_data.pkl - it has all 14 risky tickers
    # AND already had CASH injected if cash.enabled; we'll re-derive cleanly).
    # We only need the OHLCV part; everything else is recomputed below from
    # the bootstrapped prices.
    real_full = source_data_dir / "full_data.pkl"
    if not real_full.exists():
        raise FileNotFoundError(
            f"{real_full} not found. Run 01_get_data.py on the REAL config first "
            f"to get the input panel that this script bootstraps from."
        )
    df_real = pd.read_pickle(real_full)
    cash_ticker = str(config.get("cash", {}).get("ticker", "CASH"))
    # Strip CASH; we re-inject it after the bootstrap so its synthetic price
    # tracks the same risk-free-rate accumulation as on real data.
    df_no_cash = df_real[df_real["tic"] != cash_ticker].copy()
    print(f"Loaded {real_full}: {df_no_cash.shape}, "
          f"tickers={sorted(df_no_cash.tic.unique())}")

    # ---- 2. Extract per-asset simple-return panel ----
    # Wide pivot of close prices, compute pct_change, drop the first NaN row.
    close_wide = df_no_cash.pivot(index="date", columns="tic",
                                  values="close").sort_index()
    close_wide.index = pd.to_datetime(close_wide.index)
    rets_panel = close_wide.pct_change().dropna()
    print(f"Real return panel: {rets_panel.shape}   "
          f"date range {rets_panel.index.min().date()} -> "
          f"{rets_panel.index.max().date()}")

    # ---- 3. Block-bootstrap the return panel by DATE ----
    rets_boot = block_bootstrap_returns(rets_panel, args.block_len, rng)
    print(f"Block-bootstrapped (block_len={args.block_len}): {rets_boot.shape}")

    # ---- 4. Reconstruct synthetic OHLCV ----
    synthetic_dates = rets_boot.index
    df_syn_raw = returns_to_synthetic_ohlcv(rets_boot, synthetic_dates)
    print(f"Synthetic OHLCV (long): {df_syn_raw.shape}")

    # ---- 5. stockstats indicators + turbulence ----
    print(f"Computing indicators ({len(data_cfg['indicators'])}) + "
          f"turbulence ({data_cfg['use_turbulence']})...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=data_cfg["indicators"],
        use_turbulence=data_cfg["use_turbulence"],
        user_defined_feature=False,
    )
    df_proc = fe.preprocess_data(df_syn_raw)
    df_proc["date"] = df_proc["date"].astype(str)

    # ---- 6. benchmark_return (equal-weight of the synthetic risky names) ----
    df_proc = F.add_benchmark_return(df_proc,
                                     config.get("benchmark", {}) or {},
                                     None)

    # ---- 7. Plant alpha_signal BEFORE custom-feature computation.
    # The window matches the env-time rebalance cadence (weekly = 5 bars).
    rb_cfg = config.get("rebalance", {}) or {}
    cadence = str(rb_cfg.get("cadence", "weekly"))
    weekly_day = str(rb_cfg.get("weekly_day", "FRI"))
    window = 5 if cadence == "weekly" else 1
    rebalance_dates = compute_rebalance_dates(
        sorted(set(df_proc["date"].unique())),
        cadence=cadence, weekly_day=weekly_day,
    )
    print(f"Planting alpha_signal (target IC={args.ic:+.4f}, "
          f"window={window} bars, {len(rebalance_dates)} rebalance dates)...")
    df_proc = plant_alpha_signal(df_proc, rebalance_dates,
                                 target_ic=args.ic, window=window, rng=rng)

    # ---- 8. custom per-asset + global features ----
    cf            = data_cfg.get("custom_features", {}) or {}
    per_asset_keys = cf.get("per_asset", []) or []
    global_keys   = cf.get("global", []) or []
    params        = cf.get("params", {}) or {}
    if "alpha_signal" not in per_asset_keys:
        raise ValueError(
            "config.data.custom_features.per_asset must list 'alpha_signal' for "
            "phase 3 synthetic runs. Without it the planted column is dropped "
            "from the env state and the recovery test is meaningless."
        )
    if "vix" in global_keys:
        raise ValueError(
            "Phase 3 synthetic configs must NOT include 'vix' in "
            "custom_features.global — real VIX data is not available for "
            "the bootstrapped period. Drop 'vix' from the synthetic config."
        )
    df_proc, added = F.add_custom_features(df_proc, per_asset_keys,
                                           global_keys, params)
    print(f"Added {len(added)} custom feature columns: {added}")
    before = len(df_proc)
    df_proc = df_proc.dropna(subset=added).reset_index(drop=True)
    print(f"Dropped {before - len(df_proc)} warmup-NaN rows.")

    # ---- 9. Optionally re-inject CASH (mirror the real pipeline) ----
    cash_cfg = config.get("cash", {}) or {}
    if bool(cash_cfg.get("enabled", False)):
        rf            = float(cash_cfg.get("risk_free_rate", 0.0))
        per_asset_cols = list(data_cfg.get("indicators", [])) + \
            F.expected_columns(per_asset_keys, [])
        df_proc = F.inject_cash_asset(df_proc, cash_ticker, rf, per_asset_cols)
        print(f"Re-injected synthetic CASH '{cash_ticker}'")

    # ---- 10. Rolling covariance and split ----
    print(f"Computing rolling covariance (lookback={data_cfg['lookback']}d)...")
    df_full = compute_cov_features(df_proc, lookback=data_cfg["lookback"])
    print(f"After covariance: {df_full.shape}")

    needed = resolve_indicator_list(config)
    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise RuntimeError(f"Columns required by the env state are missing: "
                           f"{missing}")

    train_df = data_split(df_full, data_cfg["train_start_date"],
                          data_cfg["train_end_date"])
    trade_df = data_split(df_full, data_cfg["trade_start_date"],
                          data_cfg["trade_end_date"])
    print(f"Train rows: {len(train_df)}   Trade rows: {len(trade_df)}")

    # ---- 11. Save to OUTPUT data_dir (config.paths.data_dir) with the
    # standard filenames so 02_train.py / 03_backtest.py work unchanged.
    out_full  = data_dir / "full_data.pkl"
    out_train = data_dir / "train_data.pkl"
    out_trade = data_dir / "trade_data.pkl"
    df_full.to_pickle(out_full)
    train_df.to_pickle(out_train)
    trade_df.to_pickle(out_trade)
    print(f"Saved {out_full}")
    print(f"Saved {out_train}")
    print(f"Saved {out_trade}")


if __name__ == "__main__":
    main()
