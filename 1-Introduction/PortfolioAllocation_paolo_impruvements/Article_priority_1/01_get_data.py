"""Stage 1 - download + feature engineering + covariance feature + split.

Pipeline (priority-1 article features):
  1. Download OHLCV for the universe tickers (Yahoo).
  2. FeatureEngineer: stockstats technical indicators + turbulence.
  3. Add a benchmark_return column (equal-weight or a benchmark ticker).
  4. Download ^VIX and attach a vix_level column (broadcast per date).
  5. Compute custom per-asset + global features (features.py).
  6. If cash.enabled: inject a synthetic CASH asset (price compounding at the
     risk-free rate; per-asset features zeroed) so the env allocates across
     N risky assets + cash with no env surgery.
  7. Rolling covariance (now (N[+1])x(N[+1])), then train/trade split.

Usage:
    python 01_get_data.py --config config.json
"""
from __future__ import annotations

import argparse

import pandas as pd
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

import features as F
from utils import (
    compute_cov_features,
    fetch_yahoo_with_retry,
    load_config,
    resolve_indicator_list,
    resolve_path,
)


def add_vix_level(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Download ^VIX and attach a per-date vix_level column (broadcast)."""
    print("Downloading ^VIX for global features...")
    vix = fetch_yahoo_with_retry("^VIX", start, end)
    vix = vix[["date", "close"]].rename(columns={"close": "vix_level"})
    vix["date"] = vix["date"].astype(str)
    vix_map = vix.drop_duplicates("date").set_index("date")["vix_level"]
    df = df.copy()
    df["date"] = df["date"].astype(str)
    df["vix_level"] = df["date"].map(vix_map)
    # Forward/back-fill the few dates where VIX and the equity calendar differ.
    df["vix_level"] = (df.sort_values("date")
                         .groupby("tic")["vix_level"]
                         .ffill().bfill().reset_index(level=0, drop=True))
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config   = load_config(args.config)
    data_cfg = config["data"]
    data_dir = resolve_path(config, "data_dir")

    # ---- 1. Download universe ----
    print(f"Downloading {len(data_cfg['ticker_list'])} tickers "
          f"({data_cfg['download_start_date']} -> {data_cfg['trade_end_date']})...")
    df_raw = YahooDownloader(
        start_date=data_cfg["download_start_date"],
        end_date=data_cfg["trade_end_date"],
        ticker_list=data_cfg["ticker_list"],
    ).fetch_data()
    print(f"  Raw shape: {df_raw.shape}   tickers downloaded: {df_raw.tic.nunique()}")

    # ---- 2. stockstats indicators + turbulence ----
    print(f"Computing {len(data_cfg['indicators'])} indicators "
          f"(turbulence={data_cfg['use_turbulence']})...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=data_cfg["indicators"],
        use_turbulence=data_cfg["use_turbulence"],
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df_raw)
    df_processed["date"] = df_processed["date"].astype(str)
    print(f"  After indicators: {df_processed.shape}")

    # ---- 3. benchmark_return column ----
    bench_cfg = config.get("benchmark", {}) or {}
    btype = bench_cfg.get("type", "equal_weight")
    print(f"Adding benchmark_return (type={btype})...")
    ticker_ret = None
    if btype == "ticker":
        bt = bench_cfg.get("ticker", "^DJI")
        bench_raw = fetch_yahoo_with_retry(bt,
                                           data_cfg["download_start_date"],
                                           data_cfg["trade_end_date"])
        bench_raw["date"] = bench_raw["date"].astype(str)
        bser = (bench_raw.drop_duplicates("date").set_index("date")["close"]
                .sort_index().pct_change())
        ticker_ret = bser
    df_processed = F.add_benchmark_return(df_processed, bench_cfg, ticker_ret)

    # ---- 4. VIX (only if a vix feature is requested) ----
    cf = data_cfg.get("custom_features", {}) or {}
    global_keys = cf.get("global", []) or []
    if "vix" in global_keys:
        df_processed = add_vix_level(df_processed,
                                     data_cfg["download_start_date"],
                                     data_cfg["trade_end_date"])

    # ---- 5. custom per-asset + global features ----
    per_asset_keys = cf.get("per_asset", []) or []
    params = cf.get("params", {}) or {}
    if per_asset_keys or global_keys:
        print(f"Computing custom features: per_asset={per_asset_keys}  global={global_keys}")
        df_processed, added = F.add_custom_features(df_processed, per_asset_keys,
                                                    global_keys, params)
        print(f"  Added {len(added)} custom columns: {added}")
        # Drop warmup rows that contain NaN in any custom column (the longest
        # warmup window is well under the 252-day covariance lookback, so this
        # only trims the very start; compute_cov_features drops the rest).
        before = len(df_processed)
        df_processed = df_processed.dropna(subset=added).reset_index(drop=True)
        print(f"  Dropped {before - len(df_processed)} warmup rows with NaN features.")

    # ---- 6. synthetic CASH asset ----
    cash_cfg = config.get("cash", {}) or {}
    if bool(cash_cfg.get("enabled", False)):
        cash_ticker = str(cash_cfg.get("ticker", "CASH"))
        rf = float(cash_cfg.get("risk_free_rate", 0.0))
        # Per-asset feature columns to zero out for cash = stockstats indicators
        # + custom per-asset feature columns (NOT the global ones).
        per_asset_cols = list(data_cfg.get("indicators", [])) + \
            F.expected_columns(per_asset_keys, [])
        print(f"Injecting synthetic CASH asset '{cash_ticker}' (rf={rf:.4f}/yr)...")
        df_processed = F.inject_cash_asset(df_processed, cash_ticker, rf, per_asset_cols)
        print(f"  Tickers now: {sorted(df_processed.tic.unique())}")

    # ---- 7. rolling covariance + split ----
    print(f"Computing rolling covariance (lookback={data_cfg['lookback']}d)...")
    df_full = compute_cov_features(df_processed, lookback=data_cfg["lookback"])
    print(f"  After covariance: {df_full.shape}")

    # Sanity: the env's combined indicator list must all be present.
    needed = resolve_indicator_list(config)
    missing = [c for c in needed if c not in df_full.columns]
    if missing:
        raise RuntimeError(f"Columns required by the env state are missing from the "
                           f"data: {missing}. Check config.data.custom_features.")

    train_df = data_split(df_full, data_cfg["train_start_date"], data_cfg["train_end_date"])
    trade_df = data_split(df_full, data_cfg["trade_start_date"], data_cfg["trade_end_date"])
    print(f"  Train rows: {len(train_df)}   Trade rows: {len(trade_df)}   "
          f"state rows = cov({df_full.tic.nunique()}) + indicators({len(needed)})")

    full_path  = data_dir / "full_data.pkl"
    train_path = data_dir / "train_data.pkl"
    trade_path = data_dir / "trade_data.pkl"
    df_full.to_pickle(full_path)
    train_df.to_pickle(train_path)
    trade_df.to_pickle(trade_path)
    print(f"Saved {full_path}")
    print(f"Saved {train_path}")
    print(f"Saved {trade_path}")


if __name__ == "__main__":
    main()
