"""Data pipeline: download → feature engineering → train/trade split.

Consolidates what used to be 01_get_data.py + the helpers from utils.py +
the feature registry from features.py.

Step 0 keeps the old feature set as-is. Step 1 (per the analyst's
roadmap) will rewrite this module to (a) reduce the per-asset feature
set to 6-8 columns and (b) rank-normalize them cross-sectionally so the
env can drop VecNormalize on the per-asset features.

Public surface:
  * load_config(path)              - load the new minimal config.json
  * resolve_path(config, key)      - resolve a relative path under config.paths
  * feature_columns(config)        - the ordered list of state columns
  * prepare_dataset(config)        - run the full pipeline and write pickles
  * load_pickles(config)           - load full/train/trade pickles
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# The feature registry stays in features.py for now (it's a clean module
# already and the @register decorator pattern is fine). We re-export the
# bits data.py needs so callers can `from data import ...`.
import features as _F


# ---------------------------------------------------------------------------
# Config + paths
# ---------------------------------------------------------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def resolve_path(config: dict, key: str) -> Path:
    p = Path(config["paths"][key])
    if not p.is_absolute():
        p = project_root() / p
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Feature column list (ordered) — what the env state vector contains
# ---------------------------------------------------------------------------

def feature_columns(config: dict) -> list[str]:
    """Ordered list of column names that form the env state.

    = config.data.indicators (stockstats keys) + custom per-asset columns
      + custom global columns.

    Pulls produced-column names from the features.py registry so config
    keys ('mom') map to their actual columns ('mom_1', 'mom_5', ...).
    """
    indicators = list(config["data"].get("indicators", []))
    cf = config["data"].get("custom_features", {}) or {}
    per_asset = cf.get("per_asset", []) or []
    global_   = cf.get("global", []) or []
    if per_asset or global_:
        indicators = indicators + _F.expected_columns(per_asset, global_)
    return indicators


# ---------------------------------------------------------------------------
# Covariance feature (kept verbatim — gen_synthetic uses it)
# ---------------------------------------------------------------------------

def compute_cov_features(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Attach trailing-`lookback` covariance + return matrices per date.

    cov_list[t]    : (N, N) sample covariance of the trailing returns.
    return_list[t] : (lookback-1, N) DataFrame of the trailing returns.

    Both stored as object cells. Used by the static-prior baselines AND
    by gen_synthetic_data when it re-runs the feature pipeline on
    bootstrapped prices.
    """
    df = df.sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    cov_list: list[np.ndarray] = []
    return_list: list[pd.DataFrame] = []

    for i in range(lookback, len(df.index.unique())):
        data_lookback   = df.loc[i - lookback : i, :]
        price_lookback  = data_lookback.pivot_table(
            index="date", columns="tic", values="close"
        )
        return_lookback = price_lookback.pct_change().dropna()
        return_list.append(return_lookback)
        cov_list.append(return_lookback.cov().values)

    df_cov = pd.DataFrame({
        "date":        df.date.unique()[lookback:],
        "cov_list":    cov_list,
        "return_list": return_list,
    })
    df = df.merge(df_cov, on="date")
    return df.sort_values(["date", "tic"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Full pipeline (mirrors 01_get_data.py)
# ---------------------------------------------------------------------------

def prepare_dataset(config: dict) -> None:
    """Download + feature-engineer + split + write pickles.

    Side-effect-only function. Writes:
      data_dir/full_data.pkl    full panel with all features + cov_list
      data_dir/train_data.pkl   train slice
      data_dir/trade_data.pkl   trade slice

    External deps for Step 0: FinRL's YahooDownloader + FeatureEngineer.
    Step 1 will replace these with our own minimal versions.
    """
    # Lazy import — only needed when downloading.
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    data_cfg = config["data"]
    data_dir = resolve_path(config, "data_dir")

    print(f"Downloading {len(data_cfg['ticker_list'])} tickers...")
    df_raw = YahooDownloader(
        start_date=data_cfg["download_start_date"],
        end_date=data_cfg["trade_end_date"],
        ticker_list=data_cfg["ticker_list"],
    ).fetch_data()

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=data_cfg["indicators"],
        use_turbulence=data_cfg.get("use_turbulence", True),
        user_defined_feature=False,
    )
    df_proc = fe.preprocess_data(df_raw)
    df_proc["date"] = df_proc["date"].astype(str)

    # Benchmark return (equal-weight of the risky assets).
    df_proc = _F.add_benchmark_return(df_proc,
                                     config.get("benchmark", {}) or {},
                                     None)

    # Custom features.
    cf = data_cfg.get("custom_features", {}) or {}
    per_asset = cf.get("per_asset", []) or []
    global_   = cf.get("global", []) or []
    params    = cf.get("params", {}) or {}
    if per_asset or global_:
        df_proc, added = _F.add_custom_features(df_proc, per_asset,
                                                global_, params)
        before = len(df_proc)
        df_proc = df_proc.dropna(subset=added).reset_index(drop=True)
        print(f"  Added {len(added)} custom features; dropped "
              f"{before - len(df_proc)} warmup-NaN rows.")

    # Cash injection.
    cash_cfg = config.get("cash", {}) or {}
    if bool(cash_cfg.get("enabled", False)):
        per_asset_cols = list(data_cfg.get("indicators", [])) + \
                         _F.expected_columns(per_asset, [])
        df_proc = _F.inject_cash_asset(df_proc,
                                       str(cash_cfg.get("ticker", "CASH")),
                                       float(cash_cfg.get("risk_free_rate", 0.0)),
                                       per_asset_cols)

    # Rolling covariance + split.
    df_full = compute_cov_features(df_proc, lookback=data_cfg["lookback"])

    train_df = data_split(df_full, data_cfg["train_start_date"],
                          data_cfg["train_end_date"])
    trade_df = data_split(df_full, data_cfg["trade_start_date"],
                          data_cfg["trade_end_date"])

    df_full.to_pickle(data_dir / "full_data.pkl")
    train_df.to_pickle(data_dir / "train_data.pkl")
    trade_df.to_pickle(data_dir / "trade_data.pkl")
    print(f"Wrote {data_dir}/full_data.pkl  ({len(df_full)} rows)")


# ---------------------------------------------------------------------------
# Load pickles
# ---------------------------------------------------------------------------

def load_pickles(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load (full, train, trade) pickles. Fails fast if any is missing."""
    data_dir = resolve_path(config, "data_dir")
    paths = [data_dir / f for f in ("full_data.pkl",
                                     "train_data.pkl",
                                     "trade_data.pkl")]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Run `prepare_dataset(config)` first."
            )
    return tuple(pd.read_pickle(p) for p in paths)
