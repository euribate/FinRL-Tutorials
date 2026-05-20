"""Stage 1 - download + feature engineering + covariance feature + split.

Replicates the data-preparation cells of FinRL_PortfolioAllocation_NeurIPS_2020.

Usage:
    python 01_get_data.py --config config.json
"""
from __future__ import annotations

import argparse

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from utils import compute_cov_features, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config   = load_config(args.config)
    data_cfg = config["data"]
    data_dir = resolve_path(config, "data_dir")

    print(f"Downloading {len(data_cfg['ticker_list'])} tickers "
          f"({data_cfg['download_start_date']} -> {data_cfg['trade_end_date']})...")
    df_raw = YahooDownloader(
        start_date=data_cfg["download_start_date"],
        end_date=data_cfg["trade_end_date"],
        ticker_list=data_cfg["ticker_list"],
    ).fetch_data()
    print(f"  Raw shape: {df_raw.shape}   tickers downloaded: {df_raw.tic.nunique()}")

    print(f"Computing {len(data_cfg['indicators'])} indicators "
          f"(turbulence={data_cfg['use_turbulence']})...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=data_cfg["indicators"],
        use_turbulence=data_cfg["use_turbulence"],
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df_raw)
    print(f"  After indicators: {df_processed.shape}")

    print(f"Computing rolling covariance (lookback={data_cfg['lookback']}d)...")
    df_full = compute_cov_features(df_processed, lookback=data_cfg["lookback"])
    print(f"  After covariance: {df_full.shape}")

    train_df = data_split(df_full,
                          data_cfg["train_start_date"],
                          data_cfg["train_end_date"])
    trade_df = data_split(df_full,
                          data_cfg["trade_start_date"],
                          data_cfg["trade_end_date"])
    print(f"  Train rows: {len(train_df)}   Trade rows: {len(trade_df)}")

    # The full processed frame is the source-of-truth for walk-forward
    # (stages 2 and 3 slice it per window). The legacy train/trade pickles
    # are kept so non-walk-forward configs and stages 4/5 still work unchanged.
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
