"""Stage 1 — download + feature engineering + chronological split.

Usage:
    python 01_get_data.py --config config.json
"""
from __future__ import annotations

import argparse

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from utils import load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    data_dir = resolve_path(config, "data_dir")

    print(f"Downloading {len(data_cfg['ticker_list'])} tickers "
          f"({data_cfg['train_start_date']} -> {data_cfg['trade_end_date']})...")
    df_raw = YahooDownloader(
        start_date=data_cfg["train_start_date"],
        end_date=data_cfg["trade_end_date"],
        ticker_list=data_cfg["ticker_list"],
    ).fetch_data()
    print(f"  Raw shape: {df_raw.shape}")

    print(f"Computing {len(data_cfg['indicators'])} indicators "
          f"(vix={data_cfg['use_vix']}, turbulence={data_cfg['use_turbulence']})...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=data_cfg["indicators"],
        use_vix=data_cfg["use_vix"],
        use_turbulence=data_cfg["use_turbulence"],
        user_defined_feature=False,
    )
    df_processed = fe.preprocess_data(df_raw).fillna(0).reset_index(drop=True)
    print(f"  Processed shape: {df_processed.shape}")

    train_df = data_split(df_processed,
                          data_cfg["train_start_date"],
                          data_cfg["train_end_date"])
    trade_df = data_split(df_processed,
                          data_cfg["trade_start_date"],
                          data_cfg["trade_end_date"])
    print(f"  Train rows: {len(train_df)}   Trade rows: {len(trade_df)}")

    train_path = data_dir / "train_data.csv"
    trade_path = data_dir / "trade_data.csv"
    train_df.to_csv(train_path)
    trade_df.to_csv(trade_path)
    print(f"Saved {train_path}")
    print(f"Saved {trade_path}")


if __name__ == "__main__":
    main()
