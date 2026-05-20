"""Stage 5 - QuantStats HTML tearsheet from the backtrader equity curve.

Reads the equity-curve CSV produced by 04_backtrader_replay.py, converts it
into a daily-returns series, downloads the configured benchmark from Yahoo
(default: ^DJI), and renders a QuantStats tearsheet HTML comparing the two.

Usage:
    python 05_quantstats_report.py --config quantstats_config.json

Requires:
    pip install quantstats
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import quantstats as qs
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

from utils import load_config, project_root


def load_configs(qs_config_path: str) -> tuple[dict, dict, dict, Path]:
    """Load quantstats config + the upstream backtrader and main configs."""
    qs_cfg = load_config(qs_config_path)

    bt_path = Path(qs_cfg["source_config"])
    if not bt_path.is_absolute():
        bt_path = project_root() / bt_path
    bt_cfg = load_config(bt_path)

    main_path = Path(bt_cfg["source_config"])
    if not main_path.is_absolute():
        main_path = bt_path.parent / main_path
    main_cfg = load_config(main_path)

    return qs_cfg, bt_cfg, main_cfg, bt_path.parent


def load_equity_returns(qs_cfg: dict, project_dir: Path, name: str) -> pd.Series:
    """Read the equity CSV and convert to a daily-returns series."""
    eq_path = Path(qs_cfg["inputs"]["equity_csv"])
    if not eq_path.is_absolute():
        eq_path = project_dir / eq_path
    if not eq_path.exists():
        raise FileNotFoundError(
            f"{eq_path} not found - run 04_backtrader_replay.py first."
        )

    df = pd.read_csv(eq_path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    if "equity" not in df.columns:
        raise ValueError(
            f"{eq_path} must contain an 'equity' column; found {list(df.columns)}."
        )
    returns = df["equity"].pct_change().dropna()
    returns.name = name
    return returns


def resolve_benchmark(qs_cfg: dict, main_cfg: dict) -> str:
    bench = qs_cfg["benchmark"]
    if bench.get("use_source", True):
        return main_cfg["baselines"]["dji_ticker"]
    ticker = bench.get("ticker")
    if not ticker:
        raise ValueError(
            "benchmark.use_source=false but benchmark.ticker is null."
        )
    return ticker


def fetch_benchmark_returns(ticker: str, returns_index: pd.DatetimeIndex) -> pd.Series:
    """Pre-fetch benchmark daily returns via FinRL's YahooDownloader.

    QuantStats's internal `yfinance.download()` call relies on `fc.yahoo.com`
    for cookie/crumb retrieval and can fail silently on intermittent DNS issues,
    leaving the benchmark series empty and the report's comparison panels
    meaningless. Pre-fetching here lets us validate the series and produce a
    clear error instead of a degenerate report.
    """
    # Pad the start by 5 trading days so pct_change has a baseline observation
    # aligned to the strategy's first day.
    start = (returns_index.min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end   = (returns_index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = YahooDownloader(start_date=start, end_date=end,
                         ticker_list=[ticker]).fetch_data()[["date", "close"]]
    if df.empty:
        raise RuntimeError(
            f"Benchmark download for {ticker!r} returned no rows "
            f"({start} -> {end}). Check network / DNS to Yahoo Finance."
        )
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    ret = df["close"].pct_change().dropna()
    if ret.empty:
        raise RuntimeError(
            f"Benchmark series for {ticker!r} is empty after pct_change."
        )
    ret.name = ticker
    return ret


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="quantstats_config.json")
    args = parser.parse_args()

    qs_cfg, _bt_cfg, main_cfg, project_dir = load_configs(args.config)

    strategy_name = qs_cfg["report"].get("strategy_name", "Strategy")
    returns       = load_equity_returns(qs_cfg, project_dir, strategy_name)
    bench_ticker  = resolve_benchmark(qs_cfg, main_cfg)

    out_dir = Path(qs_cfg["output"]["report_dir"])
    if not out_dir.is_absolute():
        out_dir = project_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / qs_cfg["output"]["html_filename"]
    title     = qs_cfg["report"].get("title", "Strategy Tearsheet")
    rf        = float(qs_cfg["report"].get("risk_free_rate", 0.0))

    print(f"Equity curve: {qs_cfg['inputs']['equity_csv']}")
    print(f"Date range:   {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"Days:         {len(returns)}")
    print(f"Strategy:     {strategy_name}")
    print(f"Benchmark:    {bench_ticker}")
    print(f"Output:       {html_path}\n")

    benchmark = fetch_benchmark_returns(bench_ticker, returns.index)
    print(f"Benchmark series: {len(benchmark)} rows "
          f"({benchmark.index.min().date()} -> {benchmark.index.max().date()})\n")

    qs.extend_pandas()
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=str(html_path),
        title=title,
        rf=rf,
    )
    print(f"Saved {html_path}")

    metrics_csv = qs_cfg["output"].get("metrics_csv")
    if metrics_csv:
        metrics_path = out_dir / metrics_csv
        metrics = qs.reports.metrics(
            returns, benchmark=benchmark, mode="full", rf=rf, display=False,
        )
        if isinstance(metrics, pd.DataFrame):
            metrics.to_csv(metrics_path)
            print(f"Saved {metrics_path}")
        else:
            print(f"qs.reports.metrics returned {type(metrics).__name__}; "
                  f"skipped {metrics_path}.")


if __name__ == "__main__":
    main()
