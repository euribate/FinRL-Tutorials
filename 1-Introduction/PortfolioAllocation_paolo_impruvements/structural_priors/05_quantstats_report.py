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

from utils import fetch_yahoo_with_retry, load_config, project_root


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


def resolve_benchmark_returns(qs_cfg: dict, main_cfg: dict,
                              returns_index: pd.DatetimeIndex) -> tuple[pd.Series, str]:
    """Return (benchmark_daily_returns, label) for the tearsheet.

    use_source=true (default): follow the MAIN config's benchmark block by
    reading the benchmark_return series stage 1 stored in full_data.pkl. This
    matches config.benchmark exactly (equal_weight or ticker) and is identical
    to the series the reward / features / stage-3 baseline use.

    use_source=false: download the explicit override ticker from Yahoo.
    """
    bench = qs_cfg["benchmark"]
    if bench.get("use_source", True):
        from utils import benchmark_label, load_benchmark_returns
        label = benchmark_label(main_cfg)
        s = load_benchmark_returns(main_cfg).reindex(returns_index).fillna(0.0)
        s.name = label
        return s, label
    ticker = bench.get("ticker")
    if not ticker:
        raise ValueError("benchmark.use_source=false but benchmark.ticker is null.")
    return fetch_benchmark_returns(ticker, returns_index), ticker


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
    df = fetch_yahoo_with_retry(ticker, start, end)[["date", "close"]]
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

    out_dir = Path(qs_cfg["output"]["report_dir"])
    if not out_dir.is_absolute():
        out_dir = project_dir / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    html_path = out_dir / qs_cfg["output"]["html_filename"]
    title     = qs_cfg["report"].get("title", "Strategy Tearsheet")
    rf        = float(qs_cfg["report"].get("risk_free_rate", 0.0))

    benchmark, bench_label = resolve_benchmark_returns(qs_cfg, main_cfg, returns.index)

    print(f"Equity curve: {qs_cfg['inputs']['equity_csv']}")
    print(f"Date range:   {returns.index.min().date()} -> {returns.index.max().date()}")
    print(f"Days:         {len(returns)}")
    print(f"Strategy:     {strategy_name}")
    print(f"Benchmark:    {bench_label}")
    print(f"Output:       {html_path}\n")

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

    # Cash-inclusive variant report: mirrors stage-3's EqualWeight_w_Cash row
    # and stage-4's dotted overlay. QuantStats is single-benchmark, so we
    # generate a SECOND html (and metrics CSV) using the cash-inclusive
    # EqualWeight as the benchmark. Math: with cash earning 0, a 1/(M+1)
    # daily-rebalanced portfolio's return is exactly M/(M+1) * r_eq_no_cash,
    # so we scale the existing series. Only applies when the benchmark is
    # the equal-weight series from stage 1 (use_source=true, type=equal_weight)
    # with cash.enabled=true; controlled by benchmark.show_cash_inclusive.
    bench_cfg_main = main_cfg.get("benchmark", {}) or {}
    cash_cfg_main  = main_cfg.get("cash", {}) or {}
    if (bench_cfg_main.get("show_cash_inclusive", True)
            and cash_cfg_main.get("enabled", False)
            and bench_cfg_main.get("type", "equal_weight") == "equal_weight"
            and qs_cfg.get("benchmark", {}).get("use_source", True)):
        M = len(main_cfg.get("data", {}).get("ticker_list", []) or [])
        if M > 0:
            bench_wc     = (M / (M + 1)) * benchmark
            label_wc     = f"{bench_label} w/ Cash"
            html_wc_path = html_path.with_name(f"{html_path.stem}_w_cash{html_path.suffix}")
            title_wc     = f"{title} (vs {label_wc})"
            print(f"\nGenerating cash-inclusive variant report (benchmark = {label_wc})...")
            qs.reports.html(
                returns,
                benchmark=bench_wc,
                output=str(html_wc_path),
                title=title_wc,
                rf=rf,
            )
            print(f"Saved {html_wc_path}")
            if metrics_csv:
                stem  = Path(metrics_csv).stem
                suf   = Path(metrics_csv).suffix
                metrics_wc_path = out_dir / f"{stem}_w_cash{suf}"
                metrics_wc = qs.reports.metrics(
                    returns, benchmark=bench_wc, mode="full", rf=rf, display=False,
                )
                if isinstance(metrics_wc, pd.DataFrame):
                    metrics_wc.to_csv(metrics_wc_path)
                    print(f"Saved {metrics_wc_path}")

    # Structural-prior variant report: emitted when policy_prior is active.
    # The right benchmark for a residual-policy PPO is the STATIC prior
    # portfolio (Prior_{type} column written by stage 3), NOT EqualWeight -
    # because the agent's job is to add tilt on top of that prior. This
    # report tells you how much of the system Sharpe is the agent's learned
    # tilt vs the prior alone. Disabled when policy_prior.enabled=false or
    # type='none'; controlled by benchmark.show_prior_baseline (default true).
    pp_cfg     = main_cfg.get("policy_prior", {}) or {}
    if (bench_cfg_main.get("show_prior_baseline", True)
            and pp_cfg.get("enabled", False)
            and pp_cfg.get("type", "none") != "none"):
        ptype = str(pp_cfg.get("type"))
        prior_col = f"Prior_{ptype}"
        # Stage 3 writes results/equity_curves.csv with the Prior_{type} column.
        s3_results_dir = main_cfg.get("paths", {}).get("results_dir", "results")
        s3_curves_path = Path(s3_results_dir)
        if not s3_curves_path.is_absolute():
            s3_curves_path = project_dir / s3_curves_path
        s3_curves_path = s3_curves_path / "equity_curves.csv"
        if not s3_curves_path.exists():
            print(f"WARNING: {s3_curves_path} not found; skipping prior-baseline report.")
        else:
            df_s3 = pd.read_csv(s3_curves_path)
            date_col = df_s3.columns[0]
            df_s3[date_col] = pd.to_datetime(df_s3[date_col])
            df_s3 = df_s3.set_index(date_col).sort_index()
            if prior_col not in df_s3.columns:
                print(f"WARNING: stage-3 equity_curves.csv has no {prior_col!r} "
                      f"column (re-run 03_backtest.py with policy_prior enabled). "
                      f"Skipping prior-baseline report.")
            else:
                prior_eq      = df_s3[prior_col].astype(float)
                bench_prior   = prior_eq.pct_change().reindex(returns.index).fillna(0.0)
                label_prior   = f"Prior ({ptype})"
                html_pr_path  = html_path.with_name(f"{html_path.stem}_vs_prior{html_path.suffix}")
                title_pr      = f"{title} (vs {label_prior})"
                print(f"\nGenerating prior-baseline report (benchmark = {label_prior})...")
                qs.reports.html(
                    returns,
                    benchmark=bench_prior,
                    output=str(html_pr_path),
                    title=title_pr,
                    rf=rf,
                )
                print(f"Saved {html_pr_path}")
                if metrics_csv:
                    stem = Path(metrics_csv).stem
                    suf  = Path(metrics_csv).suffix
                    metrics_pr_path = out_dir / f"{stem}_vs_prior{suf}"
                    metrics_pr = qs.reports.metrics(
                        returns, benchmark=bench_prior, mode="full", rf=rf, display=False,
                    )
                    if isinstance(metrics_pr, pd.DataFrame):
                        metrics_pr.to_csv(metrics_pr_path)
                        print(f"Saved {metrics_pr_path}")


if __name__ == "__main__":
    main()
