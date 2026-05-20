"""Stage 3 - backtest enabled algorithms + Min-Variance + DJIA baselines.

Replicates the trading and comparison cells of
FinRL_PortfolioAllocation_NeurIPS_2020.

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
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from pypfopt.efficient_frontier import EfficientFrontier

from utils import (
    ALGO_REGISTRY,
    enabled_models,
    load_config,
    make_portfolio_env,
    resolve_path,
)


def load_pickle(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run 01_get_data.py first.")
    return pd.read_pickle(path)


def predict_one(algo: str, trade_df: pd.DataFrame, config: dict,
                model_dir: Path) -> pd.DataFrame:
    """Run an SB3 agent against a fresh StockPortfolioEnv variant.

    The env class (upstream vs log-return) is selected by config.env.reward_mode.
    At inference time the reward signal is irrelevant - DRL_prediction only
    reads actions and observations - but using the same env class at training
    and inference keeps observation semantics identical.
    """
    AlgoClass = ALGO_REGISTRY[algo]
    model     = AlgoClass.load(str(model_dir / f"agent_{algo}.zip"))

    stock_dim = len(trade_df.tic.unique())
    env       = make_portfolio_env(trade_df, config, stock_dim)

    daily_return, _ = DRLAgent.DRL_prediction(model=model, environment=env)
    return daily_return


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
        # clean_weights() returns {ticker: weight}; values() preserves the
        # ticker ordering of Sigma's columns, which matches df_temp.close
        # (both alphabetical, since preprocessing sorts by ['date', 'tic']).
        cash_per_stock = np.array([cap * w for w in weights.values()])
        shares = cash_per_stock / np.array(df_temp.close, dtype=float)
        next_prices = np.array(df_temp_next.close, dtype=float)
        portfolio.iloc[i + 1] = float(np.dot(shares, next_prices))

    portfolio.index = pd.to_datetime(portfolio.index)
    portfolio.name = "MinVariance"
    return portfolio


def compute_dji_baseline(start: str, end: str, ticker: str, initial: float) -> pd.Series:
    df_dji = YahooDownloader(start_date=start, end_date=end,
                             ticker_list=[ticker]).fetch_data()[["date", "close"]]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()

    config      = load_config(args.config)
    data_dir    = resolve_path(config, "data_dir")
    model_dir   = resolve_path(config, "model_dir")
    results_dir = resolve_path(config, "results_dir")

    initial = float(config["env"]["initial_amount"])
    trade_df = load_pickle(data_dir / "trade_data.pkl")

    equity: dict[str, pd.Series] = {}

    for algo in enabled_models(config):
        print(f"\nBacktesting {algo.upper()}...")
        daily_return = predict_one(algo, trade_df, config, model_dir)
        equity[algo.upper()] = daily_return_to_equity(daily_return, initial)

    print("\nComputing Min-Variance baseline...")
    weight_bounds = config["baselines"]["min_variance"]["weight_bounds"]
    equity["MinVariance"] = compute_min_variance_baseline(trade_df, initial, weight_bounds)

    print("Computing DJIA baseline...")
    equity["DJIA"] = compute_dji_baseline(
        start=config["data"]["trade_start_date"],
        end=config["data"]["trade_end_date"],
        ticker=config["baselines"]["dji_ticker"],
        initial=initial,
    )

    result = pd.concat(equity, axis=1).sort_index().ffill()

    csv_path = results_dir / "equity_curves.csv"
    result.to_csv(csv_path)
    print(f"\nSaved {csv_path}")

    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    plt.title("Portfolio Allocation: RL Agents vs Min-Variance vs DJIA")
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
