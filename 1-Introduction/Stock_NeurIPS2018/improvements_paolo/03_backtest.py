"""Stage 3 — backtest every enabled algorithm + MVO + DJIA baselines.

Outputs:
    results/equity_curves.csv   — daily portfolio value per strategy
    results/equity_plot.png     — overlaid equity curves
    stdout                       — cumulative return / Sharpe / max drawdown

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
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from pypfopt.efficient_frontier import EfficientFrontier
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from utils import (
    ALGO_REGISTRY,
    build_env_kwargs,
    enabled_models,
    load_config,
    resolve_path,
)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.set_index(df.columns[0])
    df.index.names = [""]
    return df


def build_trade_env(config: dict, trade_df: pd.DataFrame) -> StockTradingEnv:
    stock_dim = len(trade_df.tic.unique())
    env_kwargs = build_env_kwargs(config, stock_dim, mode="trade")
    return StockTradingEnv(df=trade_df, **env_kwargs)


def predict_finrl(algo: str, env: StockTradingEnv, model_dir: Path) -> pd.DataFrame:
    """Standard FinRL inference path."""
    AlgoClass = ALGO_REGISTRY[algo]
    model = AlgoClass.load(str(model_dir / f"agent_{algo}.zip"))
    df_account, _ = DRLAgent.DRL_prediction(model=model, environment=env)
    return df_account


def predict_vecnormalize(algo: str, env: StockTradingEnv, model_dir: Path) -> pd.DataFrame:
    """Manual rollout against a VecNormalize-wrapped env (stats frozen)."""
    AlgoClass = ALGO_REGISTRY[algo]
    model = AlgoClass.load(str(model_dir / f"agent_{algo}.zip"))

    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(str(model_dir / f"vecnormalize_{algo}.pkl"), venv)
    venv.training = False
    venv.norm_reward = False

    obs = venv.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = venv.step(action)
        done = bool(dones[0])

    underlying = venv.envs[0]
    dates = underlying.date_memory
    values = underlying.asset_memory
    if len(values) == len(dates) + 1:
        values = values[1:]
    return pd.DataFrame({"date": dates, "account_value": values})


def compute_mvo_baseline(train_df: pd.DataFrame, trade_df: pd.DataFrame,
                         initial_amount: float) -> pd.Series:
    """Markowitz max-Sharpe portfolio computed on train returns, held passively."""
    stock_dim = len(train_df.tic.unique())

    def to_price_matrix(df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["date", "tic"], ignore_index=True)[["date", "tic", "close"]]
        tics = df.iloc[0:stock_dim]["tic"].tolist()
        mvo = pd.DataFrame(columns=tics)
        for i in range(df.shape[0] // stock_dim):
            block = df.iloc[i * stock_dim:(i + 1) * stock_dim]
            mvo.loc[block["date"].iloc[0]] = block["close"].tolist()
        return mvo

    train_prices = to_price_matrix(train_df)
    trade_prices = to_price_matrix(trade_df)

    arr = np.asarray(train_prices, dtype=float)
    n_rows, n_cols = arr.shape
    returns = (arr[1:] - arr[:-1]) / arr[:-1] * 100
    mean_returns = returns.mean(axis=0)
    cov_returns = np.cov(returns, rowvar=False)

    ef = EfficientFrontier(mean_returns, cov_returns, weight_bounds=(0, 0.5))
    ef.max_sharpe()
    weights = ef.clean_weights()
    cash_per_stock = np.array([initial_amount * weights[i] for i in range(stock_dim)])

    last_train_prices = train_prices.tail(1).to_numpy(dtype=float)[0]
    shares = cash_per_stock / last_train_prices

    portfolio_value = trade_prices.to_numpy(dtype=float) @ shares
    return pd.Series(portfolio_value, index=trade_prices.index, name="MVO")


def compute_dji_baseline(config: dict, initial_amount: float) -> pd.Series:
    df_dji = YahooDownloader(
        start_date=config["data"]["trade_start_date"],
        end_date=config["data"]["trade_end_date"],
        ticker_list=["dji"],
    ).fetch_data()[["date", "close"]]
    first = df_dji["close"].iloc[0]
    scaled = df_dji["close"] / first * initial_amount
    return pd.Series(scaled.values, index=df_dji["date"].values, name="DJIA")


def compute_metrics(equity: pd.Series) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {"cum_return": np.nan, "sharpe": np.nan, "max_dd": np.nan}
    returns = equity.pct_change().dropna()
    cum_return = equity.iloc[-1] / equity.iloc[0] - 1
    sharpe = (returns.mean() / (returns.std() + 1e-12)) * np.sqrt(252)
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

    config = load_config(args.config)
    data_dir    = resolve_path(config, "data_dir")
    model_dir   = resolve_path(config, "model_dir")
    results_dir = resolve_path(config, "results_dir")

    normalize      = config["normalization"]["normalize_observations"]
    initial_amount = config["env"]["initial_amount"]

    trade_df = load_csv(data_dir / "trade_data.csv")
    train_df = load_csv(data_dir / "train_data.csv")

    print(f"Mode: {'NORMALIZED' if normalize else 'STANDARD'}")

    equity = {}
    for algo in enabled_models(config):
        print(f"\nBacktesting {algo.upper()}...")
        env = build_trade_env(config, trade_df)
        if normalize:
            df = predict_vecnormalize(algo, env, model_dir)
        else:
            df = predict_finrl(algo, env, model_dir)
        df = df.set_index(df.columns[0])
        equity[algo.upper()] = df[df.columns[0]]

    print("\nComputing MVO baseline...")
    equity["MVO"] = compute_mvo_baseline(train_df, trade_df, initial_amount)

    print("Computing DJIA baseline...")
    equity["DJIA"] = compute_dji_baseline(config, initial_amount)

    result = pd.concat(equity, axis=1).sort_index().ffill()

    csv_path = results_dir / "equity_curves.csv"
    result.to_csv(csv_path)
    print(f"\nSaved {csv_path}")

    plt.figure(figsize=(15, 6))
    result.plot(ax=plt.gca())
    plt.title("Equity Curves: RL Agents vs. MVO vs. DJIA")
    plt.ylabel("Portfolio Value ($)")
    plt.xlabel("Date")
    plt.legend(loc="best")
    plt.tight_layout()
    plot_path = results_dir / "equity_plot.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"Saved {plot_path}")

    print("\n" + "=" * 60)
    print(f"{'Strategy':<10} {'CumReturn':>12} {'Sharpe':>10} {'MaxDD':>10}")
    print("-" * 60)
    for name in result.columns:
        m = compute_metrics(result[name])
        print(f"{name:<10} {m['cum_return']*100:>11.2f}% "
              f"{m['sharpe']:>10.3f} {m['max_dd']*100:>9.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
