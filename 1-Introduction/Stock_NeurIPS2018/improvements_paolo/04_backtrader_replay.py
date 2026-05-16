"""Stage 4 — Backtrader replay of a trained RL agent (Approach B).

The trained SB3 model is called live inside `bt.Strategy.next()`. On every
bar the strategy reconstructs the same state vector that StockTradingEnv
produced during training, optionally re-applies the saved VecNormalize
statistics, calls model.predict(), and converts the action into per-ticker
buy/sell orders. Backtrader handles fills, commissions, slippage, and
analytics.

Usage:
    python 04_backtrader_replay.py --config backtrader_config.json

Requires:
    pip install backtrader
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

from utils import ALGO_REGISTRY, load_config, project_root


# ---------- config loading ----------

def load_configs(bt_config_path: str) -> tuple[dict, dict, Path]:
    """Load backtrader config + the referenced training config."""
    bt_cfg = load_config(bt_config_path)
    src_path = Path(bt_cfg["source_config"])
    if not src_path.is_absolute():
        src_path = project_root() / src_path
    src_cfg = load_config(src_path)
    return bt_cfg, src_cfg, src_path.parent


# ---------- VecNormalize ----------

def load_vecnormalize_stats(model_dir: Path, algo: str):
    stats_path = model_dir / f"vecnormalize_{algo}.pkl"
    if not stats_path.exists():
        return None
    with open(stats_path, "rb") as f:
        return pickle.load(f)


def apply_obs_normalization(state: np.ndarray, vn) -> np.ndarray:
    eps = getattr(vn, "epsilon", 1e-8)
    clip = getattr(vn, "clip_obs", 10.0)
    normed = (state - vn.obs_rms.mean) / np.sqrt(vn.obs_rms.var + eps)
    return np.clip(normed, -clip, clip).astype(np.float32)


# ---------- data feeds ----------

def build_data_feed_class(indicators: list[str]) -> type:
    """Dynamically build a PandasData subclass exposing indicator columns as lines."""
    lines = tuple(indicators)
    params = tuple((ind, ind) for ind in indicators)
    return type(
        "StockDataWithIndicators",
        (bt.feeds.PandasData,),
        {"lines": lines, "params": params},
    )


def split_long_format(trade_df: pd.DataFrame, indicators: list[str]) -> dict[str, pd.DataFrame]:
    """Pivot long-format trade CSV into one DataFrame per ticker, datetime-indexed."""
    cols_needed = ["open", "high", "low", "close", "volume"] + indicators
    out: dict[str, pd.DataFrame] = {}
    for tic in sorted(trade_df["tic"].unique()):
        sub = trade_df[trade_df["tic"] == tic].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        sub = sub.set_index("date").sort_index()
        for col in cols_needed:
            if col not in sub.columns:
                sub[col] = 0.0
        out[tic] = sub[cols_needed]
    return out


# ---------- strategy ----------

class RLStrategy(bt.Strategy):
    """Mirrors StockTradingEnv's state vector and calls the SB3 model each bar."""

    params = (
        ("model", None),
        ("indicators", None),
        ("vecnormalize", None),
        ("hmax", 100),
        ("deterministic", True),
        ("round_shares", True),
        ("min_order_size", 1),
        ("verbosity", "info"),
        ("log_every_trade", False),
    )

    def __init__(self):
        self.equity_curve: list[tuple] = []
        self.transactions_log: list[dict] = []
        self.tickers = [d._name for d in self.datas]
        if self.p.verbosity != "silent":
            print(f"RLStrategy: {len(self.tickers)} tickers, "
                  f"indicators={self.p.indicators}, hmax={self.p.hmax}, "
                  f"vecnormalize={'on' if self.p.vecnormalize else 'off'}")

    def _build_state(self) -> np.ndarray:
        """Match StockTradingEnv._initiate_state layout:
            [cash, price_1..N, shares_1..N, ind1_1..N, ind2_1..N, ...]
        Ticker order is the same alphabetic order used by FinRL's data_split.
        """
        cash = self.broker.getcash()
        prices = [d.close[0] for d in self.datas]
        holdings = [int(self.getposition(d).size) for d in self.datas]
        ind_flat: list[float] = []
        for ind in self.p.indicators:
            ind_flat.extend([getattr(d, ind)[0] for d in self.datas])
        return np.array([cash] + prices + holdings + ind_flat, dtype=np.float32)

    def next(self):
        state = self._build_state()
        if self.p.vecnormalize is not None:
            state = apply_obs_normalization(state, self.p.vecnormalize)

        action, _ = self.p.model.predict(state, deterministic=self.p.deterministic)
        if action.ndim > 1:
            action = action[0]

        scaled = action * self.p.hmax
        if self.p.round_shares:
            scaled = scaled.astype(int)

        for d, delta in zip(self.datas, scaled):
            if abs(delta) < self.p.min_order_size:
                continue
            if delta > 0:
                self.buy(data=d, size=int(delta))
                if self.p.log_every_trade:
                    print(f"BUY  {d._name:>6} x{int(delta):>4} @ {d.close[0]:>8.2f}")
            else:
                pos = int(self.getposition(d).size)
                qty = min(int(-delta), pos)
                if qty > 0:
                    self.sell(data=d, size=qty)
                    if self.p.log_every_trade:
                        print(f"SELL {d._name:>6} x{qty:>4} @ {d.close[0]:>8.2f}")

        self.equity_curve.append((self.datetime.date(0), self.broker.getvalue()))

    def notify_order(self, order):
        if order.status == order.Completed:
            self.transactions_log.append({
                "date":   self.datetime.date(0).isoformat(),
                "ticker": order.data._name,
                "side":   "BUY" if order.isbuy() else "SELL",
                "size":   int(order.executed.size),
                "price":  float(order.executed.price),
                "value":  float(order.executed.value),
                "comm":   float(order.executed.comm),
            })


# ---------- broker / analyzers ----------

def configure_broker(cerebro: bt.Cerebro, broker_cfg: dict) -> None:
    cerebro.broker.set_cash(broker_cfg["initial_cash"])

    comm = broker_cfg["commission"]
    if comm["value"] > 0:
        if comm["type"] == "percentage":
            cerebro.broker.setcommission(
                commission=comm["value"],
                percabs=True,
                stocklike=comm.get("stocklike", True),
            )
        elif comm["type"] == "fixed":
            cerebro.broker.setcommission(
                commission=comm["value"],
                commtype=bt.CommInfoBase.COMM_FIXED,
                stocklike=comm.get("stocklike", True),
            )

    slip = broker_cfg["slippage"]
    if slip["type"] != "none" and slip["value"] > 0:
        slip_kwargs = {
            "slip_open":  slip["apply_to"]["open"],
            "slip_limit": slip["apply_to"]["limit"],
            "slip_match": slip["apply_to"]["match"],
            "slip_out":   slip["apply_to"]["out"],
        }
        if slip["type"] == "percentage":
            cerebro.broker.set_slippage_perc(perc=slip["value"], **slip_kwargs)
        elif slip["type"] == "fixed":
            cerebro.broker.set_slippage_fixed(fixed=slip["value"], **slip_kwargs)

    cerebro.broker.set_checksubmit(broker_cfg.get("check_submit", True))
    cerebro.broker.set_shortcash(broker_cfg.get("allow_short", False))
    if broker_cfg.get("coc", False):
        cerebro.broker.set_coc(True)


def add_analyzers(cerebro: bt.Cerebro, analyzers_cfg: dict) -> None:
    if analyzers_cfg.get("sharpe", {}).get("enabled"):
        s = analyzers_cfg["sharpe"]
        cerebro.addanalyzer(
            bt.analyzers.SharpeRatio,
            _name="sharpe",
            riskfreerate=s.get("riskfreerate", 0.0),
            annualize=s.get("annualize", True),
            timeframe=getattr(bt.TimeFrame, s.get("timeframe", "Days")),
        )
    if analyzers_cfg.get("drawdown", {}).get("enabled"):
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    if analyzers_cfg.get("returns", {}).get("enabled"):
        cerebro.addanalyzer(
            bt.analyzers.Returns,
            _name="returns",
            tann=analyzers_cfg["returns"].get("tann", 252),
        )
    if analyzers_cfg.get("trade_analyzer", {}).get("enabled"):
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")
    if analyzers_cfg.get("annual_return", {}).get("enabled"):
        cerebro.addanalyzer(bt.analyzers.AnnualReturn, _name="annual_return")
    if analyzers_cfg.get("transactions", {}).get("enabled"):
        cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")
    if analyzers_cfg.get("position_value", {}).get("enabled"):
        cerebro.addanalyzer(bt.analyzers.PositionsValue, _name="position_value")
    if analyzers_cfg.get("pyfolio", {}).get("enabled"):
        cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")


# ---------- output helpers ----------

def _coerce_for_json(o):
    if isinstance(o, dict):
        return {str(k): _coerce_for_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce_for_json(x) for x in o]
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)


def export_outputs(strat: RLStrategy, cerebro: bt.Cerebro, bt_cfg: dict, out_dir: Path) -> None:
    # Equity curve
    eq_df = pd.DataFrame(strat.equity_curve, columns=["date", "equity"]).set_index("date")
    eq_path = out_dir / bt_cfg["output"]["equity_csv"]
    eq_df.to_csv(eq_path)
    print(f"Saved {eq_path}")

    # Trade log (per fill)
    if strat.transactions_log:
        tx_df = pd.DataFrame(strat.transactions_log)
        tx_path = out_dir / bt_cfg["output"]["trade_log_csv"]
        tx_df.to_csv(tx_path, index=False)
        print(f"Saved {tx_path}")

    # Analyzer summary JSON
    summary = {}
    for name, an in strat.analyzers.getitems():
        try:
            summary[name] = an.get_analysis()
        except Exception as e:
            summary[name] = {"error": str(e)}
    summary_path = out_dir / bt_cfg["output"]["summary_json"]
    with open(summary_path, "w") as f:
        json.dump(_coerce_for_json(summary), f, indent=2)
    print(f"Saved {summary_path}")

    # Transactions analyzer -> CSV
    if bt_cfg["analyzers"]["transactions"]["enabled"]:
        try:
            txn = strat.analyzers.transactions.get_analysis()
            rows = []
            for dt, items in txn.items():
                for entry in items:
                    rows.append({
                        "date":  str(dt),
                        "size":  entry[0],
                        "price": entry[1],
                        "sid":   entry[2],
                        "symbol": entry[3] if len(entry) > 3 else None,
                        "value": entry[4] if len(entry) > 4 else None,
                    })
            if rows:
                tdf = pd.DataFrame(rows)
                tpath = out_dir / bt_cfg["output"]["transactions_csv"]
                tdf.to_csv(tpath, index=False)
                print(f"Saved {tpath}")
        except Exception as e:
            print(f"Could not export transactions analyzer: {e}")

    # Plot (best-effort; backtrader plotting can be finicky on headless macOS)
    if bt_cfg["output"]["plot"]["enabled"]:
        try:
            figs = cerebro.plot(
                style=bt_cfg["output"]["plot"]["style"],
                volume=bt_cfg["output"]["plot"]["show_volume"],
                iplot=False,
            )
            if figs and figs[0]:
                fig_path = out_dir / bt_cfg["output"]["plot"]["filename"]
                figs[0][0].savefig(fig_path, dpi=120)
                print(f"Saved {fig_path}")
        except Exception as e:
            print(f"Plot failed (non-fatal): {e}")


def print_summary(strat: RLStrategy, initial: float, final: float) -> None:
    print("\n" + "=" * 60)
    print("BACKTRADER REPLAY SUMMARY")
    print("=" * 60)
    print(f"Start equity:        ${initial:>14,.2f}")
    print(f"End equity:          ${final:>14,.2f}")
    print(f"Total return:        {(final / initial - 1) * 100:>14.2f}%")

    a = strat.analyzers
    if hasattr(a, "sharpe"):
        sr = a.sharpe.get_analysis().get("sharperatio")
        if sr is not None:
            print(f"Sharpe (annualized): {sr:>14.3f}")
    if hasattr(a, "drawdown"):
        dd = a.drawdown.get_analysis().get("max", {}).get("drawdown")
        if dd is not None:
            print(f"Max drawdown:        {dd:>14.2f}%")
    if hasattr(a, "trade_analyzer"):
        ta = a.trade_analyzer.get_analysis()
        total = ta.get("total", {}).get("total", 0)
        won = ta.get("won", {}).get("total", 0)
        lost = ta.get("lost", {}).get("total", 0)
        print(f"Trades total/won/lost: {total} / {won} / {lost}")
    print("=" * 60)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backtrader_config.json")
    args = parser.parse_args()

    bt_cfg, src_cfg, src_dir = load_configs(args.config)

    model_dir = src_dir / src_cfg["paths"]["model_dir"]
    data_dir  = src_dir / src_cfg["paths"]["data_dir"]
    out_dir   = src_dir / bt_cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- load model + VecNormalize -----
    algo = bt_cfg["model"]["algorithm"]
    AlgoClass = ALGO_REGISTRY[algo]
    model_path = model_dir / f"agent_{algo}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found — run 02_train.py first.")
    print(f"Loading model: {model_path}")
    model = AlgoClass.load(str(model_path))

    vn_setting = bt_cfg["state_reconstruction"]["vecnormalize_load"]
    vecnormalize = None
    if vn_setting == "auto":
        if src_cfg["normalization"]["obs_mode"] != "off":
            vecnormalize = load_vecnormalize_stats(model_dir, algo)
            if vecnormalize is not None:
                print(f"  VecNormalize: loaded (obs count={int(vecnormalize.obs_rms.count):,})")
            else:
                print("  VecNormalize: requested but no stats file found.")
    elif vn_setting == "manual":
        manual_path = bt_cfg["state_reconstruction"]["vecnormalize_path"]
        if manual_path:
            with open(manual_path, "rb") as f:
                vecnormalize = pickle.load(f)

    # ----- load + filter trade CSV -----
    raw_csv = bt_cfg["data"].get("data_csv") or "data/trade_data.csv"
    trade_csv = Path(raw_csv)
    if not trade_csv.is_absolute():
        trade_csv = src_dir / trade_csv
    print(f"Loading trade data: {trade_csv}")
    trade_df = pd.read_csv(trade_csv)
    if "date" not in trade_df.columns:
        trade_df = pd.read_csv(trade_csv, index_col=0)

    if not bt_cfg["data"]["use_source_dates"]:
        sd = bt_cfg["data"]["trade_start_date"]
        ed = bt_cfg["data"]["trade_end_date"]
        if sd: trade_df = trade_df[trade_df["date"] >= sd]
        if ed: trade_df = trade_df[trade_df["date"] <  ed]

    # ----- resolve indicators -----
    if bt_cfg["state_reconstruction"]["use_source_indicators"]:
        indicators = src_cfg["data"]["indicators"]
    else:
        indicators = bt_cfg["state_reconstruction"]["indicators_override"]
        if not indicators:
            raise ValueError(
                "use_source_indicators=false but indicators_override is null."
            )

    # ----- build cerebro -----
    cerebro = bt.Cerebro(stdstats=False)
    configure_broker(cerebro, bt_cfg["broker"])

    FeedCls = build_data_feed_class(indicators)
    per_ticker = split_long_format(trade_df, indicators)
    for tic in sorted(per_ticker.keys()):
        cerebro.adddata(FeedCls(dataname=per_ticker[tic]), name=tic)
    print(f"Added {len(per_ticker)} data feeds.")

    hmax = bt_cfg["execution"].get("hmax_override") or src_cfg["env"]["hmax"]

    cerebro.addstrategy(
        RLStrategy,
        model=model,
        indicators=indicators,
        vecnormalize=vecnormalize,
        hmax=hmax,
        deterministic=bt_cfg["model"]["deterministic"],
        round_shares=bt_cfg["execution"]["round_shares"],
        min_order_size=bt_cfg["execution"]["min_order_size"],
        verbosity=bt_cfg["logging"]["verbosity"],
        log_every_trade=bt_cfg["logging"]["log_every_trade"],
    )

    add_analyzers(cerebro, bt_cfg["analyzers"])

    # ----- run -----
    print("\nRunning backtrader cerebro...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    export_outputs(strat, cerebro, bt_cfg, out_dir)
    print_summary(strat, initial_value, final_value)


if __name__ == "__main__":
    main()
