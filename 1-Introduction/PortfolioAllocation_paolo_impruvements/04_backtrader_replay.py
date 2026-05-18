"""Stage 4 - Backtrader replay of a trained portfolio-allocation agent.

The trained SB3 model is called live inside `bt.Strategy.next()`. On every bar
the strategy reconstructs the same (N + K, N) state matrix that StockPortfolioEnv
produced during training (covariance block from the pickled trade DataFrame plus
the K technical-indicator rows from the data feeds), feeds it to model.predict(),
softmaxes the raw action into target weights, and rebalances each ticker via
order_target_percent. Backtrader handles fills, commissions, slippage, and
analytics.

Usage:
    python 04_backtrader_replay.py --config backtrader_config.json

Requires:
    pip install backtrader
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

from utils import ALGO_REGISTRY, load_config, project_root


# ---------- config loading ----------

def load_configs(bt_config_path: str) -> tuple[dict, dict, Path]:
    bt_cfg = load_config(bt_config_path)
    src_path = Path(bt_cfg["source_config"])
    if not src_path.is_absolute():
        src_path = project_root() / src_path
    src_cfg = load_config(src_path)
    return bt_cfg, src_cfg, src_path.parent


# ---------- data feeds ----------

def build_data_feed_class(indicators: list[str]) -> type:
    """PandasData subclass that exposes indicator columns as lines."""
    lines = tuple(indicators)
    params = tuple((ind, ind) for ind in indicators)
    return type(
        "StockDataWithIndicators",
        (bt.feeds.PandasData,),
        {"lines": lines, "params": params},
    )


def split_long_format(trade_df: pd.DataFrame,
                      indicators: list[str]) -> dict[str, pd.DataFrame]:
    """Pivot the long-format trade DF into one DataFrame per ticker."""
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


def build_cov_lookup(trade_df: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
    """Map each trade date -> covariance matrix (read from cov_list column)."""
    out: dict[pd.Timestamp, np.ndarray] = {}
    by_date = trade_df.drop_duplicates(subset=["date"])
    for _, row in by_date.iterrows():
        out[pd.Timestamp(row["date"])] = np.asarray(row["cov_list"])
    return out


# ---------- strategy ----------

class RLPortfolioStrategy(bt.Strategy):
    """Mirrors StockPortfolioEnv's state matrix and calls the SB3 model per bar."""

    params = (
        ("model", None),
        ("indicators", None),
        ("cov_lookup", None),
        ("ticker_order", None),
        ("deterministic", True),
        ("min_weight_delta", 1e-4),
        ("verbosity", "info"),
        ("log_every_trade", False),
    )

    def __init__(self):
        self.equity_curve: list[tuple] = []
        self.transactions_log: list[dict] = []
        self.tickers = [d._name for d in self.datas]
        if self.p.ticker_order is not None and \
           tuple(self.tickers) != tuple(self.p.ticker_order):
            raise RuntimeError(
                "Data feed order does not match ticker order used at training. "
                f"feeds={self.tickers}  expected={list(self.p.ticker_order)}"
            )
        self._last_weights = np.zeros(len(self.tickers), dtype=np.float64)
        if self.p.verbosity != "silent":
            print(f"RLPortfolioStrategy: {len(self.tickers)} tickers, "
                  f"indicators={self.p.indicators}, "
                  f"deterministic={self.p.deterministic}")

    def _current_cov(self) -> np.ndarray:
        ts = pd.Timestamp(self.datetime.date(0))
        cov = self.p.cov_lookup.get(ts)
        if cov is None:
            # Tolerant lookup: latest date <= current bar.
            keys = [k for k in self.p.cov_lookup.keys() if k <= ts]
            if not keys:
                raise RuntimeError(f"No covariance entry available for {ts.date()}.")
            cov = self.p.cov_lookup[max(keys)]
        return cov

    def _build_state(self) -> np.ndarray:
        cov = self._current_cov()
        ind_rows = []
        for ind in self.p.indicators:
            ind_rows.append([getattr(d, ind)[0] for d in self.datas])
        state = np.append(np.array(cov, dtype=np.float32),
                          np.array(ind_rows, dtype=np.float32), axis=0)
        return state

    def next(self):
        state = self._build_state()
        action, _ = self.p.model.predict(state, deterministic=self.p.deterministic)
        if action.ndim > 1:
            action = action[0]

        a = np.asarray(action, dtype=np.float64)
        e = np.exp(a - a.max())
        weights = e / e.sum()

        for i, (d, w) in enumerate(zip(self.datas, weights)):
            if abs(w - self._last_weights[i]) < self.p.min_weight_delta:
                continue
            self.order_target_percent(data=d, target=float(w))
            if self.p.log_every_trade:
                print(f"REBAL {d._name:>6} -> {w*100:>5.2f}% @ {d.close[0]:>8.2f}")
        self._last_weights = weights

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


def export_outputs(strat: RLPortfolioStrategy, cerebro: bt.Cerebro,
                   bt_cfg: dict, out_dir: Path) -> None:
    eq_df = pd.DataFrame(strat.equity_curve, columns=["date", "equity"]).set_index("date")
    eq_path = out_dir / bt_cfg["output"]["equity_csv"]
    eq_df.to_csv(eq_path)
    print(f"Saved {eq_path}")

    if strat.transactions_log:
        tx_df = pd.DataFrame(strat.transactions_log)
        tx_path = out_dir / bt_cfg["output"]["trade_log_csv"]
        tx_df.to_csv(tx_path, index=False)
        print(f"Saved {tx_path}")

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


def print_summary(strat: RLPortfolioStrategy, initial: float, final: float) -> None:
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

    # ----- model -----
    algo = bt_cfg["model"]["algorithm"]
    AlgoClass = ALGO_REGISTRY[algo]
    model_path = model_dir / f"agent_{algo}.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found - run 02_train.py first.")
    print(f"Loading model: {model_path}")
    model = AlgoClass.load(str(model_path))

    # ----- trade pickle -----
    raw_pkl = bt_cfg["data"].get("data_pickle") or "data/trade_data.pkl"
    trade_pkl = Path(raw_pkl)
    if not trade_pkl.is_absolute():
        trade_pkl = src_dir / trade_pkl
    print(f"Loading trade data: {trade_pkl}")
    trade_df = pd.read_pickle(trade_pkl)

    if not bt_cfg["data"]["use_source_dates"]:
        sd = bt_cfg["data"]["trade_start_date"]
        ed = bt_cfg["data"]["trade_end_date"]
        if sd: trade_df = trade_df[trade_df["date"] >= sd]
        if ed: trade_df = trade_df[trade_df["date"] <  ed]

    # ----- indicators -----
    if bt_cfg["state_reconstruction"]["use_source_indicators"]:
        indicators = src_cfg["data"]["indicators"]
    else:
        indicators = bt_cfg["state_reconstruction"]["indicators_override"]
        if not indicators:
            raise ValueError(
                "use_source_indicators=false but indicators_override is null."
            )

    # ----- cov lookup -----
    cov_lookup = build_cov_lookup(trade_df)
    print(f"Built covariance lookup with {len(cov_lookup)} dates.")

    # ----- cerebro -----
    cerebro = bt.Cerebro(stdstats=False)
    configure_broker(cerebro, bt_cfg["broker"])

    FeedCls    = build_data_feed_class(indicators)
    per_ticker = split_long_format(trade_df, indicators)
    ticker_order = sorted(per_ticker.keys())
    for tic in ticker_order:
        cerebro.adddata(FeedCls(dataname=per_ticker[tic]), name=tic)
    print(f"Added {len(per_ticker)} data feeds.")

    cerebro.addstrategy(
        RLPortfolioStrategy,
        model=model,
        indicators=indicators,
        cov_lookup=cov_lookup,
        ticker_order=ticker_order,
        deterministic=bt_cfg["model"]["deterministic"],
        min_weight_delta=bt_cfg["execution"]["min_weight_delta"],
        verbosity=bt_cfg["logging"]["verbosity"],
        log_every_trade=bt_cfg["logging"]["log_every_trade"],
    )

    add_analyzers(cerebro, bt_cfg["analyzers"])

    print("\nRunning backtrader cerebro...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    export_outputs(strat, cerebro, bt_cfg, out_dir)
    print_summary(strat, initial_value, final_value)


if __name__ == "__main__":
    main()
