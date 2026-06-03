"""Stage 4 - Backtrader replay of pre-recorded daily portfolio weights.

This stage no longer runs the SB3 model inside the strategy. Instead it reads
the per-day post-softmax weight vector that stage 3 already produced
(results/weights_<algo>.csv) and replays it through backtrader as a sequence
of `order_target_percent` calls. The strategy holds no model and no state-
reconstruction logic; it is a pure execution simulator.

Semantics:
  * Same-day-close fills via `cerebro.broker.set_coc(True)`. The weight
    recorded for date d_t is the target the portfolio should be holding
    at the CLOSE of d_t. Backtrader's "cheat on close" matches market
    orders submitted in next() against the SAME bar's close price.
  * Cash buffer (default 5%): target weights from the CSV (which sum to
    1.0 after softmax) are scaled by (1 - cash_buffer) so the broker has
    headroom for slippage / fractional-share rounding without rejecting
    orders for insufficient cash. Configurable via execution.cash_buffer.
  * Sell-before-buy: orders are sorted by (target_weight - current_weight)
    ascending and submitted in that order, so sells (negative delta) hit
    the broker before buys (positive delta) and free up cash for the buys
    within the same bar. Configurable via execution.sell_first.

Usage:
    python 04_backtrader_replay.py --config backtrader_config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import backtrader as bt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import fetch_yahoo_with_retry, load_config, project_root


# ---------- config loading ----------

def load_configs(bt_config_path: str) -> tuple[dict, dict, Path]:
    bt_cfg = load_config(bt_config_path)
    src_path = Path(bt_cfg["source_config"])
    if not src_path.is_absolute():
        src_path = project_root() / src_path
    src_cfg = load_config(src_path)
    return bt_cfg, src_cfg, src_path.parent


# ---------- inputs ----------

def load_weights_csv(path: Path) -> pd.DataFrame:
    """Load per-day post-softmax weights, indexed by date."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run 03_backtest.py first to produce the "
            f"weights CSV that stage 4 replays."
        )
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def load_trade_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run 01_get_data.py first."
        )
    return pd.read_pickle(path)


def split_long_format(trade_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot the long-format trade DF into one OHLCV DataFrame per ticker.

    Indicators are no longer needed - the strategy reads weights from a CSV
    rather than reconstructing the env's observation matrix per bar.
    """
    cols_needed = ["open", "high", "low", "close", "volume"]
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

class WeightReplayStrategy(bt.Strategy):
    """Read target weights from a DataFrame and rebalance via order_target_percent.

    No model, no state reconstruction, no indicators - just a CSV lookup and
    one rebalance per bar.
    """

    params = (
        ("weights_df", None),
        ("cash_buffer", 0.05),
        ("sell_first", True),
        ("min_weight_delta", 1e-4),
        ("verbosity", "info"),
        ("log_every_trade", False),
        # Approach A: inference-only rebalance cadence. cadence='daily' is the
        # default and preserves the previous behaviour; 'weekly' / 'monthly'
        # only act on configured days, holding weights in between.
        ("cadence", "daily"),
        ("weekly_day_iso", 5),  # 5 = Friday (ISO weekday: MON=1 ... SUN=7)
        ("monthly_day", 1),     # day of month for monthly cadence (1..28)
    )

    def __init__(self):
        self.equity_curve: list[tuple] = []
        self.transactions_log: list[dict] = []
        self.tickers = [d._name for d in self.datas]
        missing = set(self.tickers) - set(self.p.weights_df.columns)
        if missing:
            raise RuntimeError(
                f"Tickers missing from weights CSV: {sorted(missing)}. "
                f"Make sure stage 3 was run with the same universe."
            )
        # Cadence-tracking state.
        self._cadence = str(self.p.cadence).lower()
        if self._cadence not in ("daily", "weekly", "monthly"):
            raise ValueError(
                f"Unknown rebalance.cadence={self._cadence!r}; "
                f"expected 'daily', 'weekly' or 'monthly'."
            )
        self._last_rebalance_date: pd.Timestamp | None = None
        self._n_rebalances = 0
        self._n_bars_total = 0
        if self.p.verbosity != "silent":
            cad_desc = (self._cadence if self._cadence == "daily"
                        else f"{self._cadence} (weekly_day_iso={self.p.weekly_day_iso}, "
                             f"monthly_day={self.p.monthly_day})")
            print(f"WeightReplayStrategy: {len(self.tickers)} tickers, "
                  f"cash_buffer={self.p.cash_buffer}, "
                  f"sell_first={self.p.sell_first}, "
                  f"weight rows={len(self.p.weights_df)}, "
                  f"cadence={cad_desc}")

    def _is_rebalance_day(self, current_date: pd.Timestamp) -> bool:
        """Decide whether `current_date` is a rebalance day per the cadence.

        - 'daily': always True (existing behaviour).
        - 'weekly': True on the configured ISO weekday, OR if 7+ days have
          passed since the last rebalance (catches holiday-rolled weeks
          where the configured weekday was not a trading day).
        - 'monthly': True only when the calendar month differs from the
          last rebalance AND today.day >= monthly_day (so monthly_day=15
          means 'first trading day on or after the 15th of each month';
          monthly_day=1 effectively means 'first trading day of each month').
        - On the first bar that has recorded weights, always True so the
          replay opens a position.
        """
        if self._cadence == "daily":
            return True
        if self._last_rebalance_date is None:
            return True
        if self._cadence == "weekly":
            if current_date.isoweekday() == int(self.p.weekly_day_iso):
                return True
            return (current_date - self._last_rebalance_date).days >= 7
        if self._cadence == "monthly":
            same_month = (
                current_date.year == self._last_rebalance_date.year
                and current_date.month == self._last_rebalance_date.month
            )
            if same_month:
                return False
            return current_date.day >= int(self.p.monthly_day)
        return False  # unreachable

    def next(self):
        current_date = pd.Timestamp(self.datetime.date(0))
        # Record equity at the start of the bar (post-fills from previous bar
        # with coc=True - so this is end-of-previous-day equity).
        self.equity_curve.append((current_date.date(), self.broker.getvalue()))
        self._n_bars_total += 1

        if current_date not in self.p.weights_df.index:
            return  # no recorded weights for this date

        # Cadence gate (Approach A): on non-rebalance days, hold the existing
        # position - the broker drifts naturally with prices. The agent's
        # daily-recorded target weights are simply not acted upon.
        if not self._is_rebalance_day(current_date):
            return
        self._last_rebalance_date = current_date
        self._n_rebalances += 1

        raw = self.p.weights_df.loc[current_date]
        invested = 1.0 - float(self.p.cash_buffer)
        target_weights = {t: float(raw.get(t, 0.0)) * invested for t in self.tickers}

        portfolio_value = self.broker.getvalue()
        orders = []
        for d in self.datas:
            tic = d._name
            cur_value  = self.getposition(d).size * d.close[0]
            cur_weight = (cur_value / portfolio_value) if portfolio_value > 0 else 0.0
            tgt = target_weights.get(tic, 0.0)
            delta = tgt - cur_weight
            # B-only no-trade band: skip iff the ACTUAL required trade is smaller
            # than min_weight_delta. delta = target - current_DRIFTED_weight is
            # exactly the trade size order_target_percent would place, so this is
            # the economically correct gate (drift-aware by construction). The
            # previous AND clause also tested |target - previous_target|, which
            # was redundant and could skip a real drift-driven rebalance.
            if abs(delta) < self.p.min_weight_delta:
                continue
            orders.append((delta, d, tgt))

        if self.p.sell_first:
            orders.sort(key=lambda x: x[0])  # most negative (sells) first

        for delta, d, tgt in orders:
            self.order_target_percent(data=d, target=tgt)
            if self.p.log_every_trade:
                kind = "SELL" if delta < 0 else "BUY "
                print(f"{kind} {d._name:>6} delta={delta*100:+5.2f}%  "
                      f"target={tgt*100:5.2f}%  @ {d.close[0]:>8.2f}")

    def stop(self):
        # Capture the final bar's equity after fills.
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
    # Cheat-on-close: market orders submitted in next() match the SAME bar's close.
    if broker_cfg.get("coc", True):
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


def export_outputs(strat: WeightReplayStrategy, bt_cfg: dict,
                   src_cfg: dict, out_dir: Path) -> None:
    eq_df = pd.DataFrame(strat.equity_curve, columns=["date", "equity"])
    eq_df["date"] = pd.to_datetime(eq_df["date"])
    eq_df = eq_df.drop_duplicates(subset=["date"], keep="last").set_index("date")
    eq_path = out_dir / bt_cfg["output"]["equity_csv"]
    eq_df.to_csv(eq_path)
    print(f"Saved {eq_path}")

    if strat.transactions_log:
        tx_df = pd.DataFrame(strat.transactions_log)
        tx_path = out_dir / bt_cfg["output"]["trade_log_csv"]
        tx_df.to_csv(tx_path, index=False)
        print(f"Saved {tx_path}")

        # Focused per-day trade log: date, ticker, side, quantity, price.
        # Sorted chronologically with deterministic intra-day ordering
        # (sells first, then alphabetical ticker) so diffing two runs is easy.
        daily_name = bt_cfg["output"].get("daily_trades_csv")
        if daily_name:
            daily_df = tx_df[["date", "ticker", "side", "size", "price"]].copy()
            daily_df = daily_df.rename(columns={"size": "quantity"})
            daily_df["_side_order"] = (daily_df["side"] == "BUY").astype(int)
            daily_df = daily_df.sort_values(
                by=["date", "_side_order", "ticker"],
                kind="mergesort",
            ).drop(columns=["_side_order"]).reset_index(drop=True)
            daily_path = out_dir / daily_name
            daily_df.to_csv(daily_path, index=False)
            print(f"Saved {daily_path}")

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

    if bt_cfg["analyzers"].get("transactions", {}).get("enabled"):
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
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(eq_df.index, eq_df["equity"],
                    label="Replay (recorded weights)", linewidth=1.5)
            if bt_cfg["output"]["plot"].get("overlay_djia", True):
                try:
                    # Overlay the benchmark defined in the MAIN config (equal_weight
                    # or ticker), using the benchmark_return series stage 1 stored -
                    # so the overlay matches the reward / features / stage-3 baseline.
                    from utils import benchmark_label, load_benchmark_returns
                    label = benchmark_label(src_cfg)
                    bret  = load_benchmark_returns(src_cfg)
                    bret  = bret.reindex(eq_df.index).fillna(0.0)
                    curve = bt_cfg["broker"]["initial_cash"] * (1.0 + bret).cumprod()
                    ax.plot(curve.index, curve.values, label=label,
                            linewidth=1.2, alpha=0.85, linestyle="--")
                    # Cash-inclusive EqualWeight overlay (mirrors stage-3
                    # EqualWeight_w_Cash). Since CASH earns 0%, a daily-rebalanced
                    # 1/(M+1) portfolio of M risky + cash has daily return
                    # r' = M/(M+1) * r_eq_no_cash exactly. Controlled by
                    # benchmark.show_cash_inclusive (default true); applied only
                    # when the benchmark is equal_weight AND cash is enabled.
                    bench_cfg_l = src_cfg.get("benchmark", {}) or {}
                    cash_cfg_l  = src_cfg.get("cash", {}) or {}
                    if (bench_cfg_l.get("show_cash_inclusive", True)
                            and cash_cfg_l.get("enabled", False)
                            and bench_cfg_l.get("type", "equal_weight") == "equal_weight"):
                        M = len(src_cfg.get("data", {}).get("ticker_list", []) or [])
                        if M > 0:
                            bret_wc  = (M / (M + 1)) * bret
                            curve_wc = bt_cfg["broker"]["initial_cash"] * (1.0 + bret_wc).cumprod()
                            ax.plot(curve_wc.index, curve_wc.values,
                                    label=f"{label} w/ Cash",
                                    linewidth=1.0, alpha=0.75, linestyle=":")
                    # Structural-prior overlay: stage 3 wrote the static daily-
                    # rebalanced Prior_{type} portfolio into results/equity_curves.csv;
                    # read that column and overlay it (dash-dot). The right
                    # comparator for a residual-policy PPO. Controlled by
                    # benchmark.show_prior_baseline (default true); only fires
                    # when policy_prior is enabled and the stage-3 column exists.
                    pp_cfg = src_cfg.get("policy_prior", {}) or {}
                    if (bench_cfg_l.get("show_prior_baseline", True)
                            and pp_cfg.get("enabled", False)
                            and pp_cfg.get("type", "none") != "none"):
                        ptype     = str(pp_cfg.get("type"))
                        prior_col = f"Prior_{ptype}"
                        from utils import resolve_path
                        s3_csv = resolve_path(src_cfg, "results_dir") / "equity_curves.csv"
                        if s3_csv.exists():
                            df_s3 = pd.read_csv(s3_csv)
                            dc    = df_s3.columns[0]
                            df_s3[dc] = pd.to_datetime(df_s3[dc])
                            df_s3 = df_s3.set_index(dc).sort_index()
                            if prior_col in df_s3.columns:
                                prior_eq = df_s3[prior_col].astype(float)
                                # Rescale to start at backtrader's initial cash so
                                # the overlay is comparable with the replay equity.
                                first = float(prior_eq.iloc[0])
                                if first > 0:
                                    scaled = prior_eq * (bt_cfg["broker"]["initial_cash"] / first)
                                    scaled = scaled.reindex(eq_df.index).ffill()
                                    ax.plot(scaled.index, scaled.values,
                                            label=f"Prior ({ptype}, static)",
                                            linewidth=1.0, alpha=0.85, linestyle="-.")
                            else:
                                print(f"NOTE: stage-3 equity_curves.csv has no "
                                      f"{prior_col!r} column; skipping prior overlay.")
                        else:
                            print(f"NOTE: {s3_csv} not found; skipping prior overlay "
                                  f"(re-run 03_backtest.py to produce it).")
                except Exception as e:
                    print(f"Benchmark overlay failed (non-fatal): {e}")
            ax.set_title("Portfolio Equity - Backtrader Replay (recorded weights)")
            ax.set_ylabel("Portfolio Value ($)")
            ax.set_xlabel("Date")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")
            fig.tight_layout()
            fig_path = out_dir / bt_cfg["output"]["plot"]["filename"]
            fig.savefig(fig_path, dpi=120)
            plt.close(fig)
            print(f"Saved {fig_path}")
        except Exception as e:
            print(f"Plot failed (non-fatal): {e}")


def print_summary(strat: WeightReplayStrategy, initial: float, final: float) -> None:
    print("\n" + "=" * 60)
    print("BACKTRADER REPLAY SUMMARY (recorded weights)")
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
    # Approach A cadence summary: tells you exactly how many bars the
    # strategy actually attempted a rebalance on, vs. how many bars
    # received daily target weights from the agent.
    cadence  = getattr(strat, "_cadence", "daily")
    n_rebal  = getattr(strat, "_n_rebalances", None)
    n_bars   = getattr(strat, "_n_bars_total", None)
    if n_rebal is not None and n_bars:
        pct = 100.0 * n_rebal / max(n_bars, 1)
        print(f"Rebalance cadence:   {cadence:>14}  "
              f"({n_rebal} rebalance bars of {n_bars} total = {pct:.1f}%)")
    print("=" * 60)


# ---------- main ----------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backtrader_config.json")
    args = parser.parse_args()

    bt_cfg, src_cfg, src_dir = load_configs(args.config)

    out_dir = src_dir / bt_cfg["output"]["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- weights CSV -----
    weights_csv = Path(bt_cfg["inputs"]["weights_csv"])
    if not weights_csv.is_absolute():
        weights_csv = src_dir / weights_csv
    print(f"Loading weights: {weights_csv}")
    weights_df = load_weights_csv(weights_csv)
    print(f"  {len(weights_df)} rows  "
          f"{weights_df.index.min().date()} -> {weights_df.index.max().date()}  "
          f"{len(weights_df.columns)} tickers")

    # ----- trade pickle (OHLCV per ticker) -----
    raw_pkl = bt_cfg["inputs"].get("trade_pickle") or "data/full_data.pkl"
    trade_pkl = Path(raw_pkl)
    if not trade_pkl.is_absolute():
        trade_pkl = src_dir / trade_pkl
    print(f"Loading trade data: {trade_pkl}")
    trade_df = load_trade_df(trade_pkl)

    # Clip the trade frame to the weights CSV date range so backtrader iterates
    # over exactly the period that has recorded weights. Avoids the pitfall of
    # using data/trade_data.pkl (16 months) with a walk-forward weights CSV
    # that spans 7 years - or vice versa.
    if bt_cfg["inputs"].get("auto_slice", True):
        w_start = weights_df.index.min().strftime("%Y-%m-%d")
        w_end   = (weights_df.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        before = len(trade_df)
        trade_df = trade_df[(trade_df["date"] >= w_start) & (trade_df["date"] < w_end)]
        print(f"  auto_slice: {w_start} -> {w_end}  "
              f"({before} -> {len(trade_df)} rows)")
    else:
        sd = bt_cfg["inputs"].get("trade_start_date")
        ed = bt_cfg["inputs"].get("trade_end_date")
        if sd: trade_df = trade_df[trade_df["date"] >= sd]
        if ed: trade_df = trade_df[trade_df["date"] <  ed]

    if len(trade_df) == 0:
        raise RuntimeError(
            f"No trade rows survive after slicing. weights_csv covers "
            f"{weights_df.index.min().date()} -> {weights_df.index.max().date()} "
            f"but trade_pickle {trade_pkl.name} has no data in that range. "
            f"Re-run 01_get_data.py to produce data/full_data.pkl, or set "
            f"inputs.trade_pickle to a file that covers the right window."
        )

    # ----- synthetic CASH handling -----
    # When the env trades an explicit CASH asset, it appears as a weight column
    # AND as a ticker in the trade frame. Backtrader must NOT receive a CASH
    # data feed: the broker holds the un-invested fraction as actual cash, so the
    # real-ticker order_target_percent calls (summing to 1 - w_cash) leave w_cash
    # in cash automatically. Drop the CASH ticker/column here, and require
    # cash_buffer == 0 to avoid double-counting cash.
    cash_cfg = src_cfg.get("cash", {}) or {}
    if cash_cfg.get("enabled", False):
        cash_ticker = str(cash_cfg.get("ticker", "CASH"))
        cb = float(bt_cfg.get("execution", {}).get("cash_buffer", 0.0))
        if cb != 0.0:
            raise ValueError(
                f"cash.enabled=true (env trades an explicit CASH asset) but "
                f"backtrader execution.cash_buffer={cb} != 0. This double-counts "
                f"cash. Set execution.cash_buffer to 0.0 in {args.config}."
            )
        before = len(trade_df)
        trade_df = trade_df[trade_df["tic"] != cash_ticker]
        print(f"  cash asset '{cash_ticker}' excluded from backtrader feeds "
              f"({before} -> {len(trade_df)} rows); broker holds the cash weight.")

    # ----- cerebro -----
    cerebro = bt.Cerebro(stdstats=False)
    configure_broker(cerebro, bt_cfg["broker"])

    per_ticker = split_long_format(trade_df)
    for tic in sorted(per_ticker.keys()):
        cerebro.adddata(bt.feeds.PandasData(dataname=per_ticker[tic]), name=tic)
    print(f"Added {len(per_ticker)} data feeds.")

    # Approach A: inference-only rebalance cadence (see backtrader_config.json
    # 'rebalance' block; default 'daily' preserves the existing behaviour).
    rb_cfg          = bt_cfg.get("rebalance", {}) or {}
    cadence         = str(rb_cfg.get("cadence", "daily")).lower()
    weekly_day_name = str(rb_cfg.get("weekly_day", "FRI")).upper()
    iso_map         = {"MON": 1, "TUE": 2, "WED": 3, "THU": 4,
                       "FRI": 5, "SAT": 6, "SUN": 7}
    weekly_day_iso  = int(iso_map.get(weekly_day_name, 5))
    monthly_day     = max(1, min(28, int(rb_cfg.get("monthly_day", 1))))

    cerebro.addstrategy(
        WeightReplayStrategy,
        weights_df=weights_df,
        cash_buffer=bt_cfg["execution"]["cash_buffer"],
        sell_first=bt_cfg["execution"].get("sell_first", True),
        min_weight_delta=bt_cfg["execution"].get("min_weight_delta", 1e-4),
        verbosity=bt_cfg["logging"]["verbosity"],
        log_every_trade=bt_cfg["logging"]["log_every_trade"],
        cadence=cadence,
        weekly_day_iso=weekly_day_iso,
        monthly_day=monthly_day,
    )

    add_analyzers(cerebro, bt_cfg["analyzers"])

    print("\nRunning backtrader cerebro...")
    initial_value = cerebro.broker.getvalue()
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()

    export_outputs(strat, bt_cfg, src_cfg, out_dir)
    print_summary(strat, initial_value, final_value)


if __name__ == "__main__":
    main()
