"""Custom feature engineering for the portfolio-allocation pipeline.

This module is the single place to define custom per-asset and global features
that go beyond FinRL's stockstats indicators. It is designed to be extended:
to add a new feature, write a builder function, decorate it with @register,
and reference its output column name(s) in config.custom_features.

Two feature kinds:
  * "per_asset" - a column with a DIFFERENT value per ticker per date. Becomes
    one row of the env state matrix (one value per ticker).
  * "global"    - a column with the SAME value for every ticker on a given date
    (broadcast). Also becomes one state row, but flat across the asset axis.
    This is exactly how the reference paper injects market-wide features.

Contract for builder functions:
  builder(df, params) -> dict[str, pd.Series]
    df:     long-format frame, columns at least
            [date, tic, open, high, low, close, volume], plus any columns added
            by earlier stages (benchmark_return, vix_level). Index is arbitrary;
            do NOT rely on it. Sort inside the builder if order matters.
    params: the dict from config.custom_features.params (may be empty).
    returns: mapping {column_name: Series aligned to df.index}.

The orchestrator add_custom_features() applies the requested builders and
returns (df_with_columns, ordered_list_of_added_column_names).

Inputs the builders may depend on, produced by stage 1 BEFORE this module runs:
  * benchmark_return : daily return of the configured benchmark, broadcast to
    every row of a given date. Used by beta and market-proxy features.
  * vix_level        : VIX close level, broadcast per date. Used by VIX features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# name -> {"func": builder, "kind": "per_asset"|"global", "produces": [colnames]}
FEATURE_REGISTRY: dict[str, dict] = {}


def register(name: str, kind: str, produces: list[str] | None = None):
    """Decorator registering a feature builder.

    name:     the key used in config.custom_features.per_asset / .global lists.
    kind:     "per_asset" or "global".
    produces: the column names the builder adds. Defaults to [name].
    """
    if kind not in ("per_asset", "global"):
        raise ValueError(f"kind must be 'per_asset' or 'global'; got {kind!r}")

    def deco(func):
        FEATURE_REGISTRY[name] = {
            "func":     func,
            "kind":     kind,
            "produces": produces or [name],
        }
        return func

    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["tic", "date"]).copy()


def _per_tic_logret(df: pd.DataFrame) -> pd.Series:
    """Daily log return per ticker, aligned to df.index (NaN on each tic's first day)."""
    d = _sorted(df)
    lr = np.log(d["close"] / d.groupby("tic")["close"].shift(1))
    return lr.reindex(df.index)


# ---------------------------------------------------------------------------
# Per-asset feature builders
# ---------------------------------------------------------------------------

@register("mom", "per_asset", produces=["mom_1", "mom_5", "mom_20", "mom_60"])
def _momentum(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Momentum log-returns over multiple horizons (paper: 1,5,20,60 days)."""
    horizons = params.get("mom_horizons", [1, 5, 20, 60])
    d = _sorted(df)
    out: dict[str, pd.Series] = {}
    g = d.groupby("tic")["close"]
    for h in horizons:
        col = f"mom_{h}"
        s = np.log(d["close"] / g.shift(h))
        out[col] = s.reindex(df.index)
    return out


@register("vol", "per_asset", produces=["std_5", "std_20"])
def _volatility(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Rolling std of daily log returns (paper: 5, 20 days)."""
    windows = params.get("vol_windows", [5, 20])
    d = _sorted(df)
    lr = np.log(d["close"] / d.groupby("tic")["close"].shift(1))
    d = d.assign(_lr=lr)
    out: dict[str, pd.Series] = {}
    for w in windows:
        col = f"std_{w}"
        s = d.groupby("tic")["_lr"].transform(lambda x: x.rolling(w, min_periods=w).std())
        out[col] = s.reindex(df.index)
    return out


@register("bb_pctb", "per_asset")
def _bollinger_pctb(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Bollinger %B = (close - lower) / (upper - lower), bands = SMA20 +/- 2*std20."""
    w = int(params.get("bb_window", 20))
    k = float(params.get("bb_k", 2.0))
    d = _sorted(df)
    ma = d.groupby("tic")["close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
    sd = d.groupby("tic")["close"].transform(lambda x: x.rolling(w, min_periods=w).std())
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower).replace(0.0, np.nan)
    pctb = (d["close"] - lower) / width
    return {"bb_pctb": pctb.reindex(df.index)}


@register("dist_high_20", "per_asset")
def _dist_from_high(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Distance from the rolling high: close / max(high, w) - 1 (<= 0)."""
    w = int(params.get("dist_high_window", 20))
    d = _sorted(df)
    roll_high = d.groupby("tic")["high"].transform(lambda x: x.rolling(w, min_periods=w).max())
    s = d["close"] / roll_high - 1.0
    return {"dist_high_20": s.reindex(df.index)}


@register("meanrev_20", "per_asset")
def _mean_reversion(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Deviation from the rolling MA: close / SMA(close, w) - 1."""
    w = int(params.get("meanrev_window", 20))
    d = _sorted(df)
    ma = d.groupby("tic")["close"].transform(lambda x: x.rolling(w, min_periods=w).mean())
    s = d["close"] / ma - 1.0
    return {"meanrev_20": s.reindex(df.index)}


@register("beta_60", "per_asset")
def _rolling_beta(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Rolling beta of each asset's return vs the benchmark return over w days.

    Requires a 'benchmark_return' column (added by stage 1). beta =
    cov(asset_ret, bench_ret) / var(bench_ret) over the trailing window.
    """
    w = int(params.get("beta_window", 60))
    if "benchmark_return" not in df.columns:
        raise KeyError("beta_60 requires a 'benchmark_return' column. "
                       "Stage 1 must add it before computing custom features.")
    d = _sorted(df)
    asset_ret = d["close"] / d.groupby("tic")["close"].shift(1) - 1.0
    d = d.assign(_ar=asset_ret, _br=d["benchmark_return"])

    def _beta(group: pd.DataFrame) -> pd.Series:
        cov = group["_ar"].rolling(w, min_periods=w).cov(group["_br"])
        var = group["_br"].rolling(w, min_periods=w).var()
        return cov / var.replace(0.0, np.nan)

    s = d.groupby("tic", group_keys=False).apply(_beta)
    return {"beta_60": s.reindex(df.index)}


@register("xsec_mom_rank", "per_asset")
def _cross_sectional_mom_rank(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Cross-sectional percentile rank (0..1) of w-day momentum on each date.

    Scale-free relative-strength signal: 1.0 = strongest momentum among the
    universe that day, 0.0 = weakest. Robust to absolute return magnitude and
    a strong complement to the raw momentum features. (Suggested, not in paper.)
    """
    w = int(params.get("xsec_mom_window", 60))
    d = _sorted(df)
    mom = np.log(d["close"] / d.groupby("tic")["close"].shift(w))
    d = d.assign(_mom=mom)
    rank = d.groupby("date")["_mom"].rank(pct=True)
    return {"xsec_mom_rank": rank.reindex(df.index)}


@register("downside_semidev_20", "per_asset")
def _downside_semidev(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Rolling std of NEGATIVE daily returns only (downside semi-deviation).

    Captures return asymmetry the symmetric std misses. (Suggested.)
    """
    w = int(params.get("semidev_window", 20))
    d = _sorted(df)
    ret = d["close"] / d.groupby("tic")["close"].shift(1) - 1.0
    downside = ret.clip(upper=0.0)
    d = d.assign(_dn=downside)
    s = d.groupby("tic")["_dn"].transform(lambda x: x.rolling(w, min_periods=w).std())
    return {"downside_semidev_20": s.reindex(df.index)}


@register("drawdown_60", "per_asset")
def _drawdown(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Drawdown from the trailing close peak: close / max(close, w) - 1 (<= 0).

    Longer-horizon stress gauge on close (vs dist_high_20 on intraday high).
    (Suggested.)
    """
    w = int(params.get("drawdown_window", 60))
    d = _sorted(df)
    peak = d.groupby("tic")["close"].transform(lambda x: x.rolling(w, min_periods=w).max())
    s = d["close"] / peak - 1.0
    return {"drawdown_60": s.reindex(df.index)}


# ---------------------------------------------------------------------------
# Global feature builders (same value across tickers on a given date)
# ---------------------------------------------------------------------------

@register("vix", "global", produces=["vix", "vix_chg5"])
def _vix(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """VIX level and its 5-day change. Requires a 'vix_level' column (stage 1)."""
    if "vix_level" not in df.columns:
        raise KeyError("vix features require a 'vix_level' column. Stage 1 must "
                       "download ^VIX and add it before computing custom features.")
    chg = int(params.get("vix_change_days", 5))
    # vix_level is already broadcast per date; build a per-date series, diff it, re-broadcast.
    per_date = df.drop_duplicates("date").set_index("date")["vix_level"].sort_index()
    chg_series = per_date.diff(chg)
    vix_col = df["vix_level"]
    vix_chg = df["date"].map(chg_series)
    return {"vix": vix_col.reindex(df.index), "vix_chg5": vix_chg.reindex(df.index)}


@register("xsec_avg_ret", "global", produces=["xsec_avg_ret", "xsec_avg_ret_vol5"])
def _cross_sectional_avg_return(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Cross-sectional average daily return + its 5-day rolling vol (market pulse)."""
    vol_w = int(params.get("xsec_vol_window", 5))
    d = _sorted(df)
    ret = d["close"] / d.groupby("tic")["close"].shift(1) - 1.0
    d = d.assign(_ret=ret)
    per_date = d.groupby("date")["_ret"].mean().sort_index()
    per_date_vol = per_date.rolling(vol_w, min_periods=vol_w).std()
    avg = df["date"].map(per_date)
    avg_vol = df["date"].map(per_date_vol)
    return {"xsec_avg_ret": avg.reindex(df.index),
            "xsec_avg_ret_vol5": avg_vol.reindex(df.index)}


@register("breadth", "global")
def _breadth(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Market breadth = fraction of tickers with positive w-day return on each date."""
    w = int(params.get("breadth_window", 20))
    d = _sorted(df)
    mom = d["close"] / d.groupby("tic")["close"].shift(w) - 1.0
    d = d.assign(_pos=(mom > 0).astype(float))
    per_date = d.groupby("date")["_pos"].mean().sort_index()
    s = df["date"].map(per_date)
    return {"breadth": s.reindex(df.index)}


@register("mkt_ret", "global", produces=["mkt_ret_5", "mkt_ret_20"])
def _market_proxy_returns(df: pd.DataFrame, params: dict) -> dict[str, pd.Series]:
    """Benchmark cumulative returns over 5- and 20-day horizons.

    Built from the per-date benchmark_return series (added by stage 1).
    """
    if "benchmark_return" not in df.columns:
        raise KeyError("mkt_ret features require a 'benchmark_return' column.")
    horizons = params.get("mkt_ret_horizons", [5, 20])
    per_date = df.drop_duplicates("date").set_index("date")["benchmark_return"].sort_index()
    growth = (1.0 + per_date)
    out: dict[str, pd.Series] = {}
    for h in horizons:
        col = f"mkt_ret_{h}"
        cum = growth.rolling(h, min_periods=h).apply(np.prod, raw=True) - 1.0
        out[col] = df["date"].map(cum).reindex(df.index)
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def add_custom_features(df: pd.DataFrame,
                        per_asset: list[str] | None,
                        global_: list[str] | None,
                        params: dict | None = None
                        ) -> tuple[pd.DataFrame, list[str]]:
    """Apply the requested feature builders; return (df, ordered added columns).

    per_asset / global_ are lists of registry keys (e.g. ["mom","vol","beta_60"]).
    Order of returned columns: per-asset first (in the order listed), then global.
    Unknown keys raise. The same registry key may produce multiple columns (e.g.
    "mom" -> mom_1..mom_60); all of them are added.
    """
    params = params or {}
    df = df.copy()
    added: list[str] = []

    for key in (per_asset or []):
        if key not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown per-asset feature {key!r}. "
                           f"Registered: {sorted(FEATURE_REGISTRY)}")
        entry = FEATURE_REGISTRY[key]
        if entry["kind"] != "per_asset":
            raise ValueError(f"Feature {key!r} is registered as {entry['kind']}, "
                             f"not per_asset.")
        cols = entry["func"](df, params)
        for name, series in cols.items():
            df[name] = series
            added.append(name)

    for key in (global_ or []):
        if key not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown global feature {key!r}. "
                           f"Registered: {sorted(FEATURE_REGISTRY)}")
        entry = FEATURE_REGISTRY[key]
        if entry["kind"] != "global":
            raise ValueError(f"Feature {key!r} is registered as {entry['kind']}, "
                             f"not global.")
        cols = entry["func"](df, params)
        for name, series in cols.items():
            df[name] = series
            added.append(name)

    return df, added


def expected_columns(per_asset: list[str] | None,
                     global_: list[str] | None) -> list[str]:
    """Return the ordered list of column names the given feature keys produce.

    Used by callers that need the indicator list WITHOUT recomputing (e.g.
    build_env_kwargs assembling tech_indicator_list).
    """
    cols: list[str] = []
    for key in (per_asset or []) + (global_ or []):
        if key not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature {key!r}. Registered: {sorted(FEATURE_REGISTRY)}")
        cols.extend(FEATURE_REGISTRY[key]["produces"])
    return cols


# ---------------------------------------------------------------------------
# Benchmark return + synthetic cash asset (stage-1 / inference data prep)
# ---------------------------------------------------------------------------

def add_benchmark_return(df: pd.DataFrame,
                         benchmark_cfg: dict,
                         ticker_return_series: pd.Series | None = None
                         ) -> pd.DataFrame:
    """Add a 'benchmark_return' column (broadcast per date).

    benchmark_cfg.type:
      * 'equal_weight' (default): daily-rebalanced equal-weight portfolio of the
        universe tickers; benchmark return = cross-sectional mean of per-ticker
        daily returns. No external data needed.
      * 'ticker': use ticker_return_series (a date->return Series the caller
        downloaded for benchmark_cfg.ticker).

    Used by the article_benchmark reward, the beta_60 feature, and the mkt_ret
    global features. The CASH synthetic asset (if any) should be injected AFTER
    this so it doesn't distort the equal-weight benchmark.
    """
    df = df.copy()
    btype = (benchmark_cfg or {}).get("type", "equal_weight")
    if btype == "equal_weight":
        piv = df.pivot_table(index="date", columns="tic", values="close")
        per_date = piv.pct_change().mean(axis=1).fillna(0.0)
    elif btype == "ticker":
        if ticker_return_series is None:
            raise ValueError("benchmark.type='ticker' requires ticker_return_series.")
        per_date = ticker_return_series
    else:
        raise ValueError(f"Unknown benchmark.type={btype!r}; expected "
                         f"'equal_weight' or 'ticker'.")
    df["benchmark_return"] = df["date"].map(per_date).fillna(0.0)
    return df


def inject_cash_asset(df: pd.DataFrame,
                      cash_ticker: str,
                      risk_free_rate: float,
                      per_asset_cols: list[str]) -> pd.DataFrame:
    """Append a synthetic CASH asset to the long-format dataframe.

    The CASH row for each date has:
      * OHLC = a price that compounds at the daily risk-free rate (so its daily
        return is constant and its variance / covariance are ~0).
      * volume = 0.
      * per-asset feature columns (stockstats indicators + custom per-asset)
        set to 0 (cash has no momentum, RSI, beta, etc.).
      * every OTHER column (global features, turbulence, benchmark_return,
        vix_level, day, etc.) copied from a real ticker on the same date, since
        those are per-date broadcast values that must match across all tickers.

    Must run BEFORE compute_cov_features so the covariance becomes (N+1)x(N+1)
    (the cash row/col is ~0). risk_free_rate is ANNUAL; 0.0 => cash earns nothing.
    """
    df = df.copy()
    dates = sorted(df["date"].unique())
    n = len(dates)
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    factors = np.concatenate([[1.0], np.full(n - 1, 1.0 + daily_rf)])
    prices = 100.0 * np.cumprod(factors)
    price_by_date = dict(zip(dates, prices))

    # Template: one real-ticker row per date carries all the per-date columns.
    template = (df.sort_values(["date", "tic"])
                  .groupby("date", as_index=False)
                  .first())
    cash = template.copy()
    cash["tic"] = cash_ticker
    for col in ("open", "high", "low", "close"):
        if col in cash.columns:
            cash[col] = cash["date"].map(price_by_date)
    if "volume" in cash.columns:
        cash["volume"] = 0.0
    for c in per_asset_cols:
        if c in cash.columns:
            cash[c] = 0.0

    out = pd.concat([df, cash], ignore_index=True)
    out = out.sort_values(["date", "tic"]).reset_index(drop=True)
    return out
