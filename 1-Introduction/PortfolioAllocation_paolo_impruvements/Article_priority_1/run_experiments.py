"""Experiment runner - launch many config variants and collect a comparison table.

Each experiment = a set of dotted-path overrides on top of a base config
(experiments.json). For every experiment the runner:

  1. derives a config (overrides applied; output paths redirected to
     experiments/<name>/; data_dir pointed at the shared data/ unless the
     overrides touch data/benchmark/cash, in which case stage 1 re-runs),
  2. runs stage 2 (train) -> stage 3 (env backtest); stage 4 (backtrader)
     only with --with-backtrader,
  3. computes metrics from the saved CSVs (NOT stdout),
  4. appends a row to experiments_results.csv (resumable).

PRIMARY ranking metric: env_sharpe_minus_eqw (PPO Sharpe minus EqualWeight
Sharpe, both from stage 3's equity_curves.csv - same basis). Backtrader
columns (bt_*) are the realistic-execution reality check, filled only when
--with-backtrader is passed.

Usage:
    python run_experiments.py --experiments experiments.json
    python run_experiments.py --experiments experiments.json --with-backtrader
    python run_experiments.py --experiments experiments.json --only baseline,gate_off
    python run_experiments.py --experiments experiments.json --force
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_TOUCH_PREFIXES = ("data.", "benchmark.", "cash.")
DATA_TOUCH_KEYS = ("data", "benchmark", "cash")


# ---------- config helpers ----------

def deep_set(cfg: dict, dotted: str, value) -> None:
    """Set cfg[a][b][c] = value for dotted='a.b.c', creating dicts as needed."""
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value


def touches_data(overrides: dict) -> bool:
    return any(k in DATA_TOUCH_KEYS or k.startswith(DATA_TOUCH_PREFIXES)
               for k in overrides)


def expand_grid(grid_spec: list[dict]) -> list[dict]:
    """Expand each {name_prefix, axes} block into a list of {name, overrides}."""
    out: list[dict] = []
    for block in grid_spec or []:
        prefix = block.get("name_prefix", "g")
        axes = block["axes"]
        keys = list(axes.keys())
        for combo in itertools.product(*[axes[k] for k in keys]):
            overrides = dict(zip(keys, combo))
            tag = "_".join(f"{k.split('.')[-1]}{v}" for k, v in zip(keys, combo))
            out.append({"name": f"{prefix}_{tag}", "overrides": overrides})
    return out


def load_experiment_list(spec: dict) -> list[dict]:
    exps = list(spec.get("experiments", []))
    exps += expand_grid(spec.get("grid", []))
    seen = set()
    for e in exps:
        if e["name"] in seen:
            raise ValueError(f"Duplicate experiment name: {e['name']}")
        seen.add(e["name"])
    return exps


# ---------- metric helpers (computed from saved CSVs) ----------

def metrics_from_equity(s: pd.Series) -> dict:
    s = s.dropna()
    if len(s) < 3:
        return {"cum_return": np.nan, "sharpe": np.nan, "max_dd": np.nan, "cagr": np.nan}
    rets = s.pct_change().dropna()
    cum = float(s.iloc[-1] / s.iloc[0] - 1.0)
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252))
    dd = float(((s - s.cummax()) / s.cummax()).min())
    cagr = float((s.iloc[-1] / s.iloc[0]) ** (252.0 / len(s)) - 1.0)
    return {"cum_return": cum, "sharpe": sharpe, "max_dd": dd, "cagr": cagr}


def collect_env_metrics(results_dir: Path) -> dict:
    eq_path = results_dir / "equity_curves.csv"
    eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)
    out: dict = {}
    ppo = metrics_from_equity(eq["PPO"]) if "PPO" in eq.columns else {}
    out["env_cum_return"] = ppo.get("cum_return", np.nan)
    out["env_sharpe"]     = ppo.get("sharpe", np.nan)
    out["env_maxdd"]      = ppo.get("max_dd", np.nan)
    out["env_cagr"]       = ppo.get("cagr", np.nan)
    eqw_col = "EqualWeight" if "EqualWeight" in eq.columns else None
    out["env_eqw_sharpe"]    = metrics_from_equity(eq[eqw_col])["sharpe"] if eqw_col else np.nan
    out["env_minvar_sharpe"] = metrics_from_equity(eq["MinVariance"])["sharpe"] if "MinVariance" in eq.columns else np.nan
    out["env_sharpe_minus_eqw"] = (out["env_sharpe"] - out["env_eqw_sharpe"]
                                   if not np.isnan(out["env_eqw_sharpe"]) else np.nan)
    return out


def collect_activity_metrics(results_dir: Path, cash_ticker: str) -> dict:
    wpath = results_dir / "weights_ppo.csv"
    if not wpath.exists():
        return {"risky_std": np.nan, "ann_turnover": np.nan, "regime_drift": np.nan}
    w = pd.read_csv(wpath, index_col=0, parse_dates=True).sort_index()
    risky = [c for c in w.columns if c != cash_ticker]
    risky_std = float(w[risky].std().mean())
    turn = w.diff().abs().sum(axis=1).dropna()
    ann_turn = float(turn.mean() * 252 / 2)
    n = len(w)
    early = w.iloc[: n // 3].mean()
    late = w.iloc[2 * n // 3:].mean()
    regime_drift = float((late - early).abs().max())
    return {"risky_std": risky_std, "ann_turnover": ann_turn, "regime_drift": regime_drift}


def collect_bt_metrics(bt_results_dir: Path) -> dict:
    out = {"bt_cum_return": np.nan, "bt_sharpe": np.nan, "bt_maxdd": np.nan, "n_trades": np.nan}
    eqp = bt_results_dir / "equity_backtrader.csv"
    if eqp.exists():
        eq = pd.read_csv(eqp, index_col=0, parse_dates=True)
        col = "equity" if "equity" in eq.columns else eq.columns[0]
        m = metrics_from_equity(eq[col])
        out.update(bt_cum_return=m["cum_return"], bt_sharpe=m["sharpe"], bt_maxdd=m["max_dd"])
    sjp = bt_results_dir / "summary.json"
    if sjp.exists():
        try:
            sj = json.load(open(sjp))
            ta = sj.get("trade_analyzer", {})
            total = ta.get("total", {})
            out["n_trades"] = total.get("total", total.get("closed", np.nan)) if isinstance(total, dict) else np.nan
        except Exception:
            pass
    return out


# ---------- running ----------

def run_stage(script: str, config_path: Path) -> tuple[bool, str]:
    cmd = [sys.executable, script, "--config", str(config_path)]
    proc = subprocess.run(cmd, cwd=str(HERE), capture_output=True, text=True)
    ok = proc.returncode == 0
    tail = (proc.stdout or "")[-1500:] + (proc.stderr or "")[-1500:]
    return ok, tail


def derive_config(base_cfg: dict, exp: dict, exp_dir: Path,
                  data_dir: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for k, v in exp["overrides"].items():
        deep_set(cfg, k, v)
    cfg["paths"]["data_dir"]        = str(data_dir)
    cfg["paths"]["model_dir"]       = str(exp_dir / "models")
    cfg["paths"]["results_dir"]     = str(exp_dir / "results")
    cfg["paths"]["tensorboard_dir"] = str(exp_dir / "tb")
    return cfg


def derive_bt_config(base_bt: dict, exp_config_path: Path, exp_dir: Path,
                     data_dir: Path) -> dict:
    bt = copy.deepcopy(base_bt)
    bt["source_config"] = str(exp_config_path)
    bt.setdefault("inputs", {})
    bt["inputs"]["weights_csv"]  = str(exp_dir / "results" / "weights_ppo.csv")
    bt["inputs"]["trade_pickle"] = str(data_dir / "full_data.pkl")
    bt.setdefault("output", {})
    bt["output"]["results_dir"] = str(exp_dir / "results_backtrader")
    bt.setdefault("execution", {})
    bt["execution"]["cash_buffer"] = 0.0  # required when env cash is enabled
    return bt


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=str(HERE), capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", default="experiments.json")
    ap.add_argument("--with-backtrader", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated experiment names to run")
    ap.add_argument("--force", action="store_true", help="re-run even if already in results CSV")
    ap.add_argument("--results", default="experiments_results.csv")
    args = ap.parse_args()

    spec = json.load(open(HERE / args.experiments))
    base_config_path = HERE / spec.get("base_config", "config.json")
    base_cfg = json.load(open(base_config_path))
    base_bt_path = HERE / "backtrader_config.json"
    base_bt = json.load(open(base_bt_path)) if (args.with_backtrader and base_bt_path.exists()) else None
    cash_ticker = str((base_cfg.get("cash", {}) or {}).get("ticker", "CASH"))

    exps = load_experiment_list(spec)
    if args.only:
        want = set(args.only.split(","))
        exps = [e for e in exps if e["name"] in want]

    results_path = HERE / args.results
    done = set()
    if results_path.exists() and not args.force:
        prev = pd.read_csv(results_path)
        done = set(prev[prev.get("status", "ok") == "ok"]["name"].astype(str))

    exp_root = HERE / "experiments"
    exp_root.mkdir(exist_ok=True)
    shared_data = HERE / Path(base_cfg["paths"]["data_dir"])

    # Ensure shared data exists (run stage 1 once with the base config).
    if not (shared_data / "full_data.pkl").exists():
        print("Shared data/full_data.pkl missing - running stage 1 with the base config...")
        ok, tail = run_stage("01_get_data.py", base_config_path)
        if not ok:
            print(tail); raise SystemExit("Stage 1 failed for the base config.")

    commit = git_commit()
    print(f"Experiments: {len(exps)}  | with_backtrader={args.with_backtrader}  | "
          f"already done: {len(done)}  | commit={commit}\n")

    for i, exp in enumerate(exps, 1):
        name = exp["name"]
        if name in done:
            print(f"[{i}/{len(exps)}] {name}: SKIP (already in results)")
            continue
        print(f"[{i}/{len(exps)}] {name}: {json.dumps(exp['overrides'])}")
        t0 = time.time()
        exp_dir = exp_root / name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # data: own dir if overrides touch data, else shared
        own_data = touches_data(exp["overrides"])
        data_dir = (exp_dir / "data") if own_data else shared_data

        cfg = derive_config(base_cfg, exp, exp_dir, data_dir)
        cfg_path = exp_dir / "config.json"
        json.dump(cfg, open(cfg_path, "w"), indent=2)

        status, tail = "ok", ""
        try:
            if own_data:
                ok, tail = run_stage("01_get_data.py", cfg_path)
                if not ok:
                    raise RuntimeError("stage 1 (data) failed")
            ok, tail = run_stage("02_train.py", cfg_path)
            if not ok:
                raise RuntimeError("stage 2 (train) failed")
            ok, tail = run_stage("03_backtest.py", cfg_path)
            if not ok:
                raise RuntimeError("stage 3 (backtest) failed")

            row = {"name": name, "overrides": json.dumps(exp["overrides"])}
            row.update(collect_env_metrics(exp_dir / "results"))
            row.update(collect_activity_metrics(exp_dir / "results", cash_ticker))

            if args.with_backtrader and base_bt is not None:
                bt_cfg = derive_bt_config(base_bt, cfg_path, exp_dir, data_dir)
                bt_cfg_path = exp_dir / "backtrader_config.json"
                json.dump(bt_cfg, open(bt_cfg_path, "w"), indent=2)
                okb, tailb = run_stage("04_backtrader_replay.py", bt_cfg_path)
                if okb:
                    row.update(collect_bt_metrics(exp_dir / "results_backtrader"))
                else:
                    print(f"    backtrader failed (non-fatal):\n{tailb[-400:]}")
        except Exception as e:
            status = "failed"
            row = {"name": name, "overrides": json.dumps(exp["overrides"]), "error": str(e)}
            print(f"    FAILED: {e}\n{tail[-600:]}")

        row.update({
            "status": status,
            "seeds": json.dumps(cfg.get("seeds", {}).get("list", [])),
            "reward_mode": cfg["env"].get("reward_mode"),
            "wall_clock_s": round(time.time() - t0, 1),
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commit": commit,
        })

        # append incrementally so a crash never loses prior rows
        df_row = pd.DataFrame([row])
        if results_path.exists():
            old = pd.read_csv(results_path)
            old = old[old["name"] != name]  # replace any prior row for this name
            pd.concat([old, df_row], ignore_index=True).to_csv(results_path, index=False)
        else:
            df_row.to_csv(results_path, index=False)
        print(f"    done in {row['wall_clock_s']}s  status={status}  "
              f"env_sharpe={row.get('env_sharpe', float('nan')):.3f}  "
              f"env_sharpe_minus_eqw={row.get('env_sharpe_minus_eqw', float('nan')):.3f}")

    # ---------- ranked summary ----------
    if not results_path.exists():
        print("\nNo results written.")
        return
    res = pd.read_csv(results_path)
    ok = res[res["status"] == "ok"].copy()
    if ok.empty:
        print("\nNo successful experiments."); return
    ok = ok.sort_values("env_sharpe_minus_eqw", ascending=False)
    cols = ["name", "env_sharpe", "env_eqw_sharpe", "env_sharpe_minus_eqw",
            "env_cum_return", "env_maxdd", "ann_turnover", "risky_std", "regime_drift"]
    if "bt_sharpe" in ok.columns and ok["bt_sharpe"].notna().any():
        cols += ["bt_sharpe", "bt_maxdd", "n_trades"]
    cols = [c for c in cols if c in ok.columns]
    print("\n" + "=" * 100)
    print("RANKED BY env_sharpe_minus_eqw (PPO Sharpe - EqualWeight Sharpe)")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.3f}"):
        print(ok[cols].to_string(index=False))
    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()
