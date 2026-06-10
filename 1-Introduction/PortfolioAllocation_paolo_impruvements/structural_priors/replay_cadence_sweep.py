"""replay_cadence_sweep.py - 5 models x 3 cadences in one table.

Phase 2 of the model-vs-cadence comparison. Replays each model's recorded
weights through backtrader at WEEKLY and MONTHLY cadences and stacks them
with the DAILY results that Phase 1 already produced. No retraining;
stage 4 only.

Pre-conditions (must be satisfied by Phase 1 before running this):
  * For each algo in {ppo, a2c, ddpg, td3, sac} that you want included,
    a finished experiment must exist at experiments/model_<algo>/, with:
        - results/weights_ppo.csv         (stage 3 weights to replay)
        - backtrader_config.json          (per-experiment paths set by
                                           derive_bt_config in Phase 1)
        - results_backtrader/summary.json (Phase 1's DAILY bt run)
  * The base 04_backtrader_replay.py from this folder is unchanged.

What this does, per algo:
  1. Reads experiments/model_<algo>/results_backtrader/summary.json -> daily row.
  2. For cadence in {weekly, monthly}:
        - Copies experiments/model_<algo>/backtrader_config.json,
          flips rebalance.cadence, points output.results_dir at
          results_backtrader_<cadence>/ (so daily isn't overwritten),
          writes backtrader_config_<cadence>.json.
        - Runs stage 4 with --config backtrader_config_<cadence>.json.
        - Parses the new summary.json into a row.

Output: model_cadence_results.csv at the project root - one row per
        (model, cadence) cell, columns bt_sharpe / bt_cum_return /
        bt_max_dd / n_trades / rebalance_bars / source.

Run from this folder:
    python replay_cadence_sweep.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE     = Path(__file__).resolve().parent
EXPDIR   = HERE / "experiments"
ALGOS    = ["ppo", "a2c", "ddpg", "td3", "sac"]
CADENCES = ["daily", "weekly", "monthly"]


def _safe_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  ! could not parse {path.name}: {e}")
        return None


def extract_metrics(summary: dict | None) -> dict:
    """Pull the headline numbers out of stage 4's summary.json. Stage 4
    writes a backtrader-native nested structure (sharpe.sharperatio,
    drawdown.max.drawdown, returns.rtot, trade_analyzer.total.total),
    not flat keys. rebalance_bars is not in the summary at all (it
    only appears in the stdout cadence line) so it's left NaN here.
    """
    s = summary or {}
    sharpe = (s.get("sharpe") or {}).get("sharperatio") if isinstance(s.get("sharpe"), dict) else None
    dd_pct = (s.get("drawdown") or {}).get("max", {}).get("drawdown") if isinstance(s.get("drawdown"), dict) else None
    rtot   = (s.get("returns")  or {}).get("rtot")  if isinstance(s.get("returns"),  dict) else None
    nt     = (s.get("trade_analyzer") or {}).get("total", {}).get("total") if isinstance(s.get("trade_analyzer"), dict) else None
    return {
        "bt_sharpe":      _safe_float(sharpe),
        "bt_cum_return":  _safe_float(rtot),
        "bt_max_dd":      _safe_float(dd_pct),     # already a percentage
        "n_trades":       _safe_float(nt),
        "rebalance_bars": float("nan"),
    }


def run_one(exp_dir: Path, cadence: str, algo: str = "ppo") -> dict | None:
    """Replay stage 4 for `exp_dir` at `cadence`. Returns metric dict or None
    on failure (with stdout tail printed)."""
    base_bt_path = exp_dir / "backtrader_config.json"
    if not base_bt_path.exists():
        print(f"  ! {exp_dir.name}: no backtrader_config.json - cannot replay")
        return None

    bt = json.loads(base_bt_path.read_text())
    bt.setdefault("rebalance", {})
    bt["rebalance"]["cadence"] = cadence
    # Make sure the weights file path matches the actual algo - older
    # versions of derive_bt_config hardcoded weights_ppo.csv for every
    # algo, which is wrong for A2C / DDPG / TD3 / SAC.
    bt.setdefault("inputs", {})
    bt["inputs"]["weights_csv"] = str(exp_dir / "results" / f"weights_{algo}.csv")
    # Route output to a cadence-specific dir so the DAILY artefacts that
    # Phase 1 produced aren't clobbered.
    out_dir = exp_dir / f"results_backtrader_{cadence}"
    bt.setdefault("output", {})
    bt["output"]["results_dir"] = str(out_dir)

    new_bt_path = exp_dir / f"backtrader_config_{cadence}.json"
    new_bt_path.write_text(json.dumps(bt, indent=2))

    proc = subprocess.run(
        [sys.executable, str(HERE / "04_backtrader_replay.py"),
         "--config", str(new_bt_path)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f"FAIL  rc={proc.returncode}")
        print((proc.stdout + proc.stderr)[-800:])
        return None

    summary_path = out_dir / "summary.json"
    return extract_metrics(load_summary(summary_path))


def main() -> None:
    # Optional CLI filter so you can run a subset (e.g. PPO only).
    only = None
    if len(sys.argv) > 1 and sys.argv[1] == "--only" and len(sys.argv) > 2:
        only = set(sys.argv[2].split(","))

    print(f"Looking for finished model experiments under {EXPDIR}/")
    available = []
    for algo in ALGOS:
        if only is not None and algo not in only:
            continue
        exp_dir = EXPDIR / f"model_{algo}"
        # Algo-aware: each model writes weights_<algo>.csv in stage 3.
        weights = exp_dir / "results" / f"weights_{algo}.csv"
        if weights.exists():
            available.append((algo, exp_dir))
        else:
            print(f"  ! model_{algo} skipped - missing results/weights_{algo}.csv")
    if not available:
        print("\nNo finished model experiments. Run Phase 1 first:")
        print("  python run_experiments.py --experiments experiments.json "
              "--only model_ppo[,model_a2c,model_ddpg,model_td3,model_sac] "
              "--with-backtrader")
        sys.exit(1)
    print(f"\nReady: {[a for a, _ in available]}\n")

    rows = []
    for algo, exp_dir in available:
        # ALWAYS run all three cadences fresh - do NOT trust whatever
        # cadence the existing results_backtrader/summary.json was
        # generated at (it may have been daily, weekly, or monthly
        # depending on backtrader_config.json's state at Phase 1 time).
        for cad in ("daily", "weekly", "monthly"):
            print(f"  [model_{algo}] cadence={cad:<7} running ...", end=" ", flush=True)
            m = run_one(exp_dir, cad, algo=algo)
            if m is None:
                continue
            m.update({"model": algo, "cadence": cad})
            rows.append(m)
            print(f"sharpe={m['bt_sharpe']:.3f}  trades={m['n_trades']:.0f}  "
                  f"rebal={m['rebalance_bars']:.0f}")

    if not rows:
        print("No results collected.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    # Stable order in the output table.
    df["cadence_ord"] = df["cadence"].map({c: i for i, c in enumerate(CADENCES)})
    df["model_ord"]   = df["model"].map({a: i for i, a in enumerate(ALGOS)})
    df = df.sort_values(["model_ord", "cadence_ord"]).drop(
        columns=["model_ord", "cadence_ord"])

    out_path = HERE / "model_cadence_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    # Pretty 5x3 print.
    print("\n" + "=" * 80)
    print(" 5 models x 3 cadences - bt_sharpe / bt_max_dd / n_trades")
    print("=" * 80)
    pivot_sharpe = df.pivot(index="model", columns="cadence", values="bt_sharpe")[CADENCES]
    pivot_dd     = df.pivot(index="model", columns="cadence", values="bt_max_dd")[CADENCES]
    pivot_tr     = df.pivot(index="model", columns="cadence", values="n_trades")[CADENCES]
    print("\nbt_sharpe:");      print(pivot_sharpe.round(3))
    print("\nbt_max_dd (%):");  print(pivot_dd.round(2))
    print("\nn_trades:");       print(pivot_tr.astype(int, errors="ignore"))


if __name__ == "__main__":
    main()
