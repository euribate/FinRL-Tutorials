"""Diagnostic - rank trained seeds by training convergence quality.

PPO training is initialisation-dependent: some seeds escape the near-uniform
initial policy, others stay stuck. This script reads every per-seed history
file in models/, ranks them, and prints which seeds should be in the
deployment ensemble.

A seed is "converged" if it has >= MIN_IMPROVEMENTS validation Sharpe
improvements during training (default 3). Converged seeds have learned
something beyond the uniform-portfolio baseline; stuck seeds saved their
"best" checkpoint at the first or second evaluation and never improved.

Usage:
    # After training N candidate seeds:
    python filter_seeds.py
    # Then copy the recommended seeds into config_production.json's seeds.list
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config_production.json")
    parser.add_argument("--min-improvements", type=int, default=3,
                        help="Min improvements to count a seed as converged (default 3).")
    parser.add_argument("--keep", type=int, default=3,
                        help="How many top seeds to recommend (default 3).")
    return parser.parse_args()


def parse_seed_from_filename(p: Path) -> int | None:
    m = re.match(r"agent_ppo_s(\d+)\.history\.json$", p.name)
    return int(m.group(1)) if m else None


def main() -> None:
    args = parse_args()

    config = json.load(open(args.config))
    model_dir = Path(config["paths"]["model_dir"])
    if not model_dir.is_absolute():
        model_dir = Path(__file__).resolve().parent / model_dir

    histories = sorted(model_dir.glob("agent_ppo_s*.history.json"))
    # Filter out walk-forward histories (those have _w<i>_s<seed>)
    histories = [p for p in histories if "_w" not in p.name]

    if not histories:
        print(f"No production seed histories found in {model_dir}.")
        print("Run python 02_train.py --config config_production.json first.")
        return

    rows: list[dict] = []
    for p in histories:
        seed = parse_seed_from_filename(p)
        if seed is None:
            continue
        with open(p) as f:
            data = json.load(f)
        if not data:
            rows.append({"seed": seed, "evals": 0, "improvements": 0,
                         "best_sharpe": float("nan"), "best_step": 0,
                         "last_step": 0, "first_sharpe": float("nan"),
                         "last_sharpe": float("nan"), "trajectory_delta": 0.0})
            continue

        sharpes = [d["sharpe"] for d in data]
        n_improvements = sum(1 for d in data if d.get("improved"))
        best_i = max(range(len(sharpes)), key=lambda i: sharpes[i])

        rows.append({
            "seed":         seed,
            "evals":        len(data),
            "improvements": n_improvements,
            "best_sharpe":  sharpes[best_i],
            "best_step":    data[best_i]["timesteps"],
            "last_step":    data[-1]["timesteps"],
            "first_sharpe": sharpes[0],
            "last_sharpe":  sharpes[-1],
            "trajectory_delta": sharpes[-1] - sharpes[0],
        })

    # Sort: most improvements first, then by best_sharpe descending
    rows.sort(key=lambda r: (-r["improvements"], -r["best_sharpe"]))

    print(f"{'seed':>7}  {'evals':>5}  {'imp':>4}  {'best':>7}  {'best@':>7}  "
          f"{'last@':>7}  {'first_S':>7}  {'last_S':>7}  {'delta':>7}  status")
    print(f"{'-'*7}  {'-'*5}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  ------")
    converged: list[int] = []
    stuck: list[int] = []
    for r in rows:
        status = "converged" if r["improvements"] >= args.min_improvements else "STUCK"
        if r["improvements"] >= args.min_improvements:
            converged.append(r["seed"])
        else:
            stuck.append(r["seed"])
        print(f"{r['seed']:>7}  {r['evals']:>5}  {r['improvements']:>4}  "
              f"{r['best_sharpe']:>7.3f}  {r['best_step']:>7}  {r['last_step']:>7}  "
              f"{r['first_sharpe']:>7.3f}  {r['last_sharpe']:>7.3f}  "
              f"{r['trajectory_delta']:>+7.3f}  {status}")

    print()
    print(f"Converged seeds ({len(converged)}): {converged}")
    print(f"Stuck seeds     ({len(stuck)}): {stuck}")

    recommended = converged[: args.keep]
    print()
    if len(recommended) >= args.keep:
        print(f"==> Recommended ensemble (top {args.keep} by improvements then best Sharpe):")
        print(f"    \"seeds\": {{ \"list\": {recommended} }}")
        print(f"    Paste this into config_production.json, then re-run inspect_ensemble.py.")
    else:
        print(f"==> Only {len(converged)} seed(s) converged - not enough for an ensemble of {args.keep}.")
        print(f"    Train more candidate seeds and re-run this script. Try adding")
        print(f"    seeds to config_production.json's seeds.list (e.g. [100, 999, 12345, 31415,")
        print(f"    7777, 271828]) and run python 02_train.py --config config_production.json.")


if __name__ == "__main__":
    main()
