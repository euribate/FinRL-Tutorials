"""Quick diagnostic: print the turbulence distribution from data/full_data.pkl.

Usage:
    python inspect_turbulence.py
"""
from pathlib import Path
import sys

import pandas as pd


def main() -> None:
    pkl = Path(__file__).resolve().parent / "data" / "full_data.pkl"
    if not pkl.exists():
        print(f"ERROR: {pkl} not found. Run 01_get_data.py first.")
        sys.exit(1)

    df = pd.read_pickle(pkl)
    if "turbulence" not in df.columns:
        print("ERROR: no 'turbulence' column in full_data.pkl.")
        print("Set data.use_turbulence=true in config.json and re-run 01_get_data.py.")
        sys.exit(1)

    turb = df.drop_duplicates("date")[["date", "turbulence"]].sort_values("date").reset_index(drop=True)

    print(f"Total trading days: {len(turb)}")
    print(f"Date range:         {turb.date.min()}  ->  {turb.date.max()}")
    print(f"Turbulence range:   {turb.turbulence.min():.2f}  ->  {turb.turbulence.max():.2f}")
    print()
    print("Percentiles:")
    for p in [25, 50, 75, 90, 95, 97, 99, 99.5]:
        print(f"  {p:5.1f}th  ->  {turb.turbulence.quantile(p / 100):8.2f}")
    print()
    print("Days exceeding candidate thresholds:")
    for thr in [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150]:
        n   = int((turb.turbulence > thr).sum())
        pct = 100.0 * n / max(len(turb), 1)
        bar = "#" * min(int(pct * 2), 40)
        print(f"  > {thr:>4}  ->  {n:>5} days  ({pct:5.2f}%)  {bar}")
    print()
    print("Top 15 most turbulent days:")
    top = turb.nlargest(15, "turbulence").reset_index(drop=True)
    for _, row in top.iterrows():
        print(f"  {row['date']}   turbulence = {row['turbulence']:.2f}")


if __name__ == "__main__":
    main()
