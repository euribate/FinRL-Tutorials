"""Inspect what turnover the env actually computed per bar (proxy via weight diffs)."""
import pandas as pd
import numpy as np

w = pd.read_csv("results/weights_ppo.csv", index_col=0, parse_dates=True)
print(f"Bars: {len(w)}")
print(f"Tickers: {len(w.columns)}")
print()

# Naive turnover (what folders 0-7 used)
naive = (w - w.shift(1)).abs().sum(axis=1).dropna()
print(f"NAIVE turnover (|w_t - w_{{t-1}}|):")
print(f"  mean per bar: {naive.mean()*100:.3f}%")
print(f"  median:       {naive.median()*100:.3f}%")
print(f"  95th pct:     {naive.quantile(0.95)*100:.3f}%")
print(f"  total over backtest: {naive.sum()*100:.1f}%")
print()

# Annualised turnover
ann_naive = naive.mean() * 252
print(f"  annualised:   {ann_naive*100:.0f}%")
