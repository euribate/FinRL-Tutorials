import pandas as pd

tx = pd.read_csv("results_backtrader/trade_log.csv")
print(f"Total fills:           {len(tx):>8,}")
print(f"Total commission paid: ${tx['comm'].abs().sum():>10,.2f}")
print(f"Total value traded:    ${tx['value'].abs().sum():>10,.2f}")
print(f"Avg fill size:         ${tx['value'].abs().mean():>10,.2f}")
print(f"Implicit slippage est: ${tx['value'].abs().sum() * 0.0005:>10,.2f}")
