# Next Steps

_Memo — 2026-05-30_

## Status: model side exhausted

After a full sweep over reward functions, network architecture, encoder size, and PPO hyperparameters (47 experiments total), the conclusion is settled:

- **Env Sharpe ceiling ≈ EqualWeight (0.784).** No PPO config beats it across the sweep.
- **Deployable winner: `enc_hidden128_emb_dim1`** — env Sharpe 0.774, bt Sharpe 0.782, ann. turnover 0.85×, 10 trades over the test window.
- **Going wider (192 / 256) did not help** — confirmed inverted-U on `hidden`; 128 is the peak.
- **Reward modes:** `diff_sharpe` slightly edges `article_absolute` at the top of the table (~0.013 env Sharpe), small and likely partly noise.
- **`hidden=128, emb_dim=1`** is now the default in `config.json`.

## Net-of-cost picture (top configs)

| Config | bt_sharpe | env_sharpe | ann_turnover |
|---|---|---|---|
| big_diff_sharpe_h256_g0.999_ent0.001 | 0.807 | 0.752 | 1.54 |
| bigenc (h64 / emb_dim4) | 0.798 | 0.748 | 2.85 |
| big_diff_sharpe_h256_g0.999_ent0.0 | 0.794 | 0.755 | 1.35 |
| **enc_hidden128_emb_dim1** | **0.782** | **0.774** | **0.85** |

The 0.025 spread on `bt_sharpe` at the top is within noise (10-trade test window). Net of execution costs, the top configs are effectively tied. `enc_hidden128_emb_dim1` wins on cleanliness — best `env_sharpe` *and* lowest turnover by ~2×.

## Three options for the next push

| Path | What it is | Effort | Expected upside |
|---|---|---|---|
| **A. Ship it** | Lock `enc_hidden128_emb_dim1` + turbulence gate as the deployable policy; wire it to Alpaca paper trading using the `3-Practical` pattern. | Low | None on Sharpe, but you get a real running system. |
| **B. Regime enrichment** | Replace the binary turbulence gate with a graded 3-state regime + fast risk-off overlay (FinRL-X `adaptive_rotation` pattern). | Medium | Likely small but real — the gate is the edge; a richer gate could extend it. |
| **C. Selection** | Add fundamentals-based ML stock selection on a larger universe (FinRL-X ML pattern); the RL allocator runs over a rotating universe. | High | The biggest potential — selection is the axis untouched so far. |

## Recommendation

1. **A now** — finish the job; get a real, running system.
2. **C next** — the axis with actual untapped alpha based on both the ceiling result here and the FinRL-X architecture.
3. **B** is fine but probably marginal given how thoroughly the gate's value is already measured.
