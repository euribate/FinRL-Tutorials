Multi-seed training and ensembled actions — operative notes

What it does
-----------
For each walk-forward window (or for the single train/trade split, if walk-forward is off), train N independent PPO models with different random seeds. At inference, run all N models on the same eval data, average their post-softmax weight vectors row-by-row, and use the averaged weights as the ensemble's allocation. Compute portfolio_return from those averaged weights against the eval prices.

Default seeds
------------
seeds.list = [42, 1337, 7]   (three seeds)
Single-seed sanity check: seeds.list = [42]
Larger ensembles: seeds.list = [0, 1, 2, 3, 4]   (5 seeds, costs 5x stage-2 wall-clock)

Why three is the recommended default
------------------------------------
- One seed: high variance in headline metrics, can't separate signal from noise.
- Three seeds: ~30-50% reduction in equity-curve std, 3x training cost, manageable wall-clock.
- Five seeds: diminishing returns on variance reduction, 5x cost.
The sweet spot for a single-machine pipeline is 3. Increase if you suspect you're still seed-noise-limited.

Ensembling method: post-softmax weight averaging
------------------------------------------------
For each trading day d, take each seed's post-softmax weight vector w_seed and compute:
    w_ensemble = mean(w_seed for seed in seeds.list)
    w_ensemble = w_ensemble / sum(w_ensemble)         (re-normalise for float drift)

Properties:
- Stays on the simplex (non-negative, sums to 1).
- "One vote per seed" semantics — diversified across policies.
- Smoother day-over-day weight trajectory than any single seed.

Alternative considered: average pre-softmax logits, then softmax. Produces sharper distributions (a confident policy can override indifferent ones). Less robust. Not used.

How portfolio_return is computed from averaged weights
------------------------------------------------------
You CANNOT average per-seed daily_return series. Each seed's equity curve drifts independently, so averaging the returns doesn't give what a real trader following the ensemble would have earned.

Correct approach:
    for each day d from 1 to T:
        return_d = sum_over_tickers( (close_d / close_{d-1} - 1) * w_ensemble_d )

This is implemented in utils.daily_return_from_weights(weights_df, trade_df).

File naming convention
----------------------
Each (algo, window, seed) triple produces a distinct set of artifacts in models/:
    agent_<algo>_w<i>_s<seed>.zip                    trained policy
    agent_<algo>_w<i>_s<seed>.history.json           early-stopping history (per seed per window)
    vecnormalize_<algo>_w<i>_s<seed>.pkl             VecNormalize running stats (per seed per window)

In single-split mode (walk_forward.enabled=false):
    agent_<algo>_s<seed>.zip
    agent_<algo>_s<seed>.history.json
    vecnormalize_<algo>_s<seed>.pkl

Stage 3 reads all seeds for each window, runs each, averages, recomputes returns.

Stages 4 and 5 don't see seeds. They read results/weights_<algo>.csv which already contains the ensemble-averaged weights from stage 3. Same backtrader replay, same QuantStats tearsheet machinery.

What changed in code
--------------------
utils.py
    + get_seeds(config) -> list[int]
    + average_seed_actions(per_seed_dfs) -> DataFrame
    + daily_return_from_weights(weights_df, trade_df) -> DataFrame

02_train.py
    Outer "for s in seeds:" loop wrapping the existing training body.
    model_path / tb_log_name extended with _s<seed> suffix.

03_backtest.py
    predict_walk_forward now takes seeds, loads N models per window,
    averages actions, recomputes returns from averaged weights.
    Single-split path does the same without the window loop.

config.json
    New seeds block: { "list": [...], "_notes": "..." }

01_get_data.py, 04_backtrader_replay.py, 05_quantstats_report.py
    No changes. Stages 4/5 consume the same weights CSV format.

Subtle gotchas
--------------
1. Each seed's VecNormalize stats are per-seed. At inference each seed loads its own stats. The ensemble averaging happens AFTER each policy has produced its softmaxed weights from its own normalised view of observations. Correct.

2. Each seed's early-stopping history is independent. Seeds will stop at different timesteps. The per-seed history.json files document the trajectory per seed; the multi-seed training stdout prints best Sharpe per (algo, window, seed).

3. Each seed's TC penalty / risk-off gate applies independently inside the env. The averaged ensemble weights at inference may not be exactly equal to "what TC + gate produced for each seed individually", but the differences are small and washed out by averaging.

4. Compute is linear in len(seeds.list). Plan accordingly:
   - Stage 2 wall-clock = N x single-seed cost (with early stopping mitigating the worst case).
   - Stage 3 wall-clock scales the prediction phase by N (seconds, not minutes).
   - Stage 4 / 5: unchanged.

5. Repeated seeds (e.g. [42, 42, 42]) are useless — all three runs produce identical policies. Don't.

How to A/B test the ensemble effect
-----------------------------------
- Run once with seeds.list = [42, 1337, 7] -> note the headline metrics.
- Run once with seeds.list = [42]            -> single-seed baseline.
- Run once with seeds.list = [1337]          -> different single-seed baseline.
- Run once with seeds.list = [7]             -> another single-seed baseline.

If the 3-seed ensemble's Sharpe is within the spread of the three single-seed runs, the ensemble adds nothing for your setup. If it's above the best single-seed run, the ensembling is doing useful variance reduction.

This is also a useful diagnostic: large spread across single-seed runs means your pipeline is seed-noise-limited and ensembling has more headroom; small spread means the policies are converging to similar solutions and ensembling helps less.

Compute budget for the default config
------------------------------------
3 seeds x 7 walk-forward windows x ~30-60 min per PPO run with early stopping
= ~10-21 hours on CPU for stage 2 alone.

With faster hardware, smaller total_timesteps, or fewer seeds the time scales down proportionally. ES typically catches each run after 30-60% of the max budget, so the actual wall-clock is usually 50-60% of the worst-case estimate.

When ensembling is NOT worth it
-------------------------------
- When your single-seed std is already very small (mature pipeline, lots of training data, low variance reward like log_return without DSR).
- When you're prototyping fast and need stage 2 to finish in under an hour.
- When you're sanity-checking a config change — use single seed, see if it matters at all, then bring in the ensemble for the final reportable number.

Most other cases (and definitely with diff_sharpe + walk-forward), 3-seed ensembling is the right default.
