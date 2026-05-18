# FinRL Portfolio Allocation (NeurIPS 2020) — How the Notebook Works

This document explains the notebook `FinRL_PortfolioAllocation_NeurIPS_2020.ipynb` end to end: what it does, what data and features it uses, which models it trains, and — most importantly — how the covariance matrix enters the state and shapes the portfolio weights the agent eventually outputs.

---

## 1. The goal

The notebook builds an automated portfolio manager for the Dow Jones 30 universe. Instead of predicting prices, it learns a daily allocation policy: on every trading day the agent outputs a vector of weights `[w_1, w_2, ..., w_N]` (one per stock), the weights are constrained to sum to 1, and the portfolio is rebalanced accordingly. The objective is to maximise the change in portfolio value over time.

The whole problem is framed as a Markov Decision Process (MDP):

- **State** `s_t` — everything the agent sees on day `t`: a covariance matrix of recent returns plus a set of technical indicators (one value per stock per indicator).
- **Action** `a_t` — a raw vector of size `N` (number of stocks). The environment converts it into portfolio weights via softmax so they are non-negative and sum to 1.
- **Reward** `r_t` — the change in portfolio value from `t` to `t+1` (i.e. `v' - v`).
- **Transition** — the next day's prices and indicators (real market data, time-driven simulation).

This is a continuous-action problem, which is why the models used are continuous-action DRL algorithms (A2C, PPO, DDPG, SAC, TD3).

---

## 2. Data

- **Source:** Yahoo Finance via `YahooDownloader`.
- **Universe:** Dow Jones 30 constituents (the 2019–2021 list hard-coded in the notebook). WBA failed to download, leaving 29 → 27 usable tickers after preprocessing.
- **Period downloaded:** 2008-01-01 to 2021-10-31, OHLCV daily bars.
- **Train split:** 2009-01-01 to 2020-07-01.
- **Trade (out-of-sample) split:** 2020-07-01 to 2021-10-31.

The raw frame has ~98k rows (`date`, `tic`, OHLCV, `day`). After feature engineering it shrinks because the technical-indicator warm-up window drops some early rows, and after adding the covariance feature it shrinks further (one full year of history is needed before any state can be produced).

---

## 3. Feature engineering — the technical indicators

`FeatureEngineer(use_technical_indicator=True, use_turbulence=False, user_defined_feature=False)` adds the following per stock per day. These are the features the agent sees as part of the state, alongside the covariance matrix:

- `macd` — Moving Average Convergence Divergence. Trend / momentum signal.
- `boll_ub`, `boll_lb` — Upper and lower Bollinger Bands. Volatility envelope around the price.
- `rsi_30` — 30-day Relative Strength Index. Overbought/oversold oscillator.
- `cci_30` — 30-day Commodity Channel Index. Mean-reversion / momentum oscillator.
- `dx_30` — 30-day Directional Movement Index. Trend strength.
- `close_30_sma`, `close_60_sma` — 30- and 60-day simple moving averages of the close.

Turbulence is intentionally turned off in this notebook (the long-only weight-allocation framing does not gate trading on a turbulence threshold).

These indicators are pulled from `config.INDICATORS` inside the env, and at every step the environment stacks them on top of the covariance matrix to build the observation.

---

## 4. The covariance feature — from DataFrame to observation, step by step

This is the part that makes the notebook different from a vanilla stock-trading DRL setup. The confusing bit is that the DataFrame has `cov_list` and `return_list` stored as **objects repeated 27 times per day** (once per ticker row). That looks wasteful, but it is just a side effect of using a "long" DataFrame (one row per ticker per date). The environment never iterates over those duplicates — it grabs one copy. Below is exactly what happens, step by step.

### Step 0 — The DataFrame layout going in

After all the preprocessing the DataFrame looks like this. Take one example day, say 2020-07-01:

| date       | tic  | close | macd  | rsi_30 | ... | cov_list                   | return_list                  |
|------------|------|-------|-------|--------|-----|----------------------------|------------------------------|
| 2020-07-01 | AAPL | 91.2  | 0.42  | 64.1   | ... | [27×27 matrix object]      | [252×27 returns DataFrame]   |
| 2020-07-01 | AMGN | 235.7 | 0.11  | 52.0   | ... | [same 27×27 matrix object] | [same 252×27 returns object] |
| 2020-07-01 | AXP  | 96.5  | -0.20 | 47.3   | ... | [same 27×27 matrix object] | [same 252×27 returns object] |
| ...        | ...  | ...   | ...   | ...    | ... | ...                        | ...                          |
| 2020-07-01 | WMT  | 119.8 | 0.05  | 55.6   | ... | [same 27×27 matrix object] | [same 252×27 returns object] |

Two things to notice:

1. There are **27 rows for that single date** — one per ticker. The per-ticker columns (`close`, `macd`, `rsi_30`, …) differ across the 27 rows. The `cov_list` and `return_list` columns are **the same Python object repeated 27 times**.
2. The notebook also does `df.index = df.date.factorize()[0]`, which assigns the **same integer index** to all 27 rows of a given date. So `df.loc[0, :]` returns 27 rows (the whole 2020-07-01 slice), `df.loc[1, :]` returns the 2020-07-02 slice, and so on.

### Step 1 — On day `t`, the env grabs the 27-row slice

In `StockPortfolioEnv`:

```python
self.day = t
self.data = self.df.loc[self.day, :]      # this is a 27-row DataFrame (one row per ticker)
```

`self.data` is the full cross-section of the universe on day `t`: 27 rows, one for AAPL, one for AMGN, etc.

### Step 2 — Pull out the covariance matrix (one copy, not 27)

```python
self.covs = self.data["cov_list"].values[0]
```

- `self.data["cov_list"]` is a pandas Series of length 27 — but every element is the same 27×27 ndarray.
- `.values` turns it into a numpy array of length 27.
- `[0]` picks the first element, which is the 27×27 covariance ndarray.

So after this line, `self.covs` is a single **27×27 numpy array** — the covariance matrix that was computed from the previous 252 trading days' returns. The duplication in the DataFrame is harmless; only the first copy is ever read.

### Step 3 — Pull out the indicator values (one number per ticker per indicator)

```python
[self.data[tech].values.tolist() for tech in self.tech_indicator_list]
```

For each technical indicator in `tech_indicator_list` (there are 8: `macd`, `boll_ub`, `boll_lb`, `rsi_30`, `cci_30`, `dx_30`, `close_30_sma`, `close_60_sma`):

- `self.data[tech]` is a Series of length 27 — the value of that indicator for each of the 27 tickers on day `t`.
- `.values.tolist()` makes it a plain list of 27 numbers.

After the list comprehension you have a list of 8 lists, each of length 27. Conceptually, that is an **8×27** block: rows = indicators, columns = stocks.

### Step 4 — Stack into a single observation matrix

```python
self.state = np.append(
    np.array(self.covs),                                          # (27, 27)
    [self.data[tech].values.tolist() for tech in self.tech_indicator_list],  # (8, 27)
    axis=0,
)
```

`np.append(..., axis=0)` concatenates along the row axis, producing a **(27 + 8, 27) = (35, 27) array**:

```
        col 0 (AAPL)  col 1 (AMGN)  col 2 (AXP)   ...  col 26 (WMT)
row 0   cov(AAPL,AAPL) cov(AAPL,AMGN) cov(AAPL,AXP) ... cov(AAPL,WMT)   ← cov matrix
row 1   cov(AMGN,AAPL) cov(AMGN,AMGN) ...                               ← cov matrix
...                                                                     ← cov matrix
row 26  cov(WMT,AAPL)  cov(WMT,AMGN)  ...                               ← cov matrix
row 27  macd_AAPL      macd_AMGN      macd_AXP      ... macd_WMT        ← macd row
row 28  boll_ub_AAPL   boll_ub_AMGN   ...           ... boll_ub_WMT     ← bollinger upper
row 29  boll_lb_AAPL   ...                                              ← bollinger lower
row 30  rsi_30_AAPL    ...                                              ← rsi
row 31  cci_30_AAPL    ...                                              ← cci
row 32  dx_30_AAPL     ...                                              ← dx
row 33  sma_30_AAPL    ...                                              ← 30-day SMA
row 34  sma_60_AAPL    ...                                              ← 60-day SMA
```

That 35×27 matrix is `self.state`. It is **the observation** the environment hands to the agent on day `t`.

### Step 5 — How the policy network actually consumes the matrix

Stable Baselines 3 (the library training A2C/PPO/DDPG/SAC/TD3 here) uses an MLP policy by default. An MLP cannot accept a 2-D matrix directly, so before the input layer SB3 **flattens** the (35, 27) observation into a 1-D vector of length **945** (35 × 27). The order is row-major: first the 27 entries of cov-row 0, then the 27 entries of cov-row 1, … then macd_AAPL, macd_AMGN, …, macd_WMT, then bollinger upper, and so on.

So the policy net's input layer has 945 neurons. Out of those:

- The first 729 (= 27 × 27) carry the flattened covariance matrix.
- The remaining 216 (= 8 × 27) carry the eight per-ticker indicator rows.

The output layer has 27 neurons — one per ticker — producing the raw action vector.

### Step 6 — From raw action to weights

```python
def softmax_normalization(self, actions):
    return np.exp(actions) / np.sum(np.exp(actions))
```

The env applies softmax to the 27-dim action so that the result is non-negative and sums to 1. Those 27 numbers are the **portfolio weights** for day `t+1`.

### Step 7 — Reward

```python
portfolio_return = sum(((self.data.close.values / last_day_memory.close.values) - 1) * weights)
new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
self.reward = new_portfolio_value
```

The reward is the new portfolio value — i.e., the previous value times one plus the weighted average of per-stock daily returns. This is the signal gradients flow back from.

### Putting it together — what the covariance actually *does*

There is **no Markowitz formula** in any of this. The agent does not invert the covariance or compute an efficient frontier. The covariance is just an input feature — 729 numbers fed into the MLP every day.

So how does it influence the weights? Through the training loop:

- On each day, the policy sees the 945-vector (covariance + indicators) and outputs 27 weights.
- The market then pays out a reward equal to the portfolio's next-day value.
- Over thousands of training steps, gradient descent adjusts the network weights so that, on average, the action vectors the network produces yield higher rewards.
- If two stocks have a high covariance entry, equal weights on both is *not* diversification — it concentrates risk. Periods where the agent happens to over-weight such pairs and gets hit by a joint drawdown produce lower rewards; gradient descent pushes the network to react to those covariance entries by spreading weight elsewhere.
- Conversely, when the covariance block shows benign, low-correlation structure, the network is free to concentrate without being penalised by joint drawdowns.

Diversification is therefore **emergent**, not enforced. The covariance is the part of the observation that makes risk structure visible to the policy; whether the policy actually learns to use it depends on how informative it is for predicting reward. This is the conceptual contrast with the Min-Variance baseline later in the notebook, which feeds the same covariance into a quadratic optimiser and gets weights from a closed-form solution.

This is the conceptual difference from classical mean–variance: classical optimisation uses the covariance algebraically (`w = Σ^{-1} μ / ...`); the DRL agent uses it as a feature and lets gradient descent figure out the mapping.

The notebook's Min-Variance baseline (later in the same file) makes the contrast explicit: it uses `EfficientFrontier(None, Sigma).min_volatility()` from PyPortfolioOpt, i.e. the classical closed-form approach driven by the same covariance matrix. The DRL agent's daily weights are then compared against this baseline and against the DJIA.

---

## 5. Environment configuration

```python
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "transaction_cost_pct": 0.001,
    "state_space": 27,          # = number of stocks
    "stock_dim": 27,
    "tech_indicator_list": config.INDICATORS,
    "action_space": 27,
    "reward_scaling": 1e-4,
}
```

Two things worth noting:

- `action_space = Box(low=0, high=1, shape=(27,))` — but with softmax normalisation in `step()`, the actual scale of the raw action does not matter; only the relative magnitudes do.
- The reward in code is `new_portfolio_value` (not `Δ portfolio_value`). The `reward_scaling` field is plumbed through but commented out in the active reward line — so the absolute portfolio value is what propagates to the optimiser. In practice this still produces a usable training signal because the network sees a monotonically-growing or shrinking reward stream proportional to performance.

---

## 6. Models trained

All from `stable_baselines3`, wrapped by FinRL's `DRLAgent`. All are continuous-action algorithms suitable for a continuous weight vector:

| Model | Type | Key hyperparameters in the notebook | Timesteps |
| --- | --- | --- | --- |
| A2C | On-policy, advantage actor-critic | n_steps=5, ent_coef=0.005, lr=2e-4 | 50,000 |
| PPO | On-policy, clipped policy gradient | n_steps=2048, ent_coef=0.005, lr=1e-4, batch=128 | 80,000 |
| DDPG | Off-policy, deterministic actor-critic | buffer=50k, batch=128, lr=1e-3 | 50,000 |
| SAC | Off-policy, max-entropy actor-critic | buffer=100k, batch=128, lr=3e-4, ent_coef=auto_0.1 | 50,000 |
| TD3 | Off-policy, twin-delayed DDPG | buffer=1M, batch=100, lr=1e-3 | 30,000 |

The notebook trains all five but only **A2C** is used for the trading/backtest stage:

```python
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c, environment=e_trade_gym)
```

---

## 7. Backtest and evaluation

Out-of-sample period 2020-07-01 → 2021-10-31, initial capital $1,000,000. The notebook computes:

- DRL strategy stats via `pyfolio.timeseries.perf_stats` — annual return, Sharpe, Calmar, Sortino, max drawdown, etc.
- A DJIA (`^DJI`) baseline pulled from Yahoo with the same window.
- A **Min-Variance** classical baseline computed daily via PyPortfolioOpt:

```python
Sigma = df_temp.return_list[0].cov()                     # covariance from the stored return panel
ef_min_var = EfficientFrontier(None, Sigma, weight_bounds=(0, 0.1))
raw_weights_min_var = ef_min_var.min_volatility()        # closed-form min-variance weights
```

Same covariance input, two very different consumers: the DRL agent (uses it as a feature) versus PyPortfolioOpt (uses it as the literal objective).

A Plotly chart at the end compares cumulative returns of A2C, Min-Variance, and DJIA.

Reported run from the notebook:

- A2C: annual return ≈ 31.6%, Sharpe ≈ 2.14, max drawdown ≈ -7.7%, end value $1,443,911.
- DJIA: annual return ≈ 27.9%, Sharpe ≈ 1.84, max drawdown ≈ -8.9%.

---

## 8. Putting the moving parts together (one-page mental model)

1. Download OHLCV for 27 Dow stocks from Yahoo.
2. Add technical indicators (MACD, Bollinger Bands, RSI, CCI, DX, 30/60 SMA).
3. For each day, compute a 252-day rolling covariance matrix of stock returns.
4. Stack `[covariance matrix | indicator rows]` into a 35×27 observation for the agent.
5. Wrap as a Gym environment whose action is `R^27` and whose reward is the next-day portfolio value.
6. Train an actor-critic agent (A2C, PPO, DDPG, SAC, TD3) on 2009-07 → 2020-07 data.
7. At test time, on each day the agent emits a raw 27-vector; softmax turns it into long-only, fully-invested weights; portfolio is repriced; repeat.
8. Compare the cumulative return curve against DJIA and a covariance-driven Min-Variance allocator.

The covariance matrix's role is therefore twofold in the same notebook:

- **For the DRL agent:** an input feature that exposes the cross-asset risk structure so the policy network can learn diversification implicitly through reward maximisation.
- **For the Min-Variance baseline:** the direct argument to a quadratic optimiser whose closed-form solution is the minimum-variance long-only portfolio.

The comparison between the two is the empirical point of the notebook: can a DRL agent that *sees* the covariance match or beat a classical allocator that *optimises* the covariance?
