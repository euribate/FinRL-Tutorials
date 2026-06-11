"""Training callbacks.

ValidationIRCallback is the only callback. It computes annualised
Information Ratio of the strategy's net returns vs the benchmark_return
column on the validation slice, every `eval_freq` timesteps. The best
model so far is saved; if no improvement over `patience` consecutive
evaluations, training stops early.

Differences from the legacy ValidationSharpeCallback:
  * Single selection metric (IR). The sharpe selector was the source
    of the EW-anchored checkpoint pathology in the triage.
  * No VecNormalize plumbing — the new env exposes already-normalised
    per-asset features (rank-normalisation lives in data.py), so the
    callback can roll the policy through a fresh env directly.
  * Score field renamed: history JSON stores `ir` not `sharpe`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from env import PortfolioEnv


class ValidationIRCallback(BaseCallback):
    def __init__(self,
                 val_df: pd.DataFrame,
                 feature_cols: list[str],
                 env_kwargs: dict,
                 model_save_path: Path,
                 history_path: Path,
                 eval_freq: int = 2500,
                 patience: int = 20,
                 min_delta: float = 0.001,
                 verbose: int = 1):
        super().__init__(verbose)
        self.val_df          = val_df
        self.feature_cols    = list(feature_cols)
        self.env_kwargs      = dict(env_kwargs)
        self.model_save_path = Path(model_save_path)
        self.history_path    = Path(history_path)
        self.eval_freq       = int(eval_freq)
        self.patience        = int(patience)
        self.min_delta       = float(min_delta)

        self.best_ir              = -math.inf
        self.no_improvement_count = 0
        self.history: list[dict]  = []
        self._next_eval           = self.eval_freq
        self._saved_any           = False

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval += self.eval_freq

        ir = self._evaluate_ir()
        improved = ir > (self.best_ir + self.min_delta)
        if improved:
            prev = self.best_ir
            self.best_ir = ir
            self.no_improvement_count = 0
            self.model.save(str(self.model_save_path))
            self._saved_any = True
            if self.verbose:
                prev_str = "-inf" if prev == -math.inf else f"{prev:.4f}"
                print(f"  [step {self.num_timesteps}] val IR improved: "
                      f"{prev_str} -> {ir:.4f}, saved best to "
                      f"{self.model_save_path.name}")
        else:
            self.no_improvement_count += 1
            if self.verbose:
                print(f"  [step {self.num_timesteps}] val IR={ir:.4f}  "
                      f"(best={self.best_ir:.4f}, no-improvement "
                      f"{self.no_improvement_count}/{self.patience})")

        self.history.append({
            "timesteps":   int(self.num_timesteps),
            "ir":          float(ir),
            "improved":    bool(improved),
            "best_so_far": float(self.best_ir),
        })

        if self.no_improvement_count >= self.patience:
            if self.verbose:
                print(f"  Early stopping: {self.patience} consecutive "
                      f"evaluations without improvement >= {self.min_delta}.")
            self._dump_history()
            return False
        return True

    def _on_training_end(self) -> None:
        self._dump_history()
        if not self._saved_any:
            self.model.save(str(self.model_save_path))
            if self.verbose:
                print(f"  No improvement detected; saved final policy to "
                      f"{self.model_save_path.name}.")

    def _dump_history(self) -> None:
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            print(f"  WARNING: could not write history to "
                  f"{self.history_path}: {e}")

    def _evaluate_ir(self) -> float:
        """Roll the current policy through the val env, compute IR vs benchmark.

        Returns -inf if the rollout produced fewer than 30 daily returns
        (cannot estimate IR reliably).
        """
        env = PortfolioEnv(self.val_df, self.feature_cols, **self.env_kwargs)
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _r, terminated, _truncated, _ = env.step(action)
        ret_df = env.save_asset_memory()
        if len(ret_df) < 30:
            return -math.inf
        # Active return vs benchmark for THIS val window.
        ret_df["date"] = pd.to_datetime(ret_df["date"])
        bench_by_date = (self.val_df[["date", "benchmark_return"]]
                         .drop_duplicates("date").copy())
        bench_by_date["date"] = pd.to_datetime(bench_by_date["date"])
        merged = ret_df.merge(bench_by_date, on="date", how="inner")
        if len(merged) < 30:
            return -math.inf
        active = (merged["daily_return"] - merged["benchmark_return"]).to_numpy(
                  dtype=float)
        std = float(active.std(ddof=1))
        if std < 1e-12:
            return -math.inf
        return float(np.sqrt(252.0) * float(active.mean()) / std)
