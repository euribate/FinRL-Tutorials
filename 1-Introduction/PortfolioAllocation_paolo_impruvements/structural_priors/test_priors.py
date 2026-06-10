"""Sanity tests for the structural-priors helpers (METHODOLOGY.md sec. 5.1).

Four tests, each printing inputs / outputs / PASS-FAIL:

  1) _solve_risk_parity on a hand-picked 3-asset covariance returns weights
     whose RISK CONTRIBUTIONS are approximately equal (each ≈ 1/3).
  2) compute_prior_weights_from_cov returns a vector that sums to 1, with
     cash at the configured share and risky weights aligned with the
     configured type. (LogReturnPortfolioEnv._compute_prior_weights is a
     2-line wrapper around this helper, so testing the helper tests the
     env's training-time prior too.)
  3) softmax(0 + alpha * log(w_prior)) == w_prior exactly when alpha=1.0.
     This is the algebraic identity that justifies the residual-policy
     formulation: at action=0 the agent's allocation IS the prior.
  4) softmax(a + alpha * log(w_eq)) == softmax(a) for ANY action vector a
     and ANY alpha. This pins the equal_weight prior's mathematical no-op
     property: with uniform w_prior, log(w_prior) is a constant vector and
     softmax is shift-invariant, so the injection has no effect on the
     resulting allocation. Documented in METHODOLOGY.md sec. 6.5.

Run from this folder:

    source ../../../.venv/bin/activate
    python test_priors.py
"""
import json
import numpy as np

from utils import _solve_risk_parity, compute_prior_weights_from_cov


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)            # numerical stability
    e = np.exp(x)
    return e / e.sum()


# ─────────────────────────────────────────────────────────────────────────
# TEST 1 — _solve_risk_parity equal-risk-contribution check
# ─────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("TEST 1 — _solve_risk_parity on hand-picked 3-asset covariance")
print("=" * 70)

sigmas = np.array([0.05, 0.20, 0.10])                                # vols
rho    = np.array([[1.0, 0.2, 0.1],
                   [0.2, 1.0, 0.3],
                   [0.1, 0.3, 1.0]])                                 # corr
cov3   = (sigmas[:, None] * sigmas[None, :]) * rho

w_rp     = _solve_risk_parity(cov3)
sigma_w  = cov3 @ w_rp
rc_frac  = (w_rp * sigma_w) / float((w_rp * sigma_w).sum())

print(f"  asset volatilities:    {sigmas}")
print(f"  RP weights:            {np.round(w_rp, 4)}   sum = {w_rp.sum():.6f}")
print(f"  risk contributions:    {np.round(rc_frac, 4)}   (each should be 1/3 = 0.3333)")
ok_1a = bool(np.allclose(rc_frac, 1.0 / 3.0, atol=1e-4))

# Test 1b: NEGATIVE correlations + asymmetric vols — this regression case
# corresponds to a real financial covariance (bonds with negative correlation
# to equities). The previous iterative solver silently produced corner
# solutions on this kind of input; the SLSQP/multi-start solver should not.
print()
print("  Regression: negative correlations + asymmetric vols")
sigmas_b = np.array([0.005, 0.15, 0.012])              # SHY-like, IVE-like, TLT-like
rho_b    = np.array([[ 1.0, -0.1,  0.6],
                     [-0.1,  1.0, -0.3],
                     [ 0.6, -0.3,  1.0]])
cov_b    = (sigmas_b[:, None] * sigmas_b[None, :]) * rho_b
w_rp_b   = _solve_risk_parity(cov_b)
rc_b     = w_rp_b * (cov_b @ w_rp_b)
rc_frac_b = rc_b / float(rc_b.sum())
print(f"  asset volatilities:    {sigmas_b}")
print(f"  RP weights:            {np.round(w_rp_b, 4)}   sum = {w_rp_b.sum():.6f}")
print(f"  risk contributions:    {np.round(rc_frac_b, 4)}   (each should be 1/3 = 0.3333)")
ok_1b = bool(np.allclose(rc_frac_b, 1.0 / 3.0, atol=1e-3))

ok_1 = ok_1a and ok_1b
print(f"  RESULT:                {'PASS' if ok_1 else 'FAIL'} "
      f"(positive-corr={ok_1a}, negative-corr={ok_1b})")


# ─────────────────────────────────────────────────────────────────────────
# TEST 2 — compute_prior_weights_from_cov shape, sum, cash, type alignment
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 2 — compute_prior_weights_from_cov (env._compute_prior_weights")
print("         delegates to this; testing the helper tests both)")
print("=" * 70)

cfg     = json.load(open("config.json"))
pp_cfg  = cfg.get("policy_prior", {}) or {}
ptype   = str(pp_cfg.get("type", "inverse_vol"))
alpha   = float(pp_cfg.get("alpha", 1.0))
cs_cfg  = pp_cfg.get("cash_share", None)
cs_cfg  = None if cs_cfg is None else float(cs_cfg)

# 4-asset toy: 3 risky + 1 synthetic CASH (zero variance — mimics what
# stage 1 injects). Cash MUST be held out of the covariance-based formula
# or inverse_vol diverges (1/0).
n         = 4
cash_idx  = 3
diag_vals = np.array([0.0025, 0.04, 0.01, 1e-12])    # 5%/20%/10% vol + cash
cov4      = np.diag(diag_vals)

w_prior        = compute_prior_weights_from_cov(cov4, ptype, cash_idx=cash_idx, cash_share=cs_cfg)
expected_cash  = cs_cfg if cs_cfg is not None else (1.0 / n)
expected_risky = 1.0 - expected_cash

print(f"  configured type:        '{ptype}'   alpha = {alpha}")
print(f"  configured cash_share:  {cs_cfg}   (None means 1/N = {1.0 / n})")
print(f"  resulting weights:      {np.round(w_prior, 4)}   sum = {w_prior.sum():.6f}")
print(f"  cash weight:            {w_prior[cash_idx]:.4f}   expected {expected_cash:.4f}")
print(f"  risky weights sum:      {w_prior.sum() - w_prior[cash_idx]:.4f}   expected {expected_risky:.4f}")

# Type-specific alignment check.
if ptype in ("inverse_vol", "risk_parity"):
    # On a DIAGONAL covariance these reduce to the same formula:
    # w_i ∝ 1 / sqrt(diag_i). The asset with the lowest vol gets the
    # largest weight; with the highest vol the smallest.
    risky_w     = np.array([w_prior[0], w_prior[1], w_prior[2]])
    risky_vols  = np.sqrt(diag_vals[:3])
    expected_iv = (1.0 / risky_vols) / (1.0 / risky_vols).sum() * expected_risky
    print(f"  expected inverse-vol risky weights = {np.round(expected_iv, 4)}")
    ok_type = bool(np.allclose(risky_w, expected_iv, atol=1e-4))
elif ptype == "equal_weight":
    expected_eq = np.full(3, expected_risky / 3.0)
    print(f"  expected equal_weight risky weights = {np.round(expected_eq, 4)}")
    ok_type = bool(np.allclose(w_prior[:3], expected_eq, atol=1e-6))
else:  # 'none' or unknown — falls back to equal_weight
    print("  (type 'none' or unknown — falls back to equal_weight)")
    ok_type = True

ok_sum  = bool(abs(w_prior.sum() - 1.0) < 1e-8)
ok_cash = bool(abs(w_prior[cash_idx] - expected_cash) < 1e-6)
ok_pos  = bool(np.all(w_prior > 0.0))
ok_2    = ok_sum and ok_cash and ok_pos and ok_type
print(f"  RESULT:                 {'PASS' if ok_2 else 'FAIL'} "
      f"(sum={ok_sum}, cash={ok_cash}, positive={ok_pos}, type-aligned={ok_type})")


# ─────────────────────────────────────────────────────────────────────────
# TEST 3 — action = 0 identity: softmax(alpha * log(w_prior)) == w_prior
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 3 — softmax(0 + alpha=1.0 * log(w_prior)) == w_prior (identity)")
print("=" * 70)

action_zero    = np.zeros(n)
logits_a1      = action_zero + 1.0 * np.log(w_prior)
w_after_a1     = softmax(logits_a1)
max_diff_a1    = float(np.max(np.abs(w_after_a1 - w_prior)))

print(f"  w_prior:                {np.round(w_prior, 6)}")
print(f"  softmax(log(w_prior)):  {np.round(w_after_a1, 6)}")
print(f"  max |difference|:       {max_diff_a1:.3e}   (should be ~ 0)")
ok_3 = bool(np.allclose(w_after_a1, w_prior, atol=1e-10))
print(f"  RESULT:                 {'PASS' if ok_3 else 'FAIL'}")

# Context: show what happens at other alpha values so the identity is clear.
print()
print("  For context — same input, different alpha:")
for a in (0.0, 0.5, 2.0):
    w_a = softmax(action_zero + a * np.log(w_prior))
    print(f"    alpha={a:>3.1f}:  weights = {np.round(w_a, 4)}   "
          f"({'flat 1/N' if a == 0 else 'flatter' if a < 1 else 'more peaked'} than w_prior)")


# ─────────────────────────────────────────────────────────────────────────
# TEST 4 — equal_weight prior is a softmax-shift-invariant no-op
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("TEST 4 — softmax(a + alpha*log(w_eq)) == softmax(a) for any a, any alpha")
print("=" * 70)

# Three sanity slices: small alpha, alpha=1, large alpha. Three random
# action vectors of varying scale (small / unit / large) so we catch any
# numerical issue at the extremes too.
rng    = np.random.default_rng(0)
N      = 7  # arbitrary asset count
w_eq   = np.full(N, 1.0 / N)

max_diff_4 = 0.0
for action_scale in (0.1, 1.0, 5.0):
    a = rng.normal(scale=action_scale, size=N)
    base = softmax(a)
    for alpha in (0.25, 1.0, 3.0):
        injected = softmax(a + alpha * np.log(w_eq))
        diff = float(np.max(np.abs(injected - base)))
        max_diff_4 = max(max_diff_4, diff)

# Diagnostic: confirm log(w_eq) is a constant vector (the algebraic source).
log_we = np.log(w_eq)
print(f"  log(1/N) =              {np.round(log_we, 4)}   "
      f"(constant: var = {float(log_we.var()):.3e})")
print(f"  softmax(a + alpha*log(w_eq)) vs softmax(a):")
print(f"    max |difference| over (action_scale, alpha) sweep: "
      f"{max_diff_4:.3e}   (should be ~ 0)")
ok_4 = bool(max_diff_4 < 1e-12)
print(f"  RESULT:                 {'PASS' if ok_4 else 'FAIL'}")


# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print(f"SUMMARY:   test 1 = {'PASS' if ok_1 else 'FAIL'}    "
      f"test 2 = {'PASS' if ok_2 else 'FAIL'}    "
      f"test 3 = {'PASS' if ok_3 else 'FAIL'}    "
      f"test 4 = {'PASS' if ok_4 else 'FAIL'}")
print("=" * 70)

if not (ok_1 and ok_2 and ok_3 and ok_4):
    raise SystemExit(1)
