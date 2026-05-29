"""Custom SB3 policy components for the portfolio-allocation pipeline.

`PerAssetSharedEncoder` replaces the default flatten-then-MLP feature extractor
with the per-asset shared-weight encoder discussed in README A.22 (the
load-bearing half of the article's LSTM/Transformer architecture, minus the
expensive temporal part).

The problem it fixes
--------------------
The env state is a (n_rows, n_assets) matrix whose COLUMN j is asset j's full
feature vector (its covariance row + indicators). SB3's default `MlpPolicy`
flattens this to one long vector, scattering each asset's features across the
input and giving the network no notion of "asset". The optimiser's easy
solution is near-constant logits -> equal weight (empirically: per-asset
weights end up uncorrelated with per-asset features).

What this does instead
----------------------
Apply the SAME small MLP to EACH asset's feature column (weight sharing), so
every asset is scored by one shared rule of its OWN features:

    obs (B, n_rows, n_assets)
      -> transpose -> (B, n_assets, n_rows)
      -> shared MLP f: n_rows -> hidden -> hidden -> emb_dim   (reused per asset)
      -> (B, n_assets, emb_dim)
      -> [optional] subtract cross-asset mean  (cross-sectional comparison)
      -> flatten -> (B, n_assets * emb_dim)   = the extracted features

With the recommended `emb_dim=1`, the extractor emits ONE score per asset
(features_dim = n_assets); pair it with `net_arch={"pi": [], "vf": [64,64]}` so
the policy head is a minimal `Linear(n_assets -> n_assets)` on already-per-asset,
cross-sectionally-aware scores, while the value head keeps capacity.

Inductive-bias win: the network learns a single "features -> score" function
(updated with n_assets samples per step) and asset j's score depends only on
asset j's features -> it CAN differentiate. Caveat: SB3's final action head is
still a dense Linear, so there is a small residual cross-asset mix at the very
last layer; with emb_dim=1 it is an 11x11 map the optimiser can drive toward
identity. For a fully per-asset action head you'd need a custom policy class
(deferred - this extractor is the cheap, policy_kwargs-selectable first step).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PerAssetSharedEncoder(BaseFeaturesExtractor):
    """Shared-weight per-asset encoder over a (n_rows, n_assets) Box observation.

    Args (via policy_kwargs.features_extractor_kwargs):
      hidden:         width of the shared MLP's hidden layers (default 32).
      emb_dim:        per-asset output width (default 1 = a scalar score; the
                      features_dim becomes n_assets * emb_dim).
      cross_sectional: if True (default), subtract the cross-asset mean of the
                      per-asset embeddings so each asset is scored relative to
                      the day's cross-section.
      activation:     torch.nn activation name (default "ReLU").
      layers:         number of hidden layers in the shared MLP (default 2).
    """

    def __init__(self, observation_space: spaces.Box,
                 hidden: int = 32,
                 emb_dim: int = 1,
                 cross_sectional: bool = True,
                 activation: str = "ReLU",
                 layers: int = 2):
        if len(observation_space.shape) != 2:
            raise ValueError(
                f"PerAssetSharedEncoder expects a 2-D (n_rows, n_assets) "
                f"observation; got shape {observation_space.shape}. The env must "
                f"expose the un-flattened covariance+indicator matrix."
            )
        n_rows, n_assets = observation_space.shape
        super().__init__(observation_space, features_dim=n_assets * emb_dim)
        self.n_rows          = int(n_rows)
        self.n_assets        = int(n_assets)
        self.emb_dim         = int(emb_dim)
        self.cross_sectional = bool(cross_sectional)

        if not hasattr(nn, activation):
            raise ValueError(f"Unknown activation {activation!r}")
        act = getattr(nn, activation)

        dims = [self.n_rows] + [hidden] * int(layers)
        mods: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), act()]
        mods += [nn.Linear(dims[-1], self.emb_dim)]
        self.shared = nn.Sequential(*mods)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: (B, n_rows, n_assets)
        x = observations.transpose(1, 2)                 # (B, n_assets, n_rows)
        b, a, f = x.shape
        x = x.reshape(b * a, f)                          # (B*n_assets, n_rows)
        z = self.shared(x)                               # (B*n_assets, emb_dim)
        z = z.reshape(b, a, self.emb_dim)                # (B, n_assets, emb_dim)
        if self.cross_sectional:
            z = z - z.mean(dim=1, keepdim=True)          # centre across assets
        return z.reshape(b, a * self.emb_dim)            # (B, features_dim)


FEATURE_EXTRACTORS = {
    "per_asset": PerAssetSharedEncoder,
}
