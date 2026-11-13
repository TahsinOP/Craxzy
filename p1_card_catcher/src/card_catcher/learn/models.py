"""Sequence models with heteroscedastic Gaussian heads (Kendall & Gal, 2017).

Both predictors share one architecture: a GRU reads the track one frame at a
time; from its hidden state at frame k an MLP decodes, for each of the next
``horizon`` steps and each axis, a mean and a log-variance.

* ``kind = "residual"`` (the hybrid): inputs are features of the physics EKF
  (filtered velocity, normalised innovation, validity, drag estimate); outputs
  are the residual accelerations r_{k+j}, j = 0..horizon-1, in units of
  ``scale`` (a scalar, the residual's std on the training set).
* ``kind = "pure"`` (no physics): inputs are raw measurement features; outputs
  are future displacements p_{k+j} - z_anchor, j = 1..horizon, in units of
  ``scale[j]`` (a fixed ramp).  Gravity, drag and the integrator are not given.

Loss: mean over valid (k, j, axis) of  0.5 * [ logvar + (y - mu)^2 exp(-logvar) ].
"""
from __future__ import annotations

import torch
from torch import nn

LOGVAR_MIN, LOGVAR_MAX = -9.0, 6.0


class SeqPredictor(nn.Module):
    def __init__(self, n_in: int, horizon: int, hidden: int = 128, kind: str = "residual"):
        super().__init__()
        self.kind, self.horizon, self.hidden = kind, horizon, hidden
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.dec = nn.Sequential(nn.Linear(hidden, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, horizon * 6))
        self.register_buffer("scale", torch.ones(horizon))     # output units (per horizon step)

    def encode(self, feats: torch.Tensor, h0: torch.Tensor | None = None):
        """feats (B, K, n_in) -> hidden states (B, K, hidden) and the last state (1, B, hidden)."""
        return self.gru(self.inp(feats), h0)

    def decode(self, h: torch.Tensor):
        """h (..., hidden) -> mean, logvar of shape (..., horizon, 3), in normalised units."""
        o = self.dec(h).reshape(*h.shape[:-1], self.horizon, 6)
        return o[..., :3], o[..., 3:].clamp(LOGVAR_MIN, LOGVAR_MAX)

    def decode_physical(self, h: torch.Tensor):
        """Mean and standard deviation in physical units."""
        mu, lv = self.decode(h)
        s = self.scale[:, None]
        return mu * s, torch.exp(0.5 * lv) * s


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean of the per-element Gaussian negative log-likelihood (without the constant)."""
    nll = 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))
    m = mask.expand_as(nll)
    return (nll * m).sum() / m.sum().clamp(min=1)
