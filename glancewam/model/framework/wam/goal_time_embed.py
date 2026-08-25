"""Sinusoidal embedding of the goal-time difference g.

``GlanceWAM`` adds this to the goal tokens so the action head knows how
far ahead the goal frame is: g is drawn from U(0, H_g] during training and shrinks through
each refresh cycle at inference.
"""

from __future__ import annotations

import math

import torch


def sinusoidal_h_embed(h_rows: torch.Tensor, dim: int) -> torch.Tensor:
    """Horizon H (native rows, shape (...,)) -> (..., dim) sinusoidal embedding — the
    standard transformer formulation, period range sized for H in [1, ~100]."""
    assert dim % 2 == 0, f"h_embed dim must be even, got {dim}"
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=h_rows.device, dtype=torch.float32) / half)
    angles = h_rows.float().unsqueeze(-1) * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)
