# Copyright 2025 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""
Exponential Moving Average (EMA) of model weights.

Maintains a parallel, slowly-updated average of the *trainable* parameters during
training. EMA does not touch the live weights or optimizer state — it only keeps a
second copy — so the optimization trajectory is unchanged. At checkpoint time the
trainer writes both the live model and an EMA model (identical key structure), so
evaluation can choose between them by pointing the checkpoint path at the `*_ema.*`
file.

Design notes
------------
- Only parameters with ``requires_grad`` are tracked. Frozen parameters never
  change, so their EMA equals the frozen value; tracking them would only waste
  memory. At save time the EMA tensors are merged on top of the full live
  state_dict (:meth:`merge_into`) so the saved file keeps the complete, identical
  key set and loads under ``strict=True``.
- Under DeepSpeed ZeRO-2 parameters are *replicated* across ranks (only optimizer
  state and gradients are partitioned), so the EMA can be maintained entirely on
  rank 0 by reading the unwrapped model's live params — no gather needed.
- The EMA master copy is kept in fp32. Training params are bf16; accumulating an
  EMA at decay ~0.999 in bf16 would underflow the small per-step deltas.
"""

import torch


class EMAModel:
    """Tracks an fp32 exponential moving average of a model's trainable params.

    Args:
        named_trainable_params: iterable of ``(name, param)`` for params to track
            (typically the ``requires_grad`` subset of ``model.named_parameters()``).
        decay: target EMA decay (momentum), e.g. ``0.999``.
        warmup: if True, ramp the effective decay early via ``(1 + t) / (10 + t)``
            so the average isn't dominated by the noisy initial weights.
        device: where to hold the EMA buffers — ``"cuda"`` (fast, ~param-size VRAM)
            or ``"cpu"`` (saves VRAM, adds a per-step device-to-host copy).
    """

    def __init__(self, named_trainable_params, decay=0.999, warmup=True, device="cuda"):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.device = torch.device(device)
        # fp32 detached snapshots, keyed by parameter name.
        self.shadow = {name: param.detach().clone().float().to(self.device) for name, param in named_trainable_params}

    def _current_decay(self, step):
        if self.warmup:
            return min(self.decay, (1.0 + step) / (10.0 + step))
        return self.decay

    @torch.no_grad()
    def update(self, model, step):
        """Pull the live trainable params from ``model`` into the EMA shadow.

        ``model`` should be the unwrapped framework (so parameter names match the
        keys captured at construction). Params not in the shadow are ignored.
        """
        d = self._current_decay(step)
        for name, param in model.named_parameters():
            shadow = self.shadow.get(name)
            if shadow is None:
                continue
            new = param.detach().to(self.device, dtype=torch.float32)
            # shadow = d * shadow + (1 - d) * new
            shadow.mul_(d).add_(new, alpha=1.0 - d)

    def merge_into(self, base_state_dict, dtype=None):
        """Return a copy of ``base_state_dict`` with tracked keys overwritten by EMA.

        Args:
            base_state_dict: the full live model state_dict (all keys, e.g. from
                ``accelerator.get_state_dict(model)``).
            dtype: cast the merged EMA tensors to this dtype (typically the model's
                param dtype) so the saved file matches the live checkpoint exactly.

        Returns:
            A new dict with the same keys as ``base_state_dict``; tracked keys hold
            EMA values, the rest are the live values unchanged.
        """
        merged = dict(base_state_dict)
        for name, shadow in self.shadow.items():
            if name not in merged:
                continue
            ref = base_state_dict[name]
            target_dtype = dtype if dtype is not None else ref.dtype
            merged[name] = shadow.detach().to(device=ref.device, dtype=target_dtype)
        return merged

    def state_dict(self):
        """For resume: the fp32 shadow tensors keyed by param name."""
        return {name: shadow.clone() for name, shadow in self.shadow.items()}

    def load_state_dict(self, state_dict):
        """For resume: restore shadow tensors (matched by name; extras ignored)."""
        for name, shadow in self.shadow.items():
            saved = state_dict.get(name)
            if saved is not None:
                shadow.copy_(saved.to(self.device, dtype=torch.float32))
