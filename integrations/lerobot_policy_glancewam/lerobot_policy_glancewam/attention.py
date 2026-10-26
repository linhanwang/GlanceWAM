#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Goal-aware prefix-LM FlexAttention for the SkyReels-V2 diffusion-forcing DiT.

Video+action co-training wants the clean conditioning frames (per-frame timestep 0) isolated from the
noised future frames, so the ``t=0`` feature read-point is independent of the future noise sample, while
the noised frames still attend the clean prefix and each other. The goal (lookahead) frame adds a third
class: grounded on obs, attended by nothing. Query -> key classes, with ``clean = (timestep == 0)`` and
the last ``n_goal_frames`` frames the goal:

    obs    (clean, non-goal) -> obs
    future (noised)          -> obs + future
    goal   (clean, goal)     -> obs + goal

A dense mask through SDPA falls off FlashAttention onto the mem-efficient kernel; FlexAttention runs the
structured mask on a fused kernel that skips fully-masked blocks, so it is faster than full attention.
The per-sample ``BlockMask`` is built once per forward from the timestep matrix (outside any compiled
region) and threaded into every block as its ``attention_mask`` argument, so under per-block
``torch.compile`` it is a shape-guarded graph input. With ``n_goal_frames == 0`` the mask reduces to the
2-class prefix-LM rule, and at feature extraction with every frame clean it degenerates to full
attention. Vendored from GlanceWAM's ``speedup_patch.py``.
"""

from __future__ import annotations

import types
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import torch
from lerobot.utils.import_utils import _diffusers_available
from torch import nn

if TYPE_CHECKING or _diffusers_available:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_skyreels_v2 import (
        SkyReelsV2AttnProcessor,
        _get_qkv_projections,
    )
else:  # pragma: no cover - import guard
    Transformer2DModelOutput = None
    SkyReelsV2AttnProcessor = object
    _get_qkv_projections = None

_SDPA_BACKENDS = {
    "cudnn": "CUDNN_ATTENTION",
    "flash": "FLASH_ATTENTION",
    "mem_efficient": "EFFICIENT_ATTENTION",
    "math": "MATH",
}


def sdpa_context(backend: str | None):
    """A ``torch.nn.attention.sdpa_kernel`` context pinning `backend`, or a null context for ``auto``.

    The toggles ``torch.backends.cuda.enable_*_sdp`` apply a stricter check than the forced
    ``sdpa_kernel([...])`` path and were seen to silently fall back to the math kernel; the context
    manager dispatches reliably. It is applied *inside* the attention processor, so a checkpointed
    recompute re-enters it and the saved-tensor metadata matches the forward.
    """
    attr = _SDPA_BACKENDS.get((backend or "auto").lower())
    if attr is None:
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    return sdpa_kernel([getattr(SDPBackend, attr)])


_compiled_flex_attention = None


def get_compiled_flex_attention():
    """The process-wide compiled ``flex_attention`` (a fused kernel needs compilation)."""
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def build_goal_prefix_lm_block_mask(
    timestep: torch.Tensor,
    n_goal_frames: int,
    post_frames: int,
    frame_tok: int,
    device: torch.device | str,
):
    """Per-sample 3-class (obs / noised future / clean goal) prefix-LM ``BlockMask``.

    Args:
        timestep: ``[B, T_lat]`` per-frame timesteps; ``0`` marks a clean frame.
        n_goal_frames: The trailing frames that are the clean goal side-channel.
        post_frames: Latent frames after temporal patching.
        frame_tok: Tokens per latent frame.
        device: Device of the mask.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    clean = timestep == 0  # [B, T_lat]
    n_frames = int(post_frames)
    goal_start = n_frames - int(n_goal_frames)
    max_frame = n_frames - 1

    def goal_prefix_lm(b, h, q_idx, kv_idx):
        qf = torch.clamp(q_idx // frame_tok, max=max_frame)
        kf = torch.clamp(kv_idx // frame_tok, max=max_frame)
        q_goal = qf >= goal_start
        kv_goal = kf >= goal_start
        q_clean = clean[b, qf]
        kv_clean = clean[b, kf]
        q_obs = q_clean & (~q_goal)
        q_future = ~q_clean
        kv_obs = kv_clean & (~kv_goal)
        kv_future = ~kv_clean
        return (q_obs & kv_obs) | (q_future & (kv_obs | kv_future)) | (q_goal & (kv_obs | kv_goal))

    seq_len = n_frames * frame_tok
    return create_block_mask(
        goal_prefix_lm, B=int(timestep.shape[0]), H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )


class PrefixLMFlexAttnProcessor:
    """Drop-in SkyReels-V2 attention processor.

    Self-attention runs through FlexAttention when a ``BlockMask`` is threaded in via
    ``attention_mask``; otherwise (cross-attention, or self-attention with no mask) it delegates to the
    stock diffusers processor, bit-exact, under the configured SDPA backend context.
    """

    def __init__(self, sdpa_backend: str | None = "auto"):
        self._stock = SkyReelsV2AttnProcessor()
        self.sdpa_backend = sdpa_backend

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        from torch.nn.attention.flex_attention import BlockMask

        if encoder_hidden_states is not None or getattr(attn, "add_k_proj", None) is not None:
            with sdpa_context(self.sdpa_backend):
                return self._stock(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if not isinstance(attention_mask, BlockMask):
            with sdpa_context(self.sdpa_backend):
                return self._stock(attn, hidden_states, None, attention_mask, rotary_emb)

        query, key, value = _get_qkv_projections(attn, hidden_states, hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs_cos, freqs_sin):
                x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(x)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(x)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = get_compiled_flex_attention()(q, k, v, block_mask=attention_mask)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def goal_prefix_lm_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    enable_diffusion_forcing=False,
    fps=None,
    return_dict=True,
    attention_kwargs=None,
    n_goal_frames=0,
):
    """``SkyReelsV2Transformer3DModel.forward`` (diffusers 0.39) with one deviation.

    In diffusion-forcing mode the per-sample 3-class goal prefix-LM ``BlockMask`` (built from
    ``timestep`` and ``n_goal_frames``) is threaded into the blocks' mask slot instead of diffusers'
    unused block-causal mask. Patchify, RoPE and unpatchify are stock. Re-sync this copy if the diffusers
    forward changes.
    """
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    causal_mask = None
    if (
        enable_diffusion_forcing
        and timestep.dim() == 2
        and not getattr(self, "_goal_bidirectional_ablation", False)
    ):
        causal_mask = build_goal_prefix_lm_block_mask(
            timestep,
            int(n_goal_frames),
            post_patch_num_frames,
            post_patch_height * post_patch_width,
            hidden_states.device,
        )

    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image
    )

    timestep_proj = timestep_proj.unflatten(-1, (6, -1))

    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    if self.config.inject_sample_info:
        fps = torch.tensor(fps, dtype=torch.long, device=hidden_states.device)

        fps_emb = self.fps_embedding(fps)
        if enable_diffusion_forcing:
            timestep_proj = timestep_proj + self.fps_projection(fps_emb).unflatten(1, (6, -1)).repeat(
                timestep.shape[1], 1, 1
            )
        else:
            timestep_proj = timestep_proj + self.fps_projection(fps_emb).unflatten(1, (6, -1))

    if enable_diffusion_forcing:
        b, f = timestep.shape
        temb = temb.view(b, f, 1, 1, -1)
        timestep_proj = timestep_proj.view(b, f, 1, 1, 6, -1)
        temb = temb.repeat(1, 1, post_patch_height, post_patch_width, 1).flatten(1, 3)
        timestep_proj = timestep_proj.repeat(1, 1, post_patch_height, post_patch_width, 1, 1).flatten(1, 3)
        timestep_proj = timestep_proj.transpose(1, 2).contiguous()

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask
            )
    else:
        for block in self.blocks:
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask
            )

    if temb.dim() == 2:
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
    elif temb.dim() == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(2) + temb.unsqueeze(1)).chunk(2, dim=1)
        shift, scale = shift.squeeze(1), scale.squeeze(1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def enable_goal_prefix_lm_flex_attention(transformer: nn.Module, sdpa_backend: str | None = "auto") -> int:
    """Install the FlexAttention processor on every attention module and bind the goal-aware forward.

    Returns:
        int: The number of attention processors set.
    """
    transformer.set_attn_processor(PrefixLMFlexAttnProcessor(sdpa_backend))
    transformer.forward = types.MethodType(goal_prefix_lm_forward, transformer)
    return len(transformer.attn_processors)


def enable_goal_bidirectional_attention(transformer: nn.Module, sdpa_backend: str | None = "auto") -> int:
    """Ablation arm: bind the goal-aware forward but build no mask (every token attends every token).

    Returns:
        int: Always 0 (no flex processors set).
    """
    transformer.set_attn_processor(PrefixLMFlexAttnProcessor(sdpa_backend))
    transformer._goal_bidirectional_ablation = True
    transformer.forward = types.MethodType(goal_prefix_lm_forward, transformer)
    return 0


__all__: list[Any] = [
    "PrefixLMFlexAttnProcessor",
    "build_goal_prefix_lm_block_mask",
    "enable_goal_bidirectional_attention",
    "enable_goal_prefix_lm_flex_attention",
    "get_compiled_flex_attention",
    "goal_prefix_lm_forward",
    "sdpa_context",
]
