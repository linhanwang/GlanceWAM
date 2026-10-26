#!/usr/bin/env python

# Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team. All rights reserved.
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
"""GR00T N1.5 flow-matching action head (single embodiment), as used by GlanceWAM.

The cross-attention DiT is shared with the N1.7 port in :mod:`lerobot.policies.groot.action_head`;
the parameter names match the GR00T N1.5 head so released GlanceWAM checkpoints load unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.groot.action_head.cross_attention_dit import DiT
from torch import Tensor, nn
from torch.distributions import Beta

from .configuration_glancewam import DIT_PRESETS, GlanceWAMConfig

if TYPE_CHECKING:
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """``(B, T)`` timesteps -> ``(B, T, embedding_dim)`` sinusoidal encoding."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: Tensor) -> Tensor:
        timesteps = timesteps.float()
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.unsqueeze(-1) * exponent.exp()
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions: Tensor, timesteps: Tensor) -> Tensor:
        """``actions (B, T, D)``, ``timesteps (B,)`` -> ``(B, T, hidden)``."""
        b, t, _ = actions.shape
        if not (timesteps.dim() == 1 and timesteps.shape[0] == b):
            raise ValueError("Expected `timesteps` of shape (B,)")
        timesteps = timesteps.unsqueeze(1).expand(-1, t)
        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = swish(self.layer2(torch.cat([a_emb, tau_emb], dim=-1)))
        return self.layer3(x)


def seeded_randn(shape: tuple[int, ...], seeds: Sequence[int | None] | None, *, dtype, device) -> Tensor:
    """``torch.randn(shape)``, per-row reproducible from ``seeds`` (CPU generators) when given."""
    if seeds is None:
        return torch.randn(size=shape, dtype=dtype, device=device)
    if len(seeds) != shape[0]:
        raise ValueError(f"seeds length {len(seeds)} != batch size {shape[0]}")
    rows = []
    for s in seeds:
        if s is None:
            rows.append(torch.randn(shape[1:], dtype=torch.float32))
        else:
            g = torch.Generator(device="cpu").manual_seed(int(s) % (2**63 - 1))
            rows.append(torch.randn(shape[1:], generator=g, dtype=torch.float32))
    return torch.stack(rows).to(device=device, dtype=dtype)


class FlowmatchingActionHead(nn.Module):
    """GR00T N1.5 flow-matching head: a cross-attention DiT over ``[state?, future tokens, noisy actions]``.

    Args:
        config: The policy configuration.
        cross_attention_dim: Width of the conditioning tokens (the world model hidden size).
    """

    def __init__(self, config: GlanceWAMConfig, cross_attention_dim: int):
        super().__init__()
        preset = DIT_PRESETS[config.action_model_type]
        self.input_embedding_dim = preset["input_embedding_dim"]
        self.model = DiT(
            num_attention_heads=preset["num_attention_heads"],
            attention_head_dim=preset["attention_head_dim"],
            output_dim=config.action_hidden_size,
            num_layers=config.num_action_head_layers,
            dropout=config.action_head_dropout,
            final_dropout=True,
            interleave_self_attention=True,
            norm_type="ada_norm",
            positional_embeddings=None,
            cross_attention_dim=cross_attention_dim,
        )
        self.action_horizon = config.chunk_size
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.hidden_size = config.action_hidden_size

        self.state_encoder = (
            MLP(config.action_head_state_dim, self.hidden_size, self.input_embedding_dim)
            if config.action_head_state_dim
            else None
        )
        self.action_encoder = ActionEncoder(self.action_dim, self.input_embedding_dim)
        self.action_decoder = MLP(self.model.config.output_dim, self.hidden_size, self.action_dim)

        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        self.add_pos_embed = config.add_pos_embed
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.noise_s = config.noise_s
        self.num_timestep_buckets = config.num_timestep_buckets

        self.t_loss_weighting = config.t_loss_weighting
        if self.t_loss_weighting:
            grid = torch.arange(self.num_timestep_buckets, dtype=torch.float32) / self.num_timestep_buckets
            bell = torch.exp(-2.0 * (grid - 0.5) ** 2)
            bell = bell - bell.min()
            bell = bell * (self.num_timestep_buckets / bell.sum())
            self.register_buffer("t_loss_weights", bell, persistent=False)

    def sample_time(self, batch_size: int, device, dtype) -> Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.noise_s)
        return self.noise_s * (1 - sample)

    def _tokens(self, action_features: Tensor, state: Tensor | None, batch_size: int) -> Tensor:
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=action_features.device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
        parts = [future_tokens, action_features]
        if state is not None:
            if self.state_encoder is None:
                raise ValueError(
                    "state given but the head was built without a state encoder (use_state=False)"
                )
            parts.insert(0, self.state_encoder(state))
        return torch.cat(parts, dim=1)

    def forward(self, vl_embs: Tensor, actions: Tensor, state: Tensor | None = None) -> Tensor:
        """Flow-matching loss of ``actions (B, T, D)`` conditioned on ``vl_embs (B, S, C)``."""
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

        action_features = self.action_encoder(noisy_trajectory, t_discretized)
        sa_embs = self._tokens(action_features, state, vl_embs.shape[0])
        model_output = self.model(
            hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=t_discretized
        )
        pred_actions = self.action_decoder(model_output)[:, -actions.shape[1] :]

        se = (pred_actions.float() - velocity.float()) ** 2
        if self.t_loss_weighting:
            per_sample = se.flatten(start_dim=1).mean(dim=1)
            buckets = t_discretized.clamp(0, self.num_timestep_buckets - 1)
            return (per_sample * self.t_loss_weights[buckets].to(per_sample.dtype)).mean()
        return se.mean()

    def _velocity(self, actions: Tensor, timesteps: Tensor, state: Tensor | None, vl_embs: Tensor) -> Tensor:
        """One denoiser call: the flow velocity ``x1 - x0`` of ``actions (B, T, D)`` at ``timesteps (B,)``."""
        action_features = self.action_encoder(actions, timesteps)
        sa_embs = self._tokens(action_features, state, vl_embs.shape[0])
        model_output = self.model(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timesteps)
        return self.action_decoder(model_output)[:, -self.action_horizon :]

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: Tensor,
        state: Tensor | None = None,
        seeds: Sequence[int | None] | None = None,
        *,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """Euler flow-matching sampler: ``(B, action_horizon, action_dim)`` from ``vl_embs (B, S, C)``.

        With an enabled ``rtc_processor`` and a ``prev_chunk_left_over`` prefix (the unexecuted tail of the
        previous chunk, ``(T, D)`` or ``(B, T, D)``) every Euler step is wrapped in Real-Time Chunking
        guidance, exactly as in the pi0 / pi0.5 samplers: the first ``inference_delay`` actions are pulled
        onto the prefix and the pull fades out over the processor's ``execution_horizon``.

        The sampler state is kept in float32 (the head's linears cast under autocast anyway) so the
        guidance error and the guided velocity do not round through bfloat16.
        """
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = seeded_randn(
            (batch_size, self.action_horizon, self.action_dim), seeds, dtype=torch.float32, device=device
        )
        guided = (
            rtc_processor is not None
            and rtc_processor.rtc_config.enabled
            and prev_chunk_left_over is not None
        )
        if guided:
            prev_chunk_left_over = prev_chunk_left_over.to(device=device, dtype=torch.float32)
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        for step in range(num_steps):
            tau = step / float(num_steps)
            timesteps = torch.full((batch_size,), int(tau * self.num_timestep_buckets), device=device)
            if not guided:
                actions = actions + dt * self._velocity(actions, timesteps, state, vl_embs)
                continue

            # ``RTCProcessor`` speaks openpi's convention (time runs 1 -> 0 and v = noise - x1); the head
            # is GR00T's (tau runs 0 -> 1 and v = x1 - noise). Both predict the same clean sample,
            # x1 = x_t + (1 - tau) * v_groot = x_t - time * v_openpi with time = 1 - tau, so the head's
            # velocity is negated on the way in and the guided velocity on the way out.
            def denoise_openpi(x_t: Tensor, timesteps: Tensor = timesteps) -> Tensor:
                # The processor differentiates only its own ``x_t - time * v`` (the denoiser is called
                # before ``x_t`` requires grad), so the head runs grad-free, as compiled at warmup.
                with torch.no_grad():
                    return -self._velocity(x_t, timesteps, state, vl_embs).float()

            v_openpi = rtc_processor.denoise_step(
                x_t=actions,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=1.0 - tau,
                original_denoise_step_partial=denoise_openpi,
            )
            actions = actions - dt * v_openpi
        return actions
