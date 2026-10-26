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
"""SkyReels-V2 diffusion-forcing world model: the trainable DiT plus a frozen, out-of-tree Wan VAE.

The DiT (``self.transformer``) is the only submodule. The frozen VAE lives in a plain holder object
outside the ``nn.Module`` tree, so it never enters the state dict, the optimizer, the EMA shadow or
FSDP sharding; it is reconstructed from ``base_model_path`` at load and lazily moved to the DiT's
device on first use. Text encoding (UMT5-XXL) is handled by the preprocessor
(:mod:`lerobot_policy_glancewam.processor_glancewam`), not here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.utils.import_utils import _diffusers_available, require_package
from torch import Tensor, nn

from .attention import enable_goal_bidirectional_attention, enable_goal_prefix_lm_flex_attention
from .configuration_glancewam import GlanceWAMConfig

if TYPE_CHECKING or _diffusers_available:
    from diffusers import AutoencoderKLWan, SkyReelsV2Transformer3DModel
else:  # pragma: no cover - import guard
    AutoencoderKLWan = None
    SkyReelsV2Transformer3DModel = None

logger = logging.getLogger(__name__)

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


class FrozenVAE:
    """Holder for the frozen Wan VAE, deliberately not an ``nn.Module`` (see the module docstring)."""

    def __init__(self, vae: nn.Module, compile_encoder: bool):
        vae.requires_grad_(False)
        vae.eval()
        self.vae = vae
        self.scale_spatial = 2 ** len(vae.temperal_downsample)
        self.scale_temporal = 2 ** sum(vae.temperal_downsample)
        self.latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        self.latents_std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        self.compile_encoder = compile_encoder
        self._compiled_encoders: dict[str | None, Callable] = {}
        self._eager_encoder = vae.encoder.forward

    def to(self, device: torch.device) -> None:
        if next(self.vae.parameters()).device != device:
            self.vae.to(device)
        self.latents_mean = self.latents_mean.to(device)
        self.latents_std = self.latents_std.to(device)

    @torch.no_grad()
    def encode(self, video: Tensor, use_compiled: bool, compile_mode: str | None = None) -> Tensor:
        """Encode ``video`` ``[B, 3, T, H, W]`` in ``[-1, 1]`` to normalized latents ``[B, 16, T_lat, h, w]``.

        Args:
            video: The frames, any float dtype.
            use_compiled: Run the compiled encoder (training shapes only; eval encodes other shapes).
            compile_mode: ``torch.compile`` mode of that encoder, one compiled copy per mode.
        """
        self.to(video.device)
        encoder = self.vae.encoder
        if use_compiled and self.compile_encoder:
            if compile_mode not in self._compiled_encoders:
                self._compiled_encoders[compile_mode] = torch.compile(
                    self._eager_encoder, mode=compile_mode, dynamic=False
                )
            encoder.forward = self._compiled_encoders[compile_mode]
        else:
            encoder.forward = self._eager_encoder
        try:
            latents = self.vae.encode(video.to(self.vae.dtype)).latent_dist.sample()
        finally:
            encoder.forward = self._eager_encoder
        latents = (latents - self.latents_mean.to(latents.dtype)) / self.latents_std.to(latents.dtype)
        return latents.float()

    @torch.no_grad()
    def decode(self, latents: Tensor) -> Tensor:
        """Inverse of :meth:`encode`: normalized latents ``[B, 16, T_lat, h, w]`` -> frames in ``[0, 1]``.

        Returns:
            Tensor: ``[B, 3, T, H, W]`` float32 frames.
        """
        self.to(latents.device)
        z = latents.to(self.vae.dtype)
        z = z * self.latents_std.to(z.dtype) + self.latents_mean.to(z.dtype)
        video = self.vae.decode(z, return_dict=False)[0]
        return video.float().add(1.0).div(2.0).clamp(0.0, 1.0)


class TruncatedGoalDiT(nn.Module):
    """Goal-aware DiT forward that stops after the last tapped block and returns the taps.

    The chunk decode only needs the tapped hidden states, so the blocks after the last tap and the
    velocity head are skipped, and the taps are collected in the loop instead of through forward hooks.
    The block mask is passed in precomputed (the ``[obs, filler, goal]`` timestep pattern is fixed at
    inference), which keeps the whole forward inside one ``torch.compile`` graph. Same ops as
    ``goal_prefix_lm_forward`` up to that block, so the features match the stock path.
    """

    def __init__(self, transformer: nn.Module, taps: list[int], fps: list[int] | None):
        super().__init__()
        self.dit = transformer
        self.taps = sorted(int(t) for t in taps)
        self.n_run = self.taps[-1] + 1
        self.register_buffer(
            "fps", torch.tensor(fps if fps is not None else [0], dtype=torch.long), persistent=False
        )
        self.inject_fps = fps is not None

    def forward(
        self, hidden_states: Tensor, timestep: Tensor, encoder_hidden_states: Tensor, block_mask
    ) -> list[Tensor]:
        d = self.dit
        _, _, _t, height, width = hidden_states.shape
        _, p_h, p_w = d.config.patch_size
        pph, ppw = height // p_h, width // p_w
        rotary_emb = d.rope(hidden_states)
        hidden_states = d.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
        _temb, timestep_proj, encoder_hidden_states, _ = d.condition_embedder(
            timestep, encoder_hidden_states, None
        )
        timestep_proj = timestep_proj.unflatten(-1, (6, -1))
        if self.inject_fps:
            fps_proj = d.fps_projection(d.fps_embedding(self.fps)).unflatten(1, (6, -1))
            timestep_proj = timestep_proj + fps_proj.repeat(timestep.shape[1], 1, 1)
        b, f = timestep.shape
        timestep_proj = (
            timestep_proj.view(b, f, 1, 1, 6, -1)
            .repeat(1, 1, pph, ppw, 1, 1)
            .flatten(1, 3)
            .transpose(1, 2)
            .contiguous()
        )
        outs = []
        for i, block in enumerate(d.blocks[: self.n_run]):
            # ``block.forward`` rather than ``block(...)``: skips the tap hooks (and any compiled wrapper).
            hidden_states = block.forward(
                hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, block_mask
            )
            if i in self.taps:
                outs.append(hidden_states)
        return outs


class SkyReelsWorldModel(nn.Module):
    """The SkyReels-V2 DF DiT with goal-aware FlexAttention, feature taps and the frozen VAE holder.

    Args:
        config: The policy configuration.
        load_pretrained: Load the base DiT weights from ``config.base_model_path``. Off when a LeRobot
            checkpoint is about to be loaded on top (avoids reading 5 GB twice).
    """

    def __init__(self, config: GlanceWAMConfig, load_pretrained: bool = True):
        super().__init__()
        require_package("diffusers", extra="diffusers-dep")
        self.config = config
        transformer, vae = self.load_components(config, load_pretrained)
        self.transformer = transformer
        # Not a registered submodule on purpose (plain attribute on a non-Module object).
        self._vae_holder = FrozenVAE(vae, compile_encoder=config.compile_vae)

        if config.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        if config.clean_prefix_isolation:
            n = enable_goal_prefix_lm_flex_attention(self.transformer, config.sdpa_backend)
            logger.info(f"GlanceWAM: goal-aware 3-class prefix-LM FlexAttention on {n} attention processors.")
        else:
            enable_goal_bidirectional_attention(self.transformer, config.sdpa_backend)
            logger.info("GlanceWAM: bidirectional ablation, no attention mask.")

        self.hidden_size = int(
            self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        )
        self.fps = [0 if config.fps == 16 else 1] if self.transformer.config.inject_sample_info else None

        n_blocks = len(self.transformer.blocks)
        taps = [t if t >= 0 else n_blocks + t for t in config.extract_layers]
        if not taps or any(not 0 <= t < n_blocks for t in taps):
            raise ValueError(f"extract_layers {config.extract_layers} out of range for {n_blocks} DiT blocks")
        if taps != sorted(taps) or len(set(taps)) != len(taps):
            raise ValueError(
                f"extract_layers must be ascending and unique (capture order == block order): {taps}"
            )
        self.taps = taps
        self._features: list[Tensor] = []
        for t in taps:
            self.transformer.blocks[t].register_forward_hook(self._capture_hook)

        h, w = config.vae_input_size
        p_t, p_h, p_w = self.transformer.config.patch_size
        self.tok_h = (h // self.scale_spatial) // p_h
        self.tok_w = (w // self.scale_spatial) // p_w
        self.tokens_per_frame = self.tok_h * self.tok_w

    # ------------------------------------------------------------------ construction
    @staticmethod
    def load_components(config: GlanceWAMConfig, load_pretrained: bool) -> tuple[nn.Module, nn.Module]:
        """Build (DiT, VAE). Tests patch this to return tiny random modules."""
        dtype = _DTYPES[config.dtype]
        if load_pretrained:
            transformer = SkyReelsV2Transformer3DModel.from_pretrained(
                config.base_model_path, subfolder="transformer", torch_dtype=dtype
            )
        else:
            tf_config = SkyReelsV2Transformer3DModel.load_config(
                config.base_model_path, subfolder="transformer"
            )
            transformer = SkyReelsV2Transformer3DModel.from_config(tf_config).to(dtype)
        vae = AutoencoderKLWan.from_pretrained(
            config.base_model_path, subfolder="vae", torch_dtype=torch.bfloat16
        )
        return transformer, vae

    @property
    def scale_spatial(self) -> int:
        return self._vae_holder.scale_spatial

    @property
    def scale_temporal(self) -> int:
        return self._vae_holder.scale_temporal

    @property
    def dtype(self) -> torch.dtype:
        return self.transformer.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.transformer.patch_embedding.weight.device

    def _capture_hook(self, module, args, output):
        self._features.append(output[0] if isinstance(output, tuple) else output)

    # ------------------------------------------------------------------ encoding
    def encode_frames(self, frames: Tensor, compile_mode: str | None = None) -> Tensor:
        """VAE-encode ``frames`` ``[B, T, 3, H, W]`` in ``[0, 1]`` (already at ``vae_input_size``).

        Args:
            frames: The frames.
            compile_mode: At eval, run the encoder under ``torch.compile`` with this mode (the inference
                fast path); ``None`` keeps eval eager. Training always uses the default-mode compile.

        Returns:
            Tensor: Normalized fp32 latents ``[B, 16, T_lat, H/8, W/8]``.
        """
        video = frames.permute(0, 2, 1, 3, 4).float().mul(2.0).sub(1.0)
        if self.training:
            return self._vae_holder.encode(video, use_compiled=True)
        return self._vae_holder.encode(
            video, use_compiled=compile_mode is not None, compile_mode=compile_mode
        )

    def truncated_dit(self) -> TruncatedGoalDiT:
        """Hook-free forward of the DiT up to the last tap, for the inference fast path."""
        return TruncatedGoalDiT(self.transformer, self.taps, self.fps).to(self.device)

    def decode_latents(self, latents: Tensor) -> Tensor:
        """VAE-decode normalized latents back to ``[B, T, 3, H, W]`` frames in ``[0, 1]`` (inspection only)."""
        return self._vae_holder.decode(latents).permute(0, 2, 1, 3, 4)

    def encode_window(self, frames: Tensor, target_index: int) -> Tensor:
        """Encode one single-target window ``[history..., frames[target_index] tiled]`` (causal VAE).

        Every special frame is tiled across one temporal VAE group so it maps to exactly one latent
        frame, and each target gets its own obs-conditioned pass, free of the other targets.
        """
        nh = self.config.num_history_frames
        tiled = torch.cat(
            [
                frames[:, :nh],
                frames[:, target_index : target_index + 1].expand(-1, self.scale_temporal, -1, -1, -1),
            ],
            dim=1,
        )
        return self.encode_frames(tiled)

    # ------------------------------------------------------------------ DiT
    def run_dit(
        self, xt: Tensor, timestep: Tensor, text_embeds: Tensor, n_goal_frames: int
    ) -> tuple[list[Tensor], Tensor]:
        """One goal-aware DiT forward.

        Args:
            xt: Latent window ``[B, 16, T_lat, h, w]``.
            timestep: ``[B, T_lat]`` per-frame timesteps (``0`` = clean).
            text_embeds: ``[B, L, 4096]`` UMT5 embeddings.
            n_goal_frames: Trailing clean frames forming the goal side-channel.

        Returns:
            tuple: (tapped features in ascending block order, each ``[B, T_lat*tokens_per_frame, C]``;
            velocity ``[B, 16, T_lat, h, w]``).
        """
        self._features.clear()
        out = self.transformer(
            hidden_states=xt.to(self.dtype),
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            enable_diffusion_forcing=True,
            fps=self.fps,
            n_goal_frames=int(n_goal_frames),
        )
        velocity = out.sample if hasattr(out, "sample") else out[0]
        feats = list(self._features)
        self._features.clear()
        if len(feats) != len(self.taps):
            raise RuntimeError(f"expected {len(self.taps)} tapped features, captured {len(feats)}")
        return feats, velocity

    def stitch_views(self, views: list[Tensor]) -> Tensor:
        """Stitch the camera views of every frame into one image per ``config.camera_layout``.

        Args:
            views: Per-camera tensors ``[B, T, 3, H, W]`` in stitching order.

        Returns:
            Tensor: ``[B, T, 3, *vae_input_size]``.
        """
        size = self.config.view_size

        def _resize(v: Tensor, hw: tuple[int, int]) -> Tensor:
            b, t = v.shape[:2]
            flat = v.flatten(0, 1)
            if flat.shape[-2:] != hw:
                flat = F.interpolate(flat, size=hw, mode="bilinear", antialias=True, align_corners=False)
            return flat.unflatten(0, (b, t))

        if self.config.camera_layout == "primary_inset" and len(views) > 1:
            n_insets = len(views) - 1
            inset = size // n_insets
            primary = _resize(views[0], (size, size))
            column = torch.cat([_resize(v, (inset, inset)) for v in views[1:]], dim=-2)
            if column.shape[-2] < size:  # top-anchored on black when the insets do not span the height
                pad = column.new_zeros((*column.shape[:-2], size - column.shape[-2], inset))
                column = torch.cat([column, pad], dim=-2)
            out = torch.cat([primary, column[..., :size, :]], dim=-1)
        else:
            out = torch.cat([_resize(v, (size, size)) for v in views], dim=-1)
        h, w = self.config.vae_input_size
        if out.shape[-2:] != (h, w):
            raise ValueError(
                f"stitched frame is {tuple(out.shape[-2:])} but vae_input_size is {(h, w)}: "
                f"set view_size / camera_order / camera_layout / vae_input_size consistently."
            )
        return out
