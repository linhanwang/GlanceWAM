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
"""GlanceWAM policy: a video world model that glances ahead.

Training (one step)
    1. Stitch the cameras of every window frame into one wide image and VAE-encode two obs-conditioned
       windows: ``[obs, future]`` (the video target at ``+future_frame_idx``) and ``[obs, goal]`` (the
       hindsight goal at ``+g``). Both special frames land on exactly one latent frame each.
    2. Noise the future latent (obs and goal stay clean) and run one goal-aware DiT pass: the tapped
       block outputs feed the action head, the velocity output feeds the video loss.
    3. Action loss = GR00T flow matching on ``[obs tokens ; goal tokens (+ goal-time embedding)]``, with
       per-sample goal dropout so a no-goal fallback stays in-distribution. Video loss = masked velocity
       MSE on the future frame.

Inference (``predict_action_chunk``)
    Encode the obs, generate the goal latent with the video DiT's own flow-matching sampler on a slow
    clock (every ``future_frame_idx`` rows), then decode an action chunk conditioned on ``[obs ; goal]``
    with the shrinking goal-time diff. The goal latent is held across chunks and refreshed in lockstep
    for the whole env batch. With ``goal_async`` the sampler runs in a worker process (``goal_worker.py``)
    on a row clock: every ``goal_refresh_rows`` the current frame is sent (queued ticks included) and
    the reply is installed as soon as it lands, so the chunk decode never shares a tick with the goal
    generation and the goal-time diff is always horizon minus the age of the goal's own frame.

Real-Time Chunking (``lerobot-rollout --inference.type=rtc``)
    The engine calls ``predict_action_chunk(batch, inference_delay=, prev_chunk_left_over=)`` from a
    background thread whenever its queue runs low. The head's sampler is guided onto the previous chunk's
    unexecuted tail through the shared ``RTCProcessor`` (the pi0 / pi0.5 math), the goal row clock runs on
    wall-clock time (``rows_per_second``), and an episode's first chunk is decoded with the no-goal
    fallback while the worker imagines the first goal, so no chunk call ever waits on the video sampler.
"""

from __future__ import annotations

import builtins
import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import TypedDict, TypeVar, Unpack

import torch
import torchvision.transforms.v2.functional as TF  # noqa: N812
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.import_utils import require_package
from torch import Tensor, nn

from .action_head import FlowmatchingActionHead
from .attention import build_goal_prefix_lm_block_mask, get_compiled_flex_attention
from .configuration_glancewam import GlanceWAMConfig
from .goal_worker import GoalWorker
from .processor_glancewam import LANGUAGE_EMBEDDING_KEY
from .world_model import SkyReelsWorldModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="GlanceWAMPolicy")


class ActionSelectKwargs(TypedDict, total=False):
    """The RTC engine's call shape for ``predict_action_chunk``."""

    inference_delay: int | None
    prev_chunk_left_over: Tensor | None


# Per-sample goal offset (native rows) emitted by the dataset next to the goal frame.
GOAL_OFFSET_KEY = "observation.goal_offset"
# Timestep the filler (future-class) frame is labelled with at inference; masked out of every read.
_FILLER_TIMESTEP = 500.0


def sinusoidal_h_embed(h_rows: Tensor, dim: int) -> Tensor:
    """Goal-time diff (native rows) -> ``(..., dim)`` sinusoidal embedding, periods sized for ``[1, ~100]``."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=h_rows.device, dtype=torch.float32) / half
    )
    angles = h_rows.float().unsqueeze(-1) * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def compile_dit_blocks(transformer: nn.Module, mode: str | None, dynamic: bool | None) -> int:
    """``torch.compile`` each DiT block's ``forward`` in place (idempotent).

    Compiling the bound ``forward`` rather than ``Module.compile()`` keeps forward hooks (the feature
    taps) and any FSDP2 hooks outside the compiled region, so it composes with sharding applied later.
    """
    n = 0
    for block in transformer.blocks:
        if getattr(block.forward, "_torchdynamo_orig_callable", None) is not None:
            continue
        block.forward = torch.compile(block.forward, mode=mode, dynamic=dynamic)
        n += 1
    return n


class FastChunkPath:
    """Inference fast path: truncated hook-free DiT + compiled encoder / DiT / action head.

    Holds the truncated wrapper (not a registered submodule, so checkpoints are unaffected), the
    ``torch.compile`` mode, and the goal-aware block masks cached per ``(batch, T_lat)``. The action
    head's ``model.forward`` is compiled in place, which keeps the module identity and its state dict.
    """

    def __init__(self, policy: GlanceWAMPolicy, compile_mode: str | None):
        self.compile_mode = compile_mode
        self.trunc = policy.backbone.truncated_dit().eval()
        self._masks: dict[tuple[int, int], object] = {}
        self._tokens_per_frame = policy.tokens_per_frame
        if compile_mode is not None:
            self.trunc.forward = torch.compile(self.trunc.forward, mode=compile_mode, dynamic=False)
            head = policy.action_model.model
            if getattr(head.forward, "_torchdynamo_orig_callable", None) is None:
                head.forward = torch.compile(head.forward, mode=compile_mode, dynamic=False)

    def block_mask(self, in_timestep: Tensor, n_goal_frames: int):
        key = (int(in_timestep.shape[0]), int(in_timestep.shape[1]))
        if key not in self._masks:
            self._masks[key] = build_goal_prefix_lm_block_mask(
                in_timestep, n_goal_frames, key[1], self._tokens_per_frame, in_timestep.device
            )
        return self._masks[key]

    def features(
        self, window: Tensor, in_timestep: Tensor, text_embeds: Tensor, n_goal_frames: int
    ) -> list[Tensor]:
        mask = self.block_mask(in_timestep, n_goal_frames)
        feats = self.trunc(window.to(self.trunc.dit.dtype), in_timestep, text_embeds, mask)
        # CUDA-graph outputs are overwritten by the next replay: detach them from the graph pool.
        return [f.clone() for f in feats]


class GlanceWAMPolicy(PreTrainedPolicy):
    """GlanceWAM: SkyReels-V2 DF video DiT + GR00T flow-matching head with lookahead-frame conditioning."""

    config_class = GlanceWAMConfig
    name = "glancewam"
    # FSDP2 wrap units: the 30 SkyReels DiT blocks and the 16 action-head DiT blocks. Patch/condition
    # embedders, output heads and the small projections stay in the root unit.
    _fsdp_wrap_modules = ["SkyReelsV2TransformerBlock", "BasicTransformerBlock"]

    def __init__(self, config: GlanceWAMConfig, *, load_pretrained_backbone: bool | None = None, **kwargs):
        require_package("diffusers", extra="diffusers-dep")
        require_package("transformers", extra="transformers-dep")
        super().__init__(config)
        config.validate_features()
        self.config = config

        if load_pretrained_backbone is None:
            load_pretrained_backbone = config.pretrained_path is None
        self.backbone = SkyReelsWorldModel(config, load_pretrained=load_pretrained_backbone)
        wm_hidden = self.backbone.hidden_size
        self.action_model = FlowmatchingActionHead(config, cross_attention_dim=wm_hidden)

        self.goal_time_proj: nn.Linear | None = None
        if config.goal_h_embed_dim > 0:
            self.goal_time_proj = nn.Linear(config.goal_h_embed_dim, wm_hidden)
            nn.init.zeros_(self.goal_time_proj.weight)
            nn.init.zeros_(self.goal_time_proj.bias)

        self.n_taps = len(self.backbone.taps)
        self.tap_embed: nn.Embedding | None = None
        if self.n_taps > 1:
            self.tap_embed = nn.Embedding(self.n_taps, wm_hidden)
            nn.init.zeros_(self.tap_embed.weight)

        scale_t = self.backbone.scale_temporal
        self.obs_latent_frames = (config.num_history_frames - 1) // scale_t + 1
        self.goal_horizon_rows = float(config.future_frame_idx)

        if config.compile_dit:
            n = compile_dit_blocks(
                self.backbone.transformer, config.compile_dit_mode, config.compile_dit_dynamic
            )
            logger.info(f"GlanceWAM: torch.compile applied to {n} DiT blocks.")

        # Goal schedule: a new goal is imagined from the frame every ``goal_refresh_rows`` rows.
        self.goal_refresh_rows = config.goal_refresh_rows or config.future_frame_idx
        self._goal_worker: GoalWorker | None = GoalWorker(config) if config.goal_async else None
        self._fast: FastChunkPath | None = None  # built by the first reset() after construction (CUDA only)
        self.rtc_processor: RTCProcessor | None = None
        self.init_rtc_processor()
        self._episode_idx = 0
        self._goal_idx = 0
        self._constructed = False
        self.reset()
        self._constructed = True

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        **kwargs,
    ) -> T:
        """Load a LeRobot checkpoint; the base DiT weights are not fetched since the checkpoint holds them."""
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "force_download",
                        "resume_download",
                        "proxies",
                        "token",
                        "cache_dir",
                        "local_files_only",
                        "revision",
                    )
                },
            )
        config.pretrained_path = Path(pretrained_name_or_path)
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    # ------------------------------------------------------------------ plumbing
    def get_optim_params(self) -> dict:
        return [p for p in self.parameters() if p.requires_grad]

    def supports_rtc(self) -> bool:
        return True

    def init_rtc_processor(self) -> None:
        """(Re)build the RTC processor from ``config.rtc_config`` (the rollout context sets it after loading)."""
        self.rtc_processor = None
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)
        if self._rtc_enabled() and self._goal_worker is None:
            logger.warning(
                "GlanceWAM: RTC with the inline goal sampler (goal_async=false). Every goal refresh lands "
                "inside a chunk call (~0.5 s) and the RTC engine sizes its overlap from the slowest of its "
                "last 100 calls, so the chunk prefix stays pinned ~10 rows deep. Use --policy.goal_async=true."
            )

    def _rtc_enabled(self) -> bool:
        return self.rtc_processor is not None and self.rtc_processor.rtc_config.enabled

    def reset(self):
        self._action_queue: deque[Tensor] = deque(maxlen=self.config.n_action_steps)
        self._drop_goal()
        # The rollout resets before each episode's clock starts: absorb the worker's load + warmup here
        # rather than on the first tick. Skipped inside __init__ so the two processes load in parallel.
        if self._goal_worker is not None and self._constructed:
            self._goal_worker.wait_ready()
        if self._constructed:
            self._warmup_fast_path()
        # Goal previews are grouped per episode. Only a reset closing an episode that actually wrote
        # previews advances the index, so setup resets and task changes don't leave gaps in the names.
        if self._goal_idx:
            self._episode_idx += 1
        self._goal_idx = 0

    def _drop_goal(self) -> None:
        """Forget the held goal, the in-flight request and the row clock (the next chunk regenerates)."""
        self._goal_latent: Tensor | None = None
        # Rows since reset: the row of the next chunk's first action after a chunk tick. Within a chunk
        # the current row is ``_tick - len(_action_queue)``.
        self._tick = 0
        self._rtc_t0: float | None = None  # wall-clock origin of the row clock under the RTC engine
        self._goal_frame_row: int | None = None  # row of the frame the held goal was imagined from
        self._goal_request_row: int | None = None  # same for the in-flight request; None = none in flight
        self._goal_request_frame: Tensor | None = None  # that observation, kept only for goal previews
        if self._goal_worker is not None:
            self._goal_worker.discard()

    @property
    def _row(self) -> int:
        """The row (since reset) of the action about to be produced or popped."""
        return self._tick - len(self._action_queue)

    def _goal_distance(self, row: int) -> float:
        """Rows from ``row`` to the held goal's horizon, floored at 1 (the head's goal-time diff)."""
        return max(1.0, self.goal_horizon_rows - (row - self._goal_frame_row))

    def _rtc_row(self, inference_delay: int) -> int:
        """Row of the chunk about to be decoded under the RTC engine: wall-clock rows since the first chunk
        call after reset (``rows_per_second``), plus the rows the engine expects the call to take."""
        now = time.perf_counter()
        if self._rtc_t0 is None:
            self._rtc_t0 = now
        return int(round((now - self._rtc_t0) * self.config.rows_per_second)) + inference_delay

    def drop_queued_actions(self) -> None:
        """Also drop the goal: it was imagined under the previous instruction."""
        super().drop_queued_actions()
        self._drop_goal()

    @property
    def tokens_per_frame(self) -> int:
        return self.backbone.tokens_per_frame

    @property
    def _fast_active(self) -> bool:
        return self.config.fast_inference and self.backbone.device.type == "cuda"

    def _warmup_fast_path(self) -> None:
        """Build the fast path and run the compile + CUDA-graph recording on a dummy batch of one.

        Runs under the grad mode the rollout engine will call the policy under, since dynamo guards on it
        and a mismatch recompiles everything on the first real tick (measured: an 11.8 s tick, then a
        0.5 s graph re-recording on the next chunk): ``inference_mode`` for the sync engine, plain
        ``no_grad`` for the RTC engine (its guidance sets ``requires_grad`` on the sampler state, which an
        inference tensor cannot carry).
        """
        if not self._fast_active or self._fast is not None:
            return
        with torch.no_grad() if self._rtc_enabled() else torch.inference_mode():
            self._warmup_fast_path_impl()

    def _warmup_fast_path_impl(self) -> None:
        cfg = self.config
        self.eval()
        # The compiled graph guards on this lazily-created global: create it before compiling, otherwise
        # the first eager attention call (the goal sampler) flips the guard and forces a recompile.
        get_compiled_flex_attention()
        self._fast = FastChunkPath(self, cfg.fast_inference_compile_mode)
        if cfg.fast_inference_compile_mode is None:
            return
        device = self.backbone.device
        text_dim = int(self.backbone.transformer.config.text_dim)
        frames = torch.zeros(1, cfg.num_history_frames, 3, *cfg.vae_input_size, device=device)
        text = torch.zeros(1, cfg.text_max_length, text_dim, device=device, dtype=torch.bfloat16)
        batch = {OBS_STATE: torch.zeros(1, cfg.action_head_state_dim, device=device)} if cfg.use_state else {}
        obs_latent = self.backbone.encode_frames(frames)
        goal = torch.zeros((1, obs_latent.shape[1], cfg.n_goal_frames, *obs_latent.shape[3:]), device=device)
        # reduce-overhead records the CUDA graph after a couple of eager warmup runs.
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            obs_latent = self._encode_obs(frames)
            self._decode_chunk(batch, obs_latent, text, goal)
        if self._goal_worker is None:
            # The inline goal sampler compiles its own FlexAttention kernels on first use: do that here too.
            self._generate_goal_latent(obs_latent, text)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        logger.info(f"GlanceWAM: fast inference path ready (compile mode {cfg.fast_inference_compile_mode}).")

    def _encode_obs(self, frames: Tensor) -> Tensor:
        mode = self.config.fast_inference_compile_mode if self._fast is not None else None
        return self.backbone.encode_frames(frames, compile_mode=mode)

    def _autocast(self, device: torch.device):
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    def _stitched_window(self, batch: dict[str, Tensor]) -> Tensor:
        """Camera views of the batch -> ``[B, T, 3, H, W]`` stitched frames (``T = 1`` for a live obs)."""
        views = []
        for key in self.config.camera_keys:
            if key not in batch:
                raise KeyError(f"GlanceWAM: image key {key!r} missing from the batch ({sorted(batch)})")
            img = batch[key]
            if img.ndim == 4:
                img = img.unsqueeze(1)
            img = img.float()
            resize = self.config.image_resize_shape
            if resize is not None and tuple(img.shape[-2:]) != tuple(resize):
                # Same antialiased op as the dataloader's image_transforms.resize (see the config).
                img = TF.resize(
                    img, list(resize), interpolation=TF.InterpolationMode.BILINEAR, antialias=True
                )
            views.append(img)
        return self.backbone.stitch_views(views)

    def _language(self, batch: dict[str, Tensor]) -> Tensor:
        if LANGUAGE_EMBEDDING_KEY not in batch:
            raise KeyError(
                f"GlanceWAM: {LANGUAGE_EMBEDDING_KEY!r} missing; the preprocessor's UMT5 text-table step "
                "must run before the policy."
            )
        return batch[LANGUAGE_EMBEDDING_KEY].to(torch.bfloat16)

    def _state(self, batch: dict[str, Tensor]) -> Tensor | None:
        if not self.config.use_state:
            return None
        state = batch[OBS_STATE]
        if state.ndim == 3:
            state = state[:, -1]
        return state.unsqueeze(1)

    def _sample_sigma(self, n: int, device: torch.device) -> Tensor:
        if self.config.sigma_sampling == "uniform":
            return torch.rand(n, device=device)
        t = torch.sigmoid(torch.randn(n, device=device))
        s = self.config.sigma_shift
        return s * t / (1 + (s - 1) * t)

    def _slice_obs_goal(self, feats: list[Tensor], goal_start: int, t_lat: int) -> tuple[Tensor, Tensor]:
        """(obs tokens, goal tokens) for the head, tap-major under multi-tap with the tap embedding added."""
        ftok = self.tokens_per_frame
        n_obs_tok = self.obs_latent_frames * ftok
        if self.n_taps == 1:
            f = feats[0]
            return f[:, :n_obs_tok], f[:, goal_start * ftok : t_lat * ftok]
        obs_parts, goal_parts = [], []
        for i, f in enumerate(feats):
            e = self.tap_embed.weight[i].to(f.dtype)
            obs_parts.append(f[:, :n_obs_tok] + e)
            goal_parts.append(f[:, goal_start * ftok : t_lat * ftok] + e)
        return torch.cat(obs_parts, dim=1), torch.cat(goal_parts, dim=1)

    def _add_goal_time_diff(self, goal_hidden: Tensor, g_rows: Tensor) -> Tensor:
        if self.goal_time_proj is None:
            return goal_hidden
        if g_rows.dim() == 1:
            g_rows = g_rows.unsqueeze(1)
        emb = self.goal_time_proj(
            sinusoidal_h_embed(g_rows, self.config.goal_h_embed_dim).to(self.goal_time_proj.weight.dtype)
        )
        per_tok = emb.repeat_interleave(self.tokens_per_frame, dim=1)
        if self.n_taps > 1:
            per_tok = per_tok.repeat(1, self.n_taps, 1)
        return goal_hidden + per_tok.to(goal_hidden.dtype)

    # ------------------------------------------------------------------ training
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        cfg = self.config
        frames = self._stitched_window(batch)
        if frames.shape[1] != cfg.window_frames:
            raise RuntimeError(
                f"GlanceWAM expects {cfg.window_frames} frames per sample (history, future @ +{cfg.future_frame_idx}, "
                f"goal) but got {frames.shape[1]}; check image_observation_delta_indices and the dataset's goal_frame_range."
            )
        text_embeds = self._language(batch)
        device = frames.device
        nh, nf, ng = cfg.num_history_frames, cfg.n_future_frames, cfg.n_goal_frames

        with torch.no_grad():
            latent_first = self.backbone.encode_window(frames, nh)
            extra = [
                self.backbone.encode_window(frames, nh + k)[:, :, self.obs_latent_frames :]
                for k in range(1, nf + ng)
            ]
        clean_latent = torch.cat([latent_first, *extra], dim=2)
        bsz, t_lat = clean_latent.shape[0], clean_latent.shape[2]
        goal_start = t_lat - ng
        if goal_start <= self.obs_latent_frames:
            raise ValueError(
                f"no room for the noised future frame: t_lat={t_lat}, obs={self.obs_latent_frames}, goal={ng}"
            )

        cond_mask = torch.zeros(bsz, 1, t_lat, 1, 1, device=device, dtype=torch.float32)
        cond_mask[:, :, : self.obs_latent_frames] = 1.0
        cond_mask[:, :, goal_start:] = 1.0
        sigma = self._sample_sigma(bsz, device).view(bsz, 1, 1, 1, 1)
        noise = torch.randn_like(clean_latent)
        target_velocity = noise - clean_latent
        xt = noise * sigma + clean_latent * (1 - sigma)
        xt = clean_latent * cond_mask + xt * (1 - cond_mask)
        in_timestep = (1.0 - cond_mask[:, 0, :, 0, 0]) * sigma.view(bsz, 1) * 1000.0

        with self._autocast(device):
            feats, velocity = self.backbone.run_dit(xt, in_timestep, text_embeds, n_goal_frames=ng)

        obs_hidden, goal_hidden = self._slice_obs_goal(feats, goal_start, t_lat)
        if self.goal_time_proj is not None:
            if GOAL_OFFSET_KEY not in batch:
                raise KeyError(
                    f"goal_h_embed_dim>0 needs per-sample {GOAL_OFFSET_KEY!r} (set by the dataset's goal_frame_range)"
                )
            goal_hidden = self._add_goal_time_diff(
                goal_hidden, batch[GOAL_OFFSET_KEY].to(device).float().view(bsz, -1)
            )
        if self.training and cfg.goal_dropout_p > 0:
            keep = (torch.rand(bsz, ng, device=device) >= cfg.goal_dropout_p).to(goal_hidden.dtype)
            m = keep.repeat_interleave(self.tokens_per_frame, dim=1)
            if self.n_taps > 1:
                m = m.repeat(1, self.n_taps)
            goal_hidden = goal_hidden * m.unsqueeze(-1)
        vl_embs = torch.cat([obs_hidden, goal_hidden], dim=1)

        with self._autocast(device):
            actions = batch[ACTION][:, -cfg.chunk_size :].to(device=device, dtype=vl_embs.dtype)
            reps = cfg.repeated_diffusion_steps
            state = self._state(batch)
            state_rep = (
                state.to(device=device, dtype=vl_embs.dtype).repeat(reps, 1, 1) if state is not None else None
            )
            action_loss = self.action_model(vl_embs.repeat(reps, 1, 1), actions.repeat(reps, 1, 1), state_rep)

        fut_mask = (1.0 - cond_mask).expand_as(velocity)
        sq_err = (velocity.float() - target_velocity.float()) ** 2
        video_loss = (sq_err * fut_mask).sum() / fut_mask.sum().clamp_min(1.0)

        loss = action_loss + cfg.video_loss_weight * video_loss
        return loss, {"action_loss": action_loss.item(), "video_loss": video_loss.item()}

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _generate_goal_latent(
        self, obs_latent: Tensor, text_embeds: Tensor, before_step: Callable[[], None] | None = None
    ) -> Tensor:
        """Euler flow-matching sampler over the video DiT: the goal latent ``[B, 16, 1, h, w]`` at ``H_g``.

        ``before_step`` runs ahead of every Euler step; the goal worker uses it to yield the GPU.
        """
        cfg = self.config
        bsz = obs_latent.shape[0]
        obs_n, nf = self.obs_latent_frames, cfg.n_future_frames
        future = torch.randn(
            (bsz, obs_latent.shape[1], nf, *obs_latent.shape[3:]),
            dtype=torch.float32,
            device=obs_latent.device,
        )
        sigmas = torch.linspace(1.0, 0.0, cfg.goal_gen_steps + 1, device=obs_latent.device)
        clean_ts = torch.zeros(bsz, obs_n, device=obs_latent.device)
        for k in range(cfg.goal_gen_steps):
            if before_step is not None:
                before_step()
            sig = sigmas[k]
            d_sig = (sigmas[k] - sigmas[k + 1]).item()
            xt = torch.cat([obs_latent.float(), future], dim=2)
            in_timestep = torch.cat([clean_ts, sig.expand(bsz, nf) * 1000.0], dim=1)
            with self._autocast(obs_latent.device):
                _, velocity = self.backbone.run_dit(xt, in_timestep, text_embeds, n_goal_frames=0)
            future = future - velocity[:, :, obs_n : obs_n + nf].float() * d_sig
        return future

    @torch.no_grad()
    def _decode_goal_image(self, goal_latent: Tensor) -> Tensor:
        """VAE-decode the goal latent ``[B, 16, 1, h, w]`` to a uint8 frame ``[B, 3, H, W]``."""
        goal = self.backbone.decode_latents(goal_latent)[:, -1]
        return goal.clamp(0.0, 1.0).mul(255).to(torch.uint8)

    def _save_goal_preview(self, obs_frame: Tensor, goal_image: Tensor) -> None:
        """Write the conditioning frame above the imagined goal frame, one PNG per batch element.

        Args:
            obs_frame: The stitched frame the goal was imagined from, ``[B, 3, H, W]`` in ``[0, 1]``.
            goal_image: The decoded goal, uint8 ``[B, 3, H, W]``.
        """
        from PIL import Image

        out_dir = Path(self.config.goal_preview_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        obs = obs_frame.float().clamp(0.0, 1.0).mul(255).to(torch.uint8).cpu()
        goal = goal_image.cpu()
        jobs = []
        for b in range(goal.shape[0]):
            pair = torch.cat([obs[b], goal[b]], dim=-2)  # observation on top, imagination below
            name = f"ep{self._episode_idx:03d}_goal{self._goal_idx:03d}"
            path = out_dir / (f"{name}.png" if goal.shape[0] == 1 else f"{name}_b{b}.png")
            jobs.append((pair.permute(1, 2, 0).numpy(), path))
        self._goal_idx += 1

        def _write():  # PNG encode + file write off the control thread (PIL releases the GIL)
            for array, path in jobs:
                Image.fromarray(array).save(path, compress_level=1)

        threading.Thread(target=_write, name="glancewam-goal-preview", daemon=True).start()

    def _submit_goal(self, obs_latent: Tensor, text_embeds: Tensor, frames: Tensor, row: int) -> None:
        """Send this tick's observation to the worker and remember when (and, for previews, what) it was."""
        preview = bool(self.config.goal_preview_dir)
        self._goal_worker.submit(obs_latent, text_embeds, decode=preview)
        self._goal_request_row = row
        self._goal_request_frame = frames[:, -1].cpu() if preview else None

    def _install_goal(self, goal: Tensor, goal_image: Tensor | None) -> None:
        """Make the worker's reply the held goal; its clock starts at the request's frame."""
        self._goal_latent = goal
        self._goal_frame_row = self._goal_request_row
        self._goal_request_row = None
        if goal_image is not None and self._goal_request_frame is not None:
            self._save_goal_preview(self._goal_request_frame, goal_image)
        self._goal_request_frame = None

    def _request_due(self, row: int) -> bool:
        """Time for the next goal frame: ``goal_refresh_rows`` after the newest goal frame (held or in flight)."""
        if self._goal_worker is None or self._goal_frame_row is None:
            return False
        newest = self._goal_request_row if self._goal_request_row is not None else self._goal_frame_row
        return row >= newest + self.goal_refresh_rows

    def _goal_step_queued(self, batch: dict[str, Tensor], row: int) -> None:
        """Goal bookkeeping on a queued tick: install a reply that has arrived, send a request that is due."""
        if self._goal_worker is None:
            return
        device = self.backbone.device
        reply = self._goal_worker.poll(device)
        if reply is not None:
            self._install_goal(*reply)
        if self._request_due(row):
            if self._fast is not None and self._fast.compile_mode is not None:
                torch.compiler.cudagraph_mark_step_begin()
            frames = self._stitched_window(batch)[:, : self.config.num_history_frames]
            self._submit_goal(self._encode_obs(frames), self._language(batch), frames, row)

    def _goal_step_chunk(
        self, obs_latent: Tensor, text_embeds: Tensor, frames: Tensor, row: int, block: bool = True
    ) -> None:
        """Goal bookkeeping on a chunk tick: make sure a valid goal is held before the decode.

        ``block=False`` (the RTC engine) never waits for the worker: with nothing held yet the first goal
        is requested and this chunk decodes with the no-goal fallback, so the call's latency, which the
        engine turns into the overlap of every later chunk, never includes the video sampler.
        """
        if self._goal_worker is None:
            if self._goal_frame_row is None or row - self._goal_frame_row >= self.goal_refresh_rows:
                self._goal_latent = self._generate_goal_latent(obs_latent, text_embeds)
                self._goal_frame_row = row
                if self.config.goal_preview_dir:
                    self._save_goal_preview(frames[:, -1], self._decode_goal_image(self._goal_latent))
            return
        reply = self._goal_worker.poll(obs_latent.device)
        if reply is not None:
            self._install_goal(*reply)
        if self._goal_frame_row is None:
            # Episode start: nothing held, so this one goal is made synchronously (unless told not to).
            if self._goal_request_row is None:
                self._submit_goal(obs_latent, text_embeds, frames, row)
            if block:
                self._install_goal(*self._goal_worker.collect(obs_latent.device))
        # Otherwise the held goal is used as is; if its frame is older than the horizon (the next reply
        # has not landed yet) the goal-time diff floors at 1, which training covered.

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """Predict ``[B, chunk_size, action_dim]`` normalized actions, refreshing the held goal on its cadence.

        ``inference_delay`` / ``prev_chunk_left_over`` are the RTC engine's call shape: the chunk is guided
        onto the prefix (``FlowmatchingActionHead.predict_action``) and, since that engine calls from a
        background thread whenever its queue runs low rather than once per row, the goal row clock runs on
        wall-clock time with the chunk's first action ``inference_delay`` rows away; nothing in the call
        waits on the goal sampler.
        """
        self.eval()
        cfg = self.config
        if self._fast is None:
            self._warmup_fast_path()  # no reset() since construction (e.g. direct use); no-op off CUDA
        if self._fast is not None and self._fast.compile_mode is not None:
            torch.compiler.cudagraph_mark_step_begin()  # one chunk decode = one CUDA-graph step
        inference_delay = kwargs.get("inference_delay")
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
        rtc_call = inference_delay is not None
        frames = self._stitched_window(batch)[:, : cfg.num_history_frames]
        text_embeds = self._language(batch)
        obs_latent = self._encode_obs(frames)
        bsz = obs_latent.shape[0]

        if self._goal_latent is not None and self._goal_latent.shape[0] != bsz:
            self._drop_goal()
        row = self._rtc_row(int(inference_delay)) if rtc_call else self._tick
        self._goal_step_chunk(obs_latent, text_embeds, frames, row, block=not rtc_call)
        busy = self._goal_worker.gpu_busy() if self._goal_worker is not None else nullcontext()
        with busy:
            actions = self._decode_chunk(
                batch,
                obs_latent,
                text_embeds,
                self._goal_latent,
                row,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
        # A goal frame that falls on this tick is sent after the decode (same frame), so the worker
        # starts in the idle window that follows.
        if self._request_due(row):
            self._submit_goal(obs_latent, text_embeds, frames, row)
        if not rtc_call:
            self._tick += cfg.n_action_steps
        return actions

    def _decode_chunk(
        self,
        batch: dict[str, Tensor],
        obs_latent: Tensor,
        text_embeds: Tensor,
        g_hat: Tensor | None,
        row: int = 0,
        *,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
    ) -> Tensor:
        """One goal-conditioned DiT feature pass plus the action head: ``[B, chunk_size, action_dim]``.

        ``g_hat=None`` (no goal held yet) decodes with the no-goal fallback training covered through goal
        dropout: zero goal tokens into the head. Obs tokens never attend the goal, so a zero latent in the
        window leaves their features untouched.
        """
        cfg = self.config
        bsz, device = obs_latent.shape[0], obs_latent.device
        obs_n, nf, ng = self.obs_latent_frames, cfg.n_future_frames, cfg.n_goal_frames
        filler = obs_latent[:, :, -1:].expand(-1, -1, nf, -1, -1)
        no_goal = g_hat is None
        if no_goal:
            g_hat = obs_latent.new_zeros((bsz, obs_latent.shape[1], ng, *obs_latent.shape[3:]))
        window = torch.cat([obs_latent, filler, g_hat], dim=2)
        t_lat = window.shape[2]
        goal_start = t_lat - ng
        in_timestep = torch.zeros(bsz, t_lat, device=device)
        in_timestep[:, obs_n:goal_start] = _FILLER_TIMESTEP
        with self._autocast(device):
            if self._fast is not None:
                feats = self._fast.features(window, in_timestep, text_embeds, ng)
            else:
                feats, _ = self.backbone.run_dit(window, in_timestep, text_embeds, n_goal_frames=ng)

        obs_hidden, goal_hidden = self._slice_obs_goal(feats, goal_start, t_lat)
        if self.goal_time_proj is not None:
            distance = (
                self._goal_distance(row) if self._goal_frame_row is not None else self.goal_horizon_rows
            )
            g_rows = torch.full((bsz,), distance, device=device)
            goal_hidden = self._add_goal_time_diff(goal_hidden, g_rows)
        if cfg.eval_drop_goal or no_goal:
            goal_hidden = torch.zeros_like(goal_hidden)
        vl_embs = torch.cat([obs_hidden, goal_hidden], dim=1)

        state = self._state(batch)
        with self._autocast(device):
            state_t = state.to(device=device, dtype=vl_embs.dtype) if state is not None else None
            actions = self.action_model.predict_action(
                vl_embs,
                state_t,
                rtc_processor=self.rtc_processor,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
            )
        # ``clone``: the compiled head's output lives in the CUDA-graph pool and the queue outlives the step.
        return actions.float().clone()

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        else:
            self._goal_step_queued(batch, self._row)
        return self._action_queue.popleft()
