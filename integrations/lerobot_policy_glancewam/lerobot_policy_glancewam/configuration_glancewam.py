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
"""GlanceWAM: sparse test-time imagination for video-world-model VLAs (https://arxiv.org/abs/2608.23927).

A SkyReels-V2 diffusion-forcing video DiT is co-trained on a video loss and an action loss. One DiT pass
over the latent window ``[obs, future @ +H_g, goal @ +g]`` yields the features a GR00T-style flow-matching
head reads and the velocity target for the video loss. A 3-class FlexAttention mask keeps the *goal*
(the paper's lookahead frame) a read-only side-channel: obs and future tokens never attend it. At
inference the same DiT generates the goal latent on a slow clock (every ``future_frame_idx`` rows) while
the head decodes action chunks at control rate.
"""

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.utils.constants import ACTION

SKYREELS_DF_1_3B = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"

# GR00T N1.5 DiT presets (``action_model_type``).
DIT_PRESETS: dict[str, dict[str, int]] = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


@PreTrainedConfig.register_subclass("glancewam")
@dataclass
class GlanceWAMConfig(PreTrainedConfig):
    """Configuration of :class:`~lerobot_policy_glancewam.modeling_glancewam.GlanceWAMPolicy`.

    Defaults are the real-robot recipe: 20 Hz data, a 3 s goal horizon (60 rows), 1.2 s action chunks
    (24 rows), two cameras stitched side by side into one 224x448 frame. The released LIBERO recipe
    (``glancewam_libero``) used ``chunk_size=8`` and a 48-row horizon; converted checkpoints carry
    their own values.
    """

    n_obs_steps: int = 1
    chunk_size: int = 24  # GR00T ``action_horizon``
    n_action_steps: int = 24

    # === World model backbone (SkyReels-V2 DF, Wan2.1 architecture) ===
    base_model_path: str = SKYREELS_DF_1_3B
    # Parameter dtype of the DiT. ``float32`` keeps fp32 masters for mixed-precision training; the
    # forward always runs under bf16 autocast.
    dtype: str = "float32"
    # (H, W) of the frame fed to the Wan VAE; both must be multiples of 16 (8x VAE, 2x patch).
    vae_input_size: tuple[int, int] = (224, 448)
    # Optional (H, W) every camera frame is shrunk to (bilinear, antialiased) before the per-view
    # ``view_size`` resize in ``stitch_views``. Serialized to the checkpoint, so rollouts replay the
    # dataloader-side ``dataset.image_transforms.resize`` used in training on the native camera frames
    # (a no-op in training itself, where the dataloader already delivers frames at this size).
    image_resize_shape: tuple[int, int] | None = None
    # Image keys in stitching order. ``None`` = sorted image feature keys.
    camera_order: list[str] | None = None
    # "side_by_side": every view resized to ``view_size`` square and concatenated on width (LIBERO,
    # 2 cams -> 224x448). "primary_inset": the first view at ``view_size`` square on the left, the
    # others stacked as ``view_size // n_insets`` insets in a right-hand column (RoboCasa kitchen,
    # 3 cams -> 224x336).
    camera_layout: str = "side_by_side"
    # Square size of the primary view before stitching.
    view_size: int = 224
    # DiT blocks whose outputs feed the action head (E7 multi-tap: several taps concatenated on the token
    # axis, each with a zero-init tap embedding). Negative indices count from the end.
    extract_layers: list[int] = field(default_factory=lambda: [5, 12, 19, 26])
    # SkyReels DF conditions on an fps token; the table has 16 and 24 only.
    fps: int = 24
    # UMT5 prompt padding length (Wan pretraining used 512).
    text_max_length: int = 512
    num_history_frames: int = 1

    # === Window (native dataset rows) ===
    # The video branch predicts the frame at t+future_frame_idx (= H_g, the goal-generation horizon).
    future_frame_idx: int = 60
    # The action head conditions on a hindsight goal frame at t+g, g ~ U{lo..hi} per training sample.
    goal_frame_range: tuple[int, int] = (1, 60)

    # === Video + action co-training ===
    video_loss_weight: float = 1.0
    sigma_sampling: str = "logitnormal"  # "uniform" | "logitnormal"
    sigma_shift: float = 5.0
    # The goal-aware 3-class mask (the mechanism). False = bidirectional ablation (no mask at all).
    clean_prefix_isolation: bool = True

    # === Goal (lookahead-frame) conditioning ===
    goal_dropout_p: float = 0.1
    goal_gen_steps: int = 10  # Euler steps of the video DiT sampler generating the goal latent at eval
    goal_h_embed_dim: int = 16  # sinusoidal width of the goal-time-diff embedding; 0 disables it
    # Zero the goal tokens at inference (the paper's no-goal control arm).
    eval_drop_goal: bool = False
    # Inspection toggle: directory to write the imagined goal to every time it is regenerated, as a PNG
    # holding the conditioning observation above the decoded goal frame. Off (None) by default -- it VAE-
    # decodes and writes inside the inference call, so it costs latency on the refresh tick.
    goal_preview_dir: str | None = None
    # Inference only: generate the goal latent in a worker process holding its own copy of the video
    # DiT, so the refresh never lands on the control thread. Goals run on a row clock: every
    # ``goal_refresh_rows`` the current tick's frame (a queued tick VAE-encodes it, ~10 ms) is sent to the
    # worker, and the reply is installed the tick it lands (~10 rows later on a 4090). A goal is always
    # ``future_frame_idx`` rows ahead of its own frame, so the head's goal-time diff is horizon minus the
    # age of that frame, floored at 1 -- exactly the training definition. Only the first goal of an
    # episode blocks. The worker loads the checkpoint at ``pretrained_path`` (two copies of the DiT).
    goal_async: bool = False
    # Rows between goal frames. None = ``future_frame_idx``: a goal is imagined every horizon. Note the
    # reply lands ~10 rows after its frame, so a chunk tick inside that window still uses the previous
    # goal, at a distance floored at 1 when the period equals the horizon; a period that is a multiple
    # of the chunk (48 for chunk 24) keeps every chunk on an unexpired goal.
    goal_refresh_rows: int | None = None

    # === Real-Time Chunking (arXiv 2506.07339) ===
    # Inference-time guidance in the action head's sampler, shared with pi0 / pi0.5 (``RTCProcessor``):
    # the first ``inference_delay`` actions of a new chunk are pulled onto the unexecuted tail of the
    # previous one and the pull fades out over ``execution_horizon``. Set by ``lerobot-rollout
    # --inference.type=rtc`` (``--inference.rtc.*``); None = plain chunking.
    rtc_config: RTCConfig | None = None
    # Rows per second of the training data (its fps). Under the RTC engine the policy is called from a
    # background thread whenever the queue runs low, not once per row, so the goal row clock (refresh
    # cadence, goal-time diff) runs on wall-clock time at this rate. Match ``lerobot-rollout --fps``.
    rows_per_second: float = 20.0

    # === Inference fast path (the paper's 48 ms/chunk action path) ===
    # Chunk decode through a hook-free DiT truncated after the last tap (no velocity head), with the VAE
    # encoder, that DiT and the action head under ``torch.compile`` and the goal-aware block mask cached
    # per batch size. 4090, c24_h60 checkpoint: 82 -> 54 ms per chunk in float32, 46 ms in bfloat16.
    # Compiles on the first ``reset()`` after construction (~30 s, before the episode clock). CUDA only.
    fast_inference: bool = True
    # ``torch.compile`` mode of the fast path; None runs the truncated DiT eagerly (no compile, ~5 ms win).
    fast_inference_compile_mode: str | None = "reduce-overhead"

    # === Action head (GR00T N1.5 flow matching) ===
    action_model_type: str = "DiT-B"
    action_hidden_size: int = 1024  # MLP width of the state/action encoders and decoder
    num_action_head_layers: int = 16
    action_head_dropout: float = 0.2
    num_target_vision_tokens: int = 32
    max_seq_len: int = 1024
    add_pos_embed: bool = True
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    num_inference_timesteps: int = 4
    # Each conditioning vector is repeated this many times with independent noise/time draws.
    repeated_diffusion_steps: int = 8
    t_loss_weighting: bool = False
    # Width of the head's state encoder (GR00T ``state_dim``); 0 builds none. Released checkpoints carry a
    # 7-wide encoder even for the stateless LIBERO recipe, so it is independent of ``use_state``.
    action_head_state_dim: int = 7
    # Feed proprioception to the head (the kitchen recipe); LIBERO is stateless.
    use_state: bool = False

    # === Speed ===
    # Per-block torch.compile of the DiT (applied to ``block.forward`` before any FSDP sharding).
    compile_dit: bool = False
    compile_dit_mode: str | None = None
    compile_dit_dynamic: bool = True
    # torch.compile the frozen Wan VAE encoder during training (eager at eval: other shapes).
    compile_vae: bool = True
    # SDPA backend for the non-flex attention paths: "auto" | "cudnn" | "flash" | "mem_efficient" | "math".
    sdpa_backend: str = "cudnn"
    gradient_checkpointing: bool = False

    # Action dims left un-normalized (the active mode's stats forced to [-1, 1]): GR00T-style masked dims such as a
    # binary gripper stored raw in {0, 1}. Indices into the action vector; negative counts from the end.
    # Checked against the dataset stats when the processors are built: a dim whose recorded range falls
    # outside [-1, 1] is rejected rather than silently passed to the loss unnormalized.
    identity_action_dims: list[int] = field(default_factory=lambda: [-1])

    # === Action post-processing ===
    # Binarize the gripper dimension and map it to the LIBERO convention (-1 open, +1 close).
    gripper_mode: str = "libero"  # "libero" | "none"
    gripper_index: int = -1

    # QUANTILES (q01/q99 -> [-1, 1]): the min/max envelope is set by single outlier frames, so most of
    # the range is wasted; q01/q99 keeps the bulk of the data on the full scale. Checkpoints converted
    # from the GlanceWAM trainer (``convert_checkpoint``) were trained MIN_MAX; their saved config keeps that.
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # === Optimizer / scheduler (released LIBERO recipe) ===
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-8
    optimizer_grad_clip_norm: float = 1.0
    # `cosine_with_min_lr`: linear warmup, then a half-cosine from `optimizer_lr` to
    # `scheduler_decay_lr` ending at `scheduler_decay_steps`. Units are trainer steps.
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 20_000
    scheduler_decay_lr: float = 5e-7
    # Warm start from a model-only checkpoint of step N: set N to continue N's LR curve (with
    # scheduler_warmup_steps / scheduler_decay_steps set to the ORIGINAL run's values).
    scheduler_step_offset: int = 0

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError(f"Invalid dtype: {self.dtype}")
        if self.action_model_type not in DIT_PRESETS:
            raise ValueError(
                f"action_model_type must be one of {sorted(DIT_PRESETS)}, got {self.action_model_type}"
            )
        h, w = self.vae_input_size
        if h % 16 or w % 16:
            raise ValueError(f"vae_input_size={self.vae_input_size} must be multiples of 16")
        if self.future_frame_idx <= 0:
            raise ValueError("future_frame_idx must be > 0 (the video target is the goal-generation horizon)")
        lo, hi = self.goal_frame_range
        if not (1 <= lo <= hi):
            raise ValueError(f"goal_frame_range must satisfy 1 <= lo <= hi, got {self.goal_frame_range}")
        if self.sigma_sampling not in ("uniform", "logitnormal"):
            raise ValueError(f"sigma_sampling must be 'uniform' or 'logitnormal', got {self.sigma_sampling}")
        if self.camera_layout not in ("side_by_side", "primary_inset"):
            raise ValueError(
                f"camera_layout must be 'side_by_side' or 'primary_inset', got {self.camera_layout}"
            )
        if self.gripper_mode not in ("libero", "none"):
            raise ValueError(f"gripper_mode must be 'libero' or 'none', got {self.gripper_mode}")
        if self.goal_h_embed_dim % 2:
            raise ValueError("goal_h_embed_dim must be even")
        if self.fps not in (16, 24):
            raise ValueError("fps must be 16 or 24 (the only tokens in the SkyReels fps table)")
        if self.rows_per_second <= 0:
            raise ValueError(f"rows_per_second must be positive, got {self.rows_per_second}")
        refresh = self.goal_refresh_rows if self.goal_refresh_rows is not None else self.future_frame_idx
        if not (0 < refresh <= self.future_frame_idx):
            raise ValueError(
                f"goal_refresh_rows ({self.goal_refresh_rows}) must be in (0, future_frame_idx={self.future_frame_idx}]"
            )

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("GlanceWAM needs at least one image feature.")
        if ACTION not in self.output_features:
            raise ValueError("GlanceWAM needs an `action` output feature.")
        if self.use_state:
            if self.robot_state_feature is None:
                raise ValueError("use_state=True but the input features carry no `observation.state`.")
            if int(self.robot_state_feature.shape[0]) != self.action_head_state_dim:
                raise ValueError(
                    f"use_state=True: observation.state has {self.robot_state_feature.shape[0]} dims but "
                    f"action_head_state_dim is {self.action_head_state_dim}."
                )
        for key in self.camera_keys:
            if key not in self.image_features:
                raise ValueError(
                    f"camera_order entry {key!r} is not an image feature ({list(self.image_features)})."
                )

    @property
    def camera_keys(self) -> list[str]:
        """Image feature keys in stitching order."""
        return list(self.camera_order) if self.camera_order else sorted(self.image_features)

    @property
    def action_dim(self) -> int:
        return int(self.output_features[ACTION].shape[0])

    @property
    def state_dim(self) -> int:
        feature: PolicyFeature | None = self.robot_state_feature
        return int(feature.shape[0]) if (self.use_state and feature is not None) else 0

    @property
    def n_future_frames(self) -> int:
        return 1

    @property
    def n_goal_frames(self) -> int:
        return 1

    @property
    def window_frames(self) -> int:
        """Frames per image key in a training sample: history, video target, goal."""
        return self.num_history_frames + self.n_future_frames + self.n_goal_frames

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> None:
        # This plugin is inference-only: training needs the per-sample hindsight goal frame
        # (``goal_frame_range``), which upstream LeRobot datasets do not provide. Train with the
        # GlanceWAM trainer (transformers' ``cosine_with_min_lr``, the ``scheduler_*`` fields above).
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def image_observation_delta_indices(self) -> list[int]:
        """History frames, then the video-target frame at +future_frame_idx.

        The per-sample random goal frame is appended from ``goal_frame_range`` at training time (by
        the GlanceWAM trainer's dataloader); it lands last in the frame axis.
        """
        return list(range(-(self.num_history_frames - 1), 1)) + [self.future_frame_idx]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def language_feature_key(self) -> str:
        return "observation.language.embedding"

    @property
    def language_feature(self) -> PolicyFeature:
        return PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.text_max_length, 4096))
