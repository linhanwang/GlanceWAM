# Copyright 2026 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
GlanceWAM — video world model + action head with lookahead-frame conditioning.

A standalone sibling of ``CotrainBaseline`` (the imagination-free video+action
co-training baseline). It adds ONE thing: the action head conditions on a
**lookahead frame** the video DiT generates for itself. Trained from the raw
SkyReels-V2 DF checkpoint on demonstrations only; the same DiT is both the
asynchronous proposer and the perception backbone.

NAMING: the paper's *lookahead frame* is called the **goal image** throughout this
file (``goal_*`` — ``goal_frame_range``, ``goal_hidden``, ``goal_time_proj``). The
identifiers are load-bearing — ``goal_time_proj`` is a weight key in released
checkpoints — so they were kept as-is rather than renamed for the release.
Elsewhere: the paper's *asynchronous proposer* is ``_generate_goal_latent``, its
*non-interfering 3-class attention mask* is ``enable_goal_prefix_lm_flex_attention``
(``modules/world_model/speedup_patch.py``), and its *staleness-robust horizon
randomization* is ``datasets.vla_data.goal_frame_range`` + ``_add_goal_time_diff``.

E8.0 multi-horizon (the goal-image design): with
``datasets.vla_data.future_frame_idxs: [30, 60]`` + ``goal_frame_ranges:
[[1, 30], [31, 60]]`` the window generalizes to [obs, f@30, f@60, g_short,
g_long] — one independently-noised video target (diffusion forcing proper) and
one clean hindsight goal per horizon; the sampler denoises both future slots
jointly (the DF chain) and the head reads both goals with per-goal time embeds
and per-goal dropout. Keys absent => the historical single-horizon path,
bit-identical.

The design (settled 2026-07-23, see the plan §2 + this conversation):

  - **Two future frames per window.** The video branch trains at a FIXED horizon
    H_g (its noised target, the plain-cotrain video loss, UNCHANGED). The action
    head conditions on a separate CLEAN **goal** frame sampled at a hindsight
    offset uniform in (0, H_g] — the distribution it faces mid-cycle at inference
    (the plan's cadence rule). The dataloader ships both; the window is
    ``[obs..., future@H_g, goal@uniform]``.

  - **Causal-VAE two-pass encode.** The Wan VAE is causal, so within one encode
    the last group's latent sees everything before it. To keep BOTH the future
    latent bit-identical to the plain cotrain AND the goal latent free of the
    ground-truth future (which does not exist at inference, where the goal is
    generated), the goal is encoded in a SECOND pass ``[obs, goal]`` — obs-
    conditioned, future-free — matching the obs-conditioned latents the video head
    generates at test time. Pass 1 ``[obs, future]`` is the exact champion encode.

  - **Goal as a read-only side-channel (3-class mask).** The goal latent is placed
    as a third DiT frame (``[obs, future, goal]``, RoPE coords 0/1/2). A goal-aware
    prefix-LM FlexAttention mask (``speedup_patch.enable_goal_prefix_lm_flex_attention``)
    isolates it: obs->obs, future->obs+future, goal->obs+goal. NOTHING attends to
    the goal, so obs tokens (the action read-point) and future tokens (the video
    branch) are BIT-IDENTICAL to the plain cotrain; the goal is grounded on obs and
    its tokens are read out and concatenated into the action head's cross-attention
    stream (``vl_embs = [obs_tokens ; goal_tokens]``). The action head is unchanged.

  - **Goal-dropout / no-goal control.** With prob ``goal_dropout_p`` the goal tokens
    are zeroed (train), and the eval control arm zeroes them (``drop_goal=True``) —
    so the model keeps a no-goal fallback that targets the plain 0.680 baseline.

  - **Self-generating inference.** ``predict_action`` runs the video DiT flow-matching
    sampler to generate the goal latent ``g_hat`` at H_g (no VAE decode — g_hat is
    already in the normalized latent space), places it at the goal slot, and decodes
    actions conditioned on it. Refreshed on the H_g cadence by the eval driver.

Model-only + fake-data validated in ``__main__`` (loss/backward, obs-token
isolation leak check, self-generating predict_action shape).
"""

import os
import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch import nn

from glancewam.dataloader.camera_utils import stitch_primary_with_insets, stitch_views_side_by_side
from glancewam.dataloader.image_tools import to_pil_preserve
from glancewam.model.framework.base_framework import baseframework
from glancewam.model.framework.share_tools import merge_framework_config, select_backbone_features
from glancewam.model.framework.wam.goal_time_embed import sinusoidal_h_embed
from glancewam.model.modules.action_model.GR00T_ActionHeader import (
    FlowmatchingActionHead,
    get_action_model,
    seeded_randn,
)
from glancewam.model.modules.world_model import get_world_model
from glancewam.model.modules.world_model.speedup_patch import (
    compile_dit_blocks,
    enable_goal_bidirectional_attention,
    enable_goal_prefix_lm_flex_attention,
)
from glancewam.model.tools import FRAMEWORK_REGISTRY
from glancewam.training.trainer_utils import initialize_overwatch
from glancewam.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


# ----------------------------------------------------------------------------
# Diffusion-forcing noising helpers — copied (NOT imported) from
# CotrainBaseline per the repo's per-file code-isolation rule.
# ----------------------------------------------------------------------------
def sample_train_sigma_t(batch_size, distribution, device, dtype=torch.float32, shift=5):
    """One shared sigma per sample for the noised future frame."""
    if distribution == "uniform":
        sigma_t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
    elif distribution == "logitnormal":
        t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
        sigma_t = shift * t / (1 + (shift - 1) * t)
    else:
        raise NotImplementedError(f"sigma_sampling {distribution!r} is not implemented.")
    return sigma_t.view(batch_size, 1, 1, 1, 1)


def get_flow_xt_and_target_v(clean_latent, sigma, cond_mask):
    """Rectified-flow interpolation + velocity target (Wan/SkyReels convention):
    x_t = (1-σ)·x0 + σ·noise, target v = noise - x0. The clean prefix (cond_mask==1,
    here obs AND goal) stays ground-truth in x_t."""
    noise = torch.randn_like(clean_latent)
    target_velocity = noise - clean_latent
    xt = noise * sigma + clean_latent * (1 - sigma)
    xt = clean_latent * cond_mask + xt * (1 - cond_mask)
    return xt, target_velocity


@dataclass
class GlanceWAMDefaultConfig:
    """SkyReelsV2-GR00T goal-image co-training defaults."""

    name: str = "GlanceWAM"

    # === World Model backbone (SkyReels-V2 DF-1.3B, Wan2.1 architecture) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
            "extract_layers": [-1],
            "combine_layers": "concat_tokens",
            "vae_input_size": [320, 576],
            "camera_concat": "none",
            "compile_dit": False,
            "compile_dit_mode": None,
            "compile_dit_dynamic": True,
            "compile_vae": True,
            "compile_vae_mode": None,
            "skip_vae": False,
            "skip_text_encoder": False,
            "num_history_frames": 1,
            "fps": 24,
            "text_max_length": 512,
            "resident_text_table": False,
            "resident_text_table_max_rows": 1024,
        }
    )

    # === Video + action co-training knobs (video branch is the plain cotrain) ===
    video_cotrain: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "lambda": 1.0,
            "sigma_sampling": "uniform",
            # The goal-aware 3-class mask (the mechanism). False = bidirectional
            # ablation: no attention mask at all — obs attends future + goal.
            "clean_prefix_isolation": True,
        }
    )

    # === Goal-image conditioning (the delta vs the plain cotrain) ===
    goal_conditioning: dict = field(
        default_factory=lambda: {
            "enabled": True,
            # Per-sample probability of zeroing the goal tokens fed to the action head
            # (train). Keeps a no-goal fallback in-distribution for the eval control arm.
            "dropout_p": 0.1,
            # Video-DiT flow-matching sampler steps used to generate the goal latent at
            # inference (predict_action). Eval-only; higher = better goal, slower refresh.
            "goal_gen_steps": 10,
            # Sinusoidal embedding width for the goal-time diff g (native rows between the
            # goal frame and the current obs) added to the goal tokens, so the action head
            # knows HOW FAR AHEAD the goal is (it trains on g ~ U(0,H_g]; at inference g
            # shrinks within each H_g refresh cycle). 0 disables the diff conditioning.
            "h_embed_dim": 16,
        }
    )

    # Legacy compat: factory functions fall back to qwenvl.base_vlm.
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
            "vl_hidden_dim": 1536,
            "num_vl_layers": 30,
        }
    )

    # === Action head (GR00T flow-matching / DiT diffusion) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "future_action_window_size": 7,
            "action_horizon": 8,
            "past_action_window_size": 0,
            "repeated_diffusion_steps": 8,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "t_loss_weighting": False,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 1536,  # aligned to WM hidden_size at runtime
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    obs_image_size: Optional[list] = None


# "SkyReelsV2GR00TGoalCotrain" is the pre-release name this framework shipped under; it is
# kept registered so checkpoints whose saved config.yaml carries it keep loading unchanged.
@FRAMEWORK_REGISTRY.register("SkyReelsV2GR00TGoalCotrain")
@FRAMEWORK_REGISTRY.register("GlanceWAM")
class GlanceWAM(baseframework):
    """SkyReels-V2 DF-1.3B backbone + GR00T flow-matching head, video+action
    co-trained with the action head conditioned on a goal image (the low level of
    the goal-image planner)."""

    _save_include_prefixes = (
        "backbone.transformer.",
        "action_model.",
        "goal_time_proj.",
        "tap_embed.",
    )
    _reconstructable_prefixes = ("backbone.text_encoder.", "backbone.vae.")

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(GlanceWAMDefaultConfig, config)

        # World model backbone (SkyReels-V2 DF DiT + UMT5 text encoder + Wan VAE).
        self.backbone = get_world_model(config=self.config)

        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        wm_cfg = self.config.framework.world_model
        if bool(wm_cfg.get("truncate_at_extract", False)):
            raise ValueError(
                "goal co-training needs the full DiT velocity output; "
                "set framework.world_model.truncate_at_extract=False."
            )
        self.combine_layers = wm_cfg.get("combine_layers", "concat_tokens")
        self.camera_concat = wm_cfg.get("camera_concat", "none")
        self.num_history_frames = int(wm_cfg.get("num_history_frames", 1))

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # Co-train knobs.
        vc = self.config.framework.get("video_cotrain", {})
        self.cotrain_enabled = bool(vc.get("enabled", True))
        self.cotrain_lambda = float(vc.get("lambda", 1.0))
        self.sigma_sampling = vc.get("sigma_sampling", "uniform")

        # Frame / latent accounting. The dataloader ships obs + ONE future frame
        # (video target @H_g) + ONE goal frame (@uniform). Each of the two special
        # frames is tiled across one temporal VAE group so the causal Wan VAE maps
        # each to exactly one latent frame — encoded in SEPARATE passes (see forward).
        tscale = int(self.backbone.vae_scale_factor_temporal)
        # E8.0 multi-horizon: datasets.vla_data.future_frame_idxs
        # + goal_frame_ranges switch the window from [obs, future, goal] to
        # [obs, f@h_1.., goal@band_1..] — one noised video target and one clean hindsight
        # goal per horizon, both futures with independent per-frame σ (diffusion forcing).
        # Keys absent => the historical single-horizon window, bit-identical.
        try:
            _ds_cfg = self.config.datasets.vla_data
            _fidxs = [int(i) for i in _ds_cfg.get("future_frame_idxs", None) or []]
            _granges = [list(r) for r in _ds_cfg.get("goal_frame_ranges", None) or []]
        except Exception:
            _fidxs, _granges = [], []
        if len(_fidxs) != len(_granges):
            raise ValueError(
                f"future_frame_idxs ({_fidxs}) and goal_frame_ranges ({_granges}) must be matched "
                f"lists (one video target + one hindsight band per horizon)."
            )
        self.n_future_frames = max(1, len(_fidxs))
        self.n_goal_frames = max(1, len(_granges))
        self._goal_horizon_rows = [float(r[1]) for r in _granges] or None  # band maxes
        self.window_frames = self.num_history_frames + self.n_future_frames + self.n_goal_frames
        self.encode_frames = self.num_history_frames + tscale  # per single-target pass
        self.n_tile = tscale
        # Clean obs occupies the leading latent frames (causal VAE, frame 0 -> latent 0).
        self.obs_latent_frames = (self.num_history_frames - 1) // tscale + 1
        if self.n_goal_frames > 1:
            logger.info(
                f">> [*] E8.0 multi-horizon goals: future_frame_idxs={_fidxs}, "
                f"goal_frame_ranges={_granges} -> window {self.window_frames} frames."
            )

        # Goal-image conditioning.
        gc = self.config.framework.get("goal_conditioning", {}) or {}
        self.goal_enabled = bool(gc.get("enabled", True))
        self.goal_dropout_p = float(gc.get("dropout_p", 0.1))
        self.goal_gen_steps = int(gc.get("goal_gen_steps", 10))
        if not self.goal_enabled:
            raise ValueError(
                "GlanceWAM requires goal_conditioning.enabled=True — "
                "use CotrainBaseline for the plain (no-goal) cotrain."
            )
        # Default: the goal-aware 3-class prefix-LM mask (the mechanism that keeps
        # obs/future tokens bit-identical while the goal side-channel is grounded on
        # obs). clean_prefix_isolation=False is the bidirectional ABLATION: no mask at
        # all, obs attends the noised future AND the goal. n_goal_frames (1 legacy,
        # >1 under E8.0 multi-horizon) trailing clean goal latent frames either way —
        # the 3-class mask generalizes as-is (goals attend obs + goals, nothing
        # attends goals).
        self.clean_prefix_isolation = bool(vc.get("clean_prefix_isolation", True))
        if self.clean_prefix_isolation:
            n_proc = enable_goal_prefix_lm_flex_attention(self.backbone.transformer)
            logger.info(
                f">> [*] goal_conditioning: goal-aware 3-class prefix-LM mask on {n_proc} attention processors "
                f"(n_goal_frames={self.n_goal_frames}, dropout_p={self.goal_dropout_p}, "
                f"gen_steps={self.goal_gen_steps})."
            )
        else:
            enable_goal_bidirectional_attention(self.backbone.transformer)
            logger.info(
                ">> [*] goal_conditioning: BIDIRECTIONAL ablation — no attention mask "
                f"(obs attends future+goal; n_goal_frames={self.n_goal_frames}, "
                f"dropout_p={self.goal_dropout_p}, gen_steps={self.goal_gen_steps})."
            )

        # Goal-time-diff conditioning: embed g (native rows between the goal frame and the
        # current obs — the goal offset at train time, the shrinking effective distance
        # within an H_g refresh cycle at inference) and ADD it to the goal tokens, so the
        # action head knows how far ahead the goal is. Zero-init projection => inert at
        # step 0. Default horizon for eval = the video-target future_frame_idx (H_g).
        self.goal_h_embed_dim = int(gc.get("h_embed_dim", 16))
        self.goal_time_proj = None
        if self.goal_h_embed_dim > 0:
            self.goal_time_proj = nn.Linear(self.goal_h_embed_dim, wm_hidden)
            nn.init.zeros_(self.goal_time_proj.weight)
            nn.init.zeros_(self.goal_time_proj.bias)
        try:
            self._default_goal_horizon = float(self.config.datasets.vla_data.get("future_frame_idx", 0)) or 60.0
        except Exception:
            self._default_goal_horizon = 60.0
        # Per-goal eval defaults: the band maxes under E8.0 (right-after-refresh distances),
        # the single H_g otherwise.
        self._default_goal_horizons = self._goal_horizon_rows or [self._default_goal_horizon]

        if bool(wm_cfg.get("compile_dit", False)):
            n = compile_dit_blocks(
                self.backbone,
                mode=wm_cfg.get("compile_dit_mode", None),
                dynamic=bool(wm_cfg.get("compile_dit_dynamic", True)),
            )
            logger.info(f">> [*] compile_dit=True: torch.compile applied to {n} DiT blocks.")

        # torch.compile the frozen Wan VAE encoder (training only; eager at eval) —
        # same off-graph swap as the plain cotrain so the saved state_dict holds one
        # canonical eager VAE-encoder copy.
        self._compile_vae = False
        if bool(wm_cfg.get("compile_vae", True)) and getattr(self.backbone, "vae", None) is not None:
            mode = wm_cfg.get("compile_vae_mode", None)
            object.__setattr__(self, "_vae_encoder_eager", self.backbone.vae.encoder)
            object.__setattr__(
                self, "_vae_encoder_compiled", torch.compile(self.backbone.vae.encoder, mode=mode, dynamic=False)
            )
            self._compile_vae = True
            logger.info(
                f">> [*] compile_vae=True: torch.compile on the Wan VAE encoder (mode={mode!r}, dynamic=False), "
                f"TRAINING ONLY."
            )

        # Latent tokens per frame (for slicing obs/goal tokens out of the flattened,
        # frame-major DiT sequence).
        H, W = wm_cfg.get("vae_input_size", [320, 576])
        spatial = int(self.backbone.vae_scale_factor_spatial)
        patch = self.backbone.transformer.config.patch_size
        if isinstance(patch, (list, tuple)):
            _, ph, pw = int(patch[0]), int(patch[1]), int(patch[2])
        else:
            ph = pw = int(patch)
        self.tok_h = (int(H) // spatial) // ph
        self.tok_w = (int(W) // spatial) // pw
        self.tokens_per_frame = self.tok_h * self.tok_w

        # --- E7.0: multi-tap conditioning (goal-image plan §5 E7) ----------------------
        # world_model.extract_layers with k > 1 taps feeds the action head per-tap
        # [obs ; goal] slices concatenated along the TOKEN axis (KV length ×k; the
        # head's cross-attention is length-agnostic, so zero new head params). A
        # zero-init per-tap embedding added to each tap's tokens makes tap identity
        # explicit while keeping step 0 bit-identical to pure concat. k == 1 keeps the
        # historical single-tap path bit-exact.
        n_blocks_dit = len(self.backbone.transformer.blocks)
        taps = [int(t) if int(t) >= 0 else n_blocks_dit + int(t) for t in wm_cfg.get("extract_layers", [-1])]
        if not all(0 <= t < n_blocks_dit for t in taps):
            raise ValueError(f"world_model.extract_layers {taps} out of range for {n_blocks_dit} DiT blocks")
        self.n_taps = len(taps)
        self.tap_embed = None
        if self.n_taps > 1:
            if taps != sorted(taps) or len(set(taps)) != self.n_taps:
                raise ValueError(
                    f"multi-tap conditioning needs ascending unique extract_layers (capture order == "
                    f"block order), got {taps}"
                )
            if self.combine_layers != "concat_tokens":
                raise ValueError("multi-tap conditioning requires world_model.combine_layers=concat_tokens")
            self.tap_embed = nn.Embedding(self.n_taps, wm_hidden)
            nn.init.zeros_(self.tap_embed.weight)
            logger.info(
                f">> [*] E7.0 multi-tap conditioning: taps={taps} -> head KV length x{self.n_taps} "
                f"({self.n_taps}x(obs+goal) tokens), zero-init per-tap embedding."
            )

        self.fps = [self.backbone._fps_index] if self.backbone.transformer.config.inject_sample_info else None

        # Resident-text table (ported from the plain cotrain; bit-exact cached UMT5).
        self._resident_text_table = None
        self._resident_text_max_rows = int(wm_cfg.get("resident_text_table_max_rows", 1024))
        self._resident_text_overflow_warned = False
        self._resident_text_pending = {}
        if bool(wm_cfg.get("resident_text_table", False)):
            self._resident_text_table = {}
            self._resident_text_pending = self._try_prefill_resident_text() or {}
            logger.info(
                f">> [*] resident_text_table=True: caching unique cached-text embeds on GPU "
                f"(max {self._resident_text_max_rows} rows; eager-prefilled {len(self._resident_text_pending)})."
            )

        self._sdpa_ctx_factory = self._build_sdpa_ctx_factory()

    # ----- Cached-text / resident-text-table path (ported from CotrainBaseline) -----
    @staticmethod
    def _check_all_or_none(examples: List[dict], key: str, label: str) -> bool:
        n_with = sum(1 for e in examples if key in e)
        if n_with == 0:
            return False
        if n_with != len(examples):
            raise RuntimeError(
                f"{label}: {n_with}/{len(examples)} samples in this batch carry `{key}` — must be all-or-none."
            )
        return True

    def _maybe_stack_cached_text(self, examples: List[dict]):
        if not examples:
            return None
        have_embeds = self._check_all_or_none(examples, "lang_embed", "UMT5 text cache")
        if self.training and self._resident_text_table is not None:
            covered = have_embeds or all(
                e["lang"] in self._resident_text_table or e["lang"] in self._resident_text_pending for e in examples
            )
            if covered:
                device = next(self.backbone.transformer.parameters()).device
                return self._resident_text_rows(examples, device)
            return None
        if not have_embeds:
            return None
        device = next(self.backbone.transformer.parameters()).device
        return torch.from_numpy(np.stack([e["lang_embed"] for e in examples])).to(device=device, dtype=torch.bfloat16)

    def _try_prefill_resident_text(self):
        try:
            ds_cfg = self.config.datasets.vla_data
            t5_cfg = ds_cfg.t5_cache
            if not bool(t5_cfg.get("enabled", False)):
                return None
            from glancewam.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
            from glancewam.dataloader.t5_cache import T5CacheReader

            root = Path(ds_cfg.data_root_dir).expanduser()
            names = sorted({entry[0] for entry in DATASET_NAMED_MIXTURES[ds_cfg.data_mix]})
            pending = {}
            for name in names:
                reader = T5CacheReader(
                    root / name,
                    t5_model_id=t5_cfg.get("model_id"),
                    max_length=int(t5_cfg.get("max_length", 512)),
                )
                for prompt in reader.prompts:
                    if prompt not in pending:
                        embed, _ = reader.lookup(prompt)
                        pending[prompt] = embed
            if len(pending) > self._resident_text_max_rows:
                return None
            return pending
        except Exception as e:
            logger.info(f">> [!] resident_text_table eager prefill unavailable ({e}); lazy fill from batch lang_embed.")
            return None

    def _resident_text_rows(self, examples: List[dict], device):
        table = self._resident_text_table
        if self._resident_text_pending:
            pending = self._resident_text_pending
            staged = torch.from_numpy(np.stack(list(pending.values()))).to(device=device, dtype=torch.bfloat16)
            for prompt, row in zip(pending, staged, strict=True):
                table[prompt] = row
            self._resident_text_pending = {}
        misses = {}
        for e in examples:
            prompt = e["lang"]
            if prompt not in table and prompt not in misses:
                if "lang_embed" not in e:
                    raise KeyError(f"resident_text_table: prompt not in table and no `lang_embed`: {prompt!r}.")
                misses[prompt] = e["lang_embed"]
        overflow = {}
        if misses:
            staged = torch.from_numpy(np.stack(list(misses.values()))).to(device=device, dtype=torch.bfloat16)
            for prompt, row in zip(misses, staged, strict=True):
                if len(table) < self._resident_text_max_rows:
                    table[prompt] = row
                else:
                    overflow[prompt] = row
        rows = [table[e["lang"]] if e["lang"] in table else overflow[e["lang"]] for e in examples]
        return torch.stack(rows)

    def _eval_text_from_table(self, instructions):
        table = self._resident_text_table
        if table is None:
            return None
        pending = self._resident_text_pending
        if not all(p in table or p in pending for p in instructions):
            return None
        device = next(self.backbone.transformer.parameters()).device
        return self._resident_text_rows([{"lang": p} for p in instructions], device)

    # ----- SDPA backend -----
    def _build_sdpa_ctx_factory(self):
        from contextlib import nullcontext

        backend_name = "auto"
        try:
            backend_name = (self.config.trainer.sdpa_backend or "auto").lower()
        except Exception:
            backend_name = "auto"
        if backend_name == "auto":
            return nullcontext
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            return nullcontext
        attr = {
            "cudnn": "CUDNN_ATTENTION",
            "flash": "FLASH_ATTENTION",
            "mem_efficient": "EFFICIENT_ATTENTION",
            "math": "MATH",
        }.get(backend_name)
        if attr is None:
            return nullcontext
        backend = getattr(SDPBackend, attr)
        return lambda: sdpa_kernel([backend])

    def _maybe_stitch_camera_views(self, batch_images):
        if self.camera_concat not in ("side_by_side", "primary_inset"):
            return batch_images
        if self.camera_concat == "primary_inset":
            _stitch = lambda views: stitch_primary_with_insets(views[0], list(views[1:]))  # noqa: E731
        else:
            _stitch = stitch_views_side_by_side

        def _stitch_one(imgs):
            if isinstance(imgs, (list, tuple)) and len(imgs) > 0 and isinstance(imgs[0], (list, tuple)):
                return [_stitch(t) if len(t) > 1 else t[0] for t in imgs]
            return _stitch(imgs) if isinstance(imgs, (list, tuple)) and len(imgs) > 1 else imgs

        return [_stitch_one(imgs) for imgs in batch_images]

    def _run_dit(self, xt, timestep, text_embeds, n_goal_frames=0):
        """One goal-aware DiT forward: threads `n_goal_frames` to the vendored forward
        (goal 3-class prefix-LM mask) and returns (merged tapped features, velocity)."""
        self.backbone._intermediate_features.clear()
        dit_output = self.backbone.transformer(
            hidden_states=xt,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            enable_diffusion_forcing=True,
            fps=self.fps,
            n_goal_frames=int(n_goal_frames),
        )
        velocity = dit_output.sample if hasattr(dit_output, "sample") else dit_output[0]

        extracted = []
        for feat in self.backbone._intermediate_features:
            if feat.dim() == 5:
                B, C, T, Hh, Ww = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * Hh * Ww, C)
            extracted.append(feat)
        if self.n_taps > 1:
            # E7.0: hand back the per-tap features (ascending-block capture order) so the
            # caller can slice obs/goal PER TAP — a premature token-concat here would put
            # the frame-position slices in the wrong segments.
            if len(extracted) != self.n_taps:
                raise RuntimeError(f"expected {self.n_taps} tapped features, captured {len(extracted)}")
            return extracted, velocity
        merged = select_backbone_features(tuple(extracted), "all", self.combine_layers)
        return merged, velocity

    def _encode_video_pass(self, batch_images_window, frame_idx: int):
        """Encode ONE single-target window [obs..., <target> tiled n_tile] to clean
        latents. `frame_idx` selects which shipped frame is the tiled target — the
        window ships [obs..., futures..., goals...], so futures sit at
        num_history_frames + i and goals at num_history_frames + n_future_frames + j."""
        tiled = [list(im[: self.num_history_frames]) + [im[frame_idx]] * self.n_tile for im in batch_images_window]
        if self._compile_vae:
            self.backbone.vae.encoder = self._vae_encoder_compiled
        try:
            latent, _ = self.backbone._encode_images(tiled, num_frames=self.encode_frames)
        finally:
            if self._compile_vae:
                self.backbone.vae.encoder = self._vae_encoder_eager
        return latent.float()

    def _add_goal_time_diff(self, goal_hidden, g_rows):
        """Add the zero-init embedding of the goal-time diff g (native rows between each goal
        frame and the current obs) to that goal's tokens. g_rows: [B] or [B, n_goal] float —
        under E8.0 each goal frame gets ITS OWN horizon embedding, broadcast over the frame's
        tokens (tap-major under E7.0 multi-tap, matching _slice_obs_goal's concat). Inert at
        step 0 (zero-init); no-op when the diff embedder is off."""
        if self.goal_time_proj is None:
            return goal_hidden
        if g_rows.dim() == 1:
            g_rows = g_rows.unsqueeze(1)
        w_dtype = self.goal_time_proj.weight.dtype
        emb = self.goal_time_proj(sinusoidal_h_embed(g_rows, self.goal_h_embed_dim).to(w_dtype))  # [B, n_goal, C]
        per_tok = emb.repeat_interleave(self.tokens_per_frame, dim=1)  # [B, n_goal*ftok, C]
        if self.n_taps > 1:
            per_tok = per_tok.repeat(1, self.n_taps, 1)
        return goal_hidden + per_tok.to(goal_hidden.dtype)

    def _goal_keep_token_mask(self, keep):
        """Per-goal keep flags [B, n_goal] -> a [B, goal_tokens, 1] multiplier matching
        goal_hidden's token layout (frame-major per tap, tap-major under E7.0)."""
        m = keep.repeat_interleave(self.tokens_per_frame, dim=1)
        if self.n_taps > 1:
            m = m.repeat(1, self.n_taps)
        return m.unsqueeze(-1)

    def _slice_obs_goal(self, feats, goal_start: int, t_lat: int):
        """(obs_hidden, goal_hidden) for the action head's cross-attention stream.

        Single tap: the historical frame-position slices, bit-exact. Multi-tap (E7.0):
        per-tap slices concatenated tap-major — [obs@t1 .. obs@tk], [goal@t1 .. goal@tk]
        — with the zero-init tap embedding added to each tap's tokens (tap identity is
        otherwise only implicit in the feature statistics)."""
        ftok = self.tokens_per_frame
        n_obs_tok = self.obs_latent_frames * ftok
        if self.n_taps == 1:
            return feats[:, :n_obs_tok, :], feats[:, goal_start * ftok : t_lat * ftok, :]
        obs_parts, goal_parts = [], []
        for i, f in enumerate(feats):
            e = self.tap_embed.weight[i].to(f.dtype)
            obs_parts.append(f[:, :n_obs_tok, :] + e)
            goal_parts.append(f[:, goal_start * ftok : t_lat * ftok, :] + e)
        return torch.cat(obs_parts, dim=1), torch.cat(goal_parts, dim=1)

    def forward(self, examples: Optional[List[dict]] = None, **kwargs) -> dict:
        if not self.cotrain_enabled:
            raise RuntimeError("GlanceWAM.forward called with video_cotrain.enabled=False.")

        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        actions = [e["action"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None

        bad = next((len(im) for im in batch_images if len(im) != self.window_frames), None)
        if bad is not None:
            raise RuntimeError(
                f"expected {self.window_frames} frames per sample (obs + futures + goals) but got "
                f"{bad}; set datasets.vla_data.future_frame_idx > 0 AND datasets.vla_data.goal_frame_range "
                f"(or future_frame_idxs + goal_frame_ranges for E8.0 multi-horizon) so the dataloader "
                f"ships every special frame."
            )

        # 1. Causal-VAE PER-TARGET encode (see module note): one obs-conditioned single-
        #    target pass per special frame, so every future/goal latent is grounded on obs
        #    and free of the other targets. Pass 1 [obs, f_1] also supplies the obs latents;
        #    legacy (1 future + 1 goal) keeps the historical two passes bit-identical.
        nh, nf, ng = self.num_history_frames, self.n_future_frames, self.n_goal_frames
        with torch.no_grad():
            latent_first = self._encode_video_pass(batch_images, nh)  # [obs..., f_1]
            extra_latents = [
                self._encode_video_pass(batch_images, nh + k)[:, :, self.obs_latent_frames :, :, :]
                for k in range(1, nf + ng)
            ]
            text_embeds = self._maybe_stack_cached_text(examples)
            if text_embeds is None:
                text_embeds, _ = self.backbone._encode_text(instructions)

        # Assemble the DiT window [obs, futures..., goals...].
        clean_latent = torch.cat([latent_first, *extra_latents], dim=2)
        bsz, _, t_lat = clean_latent.shape[0], clean_latent.shape[1], clean_latent.shape[2]
        goal_start = t_lat - self.n_goal_frames  # goal frames are the trailing ones
        if goal_start <= self.obs_latent_frames:
            raise ValueError(
                f"no room for the noised future frame: t_lat={t_lat}, obs_latent_frames={self.obs_latent_frames}, "
                f"n_goal_frames={self.n_goal_frames}."
            )

        # 2. Per-frame noise: obs (front) AND goal (trailing) clean at τ=0; the future
        #    frame(s) in between are noised (video loss). cond_mask=1 => clean.
        cond_mask = torch.zeros(bsz, 1, t_lat, 1, 1, device=clean_latent.device, dtype=torch.float32)
        cond_mask[:, :, : self.obs_latent_frames] = 1.0  # obs clean
        cond_mask[:, :, goal_start:] = 1.0  # goal clean
        if nf == 1:
            sigma = sample_train_sigma_t(bsz, self.sigma_sampling, clean_latent.device)
            xt, target_velocity = get_flow_xt_and_target_v(clean_latent, sigma, cond_mask)
            # Diffusion-forcing timestep matrix [B, T_lat]: 0 on obs+goal, σ*1000 on future.
            in_timestep = (1.0 - cond_mask[:, 0, :, 0, 0]) * sigma.view(bsz, 1) * 1000.0
        else:
            # E8.0 diffusion forcing proper: an INDEPENDENT σ per noised future frame, so
            # the DiT learns every (σ_short, σ_long) corner — including the joint two-slot
            # generation the sampler runs at inference.
            sigma_pf = sample_train_sigma_t(bsz * nf, self.sigma_sampling, clean_latent.device).view(bsz, nf)
            sigma_f = clean_latent.new_zeros(bsz, 1, t_lat, 1, 1)
            sigma_f[:, 0, self.obs_latent_frames : goal_start, 0, 0] = sigma_pf
            xt, target_velocity = get_flow_xt_and_target_v(clean_latent, sigma_f, cond_mask)
            in_timestep = (1.0 - cond_mask[:, 0, :, 0, 0]) * sigma_f[:, 0, :, 0, 0] * 1000.0

        # 3. Single goal-aware DiT pass -> action features + video velocity.
        with torch.autocast("cuda", dtype=torch.bfloat16), self._sdpa_ctx_factory():
            merged, velocity = self._run_dit(
                xt.to(self.backbone.transformer.dtype),
                in_timestep,
                text_embeds,
                n_goal_frames=self.n_goal_frames,
            )

        # 4. Action loss — condition on obs tokens (bit-identical to the plain cotrain)
        #    PLUS the goal tokens (the trailing goal frame), concatenated into the
        #    action head's cross-attention stream (per tap under E7.0 multi-tap).
        obs_hidden, goal_hidden = self._slice_obs_goal(merged, goal_start, t_lat)
        # Goal-time diff: add the (zero-init) embedding of g = goal offset (native rows) to
        # the goal tokens, so the head knows how far ahead this goal is.
        if self.goal_time_proj is not None:
            if "goal_offset_rows" not in examples[0]:
                raise RuntimeError(
                    "goal_conditioning.h_embed_dim>0 needs per-sample 'goal_offset_rows' — set "
                    "datasets.vla_data.goal_frame_range(s) so the dataloader ships the goal offset(s)."
                )
            g_np = np.stack([np.atleast_1d(e["goal_offset_rows"]).astype(np.float32) for e in examples])
            if g_np.shape[1] != self.n_goal_frames:
                raise RuntimeError(
                    f"goal_offset_rows carries {g_np.shape[1]} offsets but the window has "
                    f"{self.n_goal_frames} goal frames — dataloader/framework horizon configs disagree."
                )
            goal_hidden = self._add_goal_time_diff(goal_hidden, torch.from_numpy(g_np).to(goal_hidden.device))
        # Goal-dropout: zero the goal tokens per sample so a no-goal fallback stays
        # in-distribution (the eval control arm zeroes them too). Under E8.0 the draw is
        # INDEPENDENT per goal frame, keeping the single-goal fallbacks (the eval
        # drop-short / drop-long arms) in-distribution as well.
        keep = None
        if self.training and self.goal_dropout_p > 0:
            keep = (torch.rand(bsz, self.n_goal_frames, device=goal_hidden.device) >= self.goal_dropout_p).to(
                goal_hidden.dtype
            )
            goal_hidden = goal_hidden * self._goal_keep_token_mask(keep)
        vl_embs = torch.cat([obs_hidden, goal_hidden], dim=1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions_t = torch.tensor(np.array(actions), device=vl_embs.device, dtype=vl_embs.dtype)
            actions_target = actions_t[:, -self.action_horizon :, :]
            reps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            vl_embs_rep = vl_embs.repeat(reps, 1, 1)
            actions_target_rep = actions_target.repeat(reps, 1, 1)
            state_rep = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=vl_embs.device, dtype=vl_embs.dtype)
                state_rep = state_t.repeat(reps, 1, 1)
            action_loss = self.action_model(vl_embs_rep, actions_target_rep, state_rep)

        # 5. Video loss — flow-matching velocity MSE, masked to the noised future frame.
        fut_mask = (1.0 - cond_mask).expand_as(velocity)
        sq_err = (velocity.float() - target_velocity.float()) ** 2
        video_loss = (sq_err * fut_mask).sum() / fut_mask.sum().clamp_min(1.0)

        total_loss = action_loss + self.cotrain_lambda * video_loss
        out = {
            "action_loss": total_loss,
            "action_only_loss": action_loss.detach(),
            "video_loss": video_loss.detach(),
        }
        return out

    @torch.inference_mode()
    def _generate_goal_latent(self, obs_latent, text_embeds, num_steps=None, seeds=None):
        """Flow-matching Euler sampler over the video DiT: generate the goal latent(s)
        `g_hat` conditioned on the clean obs latent. Runs with the plain 2-class mask
        (n_goal_frames=0) over a [obs(clean), futures(noised)] window. Under E8.0
        multi-horizon all future slots are denoised JOINTLY on one uniform σ schedule —
        the frames co-constrain through temporal attention (the DF chain that anchors
        the far wrist frame on the short-horizon frame). Returns [B,16,n_future,h,w]
        in the normalized latent space (no VAE decode).

        seeds: optional per-ROW goal seeds (len == B). Given, row i's σ=1 draw comes from
        a generator seeded with seeds[i], so the sampled goal depends only on that seed —
        not on how the server batched concurrent requests. None = the historical unseeded
        draw, bit-identical."""
        num_steps = int(num_steps or self.goal_gen_steps)
        bsz = obs_latent.shape[0]
        obs_n = self.obs_latent_frames
        nf = self.n_future_frames
        # The noised future latent frame(s) after the obs prefix.
        fut_shape = (bsz, obs_latent.shape[1], nf, obs_latent.shape[3], obs_latent.shape[4])
        future = seeded_randn(fut_shape, seeds, dtype=torch.float32, device=obs_latent.device)  # σ=1
        sigmas = torch.linspace(1.0, 0.0, num_steps + 1, device=obs_latent.device)
        clean_ts = torch.zeros(bsz, obs_n, device=obs_latent.device)
        for k in range(num_steps):
            sig = sigmas[k]
            d_sig = (sigmas[k] - sigmas[k + 1]).item()
            xt = torch.cat([obs_latent.float(), future], dim=2)
            in_timestep = torch.cat([clean_ts, sig.expand(bsz, nf) * 1000.0], dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16), self._sdpa_ctx_factory():
                _, velocity = self._run_dit(
                    xt.to(self.backbone.transformer.dtype), in_timestep, text_embeds, n_goal_frames=0
                )
            v_future = velocity[:, :, obs_n : obs_n + nf, :, :].float()
            # dx/dσ = v ; integrate σ: 1 -> 0  =>  x <- x - v * Δσ.
            future = future - v_future * d_sig
        return future  # g_hat @ σ=0

    @staticmethod
    def _e2_env() -> dict:
        """E2 (goal-information ablation) eval knobs, read from the environment.

        All default to the historical behaviour, so an unset environment is bit-exact. The
        batched policy server passes no per-call kwargs, so degradations that live on the
        SERVER side (what the goal latent contains, how the DiT is queried) have to arrive as
        env; the ones that live on the CLIENT side (staleness, refresh cadence, which episode's
        goal) arrive as per-example keys from `model2robocasa_kitchen_interface.py`.

        - ``GLANCEWAM_GOAL_MODE=obs``   -> goal := the current observation latent (a "goal" that
          carries no future information at all; the generator is skipped entirely).
        - ``GLANCEWAM_GOAL_DS=k``       -> average-pool the goal latent k x k spatially and
          nearest-upsample it back: a coarse goal, at 8k pixels per latent cell. k=1 is the
          untouched goal (bit-exact), so the baseline arm needs no special case.
        - ``GLANCEWAM_QUERY_TIMESTEP``  -> the diffusion timestep the obs frames are labelled with
          in the feature-extraction pass (E3(a): the DF clean-corner probe; 0 = the trained
          all-clean corner). ``GLANCEWAM_QUERY_TIMESTEP_GOAL`` does the same for the goal frame
          and defaults to the obs value.
        """
        gen_steps = os.environ.get("GLANCEWAM_GOAL_GEN_STEPS", "")
        ds = int(os.environ.get("GLANCEWAM_GOAL_DS", "1") or 1)
        q_obs = float(os.environ.get("GLANCEWAM_QUERY_TIMESTEP", "0") or 0.0)
        q_goal = os.environ.get("GLANCEWAM_QUERY_TIMESTEP_GOAL")
        return {
            "mode": (os.environ.get("GLANCEWAM_GOAL_MODE", "") or "").strip().lower(),
            "ds": max(1, ds),
            "q_obs": q_obs,
            "q_goal": float(q_goal) if q_goal not in (None, "") else q_obs,
            # Euler steps in the goal sampler. The checkpoint default (10) is the trained
            # cadence's cost; sweeping it is the cheapest available *dose* of generator quality,
            # which is what turns "is a better generator worth it?" into a slope rather than the
            # two-point comparison E1 could never fit.
            "gen_steps": int(gen_steps) if gen_steps else None,
        }

    @staticmethod
    def _downres_latent(x, k: int):
        """Coarsen a latent frame by k x k average pooling + nearest upsampling (shape kept)."""
        if k <= 1:
            return x
        b, c, t, h, w = x.shape
        flat = x.reshape(b * c * t, 1, h, w)
        # ceil_mode keeps the edge cells when k does not divide h/w; interpolate crops back.
        pooled = torch.nn.functional.avg_pool2d(flat, k, ceil_mode=True)
        up = torch.nn.functional.interpolate(pooled, scale_factor=k, mode="nearest")[:, :, :h, :w]
        return up.reshape(b, c, t, h, w)

    @torch.inference_mode()
    def predict_action(
        self, examples: List[dict], drop_goal: bool = False, goal_horizon_rows: Optional[float] = None, **kwargs
    ) -> dict:
        """Self-generating goal-conditioned inference: encode obs, generate the goal
        latent via the video DiT, then decode actions conditioned on obs + goal tokens.

        drop_goal=True zeroes the goal tokens (the no-goal control arm).
        goal_horizon_rows: the goal-time diff g (native rows) fed to the diff embedder —
        the stateful eval server passes the SHRINKING effective distance within each H_g
        refresh cycle (H_g right after refresh, down toward 0). Defaults to H_g."""
        if type(examples) is not list:
            examples = [examples]
        instructions = [e["lang"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None

        # Per-row sampling seeds (§3 noise gate). An example may carry "seed" (derives both
        # streams) and/or "goal_seed"/"action_seed" to pin them independently — the latter is
        # what lets a branch sweep hold the action noise fixed while varying only the goal,
        # decomposing best-of-N headroom into goal- vs action-attributable. Absent on every
        # row => seeds=None => the historical unseeded draw, bit-identical.
        def _row_seed(e, key, mult):
            if key in e and e[key] is not None:
                return int(e[key])
            if e.get("seed") is not None:
                return int(e["seed"]) * 2 + mult
            return None

        goal_seeds = [_row_seed(e, "goal_seed", 0) for e in examples]
        action_seeds = [_row_seed(e, "action_seed", 1) for e in examples]
        if all(s is None for s in goal_seeds):
            goal_seeds = None
        if all(s is None for s in action_seeds):
            action_seeds = None

        batch_images = self._maybe_stitch_camera_views([to_pil_preserve(e["image"]) for e in examples])
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        cached_text = self._eval_text_from_table(instructions)
        if cached_text is None:
            # Need the live UMT5 for uncached prompts; the VAE (loaded frozen, never
            # offloaded) is already present for the obs encode.
            self.backbone.ensure_eval_encoders()

        # Encode obs (eager VAE) and text.
        obs_latent, _ = self.backbone._encode_images(
            [[im] * self.num_history_frames for im in batch_images], num_frames=self.num_history_frames
        )
        obs_latent = obs_latent.float()
        if cached_text is not None:
            text_embeds = cached_text
        else:
            text_embeds, _ = self.backbone._encode_text(instructions)

        obs_n = self.obs_latent_frames
        bsz = obs_latent.shape[0]
        # Per-example goal source (stateful cadence): reuse a client-cached goal latent when
        # provided (the "hold" phase of the H_g refresh cycle), else GENERATE a fresh one via
        # the video DiT (the "refresh"). One batch may MIX the two — envs refresh at different
        # steps — so partition and only run the (expensive) sampler on the refreshing subset.
        e2 = self._e2_env()
        g_hat = obs_latent.new_zeros(
            (bsz, obs_latent.shape[1], self.n_goal_frames, obs_latent.shape[3], obs_latent.shape[4])
        )
        keep_idx = [i for i, e in enumerate(examples) if "goal_latent" in e]
        # A row may ALSO ask for a fresh generation while conditioning on the goal it sent back
        # ("force_gen"): that is how the client emulates an ASYNCHRONOUS proposer with latency —
        # the sampler starts now, but the action keeps using the goal that is already in hand.
        gen_idx = [i for i, e in enumerate(examples) if "goal_latent" not in e or e.get("force_gen")]
        if e2["mode"] == "obs":
            # Degenerate-goal arm: the goal IS the current observation, so nothing is generated.
            gen_idx = []
            g_hat[:] = obs_latent[:, :, -1:].repeat(1, 1, self.n_goal_frames, 1, 1).to(g_hat.dtype)
            keep_idx = []
        if keep_idx:
            provided = torch.from_numpy(np.stack([examples[i]["goal_latent"] for i in keep_idx])).to(
                g_hat.device, dtype=g_hat.dtype
            )
            g_hat[torch.tensor(keep_idx, device=g_hat.device)] = provided
        g_new = None
        if gen_idx:
            gi = torch.tensor(gen_idx, device=obs_latent.device)
            # Subset the seeds to the rows actually being generated (rows that arrived with a
            # cached "goal_latent" are held, not re-sampled), so seed i still tracks example i.
            gen_seeds = [goal_seeds[i] for i in gen_idx] if goal_seeds is not None else None
            g_gen = self._generate_goal_latent(
                obs_latent[gi], text_embeds[gi], num_steps=e2["gen_steps"], seeds=gen_seeds
            )
            adopt = [j for j, i in enumerate(gen_idx) if "goal_latent" not in examples[i]]
            if adopt:  # rows with nothing in hand act on the goal they just generated
                ai = torch.tensor([gen_idx[j] for j in adopt], device=g_hat.device)
                g_hat[ai] = g_gen[torch.tensor(adopt, device=g_gen.device)].to(g_hat.dtype)
            if len(adopt) != len(gen_idx):  # some row is running the async (force_gen) path
                g_new = g_hat.clone()
                g_new[gi] = g_gen.to(g_new.dtype)
        # Coarse-goal arm: k x k average pooling of the goal latent (k=1 is a no-op). Only what
        # the action head SEES is coarsened — the client still caches the full-resolution goal,
        # so the arm changes the read, not the generator's trajectory.
        g_cond = self._downres_latent(g_hat, e2["ds"])

        # Conditioning pass: window [obs, filler(s)(noised), goals=g_hat] so each goal
        # lands at its training RoPE slot (obs_n + n_future + j). The fillers are
        # "future"-class frames (noised timestep) that neither obs nor goal attends, so
        # their content is irrelevant — obs & goal tokens are consistent with training.
        if self.n_future_frames == 1:
            filler = obs_latent  # any latent; masked out of obs/goal reads (historical path)
        else:
            filler = obs_latent[:, :, -1:].repeat(1, 1, self.n_future_frames, 1, 1)
        window = torch.cat([obs_latent, filler, g_cond], dim=2)  # [B,16,obs_n+n_fut+n_goal,h,w]
        t_lat = window.shape[2]
        goal_start = t_lat - self.n_goal_frames
        # timestep: obs+goal clean (0), filler noised (>0).
        in_timestep = torch.zeros(obs_latent.shape[0], t_lat, device=obs_latent.device)
        in_timestep[:, obs_n:goal_start] = 500.0
        # E3(a) clean-corner probe: label the clean obs / goal frames with a NONZERO timestep.
        # The latents stay clean; only the timestep the DiT is told about moves off the corner
        # it was trained to read actions at. Default 0 => untouched.
        if e2["q_obs"]:
            in_timestep[:, :obs_n] = e2["q_obs"]
        if e2["q_goal"]:
            in_timestep[:, goal_start:] = e2["q_goal"]
        with torch.autocast("cuda", dtype=torch.bfloat16), self._sdpa_ctx_factory():
            merged, _ = self._run_dit(
                window.to(self.backbone.transformer.dtype),
                in_timestep,
                text_embeds,
                n_goal_frames=self.n_goal_frames,
            )

        obs_hidden, goal_hidden = self._slice_obs_goal(merged, goal_start, t_lat)
        # Goal-time diff: the effective distance to the (held) goal — H_g right after a
        # refresh, shrinking through the cycle. Per-example (each env is at its own point in
        # the cycle); the stateful eval server passes it as e["goal_horizon_rows"], else H_g.
        if self.goal_time_proj is not None:
            defaults = self._default_goal_horizons
            rows = []
            for e in examples:
                v = e.get("goal_horizon_rows", goal_horizon_rows)
                if v is None:
                    rows.append(list(defaults))
                elif isinstance(v, (list, tuple, np.ndarray)):
                    rows.append([float(x) for x in v])
                else:
                    # Scalar = the LONG goal's shrinking effective distance (the historical
                    # client contract); the whole window ages together within a refresh
                    # cycle, so shift every horizon by the same elapsed rows.
                    elapsed = defaults[-1] - float(v)
                    rows.append([max(1.0, d - elapsed) for d in defaults])
            g_rows = torch.tensor(rows, device=goal_hidden.device)
            goal_hidden = self._add_goal_time_diff(goal_hidden, g_rows)
        # No-goal control arm: drop_goal kwarg (direct calls) OR GLANCEWAM_GOAL_DROP=1 on the
        # served control run (the batched server passes no per-call kwargs).
        goal_dropped = drop_goal or os.environ.get("GLANCEWAM_GOAL_DROP", "0") not in ("0", "", "false", "False")
        if goal_dropped:
            goal_hidden = torch.zeros_like(goal_hidden)
        elif self.n_goal_frames > 1:
            # E8.0 eval decomposition arms: GLANCEWAM_GOAL_DROP_IDX="0" (drop-short) /
            # "1" (drop-long) zeroes individual goal frames — in-distribution thanks to
            # the per-goal training dropout.
            drop_idx_env = (os.environ.get("GLANCEWAM_GOAL_DROP_IDX", "") or "").strip()
            if drop_idx_env:
                keep_arm = torch.ones(bsz, self.n_goal_frames, device=goal_hidden.device, dtype=goal_hidden.dtype)
                for s in drop_idx_env.split(","):
                    keep_arm[:, int(s)] = 0.0
                goal_hidden = goal_hidden * self._goal_keep_token_mask(keep_arm)
                if not getattr(self, "_goal_drop_idx_logged", False):
                    self._goal_drop_idx_logged = True
                    print(f">> [*] E8.0 eval goal-drop arm ACTIVE: zeroing goal frame(s) {drop_idx_env}")
        vl_embs = torch.cat([obs_hidden, goal_hidden], dim=1)

        state_t = (
            torch.from_numpy(np.array(state)).to(vl_embs.device, dtype=vl_embs.dtype) if state is not None else None
        )
        if state_t is not None and state_t.dim() == 2:
            state_t = state_t.unsqueeze(1)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(vl_embs, state_t, seeds=action_seeds)
        # Return the goal latent used (per example) so the stateful client can CACHE a
        # freshly-generated goal and send it back during the hold phase of the cycle.
        out = {
            "normalized_actions": pred_actions.detach().float().cpu().numpy(),
            "goal_latent": g_hat.detach().float().cpu().numpy(),
        }
        if g_new is not None:
            # Async (force_gen) arm only: the goal that was just SAMPLED, which the client will
            # adopt after its emulated proposer latency — distinct from "goal_latent", the goal
            # this call actually acted on. Sent only when some row asked, to keep the response
            # one latent per row in the common case.
            out["goal_latent_new"] = g_new.detach().float().cpu().numpy()
        return out


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/LIBERO/train_files/config_glancewam_libero.yaml")
    parser.add_argument(
        "--e8_smoke",
        action="store_true",
        help="E8.0 two-horizon smoke: 5-frame window [obs, f@30, f@60, g_short, g_long]",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "GlanceWAM"
    cfg.framework.world_model = {
        "base_wm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        # Default smoke = E7.0 multi-tap (4 taps, E6 spacing, champion's 19 kept).
        "extract_layers": [5, 12, 19, 26],
        "vae_input_size": [224, 448],
        "camera_concat": "side_by_side",
        "num_history_frames": 1,
        "compile_vae": False,
    }
    cfg.framework.video_cotrain = {"enabled": True, "lambda": 1.0, "sigma_sampling": "uniform"}
    cfg.framework.goal_conditioning = {"enabled": True, "dropout_p": 0.5, "goal_gen_steps": 3, "h_embed_dim": 16}
    if args.e8_smoke:
        # E8.0 two-horizon window (1.5 s / 3 s @20 fps). The dataloader normally ships
        # these; the fake batch below just needs matching frame/offset counts.
        cfg.datasets.vla_data.future_frame_idx = 60
        cfg.datasets.vla_data.future_frame_idxs = [30, 60]
        cfg.datasets.vla_data.goal_frame_ranges = [[1, 30], [31, 60]]
    cfg.framework.qwenvl = {
        "base_vlm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        "vl_hidden_dim": 1536,
        "num_vl_layers": 30,
    }

    model = GlanceWAM(cfg)

    def print_model_size(m):
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}\n{'Module':<35} {'Params':>12}  {'%':>6}\n{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}\n  {'TOTAL':<33} {total:>12,}  100.0%\n{'='*55}\n")

    print_model_size(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # window = obs + future + goal (window_frames frames).
    train_window = [image] * model.window_frames
    state_dim = int(cfg.framework.action_model.get("state_dim", 7))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": train_window,
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float16),
        # goal-time diff(s) (native rows); fed to the diff embedder. One per goal frame.
        "goal_offset_rows": [15, 45] if args.e8_smoke else 42,
    }
    sample2 = dict(sample, lang="Another fake instruction for testing.")
    batch = [sample, sample2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = model(batch)
    total, action_only, video_loss = out["action_loss"], out["action_only_loss"], out["video_loss"]
    print(
        f"[smoke] total = {total.item():.4f}  (action_only = {action_only.item():.4f}  video = {video_loss.item():.4f})"
    )
    assert torch.isfinite(total) and torch.isfinite(video_loss), "non-finite loss"

    total.backward()
    head_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.action_model.parameters())
    theta_params = [p for p in model.backbone.transformer.parameters() if p.requires_grad]
    theta_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in theta_params)
    print(f"[smoke] action-head grad present = {head_grad}; backbone-θ grad present = {theta_grad}")
    assert head_grad, "no gradient reached the action head"
    assert theta_grad, "no gradient reached the SkyReels backbone θ (video loss not flowing?)"

    # E7.0 arm: the zero-init tap embedding must receive gradient (it is added to every
    # conditioning token, so the action loss reaches it on the first backward).
    if model.tap_embed is not None:
        te_grad = model.tap_embed.weight.grad
        assert te_grad is not None and torch.isfinite(te_grad).all(), "no gradient reached tap_embed"
        assert model.tap_embed.weight.abs().max().item() == 0.0, "tap_embed must start zero-init"
        print(f"[smoke] multi-tap OK: n_taps={model.n_taps}, tap_embed grad norm = {te_grad.norm().item():.4e}")

    # Obs-token isolation leak check: obs tokens must be BIT-IDENTICAL under different
    # goal frames (the 3-class mask keeps the action read-point blind to the goal),
    # while the goal tokens must differ (the goal frame is actually processed).
    model.eval()
    obs_n = model.obs_latent_frames
    h_lat, w_lat = model.backbone._vae_input_size[0] // 8, model.backbone._vae_input_size[1] // 8
    t_lat_s = obs_n + 2  # obs + future + goal
    xt_s = torch.randn(1, 16, t_lat_s, h_lat, w_lat, device=device)
    ts_s = torch.zeros(1, t_lat_s, device=device)
    ts_s[:, obs_n : t_lat_s - 1] = 500.0  # future frame noised
    goal_slot = (t_lat_s - 1) * model.tokens_per_frame
    with torch.no_grad():
        te_s, _ = model.backbone._encode_text(["leak check"])
        xt_a = xt_s.clone()
        xt_b = xt_s.clone()
        xt_b[:, :, -1] = torch.randn_like(xt_b[:, :, -1])  # different goal frame only
        with torch.autocast("cuda", dtype=torch.bfloat16), model._sdpa_ctx_factory():
            m_a, v_a = model._run_dit(xt_a.to(model.backbone.transformer.dtype), ts_s, te_s, n_goal_frames=1)
            m_b, v_b = model._run_dit(xt_b.to(model.backbone.transformer.dtype), ts_s, te_s, n_goal_frames=1)
    n_obs_tok_s = obs_n * model.tokens_per_frame
    n_fut_tok_s = goal_slot  # obs + future tokens span [0, goal_slot)
    taps_a = m_a if isinstance(m_a, list) else [m_a]
    taps_b = m_b if isinstance(m_b, list) else [m_b]
    for tap_i, (f_a, f_b) in enumerate(zip(taps_a, taps_b, strict=True)):
        assert torch.equal(f_a[:, :n_obs_tok_s], f_b[:, :n_obs_tok_s]), (
            f"tap {tap_i}: obs tokens changed with the goal frame — "
            f"the 3-class mask is leaking into the action read-point"
        )
        assert torch.equal(
            f_a[:, n_obs_tok_s:n_fut_tok_s], f_b[:, n_obs_tok_s:n_fut_tok_s]
        ), f"tap {tap_i}: future tokens changed with the goal frame — the video branch is not goal-isolated"
        assert not torch.equal(
            f_a[:, goal_slot:], f_b[:, goal_slot:]
        ), f"tap {tap_i}: goal tokens ignored the goal frame — the goal is not being processed"
    print(f"[smoke] isolation OK across {len(taps_a)} tap(s): obs & future tokens bit-identical, goal tokens respond")

    predict_sample = dict(sample, image=[image])  # raw per-view obs (single frame)
    predict_output = model.predict_action(examples=[predict_sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[smoke] predicted action shape = {normalized_actions.shape}")
    assert normalized_actions.shape == (1, model.action_horizon, 7)
    assert "goal_latent" in predict_output, "predict_action must return goal_latent for the stateful client"
    gl = predict_output["goal_latent"]
    print(f"[smoke] returned goal_latent shape = {gl.shape}")
    assert gl.shape[0] == 1 and gl.shape[2] == model.n_goal_frames

    # Batched MIXED refresh/hold (the stateful eval server case): one env refreshes (no
    # goal_latent) while another holds (sends its cached goal back), each with its own
    # goal_horizon_rows. Verify per-example generate/reuse + goal_latent passthrough.
    hold_sample = dict(predict_sample, goal_latent=gl[0], goal_horizon_rows=12.0)
    refresh_sample = dict(predict_sample, goal_horizon_rows=60.0)
    mixed = model.predict_action(examples=[hold_sample, refresh_sample])
    assert mixed["normalized_actions"].shape == (2, model.action_horizon, 7)
    assert mixed["goal_latent"].shape[0] == 2
    # The hold arm must reuse EXACTLY the cached goal it sent (no regeneration).
    assert np.allclose(mixed["goal_latent"][0], gl[0]), "hold arm regenerated instead of reusing the cached goal"
    print("[smoke] batched mixed refresh/hold OK: hold reuses cached goal, refresh regenerates")

    ctrl = model.predict_action(examples=[predict_sample], drop_goal=True)
    assert ctrl["normalized_actions"].shape == (1, model.action_horizon, 7)
    print("[smoke] no-goal control arm (drop_goal=True) OK")
    print("[smoke] OK")
