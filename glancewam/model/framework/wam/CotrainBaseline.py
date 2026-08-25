# Copyright 2026 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
SkyReelsV2-GR00T **Video + Action Co-training** Framework.

A standalone framework file (kept self-contained on purpose — this
is the co-training research path and we do not want it coupled to the action-only
backbone wrapper or to its back-compat constraints). The backbone wrapper
(``_SkyReelsV2DF_Interface``) and the GR00T flow-matching head are reused
unchanged via the registries; everything novel lives here.

What changes vs a plain action-loss-only VLA on the same backbone:
  - An action-only VLA queries the SkyReels-V2 DF DiT with EVERY frame clean at
    timestep 0 and trains only the action head — the video backbone is a frozen-ish
    feature extractor, so action and video-dynamics learning never co-shape θ.
  - Here we run ONE forward pass with **per-frame noise** (diffusion forcing): the
    first ``obs_latent_frames`` latent frames stay clean (τ=0) and feed the action
    head; the remaining future frames are noised (τ~p(t)) and supervised by the
    native flow-matching **video loss**. Both losses backprop into the same θ
    (no detach) → real bidirectional synergy. See
    the goal-image design.

Design choices (see plan):
  - **Action read-point = obs clean-frame tokens only.** The merged DiT hidden
    states are sliced to the leading ``obs_latent_frames`` latent frames so the
    action conditioning is independent of the sampled future noise.
  - **Clean-prefix isolation** (the SkyReels-V2 LoRA video finetune's
    validated ``--clean_prefix_isolation``): a prefix-LM FlexAttention mask keeps the
    clean obs frames from attending to the noised future. At eval / feature-query
    time every frame is clean (τ=0) so the mask degenerates to full attention and
    the policy path is unchanged.
  - **θ update reuses config** — full finetune, or ``dit_lora`` if set. No freeze, no EMA.

This pass is model-only and fake-data validated (see ``__main__``). Real obs+future
video windows (dataloader) and the trainer ``video`` loss-scale tag are deferred.
"""

# E402: the heavy glancewam/diffusers imports below intentionally follow the
# sys.path bootstrap so this file is runnable standalone (python .../CotrainBaseline.py),
# matching every other framework here. RUF002/RUF003: σ/τ/θ/λ are deliberate math
# symbols in the flow-matching comments and docstring.

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from glancewam.dataloader.camera_utils import stitch_primary_with_insets, stitch_views_side_by_side
from glancewam.dataloader.image_tools import to_pil_preserve
from glancewam.model.framework.base_framework import baseframework
from glancewam.model.framework.share_tools import merge_framework_config, select_backbone_features
from glancewam.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from glancewam.model.modules.world_model import get_world_model
from glancewam.model.modules.world_model.speedup_patch import compile_dit_blocks, enable_prefix_lm_flex_attention
from glancewam.model.tools import FRAMEWORK_REGISTRY
from glancewam.training.trainer_utils import initialize_overwatch
from glancewam.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)


@dataclass
class CotrainBaselineDefaultConfig:
    """SkyReelsV2-GR00T video+action co-training defaults."""

    name: str = "CotrainBaseline"

    # === World Model backbone (SkyReels-V2 DF-1.3B, Wan2.1 architecture) ===
    world_model: dict = field(
        default_factory=lambda: {
            "base_wm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
            "extract_layers": [-1],
            "combine_layers": "concat_tokens",
            "vae_input_size": [320, 576],
            "compile_dit": False,
            "compile_dit_mode": None,
            "compile_dit_dynamic": True,
            # torch.compile the (frozen) Wan VAE encoder — the live obs+future encode is
            # the step's largest forward phase. ~1.6x on the encode; introduces a ~0.2%
            # systematic latent drift (bf16 conv lowering ≠ eager cuDNN) that is far below
            # the flow-matching velocity target and self-consistent across train/eval
            # (same compiled encoder both places), so it does not affect the dynamics the
            # model learns. Default ON. Set compile_vae=False for a bit-exact encode.
            "compile_vae": True,
            "compile_vae_mode": None,
            "skip_vae": False,
            "skip_text_encoder": False,
            # Obs pixel frames encoded at EVAL/feature-query time (all clean).
            "num_history_frames": 1,
            "fps": 24,
            "text_max_length": 512,
            # PEFT LoRA on the video DiT (parameter-efficient θ update). When False
            # the DiT trains fully (no freeze) — the whole point of co-training.
            "dit_lora": False,
            "dit_lora_rank": 16,
            "dit_lora_alpha": 16,
            "dit_lora_target_modules": ["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"],
            # Keep one GPU-resident copy of each unique prompt's cached UMT5 embed and
            # gather rows per batch instead of re-running UMT5-XXL live every step.
            # Bit-exact (frozen encoder); only the obs+future VAE encode stays live
            # (its per-sample multi-frame window is not cacheable). Default OFF. See
            # the action-only variant this mirrors.
            "resident_text_table": False,
            "resident_text_table_max_rows": 1024,
        }
    )

    # === Video + action co-training knobs ===
    video_cotrain: dict = field(
        default_factory=lambda: {
            # Master switch. This framework only makes sense with it on; exposed
            # so an ablation can disable the video loss (action-only) without
            # swapping frameworks.
            "enabled": True,
            # Weight on the video flow-matching loss in the combined objective.
            "lambda": 1.0,
            # NOTE: no future-window knob lives here. Exactly ONE future frame is trained
            # (cosmos-policy conditioning): the dataloader ships the frame at
            # t+datasets.vla_data.future_frame_idx native rows, and forward() tiles it across one
            # temporal VAE group so the causal Wan VAE maps it to exactly one noised latent
            # frame. The obs latent is unaffected (causal VAE: frame 0 -> latent 0), so the
            # action read-point never moves. None of this model's shapes depend on WHICH future
            # frame it is, so mirroring the index here would only let the two copies drift —
            # which is exactly what the removed num_future_frames/future_frame_mode pair did.
            # Per-frame noise distribution for the future frames.
            #   "uniform"     — SkyReels-V2 FoPP training (uniform timestep marginals)
            #   "logitnormal" — Cosmos recipe (logit-normal, shift=5)
            "sigma_sampling": "uniform",
            # Prefix-LM FlexAttention mask: clean obs frames attend only among
            # themselves; noised future frames stay bidirectional. Degenerates to
            # full attention at eval (all τ=0).
            "clean_prefix_isolation": True,
            # Rung 12a: action-condition the video stream — the head's action chunk is
            # flattened through a zero-init Linear into ONE clean token appended after
            # the video tokens (lingbot-va style, vs 588 tokens for a cosmos-style
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


# ----------------------------------------------------------------------------
# Diffusion-forcing noising helpers — copied (NOT imported) from
# the SkyReels-V2 LoRA video finetune. Same math as that validated recipe.
# ----------------------------------------------------------------------------
def sample_train_sigma_t(batch_size, distribution, device, dtype=torch.float32, shift=5):
    """One shared sigma per sample for the noised future frames.

    uniform:     sigma ~ U(0, 1) — SkyReels-V2 FoPP DF training marginals.
    logitnormal: the Cosmos-Predict2.5 recipe (logit-normal, shift=5).
    """
    if distribution == "uniform":
        sigma_t = torch.rand((batch_size,)).to(device=device, dtype=dtype)
    elif distribution == "logitnormal":
        t = torch.sigmoid(torch.randn((batch_size,))).to(device=device, dtype=dtype)
        sigma_t = shift * t / (1 + (shift - 1) * t)
    else:
        raise NotImplementedError(f"sigma_sampling {distribution!r} is not implemented.")
    return sigma_t.view(batch_size, 1, 1, 1, 1)


def create_condition_mask(batch_size, t_lat, num_cond_latent_frames, device):
    """[B, 1, T_lat, 1, 1] float32 with 1.0 on the first `num_cond_latent_frames`
    latent frames (clean obs prefix), broadcastable against [B, C, T_lat, H, W]."""
    mask = torch.zeros(batch_size, 1, t_lat, 1, 1, device=device, dtype=torch.float32)
    mask[:, :, : int(num_cond_latent_frames)] = 1.0
    return mask


def get_flow_xt_and_target_v(clean_latent, sigma, cond_mask):
    """Rectified-flow interpolation + velocity target (Wan/SkyReels convention):
    x_t = (1-σ)·x0 + σ·noise, target v = noise - x0. The clean obs prefix is kept
    ground-truth in x_t (DF expresses conditioning via per-frame timesteps, but the
    input prefix must also stay clean)."""
    noise = torch.randn_like(clean_latent)
    target_velocity = noise - clean_latent
    xt = noise * sigma + clean_latent * (1 - sigma)
    xt = clean_latent * cond_mask + xt * (1 - cond_mask)
    return xt, target_velocity


# "SkyReelsV2GR00TCotrain" is the pre-release name this framework shipped under; it is kept
# registered so checkpoints whose saved config.yaml carries it keep loading unchanged.
@FRAMEWORK_REGISTRY.register("SkyReelsV2GR00TCotrain")
@FRAMEWORK_REGISTRY.register("CotrainBaseline")
class CotrainBaseline(baseframework):
    """SkyReels-V2 DF-1.3B backbone + GR00T flow-matching head, trained with a
    joint per-frame video + action objective in one forward pass."""

    # Checkpoint whitelist: the owned, trained components. The Wan VAE is registered at
    # build (skip_vae=false here, loaded frozen from base_wm) but is excluded from the
    # save — it is reconstructable, so from_pretrained tolerates it being absent and keeps
    # the build-time base weights. The compiled-VAE wrapper is held off-graph (see __init__)
    # and so never appears here either. See TrainerUtils.filter_savable_state_dict.
    _save_include_prefixes = (
        "backbone.transformer.",
        "action_model.",
    )
    _reconstructable_prefixes = ("backbone.text_encoder.", "backbone.vae.")

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CotrainBaselineDefaultConfig, config)

        # World model backbone (SkyReels-V2 DF DiT + UMT5 text encoder + Wan VAE).
        self.backbone = get_world_model(config=self.config)

        # Align action cross-attention dim to WM hidden size (1536 here).
        wm_hidden = self.backbone.model.config.hidden_size
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = wm_hidden

        wm_cfg = self.config.framework.world_model
        # Co-training reads the DiT's velocity output (the full stack), so the
        # extraction-tap early-stop is incompatible — fail loud instead of silently
        # losing the video loss.
        if bool(wm_cfg.get("truncate_at_extract", False)):
            raise ValueError(
                "video co-training needs the full DiT velocity output; "
                "set framework.world_model.truncate_at_extract=False."
            )
        self.combine_layers = wm_cfg.get("combine_layers", "concat_tokens")
        self.camera_concat = wm_cfg.get("camera_concat", "none")
        self.num_history_frames = int(wm_cfg.get("num_history_frames", 1))

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        # Co-train knobs. The window is single-sourced: obs/history frames come from
        # world_model.num_history_frames; the single future frame is appended by the
        # dataloader (datasets.vla_data.future_frame_idx), so the encoded window matches the
        # dataloader's hist_deltas + future_deltas exactly.
        vc = self.config.framework.get("video_cotrain", {})
        self.cotrain_enabled = bool(vc.get("enabled", True))
        self.cotrain_lambda = float(vc.get("lambda", 1.0))
        # Rung-12b fork: `framework.video_cotrain.action_loss_weight: 0` trains the video
        # stream only — the action-head forward is skipped entirely (freeze the modules too:
        # trainer.freeze_modules "action_model"). Default 1.0 is bit-identical.
        self.action_loss_weight = float(vc.get("action_loss_weight", 1.0))
        if self.action_loss_weight != 1.0:
            logger.info(f">> [*] video_cotrain.action_loss_weight={self.action_loss_weight} (0 = video-only fork)")
        tscale = int(self.backbone.vae_scale_factor_temporal)
        # The dataloader ships obs + ONE future frame (the frame at t+future_frame_idx native
        # rows); forward() tiles it across one temporal VAE group (`tscale` copies) so it lands
        # on exactly one noised latent frame. window_frames = frames the DATALOADER ships per
        # sample; encode_frames = pixel frames handed to the VAE.
        #
        # WHICH future frame is trained is purely a dataloader concern
        # (datasets.vla_data.future_frame_idx) — none of the shapes below depend on it, so the
        # model deliberately does not read it. The pre-2026-07-16 `num_future_frames` /
        # `future_frame_mode` pair (which mirrored the same number in two places and let them
        # drift) is gone; lerobot_datasets._reject_legacy_future_window_keys fails loud on it.
        self.window_frames = self.num_history_frames + 1
        self.encode_frames = self.num_history_frames + tscale
        # Clean obs occupies the leading latent frames. The Wan VAE is causal with
        # `vae_scale_factor_temporal`x downsample (frame 0 -> latent 0, then groups),
        # so the obs read-point is exactly (num_history_frames - 1)//scale + 1 latent frames.
        self.obs_latent_frames = (self.num_history_frames - 1) // tscale + 1
        self.sigma_sampling = vc.get("sigma_sampling", "uniform")

        self.clean_prefix_isolation = bool(vc.get("clean_prefix_isolation", True))

        # Clean-prefix isolation: replace the DiT self-attn with a prefix-LM
        # FlexAttention mask (validated in the LoRA video-gen finetune). The
        # clean/noised split is read per-sample from the per-frame timestep, so this
        # is a pure ON/OFF toggle; at eval (all τ=0) it degenerates to full attention.
        if self.clean_prefix_isolation:
            n_proc = enable_prefix_lm_flex_attention(self.backbone.transformer)
            logger.info(f">> [*] clean_prefix_isolation=True: prefix-LM mask on {n_proc} attention processors.")

        if bool(wm_cfg.get("compile_dit", False)):
            n = compile_dit_blocks(
                self.backbone,
                mode=wm_cfg.get("compile_dit_mode", None),
                dynamic=bool(wm_cfg.get("compile_dit_dynamic", True)),
            )
            logger.info(f">> [*] compile_dit=True: torch.compile applied to {n} DiT blocks.")

        # torch.compile the frozen Wan VAE encoder (the live obs+future encode is the
        # step's largest forward phase). Compiled ON THE MODEL OBJECT so training and
        # predict_action (eval) share the same compiled encoder — the ~0.2% latent drift
        # is then self-consistent across train/eval. dynamic=False: the encode loop calls
        # the encoder with two fixed temporal shapes (1- and 4-frame chunks); both compile
        # once at warmup. Guarded on the VAE being loaded (skip_vae=False here).
        # TRAINING-ONLY: keep both encoders and swap per call (forward → compiled,
        # predict_action → eager). Eval encodes a DIFFERENT shape (1 frame, batch = #envs)
        # than training (13-frame window, batch 16); with dynamic=False the compiled graph
        # would recompile and stall the rollout server (repeatedly if the eval batch varies).
        # Using the eager encoder at eval costs only the ~0.2% train/eval latent gap, which
        # is the same magnitude already accepted as below the flow-matching floor.
        self._compile_vae = False
        if bool(wm_cfg.get("compile_vae", True)) and getattr(self.backbone, "vae", None) is not None:
            mode = wm_cfg.get("compile_vae_mode", None)
            # Hold BOTH the eager-encoder reference and its compiled wrapper OFF the module
            # graph (object.__setattr__ bypasses nn.Module's submodule registration). The
            # compiled wrapper shares backbone.vae.encoder's parameters, so it needs no
            # registration of its own — device moves and train/eval flags propagate through
            # the registered eager module. Consequence: the ONLY registered VAE-encoder
            # submodule is backbone.vae.encoder (eager, plain key names), so the saved
            # state_dict holds exactly one canonical copy — no _vae_encoder_eager.* or
            # _vae_encoder_compiled._orig_mod.* duplicates — and the on-disk key names never
            # depend on which encoder was swapped in at save time. forward() swaps the
            # compiled wrapper in only for the encode call and restores the eager module in a
            # finally, so backbone.vae.encoder is canonical-eager at every save boundary.
            object.__setattr__(self, "_vae_encoder_eager", self.backbone.vae.encoder)
            object.__setattr__(
                self, "_vae_encoder_compiled", torch.compile(self.backbone.vae.encoder, mode=mode, dynamic=False)
            )
            self._compile_vae = True
            logger.info(
                f">> [*] compile_vae=True: torch.compile on the Wan VAE encoder (mode={mode!r}, dynamic=False), "
                f"TRAINING ONLY — ~1.6x encode; ~0.2% systematic latent drift, below the flow-matching target floor."
            )

        # Latent tokens per frame (for slicing obs tokens out of the flattened,
        # frame-major DiT sequence). patch_size is (pt, ph, pw); the Wan patch embed
        # flattens (T, H', W') time-outermost so frame k occupies a contiguous
        # H'*W'-token block.
        H, W = wm_cfg.get("vae_input_size", [320, 576])
        spatial = int(self.backbone.vae_scale_factor_spatial)
        patch = self.backbone.transformer.config.patch_size
        if isinstance(patch, (list, tuple)):
            _, ph, pw = int(patch[0]), int(patch[1]), int(patch[2])
        else:
            ph = pw = int(patch)
        self.tokens_per_frame = ((int(H) // spatial) // ph) * ((int(W) // spatial) // pw)

        # fps token (length-1 list — see SkyReelsV2DF.forward for why not [B]).
        self.fps = [self.backbone._fps_index] if self.backbone.transformer.config.inject_sample_info else None

        # GPU-resident table of unique cached UMT5 embeds (ported from the action-only variant):
        # the per-step live UMT5-XXL encode of the (small, fixed) instruction set
        # dominated the forward after the VAE; this serves one bit-exact GPU row per
        # unique prompt instead. Populated by an eager prefill from the per-dataset t5
        # caches at init (CPU fp16; GPU upload on the first batch), else lazily from
        # batch `lang_embed`. None = flag off = live-encode path. Only the obs+future
        # VAE encode stays live — its multi-frame per-sample window is not cacheable.
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

        # SDPA backend selector (trainer.sdpa_backend).
        self._sdpa_ctx_factory = self._build_sdpa_ctx_factory()

    # ----- Cached-text / resident-text-table path (ported verbatim in behavior from
    # the action-only variant, which proves bit-exactness via
    # a table-parity check). Eval (predict_action) does
    # NOT use these — it goes through the backbone's build_inputs + eval memo. -----
    @staticmethod
    def _check_all_or_none(examples: List[dict], key: str, label: str) -> bool:
        n_with = sum(1 for e in examples if key in e)
        if n_with == 0:
            return False
        if n_with != len(examples):
            raise RuntimeError(
                f"{label}: {n_with}/{len(examples)} samples in this batch carry "
                f"`{key}` — must be all-or-none. Likely cause: a mixture of cached "
                f"and uncached datasets; enable the cache on every dataset in the "
                f"mixture or disable it everywhere."
            )
        return True

    def _maybe_stack_cached_text(self, examples: List[dict]):
        """If text can be served from cache, return [B, L, 4096] bf16 embeds on the
        transformer device (identical shape/semantics to ``_encode_text``); else None
        and the caller live-encodes. Resident table (training only) serves one GPU row
        per unique prompt; with the eager prefill armed the samples need no `lang_embed`.
        """
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
        """Load every unique prompt's cached UMT5 row (fp16, CPU) from the per-dataset
        t5 caches resolved from ``datasets.vla_data`` (data_root_dir + data_mix +
        t5_cache). Returns the prompt->row dict, or None on resolution failure / row-cap
        overflow (callers fall back to lazy per-batch population)."""
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
                logger.info(
                    f">> [!] resident_text_table eager prefill found {len(pending)} unique prompts "
                    f"> cap {self._resident_text_max_rows}; falling back to lazy population."
                )
                return None
            gib = sum(p.nbytes for p in pending.values()) / 2**30
            logger.info(
                f">> [*] resident_text_table: eager-prefilled {len(pending)} unique prompts "
                f"({gib:.2f} GiB) from {len(names)} dataset cache(s)."
            )
            return pending
        except Exception as e:  # any resolution failure means "no eager prefill"
            logger.info(f">> [!] resident_text_table eager prefill unavailable ({e}); lazy fill from batch lang_embed.")
            return None

    def _resident_text_rows(self, examples: List[dict], device):
        """Gather cached text embeds from the GPU-resident per-prompt table. Bit-exact
        vs staging: a row is uploaded once (from_numpy + .to(device, bf16)) per unique
        prompt instead of once per sample per step. Keyed by the sample's ``lang`` string
        (the precompute cache is prompt-keyed, so identical prompts share the embed row).
        Prompts beyond the cap stage per batch without insertion (one-shot warning)."""
        table = self._resident_text_table
        if self._resident_text_pending:
            pending = self._resident_text_pending
            staged = torch.from_numpy(np.stack(list(pending.values()))).to(device=device, dtype=torch.bfloat16)
            for prompt, row in zip(pending, staged, strict=True):
                table[prompt] = row
            logger.info(f">> [*] resident_text_table: uploaded {len(pending)} eager rows to {device}.")
            self._resident_text_pending = {}
        misses = {}
        for e in examples:
            prompt = e["lang"]
            if prompt not in table and prompt not in misses:
                if "lang_embed" not in e:
                    raise KeyError(
                        f"resident_text_table: prompt not in the eager-prefilled table and the sample "
                        f"carries no `lang_embed` (datasets.vla_data.t5_cache.attach_embeds is off?): {prompt!r}. "
                        f"Re-run the UMT5 precompute for this dataset or re-enable attach_embeds."
                    )
                misses[prompt] = e["lang_embed"]
        overflow = {}
        if misses:
            staged = torch.from_numpy(np.stack(list(misses.values()))).to(device=device, dtype=torch.bfloat16)
            for prompt, row in zip(misses, staged, strict=True):
                if len(table) < self._resident_text_max_rows:
                    table[prompt] = row
                else:
                    overflow[prompt] = row
            if overflow and not self._resident_text_overflow_warned:
                self._resident_text_overflow_warned = True
                logger.info(
                    f">> [!] resident_text_table full ({self._resident_text_max_rows} rows); "
                    f"further unique prompts fall back to per-batch staging. Raise "
                    f"framework.world_model.resident_text_table_max_rows if VRAM allows."
                )
        rows = [table[e["lang"]] if e["lang"] in table else overflow[e["lang"]] for e in examples]
        return torch.stack(rows)

    def _eval_text_from_table(self, instructions):
        """For ``predict_action``: serve ``[B, L, 4096]`` bf16 text embeds from the resident
        table (incl. eager-prefilled rows) when it covers EVERY instruction — bit-exact
        frozen-UMT5 embeds. This lets the in-loop held-out monitor (which scores the cached
        training task set) run WITHOUT lazy-loading the ~11 GB UMT5 via ensure_eval_encoders.
        Returns None when the table is off or any instruction is uncached, so a deployed
        rollout with novel prompts still falls back to live encoders (eval stays correct)."""
        table = self._resident_text_table
        if table is None:
            return None
        pending = self._resident_text_pending
        if not all(p in table or p in pending for p in instructions):
            return None
        device = next(self.backbone.transformer.parameters()).device
        return self._resident_text_rows([{"lang": p} for p in instructions], device)

    # ----- SDPA backend (same contract across frameworks) -----
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

        # The client sends raw per-view images in training order; primary_inset packs
        # views[0] full-res + the remaining views as insets, mirroring _pack_sample's
        # per-timestep stitch (RoboCasa kitchen: [primary, secondary, wrist] -> 336x224).
        if self.camera_concat == "primary_inset":
            _stitch = lambda views: stitch_primary_with_insets(views[0], list(views[1:]))  # noqa: E731
        else:
            _stitch = stitch_views_side_by_side

        def _stitch_one(imgs):
            if isinstance(imgs, (list, tuple)) and len(imgs) > 0 and isinstance(imgs[0], (list, tuple)):
                return [_stitch(t) if len(t) > 1 else t[0] for t in imgs]
            return _stitch(imgs) if isinstance(imgs, (list, tuple)) and len(imgs) > 1 else imgs

        return [_stitch_one(imgs) for imgs in batch_images]

    def _run_dit(self, xt, timestep, text_embeds, action_tokens=None):
        """One DiT forward that yields BOTH the tapped hidden states (via the
        backbone's registered capture hooks) and the final velocity prediction.

        We call ``self.backbone.transformer`` directly (rather than the wrapper's
        ``forward``) so the velocity output — discarded by the action-only wrapper —
        is available for the video loss. The capture hooks live on the DiT blocks and
        fire regardless of who invokes the transformer.

        ``action_tokens`` [B, n_act, inner_dim] (rung 12a) is only understood by the
        prefix-LM patched forward, so it is passed through ONLY when set — every
        other call site keeps the stock signature.
        """
        self.backbone._intermediate_features.clear()
        extra = {} if action_tokens is None else {"action_tokens": action_tokens}
        dit_output = self.backbone.transformer(
            hidden_states=xt,
            timestep=timestep,
            encoder_hidden_states=text_embeds,
            enable_diffusion_forcing=True,
            fps=self.fps,
            **extra,
        )
        velocity = dit_output.sample if hasattr(dit_output, "sample") else dit_output[0]

        # Reshape captured features to [B, N, H] (SkyReels blocks already emit 3-D;
        # 5-D fallback matches the wrapper).
        extracted = []
        for feat in self.backbone._intermediate_features:
            if feat.dim() == 5:
                B, C, T, Hh, Ww = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * Hh * Ww, C)
            extracted.append(feat)
        merged = select_backbone_features(tuple(extracted), "all", self.combine_layers)
        return merged, velocity

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Drop legacy VAE-encoder duplicate keys before delegating to ``nn.Module``.

        Pre-architectural-fix checkpoints stored the frozen Wan VAE encoder THREE times:
        the canonical ``backbone.vae.encoder.*`` plus redundant ``_vae_encoder_eager.*`` and
        ``_vae_encoder_compiled._orig_mod.*`` copies (all sharing weights). The compiled
        wrapper and eager alias are now held off the module graph, so those keys no longer
        exist on the model; under ``strict=True`` they would otherwise raise as unexpected.
        Stripping them is lossless — ``backbone.vae.encoder.*`` carries the same weights.
        """
        legacy = [k for k in state_dict if k.startswith(("_vae_encoder_eager.", "_vae_encoder_compiled."))]
        if legacy:
            state_dict = {k: v for k, v in state_dict.items() if k not in set(legacy)}
            logger.info(f">> [*] dropped {len(legacy)} legacy _vae_encoder_* duplicate key(s) on load.")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, examples: Optional[List[dict]] = None, **kwargs) -> dict:
        if not self.cotrain_enabled:
            raise RuntimeError("CotrainBaseline.forward called with video_cotrain.enabled=False.")

        # GLANCEWAM_STEP_TIMING=1 (set by trainer.step_timing): sync-bracketed sub-phase
        # walls attached as `timing` in the output dict. Diagnostic only — the syncs
        # serialize the launch pipeline slightly. Mirrors the action-only variant's forward.
        import os as _os
        import time as _time

        _timing = {} if _os.environ.get("GLANCEWAM_STEP_TIMING") else None

        def _mark(key, since):
            torch.cuda.synchronize()
            now = _time.perf_counter()
            _timing[key] = now - since
            return now

        _t = _time.perf_counter() if _timing is not None else None

        # The dataloader (_pack_sample) already packs cameras PER TIMESTEP: for
        # camera_concat=side_by_side it stitches primary|wrist into each frame, and for
        # `none` it passes single-cam frames — either way e["image"] is the flat temporal
        # window of `window_frames` frames that _encode_images consumes. Do NOT re-stitch
        # here. _maybe_stitch_camera_views (correct for predict_action's RAW per-view client
        # input) would see this flat list, mistake the T temporal frames for T camera views,
        # and horizontally tile the whole window into one mega-frame — which _preprocess_to_video
        # then squishes to a single frame and repeats T_lat times. That double-stitch trained
        # the action head on a corrupted, motionless obs (and a trivial repeated-frame video
        # target), so eval — which feeds the correct single stitched obs — saw OOD features and
        # scored 0% on LIBERO (side_by_side). RoboCasa (camera_concat=none) never tiled, hence
        # was unaffected.
        batch_images = [e["image"] for e in examples]
        instructions = [e["lang"] for e in examples]
        actions = [e["action"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None

        # The window is [obs frames..., the frame at t+future_frame_idx]. Tile that single
        # future frame across one full temporal VAE group so the causal Wan VAE maps it to
        # exactly ONE noised latent frame (cosmos-policy's num_duplicates_per_image trick).
        # Fail loud on a window-length mismatch — a longer window silently truncated to
        # encode_frames would train on the WRONG (near-obs) future frame.
        bad = next((len(im) for im in batch_images if len(im) != self.window_frames), None)
        if bad is not None:
            raise RuntimeError(
                f"expected {self.window_frames} frames per sample (obs + the single future "
                f"frame at t+future_frame_idx) but got {bad}; set "
                f"datasets.vla_data.future_frame_idx > 0 so the dataloader window matches."
            )
        n_tile = self.encode_frames - self.num_history_frames
        batch_images = [list(im[: self.num_history_frames]) + [im[-1]] * n_tile for im in batch_images]

        # 1. Encode the full obs+future window to clean latents [B, 16, T_lat, h, w].
        # The VAE encode stays live (per-sample multi-frame window). Text is served
        # from the resident table when enabled (bit-exact); else UMT5 runs live.
        # Swap the compiled VAE encoder in ONLY for the encode call (training shape is
        # fixed, so the dynamic=False graph is reused), then restore the canonical eager
        # encoder in the finally — backbone.vae.encoder is therefore the plain eager
        # module at every save / state_dict boundary, regardless of train/eval interleaving.
        with torch.no_grad():
            if self._compile_vae:
                self.backbone.vae.encoder = self._vae_encoder_compiled
            try:
                clean_latent, _ = self.backbone._encode_images(batch_images, num_frames=self.encode_frames)
            finally:
                if self._compile_vae:
                    self.backbone.vae.encoder = self._vae_encoder_eager
            if _timing is not None:
                _t = _mark("vae", _t)
            text_embeds = self._maybe_stack_cached_text(examples)
            if text_embeds is None:
                text_embeds, _ = self.backbone._encode_text(instructions)
            if _timing is not None:
                _t = _mark("text", _t)
        clean_latent = clean_latent.float()
        bsz, _, t_lat = clean_latent.shape[0], clean_latent.shape[1], clean_latent.shape[2]
        if self.obs_latent_frames >= t_lat:
            raise ValueError(
                f"obs_latent_frames={self.obs_latent_frames} must be < T_lat={t_lat}; "
                f"increase window_frames so there are future frames to denoise."
            )

        # 2. Per-frame noise: clean obs prefix at τ=0, shared σ on the future frames.
        cond_mask = create_condition_mask(bsz, t_lat, self.obs_latent_frames, clean_latent.device)
        sigma = sample_train_sigma_t(bsz, self.sigma_sampling, clean_latent.device)
        xt, target_velocity = get_flow_xt_and_target_v(clean_latent, sigma, cond_mask)
        # Diffusion-forcing timestep matrix [B, T_lat] in 0-1000 units: 0 on obs, σ*1000 on future.
        in_timestep = (1.0 - cond_mask[:, 0, :, 0, 0]) * sigma.view(bsz, 1) * 1000.0

        if _timing is not None:
            _t = _mark("noise_prep", _t)

        # 3. Single DiT pass → action features (hidden states) + video velocity.
        with torch.autocast("cuda", dtype=torch.bfloat16), self._sdpa_ctx_factory():
            merged, velocity = self._run_dit(xt.to(self.backbone.transformer.dtype), in_timestep, text_embeds)
        if _timing is not None:
            _t = _mark("dit_fwd", _t)

        # 4. Action loss — read ONLY the clean obs-frame tokens.
        n_obs_tok = self.obs_latent_frames * self.tokens_per_frame
        obs_hidden = merged[:, :n_obs_tok, :]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            actions_t = torch.tensor(np.array(actions), device=obs_hidden.device, dtype=obs_hidden.dtype)
            actions_target = actions_t[:, -self.action_horizon :, :]
            reps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            obs_hidden_rep = obs_hidden.repeat(reps, 1, 1)
            actions_target_rep = actions_target.repeat(reps, 1, 1)
            state_rep = None
            if state is not None:
                state_t = torch.tensor(np.array(state), device=obs_hidden.device, dtype=obs_hidden.dtype)
                state_rep = state_t.repeat(reps, 1, 1)

            if self.action_loss_weight != 0.0:
                action_loss = self.action_model(obs_hidden_rep, actions_target_rep, state_rep)
            else:
                action_loss = obs_hidden.new_zeros(())

        if _timing is not None:
            _t = _mark("head_fwd", _t)

        # 5. Video loss — flow-matching velocity MSE, masked to the noised future
        # frames (clean obs prefix excluded). Per-element mean → ~O(1), comparable
        # in scale to the action loss so `lambda` weights them on the same footing.
        fut_mask = (1.0 - cond_mask).expand_as(velocity)  # [B,16,T_lat,h,w]
        sq_err = (velocity.float() - target_velocity.float()) ** 2
        video_loss = (sq_err * fut_mask).sum() / fut_mask.sum().clamp_min(1.0)

        # The trainer backprops ONLY output_dict["action_loss"] (train_glancewam.py), so the
        # joint objective must live there: total = action + λ·video, one backward into both
        # the action head and the (unfrozen) DiT θ. The split is returned for logging.
        total_loss = self.action_loss_weight * action_loss + self.cotrain_lambda * video_loss
        out = {
            "action_loss": total_loss,
            "action_only_loss": action_loss.detach(),
            "video_loss": video_loss.detach(),
        }
        if _timing is not None:
            _mark("video_loss", _t)
            out["timing"] = _timing
        return out

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        """Feature-query inference: every obs frame clean (τ=0); the prefix-LM mask
        degenerates to full attention. Mirrors the action-only variant's predict_action."""
        if type(examples) is not list:
            examples = [examples]
        instructions = [e["lang"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None

        batch_images = self._maybe_stitch_camera_views([to_pil_preserve(e["image"]) for e in examples])
        train_obs_image_size = getattr(self.config.framework, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Serve text from the resident table when it covers these instructions (the in-loop
        # held-out monitor scores the cached training task set) — bit-exact frozen-UMT5 embeds,
        # so we skip lazy-loading the ~11 GB UMT5 just for eval. Fall back to live encoders for
        # any uncached instruction (e.g. a deployed rollout) or if the VAE is absent.
        cached_text = self._eval_text_from_table(instructions)
        if cached_text is None or self.backbone.vae is None:
            self.backbone.ensure_eval_encoders()

        # Eval uses the EAGER VAE encoder (the inference shape — 1 frame, batch = #envs —
        # differs from training, so the training-compiled graph would recompile and stall).
        # No swap needed: backbone.vae.encoder is permanently the eager module; the compiled
        # wrapper is held off-graph and only ever invoked inside forward()'s encode block.

        wm_inputs = self.backbone.build_inputs(
            images=batch_images,
            instructions=instructions,
            cached_text=cached_text,
            num_frames=self.num_history_frames,
        )

        with torch.autocast("cuda", dtype=torch.bfloat16), self._sdpa_ctx_factory():
            wm_outputs = self.backbone(**wm_inputs, output_hidden_states=True, return_dict=True)
            vl_cond = select_backbone_features(wm_outputs.hidden_states, "all", self.combine_layers)

        state_t = (
            torch.from_numpy(np.array(state)).to(vl_cond.device, dtype=vl_cond.dtype) if state is not None else None
        )
        if state_t is not None and state_t.dim() == 2:
            # deployed clients send flat (B, state_dim); training uses (B, 1, state_dim)
            state_t = state_t.unsqueeze(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_actions = self.action_model.predict_action(vl_cond, state_t)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./examples/LIBERO/train_files/config_cotrain_baseline_libero.yaml",
        help="Path to YAML config",
    )
    args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.name = "CotrainBaseline"
    cfg.framework.world_model = {
        "base_wm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        "vae_input_size": [224, 448],
        "camera_concat": "side_by_side",
        "num_history_frames": 1,
    }
    cfg.framework.video_cotrain = {
        "enabled": True,
        "lambda": 1.0,
        "sigma_sampling": "uniform",
        "clean_prefix_isolation": True,
    }
    cfg.framework.qwenvl = {
        "base_vlm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        "vl_hidden_dim": 1536,
        "num_vl_layers": 30,
    }

    model = CotrainBaseline(cfg)

    def print_model_size(m):
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}\n{'Module':<35} {'Params':>12}  {'%':>6}\n{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}\n  {'TOTAL':<33} {total:>12,}  100.0%\n{'='*55}\n")

    print_model_size(model)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # forward() fail-louds on a window-length mismatch, so the fake training window must
    # carry exactly window_frames frames (obs + the single future frame).
    train_window = [image] * model.window_frames

    def _fake_state_9d():
        # cosmos layout: gripper_qpos(2) + eef_pos(3) + eef_quat(4, xyzw, normalized)
        s = np.random.uniform(-1, 1, size=(1, 9)).astype(np.float16)
        q = s[0, 5:9].astype(np.float32)
        s[0, 5:9] = (q / max(np.linalg.norm(q), 1e-6)).astype(np.float16)
        return s

    state_dim = int(cfg.framework.action_model.get("state_dim", 7))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": train_window,
        "lang": "This is a fake instruction for testing.",
        "state": (
            _fake_state_9d() if state_dim == 9 else np.random.uniform(-1, 1, size=(1, state_dim)).astype(np.float16)
        ),
    }
    sample2 = sample.copy()
    sample2["lang"] = "Another fake instruction for testing."
    batch = [sample, sample2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    out = model(batch)
    # "action_loss" is the combined objective the trainer backprops; the split is logging-only.
    total, action_only, video_loss = out["action_loss"], out["action_only_loss"], out["video_loss"]
    print(
        f"[smoke] total = {total.item():.4f}  (action_only = {action_only.item():.4f}  "
        f"video = {video_loss.item():.4f})"
    )
    assert torch.isfinite(total) and torch.isfinite(video_loss), "non-finite loss"

    total.backward()

    # No-detach check: gradients must reach BOTH the action head and θ (the DiT /
    # LoRA params) — the whole point of one-forward co-training.
    head_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.action_model.parameters())
    theta_params = [p for p in model.backbone.transformer.parameters() if p.requires_grad]
    theta_grad = any(p.grad is not None and torch.isfinite(p.grad).all() for p in theta_params)
    print(f"[smoke] action-head grad present = {head_grad}; backbone-θ grad present = {theta_grad}")
    assert head_grad, "no gradient reached the action head"
    assert theta_grad, "no gradient reached the SkyReels backbone θ (video loss not flowing?)"

    model.eval()
    predict_sample = dict(sample, image=[image, image])  # raw per-view obs, not the training window
    predict_output = model.predict_action(examples=[predict_sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"[smoke] predicted action shape = {normalized_actions.shape}")
    assert normalized_actions.shape == (1, model.action_horizon, 7)
    print("[smoke] OK")
