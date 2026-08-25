# Copyright 2026 glancewam community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
SkyReels-V2 Diffusion-Forcing World Model Interface.

Wraps ``Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers`` (a Wan2.1-1.3B DiT
finetuned with diffusion forcing) as a world-model backend for glancewam action
prediction frameworks.

Why diffusion forcing: during training every latent frame received an
INDEPENDENT noise level, with clean history frames at timestep 0 — the same
clean-frame-in-main-stream conditioning that Cosmos-Predict2.5's video2world
uses (``condition_mask`` + per-frame timesteps). Querying the DiT with all
observation frames clean at timestep 0 is therefore an in-distribution corner
of the training task, and multi-frame history (``num_history_frames`` > 1) is
natively supported — the model was trained for autoregressive continuation
from variable-length clean prefixes.

Architecture (diffusers format):
  - UMT5EncoderModel (umt5-xxl): text instruction → embeddings [B, L, 4096]
  - AutoencoderKLWan (Wan2.1 VAE): z_dim=16, 8x spatial / 4x temporal — the
    SAME VAE (weights and normalization stats) as Cosmos-Predict2.5, so
    precomputed ``vae_latent`` caches are interchangeable between the two
    backbones at matching resolution/history.
  - SkyReelsV2Transformer3DModel: 30-block Wan DiT, hidden_dim=1536 (12x128),
    ``inject_sample_info=True`` (fps embedding), patch_size (1,2,2).

Key differences from CosmoPredict2_5:
  - Text encoder: UMT5-XXL (dim=4096, zero-padded, no attention mask) instead
    of Qwen2.5-VL's 100352-d layer-concat; no crossattn_proj / postproj-cache
    machinery.
  - Conditioning: per-frame timestep tensor [B, T_lat] with
    ``enable_diffusion_forcing=True`` instead of a condition_mask channel;
    no padding_mask.
  - The DiT wants an fps *index* (0 for 16 fps, 1 for 24 fps — the only two
    tokens in its fps embedding table), mirroring
    ``SkyReelsV2DiffusionForcingImageToVideoPipeline``.
"""

from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from glancewam.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

_DEFAULT_MODEL = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"


class _DiTExtractionDone(Exception):
    """Control-flow signal raised by the extraction-tap block's forward hook
    when ``truncate_at_extract`` is on: every requested feature has been
    captured, so the remaining DiT blocks (and norm_out/proj_out) are skipped.
    Same mechanism as the CosmoPredict2_5 backend — it neither re-registers
    ``blocks`` (which would break state_dict keys / checkpoint compat) nor
    copies the diffusers forward."""


class _SkyReelsV2DF_Interface(nn.Module):
    """
    World model wrapper for SkyReels-V2 DF (diffusers-based).

    API matches ``_CosmoPredict2_5_Interface``: ``build_inputs(images,
    instructions, cached_text=..., cached_vae=..., num_frames=...)`` → dict,
    ``forward(**kwargs)`` → object with ``.hidden_states`` tuple. Action-head
    frameworks written against the Cosmos backends compose unchanged.
    """

    def _sdpa_ctx(self):
        """The SDPA-backend context the framework wraps its DiT calls in.

        Rebuilt here (rather than shared with the framework) because activation
        checkpointing needs to re-enter it during the BACKWARD recompute, which
        happens long after the framework's `with` block has exited.
        """
        from contextlib import nullcontext

        try:
            name = (self.config.trainer.sdpa_backend or "auto").lower()
        except Exception:
            name = "auto"
        attr = {
            "cudnn": "CUDNN_ATTENTION",
            "flash": "FLASH_ATTENTION",
            "mem_efficient": "EFFICIENT_ATTENTION",
            "math": "MATH",
        }.get(name)
        if attr is None:
            return nullcontext()
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:
            return nullcontext()
        return sdpa_kernel([getattr(SDPBackend, attr)])

    def _sdpa_aware_checkpoint(self, module, *args):
        """`gradient_checkpointing_func` that pins the SAME SDPA backend in both
        the forward pass and the backward recompute.

        Without this, the recompute runs outside the framework's
        `sdpa_kernel([CUDNN_ATTENTION])` block, picks a different attention
        kernel, and saves tensors whose metadata no longer matches what the
        forward saved — torch then raises CheckpointError on the LSE tensor
        (`[B, heads, seq]` fp32) and the philox RNG state. `context_fn` is the
        supported hook for this: it applies one context during forward and a
        fresh one during recompute.
        """
        import torch.utils.checkpoint as _ckpt

        return _ckpt.checkpoint(
            module.__call__,
            *args,
            use_reentrant=False,
            context_fn=lambda: (self._sdpa_ctx(), self._sdpa_ctx()),
        )

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", _DEFAULT_MODEL),
        )
        revision = wm_cfg.get("revision", None) or None
        self.config = config
        self._model_name = model_name
        self._revision = revision

        from diffusers import (
            AutoencoderKLWan,
            SkyReelsV2Transformer3DModel,
            UniPCMultistepScheduler,
        )
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info(f"Loading SkyReels-V2 DF from {model_name}")

        # `skip_text_encoder=True` drops the ~11 GB UMT5-XXL encoder from VRAM
        # when every training sample carries a precomputed `lang_embed`
        # ([L, 4096] zero-padded UMT5 states). Eval paths re-load it via
        # ensure_eval_encoders().
        self._skip_text_encoder = bool(wm_cfg.get("skip_text_encoder", False))
        if self._skip_text_encoder:
            logger.info("skip_text_encoder=True; UMT5 not loaded. build_inputs() requires examples with `lang_embed`.")
            self.tokenizer = None
            self.text_encoder = None
        else:
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name, subfolder="tokenizer", revision=revision)
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                model_name, subfolder="text_encoder", revision=revision, torch_dtype=torch.bfloat16
            )

        # `world_model.transformer_from` loads the DiT weights from a DIFFERENT repo while every
        # other component (Wan VAE, UMT5, tokenizer, scheduler) still comes from `base_wm`. Its
        # one purpose is E3(b): Wan2.1-T2V-1.3B is the model
        # SkyReels-V2-DF-1.3B was diffusion-forcing-finetuned FROM, and its 825 transformer
        # tensors are an exact subset of SkyReels' 830 — the extra 5 are the `inject_sample_info`
        # fps embedding, which stays off for a Wan config. So this one key swaps DF pretraining
        # in and out with nothing else moving, which is exactly the A/B E3(b) asks for.
        transformer_src = wm_cfg.get("transformer_from", "") or model_name
        transformer_rev = wm_cfg.get("transformer_revision", None) or (
            revision if transformer_src == model_name else None
        )
        if transformer_src != model_name:
            logger.info(f">> [*]  DiT weights from {transformer_src} (rest of the stack from {model_name}).")
        self.transformer = SkyReelsV2Transformer3DModel.from_pretrained(
            transformer_src, subfolder="transformer", revision=transformer_rev, torch_dtype=torch.bfloat16
        )

        # Activation checkpointing on the DiT blocks: trade ~30% compute for the
        # per-layer activation store. Default OFF, so every existing run stays
        # bit-exact and unchanged in speed. Needed for the DF-14B backbone, whose
        # ZeRO-2 states alone (~77 GB/GPU at 4 ranks) leave no room for 40 layers
        # of activations. Both patched forwards in `speedup_patch.py`
        # (`_prefix_lm_forward`, `_goal_prefix_lm_forward`) already route through
        # `self._gradient_checkpointing_func` when this is on.
        if bool(wm_cfg.get("enable_gradient_checkpointing", False)):
            self.transformer.enable_gradient_checkpointing(gradient_checkpointing_func=self._sdpa_aware_checkpoint)
            logger.info("world_model.enable_gradient_checkpointing=True; DiT blocks recompute in backward.")

        self._skip_vae = bool(wm_cfg.get("skip_vae", False))
        if self._skip_vae:
            logger.info("skip_vae=True; VAE not loaded. build_inputs() requires examples with `vae_latent`.")
            self.vae = None
            self.video_processor = None
            self.vae_scale_factor_spatial = 8
            self.vae_scale_factor_temporal = 4
        else:
            self.vae = AutoencoderKLWan.from_pretrained(
                model_name, subfolder="vae", revision=revision, torch_dtype=torch.bfloat16
            )
            from diffusers.video_processor import VideoProcessor

            self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
            self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        self.scheduler = UniPCMultistepScheduler.from_pretrained(model_name, subfolder="scheduler", revision=revision)

        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

        # The DF DiT's fps embedding table has exactly two tokens (16 fps → 0,
        # 24 fps → 1) — mirror the pipeline's mapping. The framework's
        # effective control rate is unrelated; this only selects which token
        # the model conditions on.
        fps = int(wm_cfg.get("fps", 24))
        self._fps_index = 0 if fps == 16 else 1
        if fps not in (16, 24):
            logger.info(f">> [!] world_model.fps={fps} is not 16 or 24; using the 24-fps token.")

        # Per-frame timestep the DiT is queried at for feature extraction. The
        # latents we feed are always clean (no noise added), so `t=0` is the
        # in-distribution diffusion-forcing corner (clean conditioning frame).
        # A nonzero `t` deliberately shifts the AdaLN modulation toward a
        # noisier operating point — clean-input + noisy-t is a conditioning
        # mismatch, but intermediate-t activations are often more semantically
        # useful for downstream heads (cf. DIFT diffusion features), so this is
        # exposed as a hyperparameter. Default 0 keeps existing runs bit-exact.
        self._query_timestep = int(wm_cfg.get("query_timestep", 0))
        if self._query_timestep != 0:
            logger.info(
                f">> world_model.query_timestep={self._query_timestep}: DiT feature query at a nonzero "
                "(clean-input) timestep — off the diffusion-forcing t=0 corner."
            )

        # UMT5 prompt padding length. Wan-family pretraining used 512; shorter
        # values cut cross-attn K/V cost ~linearly (instructions are short).
        self._text_max_length = int(wm_cfg.get("text_max_length", 512))

        # DiT internal hidden = num_heads x head_dim (1.3B: 12 x 128 = 1536).
        self._hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        self._intermediate_features = []
        self._hooks = []

        # Speedup plan A2: stop the
        # DiT block loop right after the highest extraction tap instead of
        # running all blocks (layer 19/30 skips ~33% of forward FLOPs).
        # Bit-exact for every consumer of the tapped features; incompatible
        # with generate().
        self._truncate_at_extract = bool(wm_cfg.get("truncate_at_extract", False))

        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._register_hooks()

        # Both dims must be multiples of 16 (8x VAE downsample x 2x DiT patch).
        # 540P native is 544x960; the conv VAE handles any multiple-of-16 size.
        vae_input_size = wm_cfg.get("vae_input_size", [320, 576])
        h, w = int(vae_input_size[0]), int(vae_input_size[1])
        if h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"vae_input_size=({h}, {w}) must be multiples of 16 (VAE downsample × DiT patch size).")
        self._vae_input_size = (h, w)

        # Eval-only memo: instruction string -> `_encode_text` embedding row
        # (speedup plan A1, same contract as the CosmoPredict2_5 backend).
        # `predict_action` re-encodes the same constant per-episode instruction
        # through UMT5-XXL every control step; `ensure_eval_encoders()` arms
        # this memo so eval pays that cost once per unique instruction. UMT5 is
        # frozen and `_encode_text` deterministic, so a hit is bit-exact — and
        # unlike Cosmos there is no crossattn_proj, so the memoized embed IS
        # the final cross-attn input. Stays None during training (never armed);
        # `_eval_text_memo_disabled` is the kill switch (see check_wm_parity).
        self._eval_text_memo = None
        self._eval_text_memo_disabled = False

    # ~4 MB per entry ([512, 4096] bf16) — two orders of magnitude lighter than
    # the Cosmos memo; 64 covers every current benchmark's task count.
    _EVAL_TEXT_MEMO_MAX = 64

    @property
    def model(self):
        class _ModelShim:
            pass

        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def ensure_eval_encoders(self):
        """Force the eval/inference path to use live encoders, undoing the
        training-time ``skip_text_encoder`` / ``skip_vae`` cache shortcuts.
        Hardcoded here so callers can't forget — every eval path must go
        through ``ensure_eval_encoders``.
        """
        device = next(self.transformer.parameters()).device
        if self.text_encoder is None:
            from transformers import T5TokenizerFast, UMT5EncoderModel

            logger.info(f"[eval] lazy-loading UMT5 text encoder from {self._model_name}.")
            self.tokenizer = T5TokenizerFast.from_pretrained(
                self._model_name, subfolder="tokenizer", revision=self._revision
            )
            text_encoder = UMT5EncoderModel.from_pretrained(
                self._model_name, subfolder="text_encoder", revision=self._revision, torch_dtype=torch.bfloat16
            )
            text_encoder.to(device)
            text_encoder.requires_grad_(False)
            self.text_encoder = text_encoder
            self._skip_text_encoder = False
        if self.vae is None:
            from diffusers import AutoencoderKLWan
            from diffusers.video_processor import VideoProcessor

            logger.info(f"[eval] lazy-loading VAE from {self._model_name}.")
            vae = AutoencoderKLWan.from_pretrained(
                self._model_name, subfolder="vae", revision=self._revision, torch_dtype=torch.bfloat16
            )
            vae.to(device)
            vae.requires_grad_(False)
            self.vae = vae
            self.vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
            self.vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
            self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            self._skip_vae = False
        if self._eval_text_memo is None and not self._eval_text_memo_disabled:
            self._eval_text_memo = OrderedDict()
            logger.info("[eval] armed per-instruction text-embed memo (LRU %d).", self._EVAL_TEXT_MEMO_MAX)

    def _register_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.blocks)
        actual_indices = []
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                actual_indices.append(actual_idx)
        tap_idx = max(actual_indices) if actual_indices else None

        stop_attached = False
        for actual_idx in actual_indices:
            block = self.transformer.blocks[actual_idx]
            if self._truncate_at_extract and actual_idx == tap_idx and not stop_attached:
                hook = block.register_forward_hook(self._capture_and_stop_hook)
                stop_attached = True
            else:
                hook = block.register_forward_hook(self._capture_hook)
            self._hooks.append(hook)
        if stop_attached:
            logger.info("truncate_at_extract=True: DiT block loop stops after block %d/%d.", tap_idx, num_blocks - 1)

    def _capture_hook(self, module, input, output):
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _capture_and_stop_hook(self, module, input, output):
        self._capture_hook(module, input, output)
        raise _DiTExtractionDone

    def _encode_text(self, instructions):
        """Encode instructions with UMT5, Wan-pipeline style: pad to
        ``text_max_length`` and ZERO the padded positions (Wan cross-attn takes
        no attention mask — padding attends through as zero vectors).

        Returns ([B, L, 4096] bf16, None).
        """
        device = next(self.text_encoder.parameters()).device

        text_inputs = self.tokenizer(
            list(instructions),
            padding="max_length",
            max_length=self._text_max_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state

        text_embeds = text_embeds * text_inputs.attention_mask.unsqueeze(-1)
        return text_embeds.to(dtype=torch.bfloat16), None

    def _encode_text_memoized(self, instructions):
        """LRU-memoized ``_encode_text`` for the eval path: one UMT5 encode per
        unique instruction instead of per control step. Same ``(embeds, None)``
        contract as ``_encode_text``; a hit is bit-exact (frozen encoder,
        deterministic forward)."""
        memo = self._eval_text_memo
        misses = [p for p in dict.fromkeys(instructions) if p not in memo]
        if misses:
            embeds, _ = self._encode_text(misses)
            for prompt, row in zip(misses, embeds, strict=True):
                memo[prompt] = row
        rows = []
        for prompt in instructions:
            memo.move_to_end(prompt)
            rows.append(memo[prompt])
        while len(memo) > self._EVAL_TEXT_MEMO_MAX:
            memo.popitem(last=False)
        return torch.stack(rows, dim=0), None

    def _preprocess_to_video(self, images, num_frames, device, dtype, height, width):
        """Build the VAE input ``video`` tensor ``[B, C, T, H, W]`` (on ``device``, ``dtype``)
        plus per-sample ``cond_frame_counts``.

        FAST PATH (the common training/eval case): every sample is a list of PIL RGB frames
        of the SAME length, already at the target (H, W). The expensive part of the diffusers
        processor is the single-threaded uint8->float32 normalization of the whole pixel stack
        (~70 ms at B16/13f/224 in the main process, blocking the GPU). Instead we decode to a
        uint8 batch on CPU (cheap), do ONE uint8 host->device copy (4x smaller than float),
        and normalize on the GPU in fp32 — bit-exact with the diffusers float path (basic IEEE
        div/mul/sub are correctly-rounded and identical CPU vs GPU), then cast to ``dtype``.

        FALLBACK: the original per-sample diffusers path for anything else (frames need a
        resize, non-RGB / non-PIL inputs, or ragged frame counts), so behavior is unchanged.
        """
        samples = [s if isinstance(s, (list, tuple)) else [s] for s in images]
        lengths = {len(s) for s in samples}
        fast_ok = (
            device.type == "cuda"
            and len(lengths) == 1
            and all(
                isinstance(f, Image.Image) and f.mode == "RGB" and f.size == (width, height) for s in samples for f in s
            )
        )
        if fast_ok:
            t_in = next(iter(lengths))
            arr = np.stack([np.stack([np.asarray(f) for f in s]) for s in samples])  # [B,T,H,W,C] uint8
            u = torch.from_numpy(arr).to(device, non_blocking=True)
            # fp32 normalize on GPU (uint8/255*2 - 1 == diffusers), permuted to [B,C,T,H,W].
            video = u.permute(0, 4, 1, 2, 3).float().div_(255.0).mul_(2.0).sub_(1.0)
            cond_frame_counts = [t_in] * len(samples)
            target = t_in if num_frames is None else num_frames
            if target < t_in:
                video = video[:, :, :target]
                cond_frame_counts = [target] * len(samples)
            elif target > t_in:
                video = torch.cat([video, video[:, :, -1:].repeat(1, 1, target - t_in, 1, 1)], dim=2)
            return video.to(dtype), cond_frame_counts

        preprocessed = []
        cond_frame_counts = []
        for s in samples:
            vt = self.video_processor.preprocess_video(s, height=height, width=width).to(device=device, dtype=dtype)
            preprocessed.append(vt)
            cond_frame_counts.append(vt.shape[2])
        target_frames = max(cond_frame_counts) if num_frames is None else num_frames
        batch_videos = []
        for i, vt in enumerate(preprocessed):
            n = vt.shape[2]
            if n > target_frames:
                vt = vt[:, :, :target_frames]
                cond_frame_counts[i] = target_frames
            elif n < target_frames:
                vt = torch.cat([vt, vt[:, :, -1:].repeat(1, 1, target_frames - n, 1, 1)], dim=2)
            batch_videos.append(vt.squeeze(0))
        return torch.stack(batch_videos, dim=0), cond_frame_counts

    def _encode_images(self, images, num_frames=None):
        """VAE-encode observation images to normalized latents.

        Identical contract to the Cosmos backends: returns
        ([B, 16, T_lat, H/8, W/8], cond_frame_counts).
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        height, width = self._vae_input_size

        video, cond_frame_counts = self._preprocess_to_video(images, num_frames, device, dtype, height, width)

        with torch.no_grad():
            latents = self.vae.encode(video).latent_dist.sample()

        if self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            )
            latents = (latents - latents_mean) / latents_std

        return latents, cond_frame_counts

    def build_inputs(self, images, instructions, cached_text=None, cached_vae=None, num_frames=None, **kwargs):
        """Build DiT inputs for the diffusion-forcing feature query.

        Every observation frame is queried at the same per-frame timestep
        (`world_model.query_timestep`, default 0) across the whole [B, T_lat]
        grid. 0 treats each frame as a clean conditioning frame (the DF corner);
        a nonzero value shifts the feature-extraction operating point. (When a
        future denoising objective is wired, the noisy frames get their
        scheduler timesteps in the same tensor.)

        ``cached_text``: optional (B, L, 4096) UMT5 embeds from a precompute
            pass (zero-padded; no mask, matching ``_encode_text``).
        ``cached_vae``: same contract as the Cosmos backends —
            (latents, cond_frame_counts).
        """
        if cached_vae is None:
            assert len(images) == len(instructions)

        if cached_text is not None:
            text_embeds = cached_text[0] if isinstance(cached_text, (tuple, list)) else cached_text
        else:
            if self.text_encoder is None:
                raise RuntimeError(
                    "skip_text_encoder=True but build_inputs() was called without `cached_text`. "
                    "Make sure samples carry precomputed `lang_embed` and the framework forwards it."
                )
            if self._eval_text_memo is not None:
                text_embeds, _ = self._encode_text_memoized(instructions)
            else:
                text_embeds, _ = self._encode_text(instructions)

        # cond_frame_counts (second tuple element) is part of the shared cache
        # contract but unused here: with diffusion forcing every observation
        # frame is conditioning, expressed via the all-zeros per-frame timestep
        # below rather than a Cosmos-style condition_mask.
        if cached_vae is not None:
            latents, _ = cached_vae
        else:
            if self.vae is None:
                raise RuntimeError(
                    "skip_vae=True but build_inputs() was called without `cached_vae`. "
                    "Make sure samples carry precomputed `vae_latent` and the framework forwards it."
                )
            latents, _ = self._encode_images(images, num_frames=num_frames)

        batch_size = latents.shape[0]
        t_lat = latents.shape[2]
        # Per-frame timesteps (diffusion forcing): uniform across the [B, T_lat]
        # grid at `query_timestep` (0 = all-clean observation, the DF corner).
        timestep = torch.full((batch_size, t_lat), self._query_timestep, device=latents.device, dtype=torch.long)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "_is_wm_input": True,
        }

    def forward(self, **kwargs):
        kwargs.pop("_is_wm_input", False)
        kwargs.pop("output_hidden_states", False)
        kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)
        kwargs.pop("attention_mask", None)

        self._intermediate_features.clear()

        # fps token index. MUST be length-1 (not per-sample): the diffusers DF
        # fps-injection path is written for the pipeline's B=1 calls — a [B]
        # fps gives fps_proj shape (B, 6, d) which mis-broadcasts against
        # timestep_proj (B, f, 6, d) into (B, B, 6, d), silently folding the
        # batch into the feature dim. A length-1 fps yields (1, 6, d) →
        # repeat(f) → (f, 6, d), which broadcasts correctly for any batch, and
        # the token is the same for every sample anyway.
        fps = [self._fps_index] if self.transformer.config.inject_sample_info else None

        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dit_output = self.transformer(
                    hidden_states=kwargs["hidden_states"],
                    timestep=kwargs["timestep"],
                    encoder_hidden_states=kwargs["encoder_hidden_states"],
                    enable_diffusion_forcing=True,
                    fps=fps,
                )
        except _DiTExtractionDone:
            dit_output = None
            if not self._intermediate_features:
                raise RuntimeError(
                    "truncate_at_extract stopped the DiT but no features were captured — "
                    "extract_layers/_register_hooks are inconsistent."
                ) from None

        # SkyReelsV2 blocks output [B, seq_len, hidden_dim] (already flattened).
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if isinstance(out, tuple):
                out = out[0]
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss

        return _WMOutput(hidden_states_tuple=tuple(extracted))

    def generate(self, **kwargs):
        """Full SkyReels-V2 DF video generation (not used in standard VLA training)."""
        if self._truncate_at_extract:
            raise RuntimeError(
                "truncate_at_extract=True skips DiT blocks above the extraction tap; "
                "video generation needs the full stack — disable the flag to generate."
            )
        from diffusers import SkyReelsV2DiffusionForcingImageToVideoPipeline

        pipe = SkyReelsV2DiffusionForcingImageToVideoPipeline(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
        )
        return pipe(**kwargs)


if __name__ == "__main__":
    # Smoke test: instantiate, run one forward pass on synthetic data.
    import numpy as np
    from omegaconf import OmegaConf
    from PIL import Image

    cfg = OmegaConf.create(
        {
            "framework": {
                "world_model": {
                    "base_wm": "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
                    "vae_input_size": [224, 448],
                    "extract_layers": [-1],
                },
            },
        }
    )

    wm = _SkyReelsV2DF_Interface(config=cfg).cuda()
    print(f"[smoke] hidden_size = {wm._hidden_size}")
    print(f"[smoke] inject_sample_info = {wm.transformer.config.inject_sample_info}")
    print(f"[smoke] vae scale spatial/temporal = {wm.vae_scale_factor_spatial}/{wm.vae_scale_factor_temporal}")

    images = [[Image.fromarray((np.random.rand(224, 448, 3) * 255).astype(np.uint8)) for _ in range(2)]]
    instructions = ["pick up the red block and put it on the plate"]

    inputs = wm.build_inputs(images, instructions)
    print(f"[smoke] encoder_hidden_states shape = {tuple(inputs['encoder_hidden_states'].shape)}")
    print(f"[smoke] hidden_states (latents) shape = {tuple(inputs['hidden_states'].shape)}")
    print(f"[smoke] timestep shape = {tuple(inputs['timestep'].shape)}")

    out = wm(**inputs)
    for i, h in enumerate(out.hidden_states):
        print(f"[smoke] hook[{i}] hidden_states shape = {tuple(h.shape)} dtype={h.dtype}")
        assert torch.isfinite(h).all(), "NaN/Inf in extracted features"
    print("[smoke] OK")
