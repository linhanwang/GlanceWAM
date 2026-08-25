"""Per-layer torch.compile for video-DiT world-model backbones.

Mirrors ``glancewam/model/modules/vlm/speedup_patch.py`` for the VLM side: wraps
each transformer block independently with ``Module.compile`` so a graph break
inside one block doesn't kill the rest. Uses the in-place ``Module.compile``
method (which patches ``forward`` rather than replacing the module instance),
so any forward hooks already registered on the blocks keep firing — important
for ``CosmoPredict2PI``, which captures per-layer hidden states via
``register_forward_hook``.

Measured at B=64, 224x224 latent, single-GPU H200, both caches: ~2.36× on DiT
forward, ~1.70× on a full forward+backward training step, and ~32% peak-VRAM
reduction (Inductor activation reuse). See `tools/profile_cosmopredict2pi.py
--compile-dit` for the A/B harness.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def find_dit_blocks(world_model: nn.Module) -> nn.ModuleList:
    """Locate the per-block ModuleList on a Cosmos/Wan video-DiT backbone.

    Cosmos exposes ``transformer.transformer_blocks``; Wan exposes
    ``transformer.blocks``. Both are flat ``nn.ModuleList`` stacks.
    """
    transformer = getattr(world_model, "transformer", None)
    if transformer is None:
        raise RuntimeError("world_model has no `.transformer` attribute.")
    for attr in ("transformer_blocks", "blocks"):
        candidate = getattr(transformer, attr, None)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate
    raise RuntimeError(
        "Could not locate transformer block ModuleList on this world model "
        "(tried `transformer.transformer_blocks` and `transformer.blocks`)."
    )


def compile_dit_blocks(
    world_model: nn.Module,
    mode: Optional[str] = None,
    dynamic: bool = True,
    suppress_errors: bool = False,
) -> int:
    """Per-layer ``torch.compile`` on the DiT block stack.

    Args:
        world_model: instance returned by ``get_world_model(...)``.
        mode: passes through to ``torch.compile(mode=...)``. ``None`` uses the
            PyTorch default (Inductor basic fusion). ``"reduce-overhead"`` is
            incompatible with ``dynamic=True``; ``"max-autotune"`` rarely
            beats default on DiT-shape workloads and pays a much longer first
            compile.
        dynamic: forward to ``torch.compile(dynamic=...)``. Default ``True``
            lets the same compiled artifact serve the training batch size and
            the eval B=1 path without recompiling. Set ``False`` to bake the
            actual training shapes into the graph — fewer symbolic expressions,
            which dodges Inductor sympy bugs (e.g. the SANA-Video Mod-with-
            negative-dividend assert) at the cost of recompiling for eval.
        suppress_errors: if ``True``, sets ``torch._dynamo.config.suppress_errors``
            so that any subgraph that fails to compile silently falls back to
            eager instead of killing the whole training run. Used on SANA-Video,
            where a handful of shape patterns in ``GLUMBTempConv`` / the rotary
            apply still crash Inductor even with static shapes — partial compile
            still nets a real speedup on the rest of each block.

    Returns:
        Number of blocks compiled.
    """
    if suppress_errors:
        import torch._dynamo as _dynamo

        _dynamo.config.suppress_errors = True
    blocks = find_dit_blocks(world_model)
    compile_kwargs: dict = {"dynamic": dynamic}
    if mode is not None:
        compile_kwargs["mode"] = mode
    for blk in blocks:
        blk.compile(**compile_kwargs)
    return len(blocks)


# --- prefix-LM FlexAttention (clean-prefix isolation) --------------------
#
# Video+action co-training wants the clean conditioning prefix frames (per-frame
# timestep 0) to be ISOLATED from the noised future frames, so the t=0 feature
# read-point is independent of the future noise sample -- while the noised frames
# still attend to the clean prefix AND bidirectionally to each other (full video-
# denoising capacity). That is a prefix-LM mask, NOT diffusers' uniform block-
# causal `_set_ar_attention` (which would also make earlier noised frames unable
# to see later ones). An explicit dense mask through SDPA falls off FlashAttention
# onto the mem-efficient kernel (~2.3x slower on the attn op); FlexAttention runs
# the structured mask on a fused kernel that SKIPS fully-masked blocks, so it is
# actually faster than full bidirectional attention. See
# the SkyReels-V2 DF video-generation objective.
#
# The clean/noised split is read PER-SAMPLE, PER-FRAME from the timestep matrix
# (`timestep == 0` -> clean). This preserves the native conditioning mix (1 vs 2
# clean frames per sample) -- no forced prefix -- and is fully general (any clean
# pattern). The per-sample BlockMask is rebuilt once per forward (cheap, shared by
# all blocks) and THREADED as the block `attention_mask` argument, exactly like
# diffusers' own `causal_mask`, so it is torch.compile-safe (a graph input guarded
# on shape, not value) under --compile_dit.
#
# TRAINING-ONLY: at feature-extraction time every frame is clean (timestep 0) so
# the mask degenerates to full attention -- the policy side (SkyReelsV2DF.py)
# needs NO patch and stays bit-exact.

_compiled_flex_attention = None


def _get_compiled_flex():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention

        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def _build_prefix_lm_block_mask(
    timestep: torch.Tensor, post_frames: int, frame_tok: int, device, n_action_tokens: int = 0
):
    """Per-sample prefix-LM BlockMask from a [B, T_lat] per-frame timestep matrix.
    clean = (timestep == 0). A clean query frame may not attend to a noised key
    frame; everything else (noised->anything, anything->clean) is allowed.

    With ``n_action_tokens > 0`` (rung 12a: clean action-conditioning tokens appended
    after the video tokens) the mask becomes 3-class and INDEX-based — the action
    tokens are clean, so a timestep-only rule would expose them to the clean obs
    prefix and leak the ground-truth chunk into the action readout:
      - clean video queries  -> clean video keys only (action tokens excluded);
      - action queries       -> clean video + action keys (never the noised future);
      - noised video queries -> everything (this is what action-conditions the WM).
    With ``n_action_tokens == 0`` the rule reduces exactly to the original 2-class mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    clean = timestep == 0  # [B, T_lat] bool
    n_video = post_frames * frame_tok
    max_frame = post_frames - 1

    def prefix_lm(b, h, q_idx, kv_idx):
        q_act = q_idx >= n_video
        kv_act = kv_idx >= n_video
        # Clamp: action-token indices land past the last video frame; the clamped
        # lookup is discarded for them via the ``~q_act`` / ``~kv_act`` guards.
        q_clean_vid = (~q_act) & clean[b, torch.clamp(q_idx // frame_tok, max=max_frame)]
        kv_clean_vid = (~kv_act) & clean[b, torch.clamp(kv_idx // frame_tok, max=max_frame)]
        q_noised_vid = (~q_act) & (~q_clean_vid)
        return q_noised_vid | (q_clean_vid & kv_clean_vid) | (q_act & (kv_clean_vid | kv_act))

    seq_len = n_video + n_action_tokens
    return create_block_mask(prefix_lm, B=int(timestep.shape[0]), H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def _action_token_rotary_rows(rope, n_act: int, t_pos: float):
    """RoPE rows for appended action-conditioning tokens (rung 12a), lingbot-va's
    coordinate recipe: same rotary math as video tokens with hand-assigned off-grid
    coordinates — fractional temporal position ``t_pos`` (midpoint between the last
    obs frame and the first future frame) and sentinel spatial coords h = w = -1.
    Matches get_1d_rotary_pos_embed(use_real=True, repeat_interleave_real=True)
    layout, which the FlexAttention processor's apply_rotary_emb assumes.
    Returns (cos, sin), each [1, n_act, 1, attention_head_dim]."""
    theta = 10000.0  # SkyReelsV2RotaryPosEmbed init default (not stored as an attribute)
    device = rope.freqs_cos.device
    cos_parts, sin_parts = [], []
    for dim, pos in ((rope.t_dim, t_pos), (rope.h_dim, -1.0), (rope.w_dim, -1.0)):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim))
        angle = pos * freqs
        cos_parts.append(angle.cos().repeat_interleave(2))
        sin_parts.append(angle.sin().repeat_interleave(2))
    cos = torch.cat(cos_parts).to(rope.freqs_cos.dtype).view(1, 1, 1, -1).expand(1, n_act, 1, -1)
    sin = torch.cat(sin_parts).to(rope.freqs_sin.dtype).view(1, 1, 1, -1).expand(1, n_act, 1, -1)
    return cos, sin


class PrefixLMFlexAttnProcessor:
    """Drop-in SkyReelsV2 attention processor. Self-attention runs through
    FlexAttention WHEN a BlockMask is threaded in via `attention_mask` (built by
    the patched forward from the timestep); otherwise -- cross-attention, or
    self-attention with no mask (isolation off / all-clean extraction) -- it
    delegates to the stock diffusers processor unchanged (bit-exact)."""

    def __init__(self):
        from diffusers.models.transformers.transformer_skyreels_v2 import SkyReelsV2AttnProcessor

        self._stock = SkyReelsV2AttnProcessor()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        from torch.nn.attention.flex_attention import BlockMask

        # Cross-attn / I2V image context, or no BlockMask threaded -> stock path
        # (preserves any tensor mask and the all-clean / disabled full-attn case).
        if encoder_hidden_states is not None or getattr(attn, "add_k_proj", None) is not None:
            return self._stock(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        if not isinstance(attention_mask, BlockMask):
            return self._stock(attn, hidden_states, None, attention_mask, rotary_emb)

        from diffusers.models.transformers.transformer_skyreels_v2 import _get_qkv_projections

        # self-attn: passing hidden_states as encoder is identical to None here
        # (the helper aliases encoder<-hidden when None) and keeps types happy.
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

        # [B, S, H, D] -> [B, H, S, D] for FlexAttention.
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = _get_compiled_flex()(q, k, v, block_mask=attention_mask)
        out = out.transpose(1, 2).flatten(2, 3).type_as(query)

        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


def _prefix_lm_forward(
    self,
    hidden_states,
    timestep,
    encoder_hidden_states,
    encoder_hidden_states_image=None,
    enable_diffusion_forcing=False,
    fps=None,
    return_dict=True,
    attention_kwargs=None,
    action_tokens=None,
):
    """Vendored copy of diffusers SkyReelsV2Transformer3DModel.forward
    (diffusers 0.39.0.dev0) with TWO changes: when prefix-LM isolation is enabled
    it builds a per-sample prefix-LM BlockMask from `timestep` and threads it to
    the blocks in the `causal_mask` slot (instead of the unused block-causal mask);
    and `action_tokens` [B, n_act, inner_dim] (rung 12a, optional) are appended to
    the sequence as clean conditioning tokens — kept invisible to the clean obs
    prefix by the 3-class mask, given the clean τ=0 modulation row and off-grid
    RoPE coords, and sliced off before the velocity head so the video output is
    shape-identical to stock. `action_tokens=None` (every pre-existing call site)
    is bit-identical to the previous behavior.
    Re-sync this copy if the diffusers transformer_skyreels_v2 forward changes."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # --- prefix-LM mask + action-conditioning tokens (deviations from stock) ---
    n_act = 0
    if action_tokens is not None:
        if not (enable_diffusion_forcing and timestep.dim() == 2):
            raise RuntimeError(
                "action_tokens require diffusion-forcing mode with a per-frame [B, T] timestep (rung 12a)."
            )
        if not bool((timestep[:, 0] == 0).all()):
            raise RuntimeError(
                "action_tokens expect a clean (τ=0) leading obs frame — its modulation row is "
                "reused for the action token(s)."
            )
        n_act = int(action_tokens.shape[1])

    causal_mask = None
    if enable_diffusion_forcing and timestep.dim() == 2:
        causal_mask = _build_prefix_lm_block_mask(
            timestep,
            post_patch_num_frames,
            post_patch_height * post_patch_width,
            hidden_states.device,
            n_action_tokens=n_act,
        )
    # ---------------------------------------------------------------------------

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

    n_video_tok = post_patch_num_frames * post_patch_height * post_patch_width
    if n_act and action_tokens is not None:
        # Append the clean action token(s) after the video tokens. Modulation: reuse
        # frame 0's token row — in DF mode timestep_proj is token-level [B, 6, S, dim]
        # repeated per frame, so token 0 is exactly the clean τ=0 row (incl. any fps
        # injection). RoPE: fractional temporal coord at the last-obs/first-future
        # midpoint, sentinel spatial coords (lingbot-va recipe).
        hidden_states = torch.cat([hidden_states, action_tokens.to(hidden_states.dtype)], dim=1)
        clean_row = timestep_proj[:, :, :1, :]
        timestep_proj = torch.cat([timestep_proj, clean_row.expand(-1, -1, n_act, -1)], dim=2)
        n_obs_frames = int((timestep[0] == 0).sum())
        act_cos, act_sin = _action_token_rotary_rows(self.rope, n_act, t_pos=n_obs_frames - 0.5)
        freqs_cos, freqs_sin = rotary_emb
        rotary_emb = (
            torch.cat([freqs_cos, act_cos.to(device=freqs_cos.device, dtype=freqs_cos.dtype)], dim=1),
            torch.cat([freqs_sin, act_sin.to(device=freqs_sin.device, dtype=freqs_sin.dtype)], dim=1),
        )

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.blocks:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask
            )
    else:
        for block in self.blocks:
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask)

    if n_act:
        # Drop the action token(s) before the velocity head: the video output (and
        # `temb` for the final norm, which was never extended) stays shape-identical
        # to stock. Block-capture hooks fire BEFORE this slice, so tapped features
        # carry the extra trailing token — downstream consumers slice the leading
        # obs tokens and are unaffected.
        hidden_states = hidden_states[:, :n_video_tok]

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


def enable_prefix_lm_flex_attention(transformer: nn.Module) -> int:
    """Enable per-sample prefix-LM clean-prefix isolation on a SkyReelsV2 DiT:
    set a FlexAttention processor on every attention module and bind the vendored
    forward that builds + threads the per-sample BlockMask from the timestep.
    The clean/noised split is derived from the timestep, so the native 1-vs-2
    clean-frame conditioning mix is preserved (no forced prefix). Returns the
    number of attention processors set. Training-only; see module note."""
    import types

    transformer.set_attn_processor(PrefixLMFlexAttnProcessor())
    transformer.forward = types.MethodType(_prefix_lm_forward, transformer)
    return len(transformer.attn_processors)


# --- goal-conditioned cotrain: 3-class prefix-LM mask (goal side-channel) -----
#
# The goal-image planner adds a CLEAN
# "goal" frame to the DiT window purely as a read-only side-channel for the action
# head: it must be grounded on the obs (so it can attend to the obs prefix) but
# NOTHING may attend back to it — obs tokens (the action read-point) AND the noised
# future tokens (the video-loss branch) both stay bit-identical to the plain
# cotrain. That is a THIRD attention class the 2-class timestep mask cannot express,
# because the goal frame is ALSO clean (τ=0) and a timestep-only rule would fold it
# into the obs prefix (obs would then attend it, corrupting the policy path).
#
# The goal frames are the LAST `n_goal_frames` video frames (index-based, like the
# rung-12a action tokens). Classes, given clean = (timestep == 0):
#   obs query    (clean, non-goal)  -> obs keys only      (blind to goal AND future)
#   future query (noised)           -> obs + future keys   (video capacity; NOT goal)
#   goal query   (clean, goal)      -> obs + goal keys     (grounded on obs; not future)
# With n_goal_frames == 0 this reduces EXACTLY to the 2-class `_build_prefix_lm_block_mask`
# (obs->obs, future->everything), so the plain cotrain path is unchanged.


def _build_goal_prefix_lm_block_mask(timestep, n_goal_frames, post_frames, frame_tok, device):
    """Per-sample 3-class (obs / noised-future / clean-goal) prefix-LM BlockMask.

    `timestep` is [B, T_lat] (0 => clean). The last `n_goal_frames` frames are the
    clean goal side-channel; the remaining clean frames are obs, the noised frames
    are the video target. See the module note above for the class semantics. Nothing
    attends to the goal frames except the goal frames themselves, so obs/future token
    hidden states are identical to the plain 2-class mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    clean = timestep == 0  # [B, T_lat] bool
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
        q_future = ~q_clean  # noised
        kv_obs = kv_clean & (~kv_goal)
        kv_future = ~kv_clean
        return (
            (q_obs & kv_obs)  # obs   -> obs
            | (q_future & (kv_obs | kv_future))  # future -> obs + future (never goal)
            | (q_goal & (kv_obs | kv_goal))  # goal  -> obs + goal (never future)
        )

    seq_len = n_frames * frame_tok
    return create_block_mask(
        goal_prefix_lm, B=int(timestep.shape[0]), H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device
    )


def _goal_prefix_lm_forward(
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
    """Vendored SkyReelsV2Transformer3DModel.forward with ONE deviation from stock:
    when diffusion-forcing is on it threads a per-sample 3-class GOAL prefix-LM
    BlockMask (built from `timestep` + `n_goal_frames`) into the blocks' `causal_mask`
    slot. The goal frame is a real clean latent frame already present in
    `hidden_states` (no token append/slice, unlike the rung-12a action-token path), so
    patchify / RoPE / unpatchify are all stock — only the attention mask changes.
    `n_goal_frames == 0` reduces to the plain 2-class prefix-LM mask (bit-identical to
    `_prefix_lm_forward` with `action_tokens=None`). Re-sync this copy if the diffusers
    transformer_skyreels_v2 forward changes."""
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    rotary_emb = self.rope(hidden_states)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # --- goal-aware prefix-LM mask (the only deviation from stock) ---
    causal_mask = None
    if enable_diffusion_forcing and timestep.dim() == 2 and not getattr(self, "_goal_bidirectional_ablation", False):
        causal_mask = _build_goal_prefix_lm_block_mask(
            timestep,
            int(n_goal_frames),
            post_patch_num_frames,
            post_patch_height * post_patch_width,
            hidden_states.device,
        )
    # -----------------------------------------------------------------

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
            hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb, causal_mask)

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


def enable_goal_prefix_lm_flex_attention(transformer: nn.Module) -> int:
    """Enable goal-conditioned 3-class clean-prefix isolation on a SkyReelsV2 DiT: set the FlexAttention processor on every
    attention module and bind the goal-aware vendored forward. The processor is the
    same generic `PrefixLMFlexAttnProcessor` (it applies whatever BlockMask is
    threaded); only the forward differs (goal mask instead of the 2-class mask). The
    goal-frame count is passed per-call as `n_goal_frames`. Returns the number of
    attention processors set. Training + goal-conditioning eval; see module note."""
    import types

    transformer.set_attn_processor(PrefixLMFlexAttnProcessor())
    transformer.forward = types.MethodType(_goal_prefix_lm_forward, transformer)
    return len(transformer.attn_processors)


def enable_goal_bidirectional_attention(transformer: nn.Module) -> int:
    """Ablation arm (clean_prefix_isolation=False): bind the goal-aware vendored
    forward but build NO attention mask — every token attends to every token, so the
    obs (action read-point) sees the noised future AND the clean goal frame. Stock
    attention processors are kept (plain SDPA; no FlexAttention BlockMask), matching
    stock SkyReels DF attention. `n_goal_frames` is still accepted so the framework
    call-sites are unchanged. Returns 0 (no flex processors set)."""
    import types

    transformer._goal_bidirectional_ablation = True
    transformer.forward = types.MethodType(_goal_prefix_lm_forward, transformer)
    return 0
