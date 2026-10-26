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
"""GlanceWAM on a tiny random SkyReels backbone (GPU: FlexAttention needs the fused kernel)."""

import contextlib

import pytest
import torch
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.import_utils import _diffusers_available, _transformers_available

pytestmark = [
    pytest.mark.skipif(
        not (_diffusers_available and _transformers_available), reason="needs diffusers + transformers"
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention smoke needs a GPU"),
]

TEXT_DIM = 64
CAMS = [f"{OBS_IMAGES}.image", f"{OBS_IMAGES}.wrist_image"]


def _tiny_components(config, load_pretrained):
    from diffusers import AutoencoderKLWan, SkyReelsV2Transformer3DModel

    torch.manual_seed(0)
    transformer = SkyReelsV2Transformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=16,
        out_channels=16,
        text_dim=TEXT_DIM,
        freq_dim=32,
        ffn_dim=64,
        num_layers=4,
        inject_sample_info=True,
    ).to(torch.float32)
    vae = AutoencoderKLWan(base_dim=8, dim_mult=[1, 1, 1, 1], num_res_blocks=1).to(torch.bfloat16)
    return transformer, vae


@pytest.fixture
def policy(monkeypatch):
    from lerobot_policy_glancewam.configuration_glancewam import GlanceWAMConfig
    from lerobot_policy_glancewam.modeling_glancewam import GlanceWAMPolicy
    from lerobot_policy_glancewam.world_model import SkyReelsWorldModel

    monkeypatch.setattr(SkyReelsWorldModel, "load_components", staticmethod(_tiny_components))
    config = GlanceWAMConfig(
        input_features={
            **{k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 40, 40)) for k in CAMS},
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(5,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        chunk_size=4,
        n_action_steps=4,
        vae_input_size=(32, 64),
        view_size=32,
        extract_layers=[1, 3],
        future_frame_idx=6,
        goal_frame_range=(1, 6),
        text_max_length=8,
        action_model_type="DiT-B",
        num_action_head_layers=2,
        goal_gen_steps=2,
        num_inference_timesteps=2,
        repeated_diffusion_steps=2,
        use_state=True,
        action_head_state_dim=5,
        compile_vae=False,
        sdpa_backend="auto",
        device="cuda",
        fast_inference=False,  # compile is too slow for a unit test; test_fast_path covers the truncated DiT
    )
    return GlanceWAMPolicy(config, load_pretrained_backbone=False).cuda()


def _train_batch(policy, bsz=2):
    cfg = policy.config
    batch = {k: torch.rand(bsz, cfg.window_frames, 3, 40, 40, device="cuda") for k in CAMS}
    batch[OBS_STATE] = torch.rand(bsz, 5, device="cuda")
    batch[ACTION] = torch.rand(bsz, cfg.chunk_size, 7, device="cuda") * 2 - 1
    batch["observation.language.embedding"] = torch.randn(bsz, cfg.text_max_length, TEXT_DIM, device="cuda")
    batch["observation.goal_offset"] = torch.tensor([3, 6], device="cuda")
    return batch


def test_forward_backward(policy):
    policy.train()
    loss, info = policy.forward(_train_batch(policy))
    assert torch.isfinite(loss)
    assert {"action_loss", "video_loss"} <= set(info)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.action_model.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy.backbone.transformer.parameters()
    )
    assert policy.tap_embed.weight.grad is not None
    # The frozen VAE is outside the module tree.
    assert not any(k.startswith("backbone._vae") for k in policy.state_dict())
    assert all(
        k.split(".")[0] in {"backbone", "action_model", "goal_time_proj", "tap_embed"}
        for k in policy.state_dict()
    )


def test_goal_isolation(policy):
    """Obs and future tokens must be bit-identical under different goal frames; goal tokens must differ."""
    policy.eval()
    obs_n = policy.obs_latent_frames
    t_lat = obs_n + 2
    h, w = policy.config.vae_input_size[0] // 8, policy.config.vae_input_size[1] // 8
    xt = torch.randn(1, 16, t_lat, h, w, device="cuda")
    ts = torch.zeros(1, t_lat, device="cuda")
    ts[:, obs_n : t_lat - 1] = 500.0
    text = torch.randn(1, 8, TEXT_DIM, device="cuda", dtype=torch.bfloat16)
    xt_b = xt.clone()
    xt_b[:, :, -1] = torch.randn_like(xt_b[:, :, -1])
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fa, _ = policy.backbone.run_dit(xt, ts, text, n_goal_frames=1)
        fb, _ = policy.backbone.run_dit(xt_b, ts, text, n_goal_frames=1)
    ftok = policy.tokens_per_frame
    for a, b in zip(fa, fb, strict=True):
        assert torch.equal(a[:, : obs_n * ftok], b[:, : obs_n * ftok]), "goal leaked into the obs read-point"
        assert torch.equal(a[:, obs_n * ftok : (t_lat - 1) * ftok], b[:, obs_n * ftok : (t_lat - 1) * ftok])
        assert not torch.equal(a[:, (t_lat - 1) * ftok :], b[:, (t_lat - 1) * ftok :]), "goal frame ignored"


def test_predict_and_goal_cadence(policy):
    cfg = policy.config
    bsz = 3
    batch = {k: torch.rand(bsz, 3, 40, 40, device="cuda") for k in CAMS}
    batch[OBS_STATE] = torch.rand(bsz, 5, device="cuda")
    batch["observation.language.embedding"] = torch.randn(bsz, cfg.text_max_length, TEXT_DIM, device="cuda")
    policy.reset()
    a1 = policy.predict_action_chunk(batch)
    assert a1.shape == (bsz, cfg.chunk_size, 7) and torch.isfinite(a1).all()
    g1 = policy._goal_latent.clone()
    assert policy._goal_frame_row == 0 and policy._tick == cfg.n_action_steps
    policy.predict_action_chunk(batch)  # row 4 < H_g=6: hold
    assert torch.equal(policy._goal_latent, g1)
    policy.predict_action_chunk(batch)  # row 8 >= 6: refresh from this frame
    assert not torch.equal(policy._goal_latent, g1)
    assert policy._goal_frame_row == 8 and policy._tick == 12

    policy.reset()
    assert policy._goal_latent is None
    step = policy.select_action(batch)
    assert step.shape == (bsz, 7)
    assert len(policy._action_queue) == cfg.n_action_steps - 1


def test_save_load_roundtrip(policy, tmp_path):
    policy.save_pretrained(tmp_path)
    assert (tmp_path / "model.safetensors").exists() and (tmp_path / "config.json").exists()
    from lerobot_policy_glancewam.modeling_glancewam import GlanceWAMPolicy
    from lerobot_policy_glancewam.world_model import SkyReelsWorldModel

    SkyReelsWorldModel.load_components = staticmethod(_tiny_components)
    loaded = GlanceWAMPolicy.from_pretrained(tmp_path)
    for (ka, va), (kb, vb) in zip(policy.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert ka == kb and torch.equal(va.cpu(), vb.cpu())


def test_processor_steps(tmp_path):
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action

    from lerobot_policy_glancewam.processor_glancewam import (
        GlanceWAMGripperProcessorStep,
        UMT5TextTableProcessorStep,
    )

    step = UMT5TextTableProcessorStep(model_id="unused", max_length=4, prompts=["a", "b"])
    step._table = {
        "a": torch.zeros(4, 4096, dtype=torch.float16),
        "b": torch.ones(4, 4096, dtype=torch.float16),
    }
    from lerobot.lerobot_types import TransitionKey

    tr = {
        TransitionKey.OBSERVATION: {"observation.state": torch.zeros(2, 3)},
        TransitionKey.COMPLEMENTARY_DATA: {"task": ["b", "a"]},
    }
    out = step(tr)[TransitionKey.OBSERVATION]["observation.language.embedding"]
    assert out.shape == (2, 4, 4096) and out.dtype == torch.bfloat16
    assert out[0].float().mean() == 1.0 and out[1].float().mean() == 0.0
    state = step.state_dict()
    assert set(state) == {"row_0", "row_1"}
    step2 = UMT5TextTableProcessorStep(**step.get_config())
    step2.load_state_dict(state)
    assert torch.equal(step2._table["b"], step._table["b"])

    grip = PolicyProcessorPipeline(
        steps=[GlanceWAMGripperProcessorStep(mode="libero", index=-1)],
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    act = torch.tensor([[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.9], [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.2]])
    res = grip(act)
    assert res[0, -1] == -1.0 and res[1, -1] == 1.0 and torch.equal(res[:, :-1], act[:, :-1])


class _InlineGoalWorker:
    """Stand-in for ``GoalWorker``: samples on the calling thread, replies after ``latency`` polls."""

    def __init__(self, policy, latency=0):
        self.policy = policy
        self.latency = latency
        self.submitted: list[tuple] = []
        self.collected = 0
        self._pending = None
        self._polls = 0

    @property
    def pending(self):
        return self._pending is not None

    def submit(self, obs_latent, text_embeds, decode=False):
        self._pending = (obs_latent.clone(), text_embeds.clone())
        self.submitted.append(self._pending)
        self._polls = 0

    def _reply(self, device):
        obs, text = self._pending
        self._pending = None
        self.collected += 1
        return self.policy._generate_goal_latent(obs, text).to(device), None

    def collect(self, device):
        return self._reply(device)

    def poll(self, device):
        if self._pending is None:
            return None
        self._polls += 1
        return self._reply(device) if self._polls > self.latency else None

    def discard(self):
        self._pending = None

    def wait_ready(self):
        pass

    @contextlib.contextmanager
    def gpu_busy(self):
        yield


def _obs_batch(cfg, bsz=2):
    batch = {k: torch.rand(bsz, 3, 40, 40, device="cuda") for k in CAMS}
    batch[OBS_STATE] = torch.rand(bsz, 5, device="cuda")
    batch["observation.language.embedding"] = torch.randn(bsz, cfg.text_max_length, TEXT_DIM, device="cuda")
    return batch


def test_async_goal_row_clock(policy):
    """Every ``goal_refresh_rows`` the current frame is sent (queued ticks too); replies install on arrival;
    the goal-time diff is horizon minus the age of the goal's own frame, floored at 1."""
    cfg = policy.config
    worker = _InlineGoalWorker(policy, latency=1)  # reply lands on the second poll after the request
    policy._goal_worker = worker
    policy.goal_horizon_rows = 16.0
    policy.goal_refresh_rows = 10  # not a multiple of the chunk (4): requests land on queued ticks too
    batch = _obs_batch(cfg)
    policy.reset()

    log = []  # (row, goal frame row, request row) after each tick
    distances = {}
    orig = policy._decode_chunk

    def spy(b, o, t, g, row=0, **kw):
        distances[row] = policy._goal_distance(row)
        return orig(b, o, t, g, row, **kw)

    policy._decode_chunk = spy
    for _ in range(40):
        row = policy._row
        policy.select_action(batch)
        log.append((row, policy._goal_frame_row, policy._goal_request_row))

    # Row 0: nothing held, so the first goal is requested and collected inline.
    assert log[0] == (0, 0, None)
    # Frames every 10 rows: requested at 10 (queued tick), installed on the second poll at 12; then 20 -> 22 ...
    assert (
        log[9][2] is None and log[10] == (10, 0, 10) and log[11] == (11, 0, 10) and log[12] == (12, 10, None)
    )
    assert log[20] == (20, 10, 20) and log[22][1] == 20 and log[30][2] == 30 and log[32][1] == 30
    assert len(worker.submitted) == 4 and worker.collected == 4
    # Chunk decodes: horizon 16 minus the age of the held goal's frame.
    assert distances[0] == 16 and distances[4] == 12 and distances[8] == 8
    assert distances[12] == 14 and distances[16] == 10  # goal@10
    assert distances[20] == 6  # row 20: the row-20 request is sent after this decode, goal@10 still held
    assert distances[24] == 12 and distances[28] == 8  # goal@20
    assert distances[32] == 14 and distances[36] == 10  # goal@30, installed on the row-32 chunk tick

    # A reset drops the goal and the in-flight request; a task change does the same.
    policy.reset()
    assert not worker.pending and policy._goal_latent is None and policy._goal_request_row is None
    for _ in range(11):
        policy.select_action(batch)
    assert worker.pending
    policy.drop_queued_actions()
    assert not worker.pending and policy._goal_latent is None and len(policy._action_queue) == 0


def test_async_goal_late_reply_never_blocks(policy):
    """After the first goal a late reply never stalls a tick: the held goal is used with distance >= 1."""
    cfg = policy.config
    worker = _InlineGoalWorker(policy, latency=10**6)  # never ready by polling
    policy._goal_worker = worker
    policy.goal_horizon_rows = 16.0
    policy.goal_refresh_rows = 16
    batch = _obs_batch(cfg)
    policy.reset()
    seen = {}
    orig = policy._decode_chunk

    def spy(b, o, t, g, row=0, **kw):
        seen[row] = policy._goal_distance(row)
        return orig(b, o, t, g, row, **kw)

    policy._decode_chunk = spy
    for _ in range(25):
        policy.select_action(batch)
    assert worker.pending and policy._goal_frame_row == 0 and worker.collected == 1
    assert seen[16] == 1.0 and seen[20] == 1.0  # goal@0 past its horizon, floored


def test_goal_schedule_validation():
    from lerobot_policy_glancewam.configuration_glancewam import GlanceWAMConfig

    with pytest.raises(ValueError, match="goal_refresh_rows"):
        GlanceWAMConfig(future_frame_idx=6, goal_refresh_rows=7)
    GlanceWAMConfig(future_frame_idx=60, goal_refresh_rows=48)


def test_fast_path_matches_stock(policy):
    """The truncated hook-free DiT (compile off) reproduces the stock features and actions."""
    from lerobot_policy_glancewam.modeling_glancewam import FastChunkPath

    cfg = policy.config
    bsz = 2
    batch = {k: torch.rand(bsz, 3, 40, 40, device="cuda") for k in CAMS}
    batch[OBS_STATE] = torch.rand(bsz, 5, device="cuda")
    batch["observation.language.embedding"] = torch.randn(bsz, cfg.text_max_length, TEXT_DIM, device="cuda")
    frames = policy._stitched_window(batch)[:, : cfg.num_history_frames]
    text = policy._language(batch)
    obs_latent = policy.backbone.encode_frames(frames)
    goal = torch.randn((bsz, 16, cfg.n_goal_frames, *obs_latent.shape[3:]), device="cuda")

    torch.manual_seed(0)
    stock = policy._decode_chunk(batch, obs_latent, text, goal)
    policy._fast = FastChunkPath(policy, compile_mode=None)
    torch.manual_seed(0)
    fast = policy._decode_chunk(batch, obs_latent, text, goal)
    assert policy._fast.trunc.n_run == max(policy.backbone.taps) + 1
    assert torch.allclose(stock, fast, atol=1e-3, rtol=1e-3), (stock - fast).abs().max()
    policy._fast = None


def test_rtc_guidance(policy):
    """The RTC engine's call shape guides the chunk onto the previous chunk's tail (pi0.5's processor math)."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.rollout.inference.rtc import supports_rtc_inference

    cfg = policy.config
    assert supports_rtc_inference(policy)
    batch = _obs_batch(cfg, bsz=1)

    def chunk(delay, prefix):  # a fresh episode each time: same goal noise, same head noise
        policy.reset()
        torch.manual_seed(0)
        return policy.predict_action_chunk(batch, inference_delay=delay, prev_chunk_left_over=prefix)

    free = chunk(0, None)
    policy.config.rtc_config = RTCConfig(execution_horizon=3, max_guidance_weight=10.0)
    policy.init_rtc_processor()
    prefix = torch.full((3, 7), 0.5, device="cuda")  # (T, A): the engine's normalized leftover
    guided = chunk(1, prefix)
    assert guided.shape == free.shape == (1, cfg.chunk_size, 7) and torch.isfinite(guided).all()
    # Prefix weights [1, 2/3, 1/3, 0]: the delayed step is pinned to the prefix (the last Euler step of a
    # 2-step sampler lands exactly on it), the ramp pulls the next ones, the tail is left to the model.
    assert torch.allclose(guided[0, 0], prefix[0], atol=1e-3), guided[0, 0]
    err_free, err_guided = (free[0] - 0.5).abs().mean(-1), (guided[0] - 0.5).abs().mean(-1)
    assert err_guided[1] < err_free[1] and err_guided[2] < err_free[2]
    assert not torch.allclose(guided, free)
    # Without a prefix (the episode's first chunk) the sampler is the plain one.
    assert torch.allclose(chunk(0, None), free)


def test_rtc_row_clock_and_first_chunk(policy, monkeypatch):
    """Under the RTC engine the goal clock runs on wall-clock rows (+ the expected delay) and no chunk call
    waits for the worker: the first chunk decodes with the no-goal fallback while the goal is imagined."""
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    cfg = policy.config
    worker = _InlineGoalWorker(policy, latency=1)  # reply lands on the second poll after the request
    policy._goal_worker = worker
    policy.goal_horizon_rows = 16.0
    policy.goal_refresh_rows = 10
    cfg.rows_per_second = 20.0
    cfg.rtc_config = RTCConfig(execution_horizon=3)
    policy.init_rtc_processor()
    clock = [100.0]
    monkeypatch.setattr("lerobot_policy_glancewam.modeling_glancewam.time.perf_counter", lambda: clock[0])
    batch = _obs_batch(cfg, bsz=1)
    policy.reset()
    seen = []  # (row, decoded without a goal, goal-time diff) per chunk call
    orig = policy._decode_chunk

    def spy(b, o, t, g, row=0, **kw):
        dist = policy._goal_distance(row) if policy._goal_frame_row is not None else None
        seen.append((row, g is None, dist))
        return orig(b, o, t, g, row, **kw)

    policy._decode_chunk = spy

    def call(delay, prefix=None):
        return policy.predict_action_chunk(batch, inference_delay=delay, prev_chunk_left_over=prefix)

    a = call(0)
    assert seen[-1] == (0, True, None) and worker.pending and policy._tick == 0
    prefix = a[0, -3:]
    clock[0] += 0.25  # 5 rows later; the engine expects this call to take 2 rows
    call(2, prefix)
    assert seen[-1] == (7, True, None)  # first poll: the reply is not in yet, still no goal
    clock[0] += 0.25
    call(2, prefix)
    assert seen[-1] == (12, False, 4.0) and policy._goal_frame_row == 0  # goal@0 lands; 16 - 12
    assert policy._goal_request_row == 12 and worker.pending  # 12 >= 0 + 10: the next frame was sent
    clock[0] += 0.25
    call(2, prefix)
    assert seen[-1] == (17, False, 1.0)  # first poll: goal@0 past its horizon, floored
    clock[0] += 0.25
    call(2, prefix)
    assert seen[-1] == (22, False, 6.0) and policy._goal_frame_row == 12  # goal@12 lands; 16 - 10
    assert policy._goal_request_row == 22 and len(worker.submitted) == 3  # rows 0, 12, 22
    # A sync-engine call keeps the tick counter; a reset restarts the wall clock with the goal.
    policy.reset()
    assert policy._rtc_t0 is None and not worker.pending and policy._goal_latent is None
    policy.predict_action_chunk(batch)
    assert seen[-1] == (0, False, 16.0) and policy._tick == cfg.n_action_steps
