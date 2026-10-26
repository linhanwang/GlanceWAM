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
"""Goal generation in a worker process, off the control thread.

The rollout's sync engine calls ``select_action`` inline on the control thread, so every millisecond the
policy spends there is a millisecond the robot waits for its next target. The goal sampler (``goal_gen_steps``
Euler passes of the video DiT, ~0.5 s on a 4090) is the one piece that does not have to run there: the
goal is imagined ``future_frame_idx`` rows ahead and held across several action chunks, so it can be
computed from a slightly older observation while the GPU is otherwise idle between chunk decodes.

``GoalWorker`` spawns one child process that loads its own copy of the policy from the checkpoint and
serves ``_generate_goal_latent`` over a pair of queues. Payloads are small CPU tensors (the obs latent and
the goal latent are ~100 KB each; the text embedding ~4 MB), so no CUDA IPC is involved. The policy sends
the current frame every ``goal_refresh_rows`` and installs the reply the tick it lands; only the first goal
of an episode is waited for.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import time
import traceback
from copy import copy
from typing import TYPE_CHECKING

import torch
import torch.multiprocessing as mp
from torch import Tensor

if TYPE_CHECKING:
    from .configuration_glancewam import GlanceWAMConfig

logger = logging.getLogger(__name__)

# Loading the checkpoint plus the FlexAttention warmup takes ~1 min on a 4090; leave headroom.
_READY_TIMEOUT_S = 600.0
_COLLECT_TIMEOUT_S = 120.0


def _goal_worker_main(config: GlanceWAMConfig, req_q: mp.Queue, resp_q: mp.Queue, gpu_free: mp.Event) -> None:
    """Worker entry point: load the policy, warm the sampler, then serve ``(id, obs_latent, text)`` requests.

    Between Euler steps the worker drains its GPU queue and blocks while ``gpu_free`` is clear, i.e. while the
    policy is decoding an action chunk, so the two processes share the GPU by turns rather than by time slice.
    """

    def yield_gpu() -> None:
        torch.cuda.synchronize()
        gpu_free.wait()

    try:
        from .modeling_glancewam import GlanceWAMPolicy

        cfg = copy(config)
        cfg.goal_async = False  # the worker must not spawn a worker
        cfg.goal_preview_dir = None  # previews are written by the policy on the collect tick
        cfg.fast_inference = False  # the worker only samples goals; no chunk decode to compile
        assert cfg.pretrained_path is not None and cfg.device is not None  # checked by GoalWorker
        device = torch.device(cfg.device)
        policy = GlanceWAMPolicy.from_pretrained(cfg.pretrained_path, config=cfg).to(device).eval()
        # First forward compiles the FlexAttention kernels: do it here, not on the first request.
        with torch.inference_mode():
            scale = policy.backbone.scale_spatial
            h, w = (d // scale for d in cfg.vae_input_size)
            obs = torch.zeros(1, 16, policy.obs_latent_frames, h, w, device=device)
            text_dim = int(policy.backbone.transformer.config.text_dim)
            text = torch.zeros(1, cfg.text_max_length, text_dim, device=device, dtype=torch.bfloat16)
            policy._generate_goal_latent(obs, text)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        resp_q.put(("ready", None))
        while True:
            msg = req_q.get()
            if msg is None:
                return
            rid, obs, text, decode = msg
            with torch.inference_mode():
                goal = policy._generate_goal_latent(
                    obs.to(device), text.to(device, torch.bfloat16), before_step=yield_gpu
                )
                # Preview: decode here, off the control thread; ship the frame as uint8 (~300 KB).
                image = policy._decode_goal_image(goal) if decode else None
            resp_q.put((rid, goal.cpu(), image.cpu() if image is not None else None))
    except KeyboardInterrupt:
        return  # Ctrl-C reaches the child too; the parent is already shutting down
    except BaseException:
        resp_q.put(("error", traceback.format_exc()))
        raise


class GoalWorker:
    """Client side of the goal worker process: at most one request in flight."""

    def __init__(self, config: GlanceWAMConfig):
        if config.pretrained_path is None:
            raise ValueError(
                "goal_async needs pretrained_path: the worker loads the policy from the checkpoint"
            )
        ctx = mp.get_context("spawn")
        self._requests: mp.Queue = ctx.Queue()
        self._responses: mp.Queue = ctx.Queue()
        # Set while the policy's GPU is idle; cleared around each chunk decode (see ``gpu_busy``).
        self._gpu_free: mp.Event = ctx.Event()
        self._gpu_free.set()
        self._proc = ctx.Process(
            target=_goal_worker_main,
            args=(config, self._requests, self._responses, self._gpu_free),
            daemon=True,
        )
        self._proc.start()
        self._ready = False
        self._next_id = 0
        self._pending: int | None = None
        logger.info(f"GlanceWAM: goal worker process started (pid {self._proc.pid}).")

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @contextlib.contextmanager
    def gpu_busy(self):
        """Hold the worker between Euler steps for the duration of the block (the chunk decode)."""
        self._gpu_free.clear()
        try:
            yield
        finally:
            self._gpu_free.set()

    def _get(self, timeout: float) -> tuple:
        deadline = time.monotonic() + timeout
        while True:
            try:
                msg = self._responses.get(timeout=1.0)
            except queue.Empty:
                if not self._proc.is_alive():
                    raise RuntimeError(
                        f"goal worker (pid {self._proc.pid}) exited with code {self._proc.exitcode}"
                    ) from None
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"goal worker (pid {self._proc.pid}) sent nothing for {timeout:.0f} s"
                    ) from None
                continue
            if msg[0] == "error":
                raise RuntimeError(f"goal worker failed:\n{msg[1]}")
            return msg

    def wait_ready(self) -> None:
        if self._ready:
            return
        msg = self._get(_READY_TIMEOUT_S)
        if msg[0] != "ready":
            raise RuntimeError(f"goal worker sent {msg[0]!r} before 'ready'")
        self._ready = True

    def submit(self, obs_latent: Tensor, text_embeds: Tensor, decode: bool = False) -> None:
        """Send one request; the previous one, if still in flight, is discarded on collect.

        Args:
            obs_latent: The observation latent to imagine from.
            text_embeds: The task embedding.
            decode: Also VAE-decode the goal in the worker and return the frame (goal previews).
        """
        self.wait_ready()
        self._pending = self._next_id
        self._next_id += 1
        self._requests.put((self._pending, obs_latent.detach().cpu(), text_embeds.detach().cpu(), decode))

    def collect(self, device: torch.device) -> tuple[Tensor, Tensor | None]:
        """Block until the pending request's goal arrives: ``(goal latent, decoded frame or None)``."""
        if self._pending is None:
            raise RuntimeError("collect() with no request pending")
        while True:
            rid, goal, image = self._get(_COLLECT_TIMEOUT_S)
            if rid == self._pending:
                self._pending = None
                # ``clone`` moves the tensors out of the queue's shared-memory storage.
                return goal.clone().to(device), (image.clone() if image is not None else None)

    def poll(self, device: torch.device) -> tuple[Tensor, Tensor | None] | None:
        """Non-blocking ``collect``: the pending result if it has arrived, else ``None``."""
        if self._pending is None:
            return None
        while True:
            try:
                msg = self._responses.get_nowait()
            except queue.Empty:
                return None
            if msg[0] == "error":
                raise RuntimeError(f"goal worker failed:\n{msg[1]}")
            rid, goal, image = msg
            if rid == self._pending:
                self._pending = None
                return goal.clone().to(device), (image.clone() if image is not None else None)

    def discard(self) -> None:
        """Forget the in-flight request; its response is dropped when the next collect sees it."""
        self._pending = None

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._requests.put(None)
                self._proc.join(timeout=5.0)
            finally:
                if self._proc.is_alive():
                    self._proc.terminate()

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()
