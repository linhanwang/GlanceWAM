"""Policy-side bridge for RoboCasa kitchen (cosmos-policy, PandaOmron single-arm) evaluation.

Talks to the shared glancewam websocket policy server. Modeled on
the shared policy-client contract, adapted to the
single-arm 7-d OSC action / 9-d proprio layout produced by the cosmos-policy RoboCasa
conversion (see ``examples/Robocasa_kitchen/train_files/convert_cosmos_robocasa_to_lerobot.py``).

Train/eval consistency (embodied policies degrade SILENTLY on any mismatch — verify against the
training config for the checkpoint under eval):
  * images  : the 3 cameras are fed in the TRAINING order ``[primary, secondary, wrist]``
              (``data_config.video_keys``), NOT cosmos's internal ``[wrist, primary, secondary]``.
              Frames arrive already flipud-corrected from ``simulation_env.prepare_observation``;
              here we resize to the model res and (optionally) JPEG-q95 round-trip to match the
              training-time ``VideoJpeg`` augmentation.
  * action  : the server returns NORMALIZED actions ``(B, chunk, 7)``; we un-normalize client-side
              with the checkpoint's ``dataset_statistics.json`` (min_max on the 6 OSC pose dims,
              gripper kept RAW via the stats ``mask``). The 7->12 mobile-base pad happens in
              ``simulation_env`` right before ``env.step``.
  * state   : the released GR00T kitchen recipe is stateless (``include_state`` off), so proprio is
              only sent when ``include_state=True`` — raw 9-d, no sin/cos, matching ``data_config``.

The norm-stats loader reads ``dataset_statistics.json`` directly (plain json) so this client needs
NO torch / heavy-glancewam imports and runs in the slim ``robocasa-cosmos-policy`` sim venv; glancewam
is available only for the lightweight websocket client via ``PYTHONPATH``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def _load_action_norm_stats(policy_ckpt_path: str, unnorm_key: Optional[str]) -> Dict:
    """Read ``<run_dir>/dataset_statistics.json`` (run_dir = ``<ckpt>.parents[1]``) and return the
    selected dataset's ``action`` stats. Pure-json; mirrors ``share_tools.read_mode_config`` without
    importing torch/glancewam so the slim sim venv stays torch-free."""
    ckpt = Path(policy_ckpt_path)
    run_dir = ckpt.parents[1] if ckpt.suffix in {".pt", ".safetensors"} else ckpt
    stats_path = run_dir / "dataset_statistics.json"
    assert stats_path.exists(), f"Missing dataset_statistics.json at {stats_path}"
    with open(stats_path) as f:
        norm_stats = json.load(f)
    if unnorm_key is None:
        assert len(norm_stats) == 1, (
            "Checkpoint was trained on >1 dataset; pass --args.unnorm-key from: " f"{list(norm_stats.keys())}"
        )
        unnorm_key = next(iter(norm_stats))
    assert (
        unnorm_key in norm_stats
    ), f"unnorm_key '{unnorm_key}' not in dataset statistics; choose from: {list(norm_stats.keys())}"
    return norm_stats[unnorm_key]["action"]


class PolicyWarper:
    """Single-arm PandaOmron kitchen policy wrapper that talks to the websocket server."""

    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 6699,
        image_size=(224, 224),
        n_action_steps: int = 16,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        use_jpeg: bool = True,
        include_state: bool = False,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.image_size = tuple(image_size)
        self.n_action_steps = n_action_steps
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.use_jpeg = use_jpeg
        self.include_state = include_state
        self.task_description = None
        self.action_norm_stats = _load_action_norm_stats(policy_ckpt_path, unnorm_key)
        server_meta = self.client.get_server_metadata()
        # Goal-image cotrain (GlanceWAM): the server advertises H_g (goal
        # horizon, native rows) when the checkpoint is goal-conditioned. When present we run
        # the STATEFUL goal cadence — generate a goal, HOLD it for H_g, refresh, and feed the
        # shrinking goal-time diff (H_g -> 0 within a cycle) each chunk. Absent = plain policy.
        self.goal_horizon_rows = (
            float(server_meta["goal_horizon_rows"])
            if isinstance(server_meta, dict) and server_meta.get("goal_horizon_rows") is not None
            else None
        )
        self._cached_goal = None
        # ---- E2 goal-information ablations (client side) ----
        # Defaults reproduce the trained cadence exactly, so an unset environment is bit-exact.
        #   GOAL_REFRESH_ROWS   how often a new goal is REQUESTED (native rows). Default H_g, i.e.
        #                       hold each goal for its full horizon. Decoupling this from H_g is
        #                       the refresh-cadence row.
        #   GOAL_STALE_ROWS     emulated ASYNC proposer latency L: a goal requested at row r is
        #                       adopted at r+L, and the policy keeps acting on the previous goal
        #                       meanwhile. Quantized up to the next chunk boundary, since the
        #                       policy is only queried once per chunk.
        #   GOAL_PREV_EPISODE   1 = condition on the PREVIOUS episode's goal at the same phase of
        #                       the cycle (goals are still generated on schedule, so the sequence
        #                       is phase- and task-matched but belongs to another episode/scene).
        #   GOAL_HORIZON_FIXED  feed a constant goal-time diff instead of the shrinking one
        #                       (tests whether the head uses the how-far-ahead signal).
        self._refresh_rows = float(os.environ.get("GOAL_REFRESH_ROWS", "") or (self.goal_horizon_rows or 0.0))
        self._stale_rows = float(os.environ.get("GOAL_STALE_ROWS", "0") or 0.0)
        self._prev_episode_goal = os.environ.get("GOAL_PREV_EPISODE", "0") not in ("0", "", "false", "False")
        _fixed = os.environ.get("GOAL_HORIZON_FIXED", "")
        self._horizon_fixed = float(_fixed) if _fixed else None
        self._row = 0  # native rows elapsed in this episode
        self._gen_row = None  # row whose observation produced the goal in hand
        self._last_request_row = None  # row at which the newest generation was REQUESTED
        self._pending = None  # (goal, adopt_row) — sampled, not yet adopted (async latency)
        self._goals_this_ep: list = []  # goals generated this episode, in refresh order
        self._goals_prev_ep: list = []  # ... and the previous episode's, for GOAL_PREV_EPISODE
        print(
            f"*** kitchen PolicyWarper: unnorm_key={unnorm_key}, include_state={include_state}, "
            f"n_action_steps={n_action_steps}, goal_horizon_rows={self.goal_horizon_rows}, "
            f"refresh_rows={self._refresh_rows}, stale_rows={self._stale_rows}, "
            f"prev_episode_goal={self._prev_episode_goal}, horizon_fixed={self._horizon_fixed}, "
            f"server_meta={server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        # Goal cadence is per-episode: forget the held goal and restart the H_g refresh cycle.
        self._cached_goal = None
        self._row = 0
        self._gen_row = None
        self._last_request_row = None
        self._pending = None
        # Carry this episode's goal sequence over as the "other episode" source for the next one.
        self._goals_prev_ep = self._goals_this_ep
        self._goals_this_ep = []

    def get_action(self, observation: Dict, task_description: str, seed: Optional[int] = None) -> np.ndarray:
        """observation: ``{primary_image, secondary_image, wrist_image, proprio}`` (flipud already
        applied by the caller). Returns an un-normalized action chunk ``(n_action_steps, 7)``.

        seed: optional per-call sampling seed forwarded to the server. Used by the branch
        harness for COMMON RANDOM NUMBERS — every candidate's continuation replays the same
        noise sequence, so outcome differences are attributable to the branch chunk rather
        than to hundreds of unseeded continuation draws. None = historical behaviour."""
        if task_description != self.task_description:
            self.reset(task_description)

        # TRAINING camera order is [primary, secondary, wrist] (data_config.video_keys).
        images = [
            self._prep_image(observation["primary_image"]),
            self._prep_image(observation["secondary_image"]),
            self._prep_image(observation["wrist_image"]),
        ]
        example = {"image": images, "lang": task_description}
        if self.include_state:
            # raw 9-d proprio [gripper_qpos(2), eef_pos(3), eef_quat(4)]; no sin/cos (data_config).
            example["state"] = np.asarray(observation["proprio"], dtype=np.float32)

        # Stateful goal cadence: reuse the held goal (send it back) during the hold phase, or
        # let the server GENERATE a fresh one on refresh; feed the shrinking goal-time diff.
        requested_gen = False
        if self.goal_horizon_rows is not None:
            H = self.goal_horizon_rows
            # Adopt a goal whose emulated proposer latency has elapsed.
            if self._pending is not None and self._row >= self._pending[1]:
                self._cached_goal, self._gen_row = self._pending[0], self._pending[2]
                self._pending = None
            due = self._pending is None and (
                self._last_request_row is None or (self._row - self._last_request_row) >= self._refresh_rows
            )
            if self._cached_goal is None:
                # Nothing in hand (episode start): the server generates AND acts on it — there is
                # no previous goal to be stale relative to, so latency cannot apply here.
                requested_gen = True
                self._gen_row = self._row
            else:
                example["goal_latent"] = self._cached_goal
                if due:
                    requested_gen = True
                    if self._stale_rows <= 0:
                        # Synchronous proposer: drop the held goal and act on the fresh sample.
                        example.pop("goal_latent")
                        self._gen_row = self._row
                    else:
                        # Async proposer: sample now, keep acting on the goal already in hand.
                        example["force_gen"] = True
            if requested_gen:
                self._last_request_row = self._row
            elapsed = self._row - (self._gen_row if self._gen_row is not None else self._row)
            example["goal_horizon_rows"] = (
                float(self._horizon_fixed) if self._horizon_fixed is not None else float(max(1.0, H - elapsed))
            )

        if seed is not None:
            example["seed"] = int(seed)

        # GOAL_TRACE=1 prints the realized cadence per chunk — the row, the goal-time diff the
        # head is told, whether a goal was requested, and whether it was requested asynchronously
        # (acting on the held goal). This is the ground truth for "what cadence did this arm
        # ACTUALLY run", as opposed to what the env vars were meant to produce.
        if os.environ.get("GOAL_TRACE", "0") not in ("0", "", "false", "False"):
            print(
                f"[goal-trace] row={self._row:4d} h={example.get('goal_horizon_rows')} "
                f"gen={requested_gen} async={'force_gen' in example} "
                f"held={'goal_latent' in example}",
                flush=True,
            )

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
            "unnorm_key": self.unnorm_key,
        }
        response = self.client.predict_action(vla_input)
        data = response["data"]
        normalized = np.asarray(data["normalized_actions"])  # (B, chunk, 7)
        if self.goal_horizon_rows is not None:
            if requested_gen:
                # "goal_latent_new" is present only on the async path (the row acted on the goal
                # it already held); otherwise the goal it acted on IS the fresh sample.
                fresh = data.get("goal_latent_new", data.get("goal_latent"))
                if fresh is not None:
                    fresh = np.asarray(fresh)[0]
                    k = len(self._goals_this_ep)
                    self._goals_this_ep.append(fresh)
                    if self._prev_episode_goal and self._goals_prev_ep:
                        # Other-episode arm: act on the previous episode's goal from the SAME
                        # point in the cycle (clamped if that episode was shorter), while still
                        # banking this episode's own goal for the next one.
                        fresh = self._goals_prev_ep[min(k, len(self._goals_prev_ep) - 1)]
                    if self._stale_rows > 0 and "force_gen" in example:
                        self._pending = (fresh, self._row + self._stale_rows, self._row)
                    else:
                        self._cached_goal = fresh
            self._row += self.n_action_steps
        raw = self._unnormalize(normalized)
        return raw[0, : self.n_action_steps, :]  # (n_action_steps, 7)

    # ------------------------------------------------------------------ helpers
    def _prep_image(self, image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image)
        if image.shape[:2] != self.image_size:
            image = cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)
        if self.use_jpeg:  # match training-time VideoJpeg q95 (encode/decode round-trip)
            ok, enc = cv.imencode(".jpg", image, [int(cv.IMWRITE_JPEG_QUALITY), 95])
            if ok:
                image = cv.imdecode(enc, cv.IMREAD_COLOR)
        return image

    def _unnormalize(self, normalized: np.ndarray) -> np.ndarray:
        s = self.action_norm_stats
        high, low = np.array(s["max"]), np.array(s["min"])
        mask = np.array(s.get("mask", np.ones_like(low, dtype=bool)))
        normalized = np.clip(normalized, -1, 1)
        return np.where(mask, (normalized + 1) / 2 * (high - low) + low, normalized)
