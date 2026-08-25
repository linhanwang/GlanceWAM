from collections import deque
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.LIBERO.eval_files.adaptive_ensemble import AdaptiveEnsembler
from glancewam.model.tools import read_mode_config

# LIBERO sim control frequency (observation/action rate, Hz). A model trained
# with a strided history at ``hist_fps`` (e.g. 5 Hz) must be fed frames at the
# same wall-clock spacing at eval, so the client samples every
# ``round(LIBERO_CONTROL_HZ / hist_fps)`` raw steps. The LIBERO *_no_noops
# datasets are 20 Hz, so hist_fps=5 -> stride 4.
LIBERO_CONTROL_HZ = 20


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble=True,
        action_ensemble_horizon: Optional[int] = 3,  # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10095,
    ) -> None:

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None

        # Multi-frame history. Read num_history_frames + hist_fps from the saved
        # framework config so the eval buffer matches what the checkpoint was
        # trained with (hist1 / no key -> single-frame, legacy behaviour). For a
        # strided history the buffer must retain enough raw 20 Hz steps to sample
        # num_history_frames frames at the trained spacing.
        self.num_history_frames, hist_fps = self._read_history_cfg(policy_ckpt_path)
        if hist_fps and hist_fps > 0:
            self.frame_stride = max(1, round(LIBERO_CONTROL_HZ / float(hist_fps)))
        else:
            self.frame_stride = 1
        buffer_depth = (self.num_history_frames - 1) * self.frame_stride + 1
        # Buffer holds the most recent resized [primary, wrist] camera pairs
        # (oldest=left). Sized to span the full strided window.
        self.image_history = deque(maxlen=buffer_depth)
        if self.num_history_frames > 1:
            print(
                f"*** history: num_history_frames={self.num_history_frames}, hist_fps={hist_fps}, "
                f"frame_stride={self.frame_stride}, buffer_depth={buffer_depth} ***"
            )
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path=policy_ckpt_path)

        # Goal-image cotrain (GlanceWAM): the server advertises H_g (goal
        # horizon, native rows) when the checkpoint is goal-conditioned. When present we run
        # the STATEFUL goal cadence — generate a goal, HOLD it for H_g, refresh, and feed the
        # shrinking goal-time diff (H_g -> 0 within a cycle) each chunk. Absent = plain policy.
        # LIBERO control is 20 Hz = the native dataset rate, so env steps count 1:1 as rows.
        server_meta = self.client.get_server_metadata()
        self.goal_horizon_rows = (
            float(server_meta["goal_horizon_rows"])
            if isinstance(server_meta, dict) and server_meta.get("goal_horizon_rows") is not None
            else None
        )
        self._cached_goal = None
        self._steps_since_refresh = 0
        if self.goal_horizon_rows is not None:
            print(f"*** goal cadence ON: goal_horizon_rows={self.goal_horizon_rows} ***")

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # Goal cadence is per-episode: forget the held goal and restart the H_g refresh cycle.
        self._cached_goal = None
        self._steps_since_refresh = 0

    def step(self, example: dict, step: int = 0, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform one step of inference
        :param image: Input image in the format (H, W, 3), type uint8
        :param task_description: Task description text
        :return: (raw action, processed action)
        """

        task_description = example.get("lang", None)
        images = example["image"]  # current timestep: [primary, wrist] camera pair

        if example is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        cur_pair = [self._resize_image(image) for image in images]
        if self.num_history_frames > 1:
            # Append the current camera pair, then assemble the strided window
            # [t-(N-1)*s, ..., t-s, t] in chronological (oldest->newest) order.
            # Before the buffer fills, max(0, ...) clamps to the oldest pair, i.e.
            # left-pad by repetition (matches the training-time first-frame-repeat
            # padding in tools/precompute_vae_for_lerobot.py). Each entry stays a
            # [primary, wrist] pair so the framework stitches per timestep.
            self.image_history.append(cur_pair)
            buf = list(self.image_history)
            idxs = [max(0, len(buf) - 1 - k * self.frame_stride) for k in range(self.num_history_frames)]
            example["image"] = [buf[i] for i in reversed(idxs)]  # nested: N x [primary, wrist]
        else:
            example["image"] = cur_pair  # flat single timestep (unchanged)
        action_chunk_size = self.action_chunk_size
        if step % action_chunk_size == 0:
            # Stateful goal cadence: reuse the held goal (send it back) during the hold phase,
            # or let the server GENERATE a fresh one on refresh; feed the shrinking goal-time
            # diff. Only touched on query steps — the counter advances one chunk per query.
            if self.goal_horizon_rows is not None:
                H = self.goal_horizon_rows
                refresh = self._cached_goal is None or self._steps_since_refresh >= H
                if refresh:
                    self._steps_since_refresh = 0
                else:
                    example["goal_latent"] = self._cached_goal
                example["goal_horizon_rows"] = float(max(1.0, H - self._steps_since_refresh))

            vla_input = {
                "examples": [example],
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
            except KeyError:
                print(f"Response data: {response}")
                raise KeyError(f"Key 'normalized_actions' not found in response data: {response['data'].keys()}")

            if self.goal_horizon_rows is not None:
                gl = response["data"].get("goal_latent")
                if gl is not None:  # cache the goal used this call (fresh on refresh, unchanged on hold)
                    self._cached_goal = np.asarray(gl)[0]
                self._steps_since_refresh += action_chunk_size

            normalized_actions = normalized_actions[0]
            self.raw_actions = self.unnormalize_actions(
                normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
            )

        raw_actions = self.raw_actions[step % action_chunk_size][None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        return {"raw_action": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path):
        model_config, _ = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        # import ipdb; ipdb.set_trace()
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1

    @staticmethod
    def _read_history_cfg(policy_ckpt_path):
        """Pull (num_history_frames, hist_fps) from the saved framework config so
        the client buffer matches what the checkpoint was trained with.
        num_history_frames comes from framework.world_model; hist_fps from
        datasets.vla_data (None = contiguous native-rate history). Defaults to
        (1, None) for single-frame checkpoints with no such keys.
        """
        model_config, _ = read_mode_config(Path(policy_ckpt_path))
        num_history_frames = 1
        hist_fps = None
        try:
            wm_cfg = model_config["framework"]["world_model"]
            if isinstance(wm_cfg, dict):
                num_history_frames = int(wm_cfg.get("num_history_frames", 1))
        except (KeyError, TypeError):
            pass
        try:
            hist_fps = model_config["datasets"]["vla_data"].get("hist_fps", None)
        except (KeyError, TypeError):
            pass
        return num_history_frames, hist_fps

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
