"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from glancewam.dataloader.gr00t_lerobot.datasets import ModalityConfig
from glancewam.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from glancewam.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from glancewam.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from glancewam.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoJpeg,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, image_aug: bool = False, image_size: tuple[int, int] = (224, 224)):
        # APPEARANCE-ONLY image augmentation (enable via `datasets.<g>.image_aug: true`),
        # copied verbatim from the RoboCasa kitchen recipe
        # (examples/Robocasa_kitchen/train_files/data_registry/data_config.py): color jitter
        # (b.3 c.4 s.5 h.05) + JPEG q95, and deliberately NO geometric ops.
        #
        # This is NOT the reverted 2026-06-30 "cosmos_aug" LIBERO recipe, which also carried a
        # 90%-area crop + resize + rotation. That crop is a DETERMINISTIC scale/FOV shift applied
        # to every training frame, so it needed an eval-side center-crop mirror (the
        # LIBERO_IMAGE_AUG plumbing, reverted in 423eea6). Geometry-free aug keeps the train/eval
        # contract identical, so no eval-side counterpart is needed here.
        #
        # ColorJitter samples ONE param set per transform call and apply() passes the whole
        # concatenated window at once, so the obs frame and the video-loss future/goal frames get
        # IDENTICAL jitter — the video branch never sees a color shift it would have to predict.
        # It is also a no-op in eval mode (get_transform returns None), so the held-out training
        # monitor stays unjittered. Applied per-camera BEFORE the side_by_side stitch.
        #
        # The VideoResize is load-bearing, not cosmetic: LIBERO videos are 256^2 on disk and the
        # config asks for 224^2, so with no VideoResize in this chain _ensure_video_resize
        # PREPENDS one (lerobot_datasets.py) and our VideoToTensor then trips its native-resolution
        # assert (224 seen vs 256 in metadata). Carrying the resize here suppresses that prepend.
        # It is the same plain bilinear resize the no-aug path gets — no crop, no scale change.
        video_transforms = []
        if image_aug:
            h, w = int(image_size[0]), int(image_size[1])
            video_transforms = [
                VideoToTensor(apply_to=self.video_keys),
                VideoResize(apply_to=self.video_keys, height=h, width=w, interpolation="linear"),
                VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05),
                VideoToNumpy(apply_to=self.video_keys),
                VideoJpeg(apply_to=self.video_keys, quality=95),
            ]
        return ComposedModalityTransform(
            transforms=[
                *video_transforms,
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "min_max",
                        "action.y": "min_max",
                        "action.z": "min_max",
                        "action.roll": "min_max",
                        "action.pitch": "min_max",
                        "action.yaw": "min_max",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
_LIBERO_SUITES = ["object", "goal", "spatial", "10"]


def _demo_mix():
    return [(f"libero_{s}_no_noops_1.0.0_lerobot", 1.0, "libero_franka") for s in _LIBERO_SUITES]


DATASET_NAMED_MIXTURES = {
    # All four LIBERO suites (object / goal / spatial / 10).
    "libero_all": _demo_mix(),
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}
