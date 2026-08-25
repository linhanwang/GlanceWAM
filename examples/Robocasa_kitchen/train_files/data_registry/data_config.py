"""RoboCasa kitchen (cosmos-policy) — data config, embodiment tags, and mixtures.

24 per-task lerobot v3 datasets converted from NVIDIA RoboCasa-Cosmos-Policy
``all_episodes`` (PandaOmron / single Franka arm). See
``examples/Robocasa_kitchen/train_files/convert_cosmos_robocasa_to_lerobot.py``.

Conventions (cosmos-native, kept verbatim so a future eval port can feed obs directly):
* state [9] = ``[gripper_qpos(2), eef_pos(3), eef_quat(4, xyzw)]`` (modality groups below)
* action [7] = ``[eef_delta_pos(3), eef_delta_rot(3), gripper(1)]`` (gripper RAW {-1,+1})
* 3 cameras: primary (agentview_left) / secondary (agentview_right) / wrist (eye_in_hand).
  The single-stream cotrain world model packs them via ``camera_concat: primary_inset``
  (primary 224^2 + secondary & wrist 112^2 insets -> one 336x224 frame).

Mirrors the LIBERO cosmos cotrain recipe (same CotrainBaseline framework): the
``success`` column on each dataset lets the gr00t_lerobot reader auto-drop failed demos
for the policy loss, while the full set stays available for the world-model loss.
"""

from glancewam.dataloader.gr00t_lerobot.datasets import ModalityConfig
from glancewam.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from glancewam.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from glancewam.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from glancewam.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoJpeg,
    VideoToNumpy,
    VideoToTensor,
)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class RobocasaKitchenPandaDataConfig:
    # 3 cameras; _pack_sample stitches them per-timestep when camera_concat=primary_inset.
    video_keys = [
        "video.primary_image",
        "video.secondary_image",
        "video.wrist_image",
    ]
    # Modality groups match meta/modality.json written by the converter.
    state_keys = ["state.gripper_qpos", "state.eef_pos", "state.eef_quat"]
    action_keys = ["action.eef_delta_pos", "action.eef_delta_rot", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self, image_aug: bool = False):
        # APPEARANCE-ONLY image augmentation (enable via `datasets.<g>.image_aug: true`):
        # color jitter (b.3 c.4 s.5 h.05, cosmos-policy's jitter ranges) + JPEG q95. Deliberately
        # NO geometric ops (crop/resize/rotation), unlike the LIBERO cosmos recipe: the kitchen
        # gap is held-out-scene appearance generalization, and geometry-free aug keeps the
        # train/eval contract identical — no eval-side center-crop counterpart needed
        # . Shared params across the whole
        # obs+future window (one transform call); applied per-camera BEFORE the primary_inset
        # stitch in _pack_sample.
        video_transforms = []
        if image_aug:
            video_transforms = [
                VideoToTensor(apply_to=self.video_keys),
                VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.05),
                VideoToNumpy(apply_to=self.video_keys),
                VideoJpeg(apply_to=self.video_keys, quality=95),
            ]
        return ComposedModalityTransform(
            transforms=[
                *video_transforms,
                StateActionToTensor(apply_to=self.action_keys),
                # min_max-normalize the 6 OSC pose-delta dims; gripper kept RAW (no entry).
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.eef_delta_pos": "min_max",
                        "action.eef_delta_rot": "min_max",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "panda_omron": RobocasaKitchenPandaDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "panda_omron": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
_KITCHEN_TASKS = [
    "CloseDoubleDoor",
    "CloseDrawer",
    "CloseSingleDoor",
    "CoffeePressButton",
    "CoffeeServeMug",
    "CoffeeSetupMug",
    "OpenDoubleDoor",
    "OpenDrawer",
    "OpenSingleDoor",
    "PnPCabToCounter",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "PnPMicrowaveToCounter",
    "PnPSinkToCounter",
    "PnPStoveToCounter",
    "TurnOffMicrowave",
    "TurnOffSinkFaucet",
    "TurnOffStove",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "TurnOnStove",
    "TurnSinkSpout",
]


def _mix(tasks):
    return [(f"robocasa_{t}_cosmos_lerobot", 1.0, "panda_omron") for t in tasks]


DATASET_NAMED_MIXTURES = {
    # All 24 kitchen tasks (success-filter exposes 1199 demos for the policy loss).
    "robocasa_kitchen_all": _mix(_KITCHEN_TASKS),
    # 8 PnP tasks (counter<->cab/sink/stove/microwave) for faster iteration.
    "robocasa_kitchen_pnp": _mix([t for t in _KITCHEN_TASKS if t.startswith("PnP")]),
    # Single-task smoke mixture.
    "robocasa_kitchen_turnoffmicrowave": _mix(["TurnOffMicrowave"]),
}
