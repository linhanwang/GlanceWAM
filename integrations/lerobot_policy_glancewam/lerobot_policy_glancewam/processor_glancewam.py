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
"""Pre/post-processors for GlanceWAM.

Pre: rename -> batch -> normalize (state/action QUANTILES by default) -> device -> UMT5 text table.
Post: unnormalize -> gripper binarization -> cpu.

The text table replaces GlanceWAM's offline UMT5 zarr cache and GPU-resident table: the step holds one
``[text_max_length, 4096]`` fp16 row per known task string (prefilled from the dataset's task list, saved
with the checkpoint as processor state) and encodes unseen prompts lazily with UMT5-XXL, exactly as the
Wan pipeline does (pad to ``max_length``, zero the padded positions, no attention mask downstream).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
from lerobot.configs import NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import (
    PolicyAction,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    make_default_policy_processor_steps,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION, OBS_PREFIX
from torch import Tensor

from .configuration_glancewam import GlanceWAMConfig

logger = logging.getLogger(__name__)

LANGUAGE_EMBEDDING_KEY = "observation.language.embedding"


@ProcessorStepRegistry.register(name="glancewam_umt5_text_table")
@dataclass
class UMT5TextTableProcessorStep(ProcessorStep):
    """Map ``complementary_data[task_key]`` strings to cached UMT5-XXL embeddings.

    Args:
        model_id: Diffusers repo holding the ``tokenizer`` and ``text_encoder`` subfolders.
        max_length: Prompt padding length.
        prompts: Task strings known at construction (encoded on first use, in one batch).
        task_key: Complementary-data key carrying the task strings.
        output_key: Observation key receiving the ``[B, max_length, 4096]`` embeddings.
        keep_encoder: Keep UMT5 resident after encoding instead of freeing it (11 GB).
    """

    model_id: str = "Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers"
    max_length: int = 512
    prompts: list[str] = field(default_factory=list)
    task_key: str = "task"
    output_key: str = LANGUAGE_EMBEDDING_KEY
    keep_encoder: bool = False

    def __post_init__(self):
        self._table: dict[str, Tensor] = {}  # prompt -> [max_length, 4096] fp16 on CPU
        self._device_cache: dict[str, Tensor] = {}
        self._device: torch.device | None = None
        self._encoder = None
        self._tokenizer = None

    # ---------------------------------------------------------------- encoding
    def _load_encoder(self, device: torch.device) -> None:
        from transformers import T5TokenizerFast, UMT5EncoderModel

        logger.info(f"Loading UMT5 text encoder from {self.model_id} on {device}.")
        self._tokenizer = T5TokenizerFast.from_pretrained(self.model_id, subfolder="tokenizer")
        self._encoder = (
            UMT5EncoderModel.from_pretrained(
                self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
            )
            .to(device)
            .eval()
            .requires_grad_(False)
        )

    @torch.no_grad()
    def encode(self, prompts: list[str], device: torch.device | None = None, batch_size: int = 16) -> None:
        """Encode `prompts` into the table (skipping ones already present)."""
        missing = [p for p in dict.fromkeys(prompts) if p not in self._table]
        if not missing:
            return
        lazy_miss = any(p not in self.prompts for p in missing)
        device = device or self._device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self._encoder is None:
            self._load_encoder(device)
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            inputs = self._tokenizer(
                chunk,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(device)
            hidden = self._encoder(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask
            ).last_hidden_state
            hidden = hidden * inputs.attention_mask.unsqueeze(-1)
            for prompt, row in zip(chunk, hidden, strict=True):
                self._table[prompt] = row.to(torch.float16).cpu()
                if prompt not in self.prompts:
                    self.prompts.append(prompt)
        logger.info(f"UMT5 text table: encoded {len(missing)} prompt(s), {len(self._table)} cached.")
        if lazy_miss and not self.keep_encoder:
            # No prompt list was known up front, so more misses are likely: keep UMT5 resident rather
            # than reloading 11 GB of weights for every new task string.
            logger.warning(
                "UMT5 text table: prompt list unknown at construction; keeping the encoder resident."
            )
            self.keep_encoder = True
        if not self.keep_encoder:
            self._encoder = None
            self._tokenizer = None
            if device.type == "cuda":
                torch.cuda.empty_cache()

    def _row(self, prompt: str, device: torch.device) -> Tensor:
        if device != self._device:
            self._device_cache.clear()
            self._device = device
        row = self._device_cache.get(prompt)
        if row is None:
            if prompt not in self._table:
                self.encode([prompt], device)
            row = self._table[prompt].to(device=device, dtype=torch.bfloat16)
            self._device_cache[prompt] = row
        return row

    # ---------------------------------------------------------------- step API
    def __call__(self, transition: EnvTransition) -> EnvTransition:
        transition = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        tasks = (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).get(self.task_key)
        if tasks is None:
            raise ValueError(f"UMT5TextTableProcessorStep: no '{self.task_key}' in complementary data")
        if isinstance(tasks, str):
            tasks = [tasks]
        device = next(
            (v.device for v in observation.values() if isinstance(v, Tensor)), None
        ) or torch.device("cpu")
        if self.prompts and not self._table:
            self.encode(list(self.prompts), device)
        observation[self.output_key] = torch.stack([self._row(str(t), device) for t in tasks])
        transition[TransitionKey.OBSERVATION] = observation
        return transition

    def get_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_length": self.max_length,
            "prompts": list(self.prompts),
            "task_key": self.task_key,
            "output_key": self.output_key,
            "keep_encoder": self.keep_encoder,
        }

    def state_dict(self) -> dict[str, Tensor]:
        return {f"row_{i}": self._table[p] for i, p in enumerate(self.prompts) if p in self._table}

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        for i, prompt in enumerate(self.prompts):
            row = state.get(f"row_{i}")
            if row is not None:
                self._table[prompt] = row.to(torch.float16)
        self._device_cache.clear()

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {k: dict(v) for k, v in features.items()}
        obs = features.setdefault(PipelineFeatureType.OBSERVATION, {})
        from lerobot.configs import FeatureType

        obs[self.output_key] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.max_length, 4096))
        return features


@ProcessorStepRegistry.register(name="glancewam_gripper")
@dataclass
class GlanceWAMGripperProcessorStep(PolicyActionProcessorStep):
    """Binarize the gripper action dimension after unnormalization.

    ``mode="libero"``: the dataset stores the gripper raw in ``{0, 1}`` (1 = open); the value is
    thresholded at 0.5 and mapped to LIBERO's ``-1`` (open) / ``+1`` (close). ``mode="none"``: identity.
    """

    mode: str = "libero"
    index: int = -1
    threshold: float = 0.5

    def action(self, action: PolicyAction) -> PolicyAction:
        if self.mode == "none":
            return action
        action = action.clone()
        g = action[..., self.index]
        action[..., self.index] = torch.where(g > self.threshold, -1.0, 1.0).to(action.dtype)
        return action

    def get_config(self) -> dict[str, Any]:
        return {"mode": self.mode, "index": self.index, "threshold": self.threshold}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


# Both settings guarded below assume the dim already lives on a unit scale: LIBERO stores its gripper
# raw in {0, 1}, GR00T's masked dims in {-1, 1}. Stats are float32, so leave a little rounding room.
_UNIT_SCALE_TOL = 1e-3


def _check_unit_scale_dim(dim: int, amin: Tensor, amax: Tensor, setting: str, effect: str) -> None:
    """Raise unless action dim ``dim`` is already within [-1, 1] in the dataset's stats.

    Both callers make a dim bypass normalization, which is only correct for a raw {0, 1} / +-1 channel.
    Pointed at a continuous one (e.g. a gripper reported in 0..57) the mistake is silent and
    expensive: the raw channel reaches the flow-matching loss un-normalized, where its target variance
    swamps every other dim, and inference commands it on the wrong scale.
    """
    ndim = amin.numel()
    if not -ndim <= dim < ndim:
        raise ValueError(f"{setting} entry {dim} is out of range for a {ndim}-dim action.")
    lo, hi = float(amin[dim]), float(amax[dim])
    if lo < -1.0 - _UNIT_SCALE_TOL or hi > 1.0 + _UNIT_SCALE_TOL:
        raise ValueError(
            f"{setting} {effect} action dim {dim}, but the dataset's range for that dim is "
            f"[{lo:.4g}, {hi:.4g}], not the raw {{0, 1}} / +-1 channel this assumes. That is the LIBERO "
            f"gripper convention meeting a robot whose gripper is continuous. For such a dataset pass "
            f"--policy.identity_action_dims='[]' --policy.gripper_mode=none."
        )


_MODE_STAT_KEYS = {
    NormalizationMode.MIN_MAX: ("min", "max"),
    NormalizationMode.QUANTILES: ("q01", "q99"),
    NormalizationMode.QUANTILE10: ("q10", "q90"),
}


def _with_identity_action_dims(
    dataset_stats: dict[str, dict[str, Tensor]] | None, config: GlanceWAMConfig
) -> dict[str, dict[str, Tensor]] | None:
    """Copy of `dataset_stats` whose action range stats are (-1, 1) on ``config.identity_action_dims``.

    The range stats are those of the action normalization mode (min/max, q01/q99 or q10/q90), so the
    normalizer maps the dim to itself. Validates every action dim the config exempts from normalization
    against the dataset's raw min/max first; see ``_check_unit_scale_dim``.
    """
    if not dataset_stats or ACTION not in dataset_stats:
        return dataset_stats
    action_stats = dict(dataset_stats[ACTION])
    if "min" not in action_stats or "max" not in action_stats:
        return dataset_stats
    amin = torch.as_tensor(action_stats["min"]).clone().float()
    amax = torch.as_tensor(action_stats["max"]).clone().float()

    for dim in config.identity_action_dims:
        _check_unit_scale_dim(dim, amin, amax, "identity_action_dims", "leaves un-normalized")
    if getattr(config, "gripper_mode", "none") == "libero":
        _check_unit_scale_dim(
            config.gripper_index, amin, amax, "gripper_mode='libero'", "thresholds at 0.5 and binarizes"
        )

    if not config.identity_action_dims:
        return dataset_stats
    mode = config.normalization_mapping.get("ACTION", NormalizationMode.MIN_MAX)
    keys = _MODE_STAT_KEYS.get(mode)
    if keys is None:
        return dataset_stats
    lo_key, hi_key = keys
    if lo_key not in action_stats or hi_key not in action_stats:
        raise ValueError(
            f"identity_action_dims needs {lo_key}/{hi_key} action stats for {mode.value} normalization, "
            f"but the dataset stats only carry {sorted(action_stats)}."
        )
    lo = torch.as_tensor(action_stats[lo_key]).clone().float()
    hi = torch.as_tensor(action_stats[hi_key]).clone().float()
    for dim in config.identity_action_dims:
        lo[dim] = -1.0
        hi[dim] = 1.0
    action_stats[lo_key], action_stats[hi_key] = lo, hi
    return {**dataset_stats, ACTION: action_stats}


def _prompts_from_dataset_meta(dataset_meta: Any | None) -> list[str]:
    tasks = getattr(dataset_meta, "tasks", None)
    if tasks is None:
        return []
    index = getattr(tasks, "index", None)
    if index is not None:
        return [str(t) for t in index]
    return [str(t) for t in tasks]


def make_glancewam_pre_post_processors(
    config: GlanceWAMConfig,
    dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    dataset_meta: Any | None = None,
    prompts: list[str] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the GlanceWAM pre/post-processor pipelines.

    Args:
        config: The policy configuration.
        dataset_stats: Normalization statistics (state/action, keys per ``config.normalization_mapping``).
        dataset_meta: Dataset metadata; its task list prefills the UMT5 text table.
        prompts: Extra task strings to prefill (e.g. from a converted checkpoint).
    """
    dataset_stats = _with_identity_action_dims(dataset_stats, config)
    steps = make_default_policy_processor_steps(config, dataset_stats)
    # `make_policy` stashes the dataset metadata on the config; the trainer's processor factory
    # call does not forward it, so fall back to that copy for the task list.
    dataset_meta = (
        dataset_meta if dataset_meta is not None else getattr(config, "_runtime_dataset_meta", None)
    )
    known = list(dict.fromkeys([*(prompts or []), *_prompts_from_dataset_meta(dataset_meta)]))
    text_step = UMT5TextTableProcessorStep(
        model_id=config.base_model_path, max_length=config.text_max_length, prompts=known
    )
    input_steps: list[ProcessorStep] = [
        steps.rename_observations,
        steps.add_batch_dim,
        steps.normalize,
        steps.to_device,
        text_step,
    ]
    output_steps: list[ProcessorStep] = [
        steps.unnormalize,
        GlanceWAMGripperProcessorStep(mode=config.gripper_mode, index=config.gripper_index),
        steps.to_cpu,
    ]
    return make_policy_processor_pipelines(input_steps=input_steps, output_steps=output_steps)


__all__ = [
    "LANGUAGE_EMBEDDING_KEY",
    "OBS_PREFIX",
    "GlanceWAMGripperProcessorStep",
    "UMT5TextTableProcessorStep",
    "make_glancewam_pre_post_processors",
]
