# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""

import copy
import gc
import hashlib
import json
import os
import pickle
import random
import time
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from glancewam.dataloader.camera_utils import stitch_primary_with_insets, stitch_views_side_by_side
from glancewam.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from glancewam.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from glancewam.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from glancewam.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

# LeRobot v2.0 dataset file names
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_STATS_FORMAT_VERSION = 2
EPSILON = 5e-4

#  LeRobot v3.0 dataset file names
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"


def calculate_dataset_statistics(parquet_paths: list[Path]) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files."""
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(parquet_path)
        parquet_data = parquet_data
        all_low_dim_data_list.append(parquet_data)

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        try:
            np_data = np.vstack([np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]])
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _normalize_action_mode(mode: str) -> str:
    """Normalize action mode names to {abs, delta, rel}."""
    # @gaoning plz move this, we want dataloader to be independent of the action mode logic, we can move this to transform or a separate utils tool to handle lerobot dataset
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _normalize_action_mode_apply_keys(
    action_mode_apply_keys: Sequence[str] | None,
    fallback_keys: Sequence[str] | None = None,
) -> list[str]:
    source_keys = action_mode_apply_keys if action_mode_apply_keys else (fallback_keys or [])
    normalized = []
    for key in source_keys:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.append(key)
    return normalized


def _normalize_action_mode_state_map(action_mode_state_map: dict[str, str] | None) -> dict[str, str]:
    normalized = {}
    for action_key, state_key in (action_mode_state_map or {}).items():
        action_key = str(action_key)
        state_key = str(state_key)
        if not action_key.startswith("action."):
            action_key = f"action.{action_key}"
        if not state_key.startswith("state."):
            state_key = f"state.{state_key}"
        normalized[action_key] = state_key
    return normalized


def _build_stats_cache_config(
    action_mode: str,
) -> dict:
    return {
        "mode": action_mode,
    }


def _invalidate_legacy_stats_cache(stats_path: Path, reason: str) -> None:
    if not stats_path.exists():
        return
    print(f"Removing stale dataset statistics cache at {stats_path}: {reason}")
    stats_path.unlink()


def _load_stats_cache(
    stats_path: Path,
    expected_config: dict,
    *,
    invalidate_legacy: bool,
) -> dict | None:
    if not stats_path.exists():
        return None

    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, f"failed to load JSON ({exc})")
        return None

    if not isinstance(payload, dict):
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "unexpected top-level format")
        return None

    format_version = payload.get("__format_version")
    cache_config = payload.get("__cache_config")
    statistics = payload.get("statistics")
    if format_version != LE_ROBOT_STATS_FORMAT_VERSION or cache_config is None or statistics is None:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "legacy statistics format detected")
        return None

    if cache_config != expected_config:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "statistics config mismatch, rebuilding cache")
        return None

    return statistics


def _save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": LE_ROBOT_STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    tmp_path = stats_path.with_suffix(".tmp")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=4)
    os.replace(tmp_path, stats_path)


def _compute_statistics_for_mode(
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    print(f"[RANK 0] Calculating dataset statistics for {dataset_name} (mode={action_mode})")

    base_stats = calculate_dataset_statistics(parquet_paths)

    if action_mode == "abs":
        return base_stats

    if action_indices is None or state_indices is None:
        raise ValueError(
            "Both action and state modalities are required to compute " f"{action_mode} action mode statistics."
        )

    if action_mode == "delta":
        return calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    if action_mode == "rel":
        return calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    raise ValueError(f"Unsupported action mode for statistics: {action_mode}")


def _load_or_compute_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
) -> dict:
    le_statistics = _load_stats_cache(
        stats_path,
        stats_cache_config,
        invalidate_legacy=True,
    )
    if le_statistics is not None:
        return le_statistics

    le_statistics = _compute_statistics_for_mode(
        parquet_paths=parquet_paths,
        dataset_name=dataset_name,
        action_mode=action_mode,
        lerobot_modality_meta=lerobot_modality_meta,
        action_keys_full=action_keys_full,
        state_keys_full=state_keys_full,
        action_indices=action_indices,
        state_indices=state_indices,
        action_mode_apply_keys=action_mode_apply_keys,
        action_mode_state_map=action_mode_state_map,
    )
    _save_stats_cache(stats_path, stats_cache_config, le_statistics)
    return le_statistics


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_mode_state_map = _normalize_action_mode_state_map(action_mode_state_map)

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    # Build per-column mapping: action column -> list of (action_slice, state_column, state_slice)
    action_col_slices: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using delta mode.

    Rule:
      - For t>0: a_t - a_{t-1}
      - For t=0: a_0 - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using rel mode.

    Rule:
      - For all t: a_t - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats


class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        eval_split: str = "train",
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        self._lerobot_version = self.data_cfg.get("lerobot_version", "v2.0")  # self._indict_lerobot_version(**kwargs)

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None

        self.delete_pause_frame = delete_pause_frame

        # Rung-3 AWR: `datasets.vla_data.advantage_label_id` names an out-of-fold
        # advantage sidecar (examples/Robocasa_kitchen/train_files/label_advantages.py)
        # under <dataset>/glancewam_cache/advantage/<label_id>/. Datasets WITH the sidecar
        # keep their failed episodes (the whole point of AWR) and every sample carries
        # the labeled `advantage`. Sidecar-less datasets are UNCHANGED vs the baseline
        # loader — the success filter still applies (demo sets carry a success column
        # and even a few failed episodes) — and contribute A = 0 (weight 1 == plain BC).
        # Default "" is bit-identical to the pre-AWR loader everywhere.
        self._advantage_label_id = (
            str(self.data_cfg.get("advantage_label_id", "") or "") if self.data_cfg is not None else ""
        )
        self._advantage: dict[int, np.ndarray] | None = None
        self._advantage_sidecar = (
            Path(dataset_path) / "glancewam_cache" / "advantage" / self._advantage_label_id / "advantage.parquet"
            if self._advantage_label_id
            else None
        )
        # `advantage_success_only`: surprise-weighted filtered BC — keep the baseline
        # success filter (failed episodes are dropped, same trajectory set as fBC) but
        # still attach the sidecar A so successes get their exp(A/beta) weights.
        self._advantage_success_only = (
            bool(self.data_cfg.get("advantage_success_only", False)) if self.data_cfg is not None else False
        )
        self._advantage_has_sidecar = self._advantage_sidecar is not None and self._advantage_sidecar.exists()
        self._advantage_keep_failed = self._advantage_has_sidecar and not self._advantage_success_only
        self._advantage_n_failed = 0  # failed episodes actually kept

        # Phase-2 rungs 6/7/9: sidecar-independent failure plumbing. With
        # `datasets.vla_data.keep_failed_episodes: true` the per-episode success filter is
        # bypassed for EVERY dataset in the mix (demo sets included — their few failed
        # episodes are treated like rollout failures) and every sample carries a `success`
        # bit (1.0 for episodes of datasets without a success column) so the framework can
        # mask failures out of the action loss (framework.mask_failed_action_loss) or
        # condition on the outcome (framework.cfgrl). Default False is bit-identical to
        # the baseline loader. Orthogonal to the advantage sidecar above.
        self._keep_failed_episodes = (
            bool(self.data_cfg.get("keep_failed_episodes", False)) if self.data_cfg is not None else False
        )
        self._episode_success: dict[int, float] | None = {} if self._keep_failed_episodes else None
        self._n_failed_kept = 0  # failed episodes kept by keep_failed_episodes

        # Rung-12b episode holdout: with `datasets.vla_data.holdout_episode_fold: k`,
        # episodes whose crc32("<dataset-dir-name>/<episode_index>") % holdout_num_folds
        # equals k are EXCLUDED from training — the same deterministic fold hash the
        # value-critic stack uses (train_value_critic.py), so an eval tool can probe
        # exactly the held-out episodes without any list file. Default None = off,
        # bit-identical to the baseline loader.
        _hf = self.data_cfg.get("holdout_episode_fold", None) if self.data_cfg is not None else None
        self._holdout_fold = None if _hf is None else int(_hf)
        self._holdout_num_folds = int(self.data_cfg.get("holdout_num_folds", 5)) if self.data_cfg is not None else 5
        self._n_holdout_dropped = 0

        # §8 state-conditioning plumbing.
        # All three knobs default OFF and are forced off on the eval split (which already
        # drops the future frame). Validation of `future_frame_range` against
        # future_frame_idx / the VAE cache happens in make_LeRobotSingleDataset
        # (_parse_future_frame_range); here we only re-read the values for runtime use.
        _cfg = self.data_cfg if self.data_cfg is not None else {}
        _train = eval_split != "eval"
        # attach_future_state: ship the raw 9-d achieved proprio row at t+H
        # (sample["future_state"]) plus H itself (sample["cond_horizon_rows"]).
        self._attach_future_state = _train and bool(_cfg.get("attach_future_state", False))
        # attach_future_state_traj: ship the raw per-step rows at t+1..t+Hp
        # (sample["future_state_traj"], (Hp, state_dim)) — hindsight targets for the
        # §8.3 state head. Past-end rows clamp to the last row ("first_last"), which is
        # consistent with the zero-padded action targets: zero delta <=> frozen state.
        self._attach_future_state_traj = _train and bool(_cfg.get("attach_future_state_traj", False))
        # future_frame_range: per-sample H ~ U{min..max} replacing the fixed
        # future_frame_idx for BOTH the decoded future frame and the future-state row.
        _ffr = _cfg.get("future_frame_range", None) if _train else None
        self._future_frame_range = (int(_ffr[0]), int(_ffr[1])) if _ffr is not None else None
        self._future_offset_fixed = int(_cfg.get("future_frame_idx", 0)) if _train else 0
        if (self._attach_future_state or self._future_frame_range) and self._future_offset_fixed <= 0:
            raise ValueError(
                f"[{Path(dataset_path).name}] attach_future_state / future_frame_range need "
                f"future_frame_idx > 0 — there is no future frame to describe or move."
            )
        # goal_frame_range: the goal-image cotrain's SECOND future frame — a hindsight goal
        # at t+g, g ~ U{min..max} native rows (GlanceWAM). Appended after
        # the fixed future slot (make_LeRobotSingleDataset), so it is the LAST video delta.
        # goal_frame_ranges (E8.0 multi-horizon): the list generalization — one hindsight
        # goal per band, appended after the future slots in band order; the single-range
        # key is normalized to a one-band list so both paths share the draw/override code.
        _gfr = _cfg.get("goal_frame_range", None) if _train else None
        _gfrs = _cfg.get("goal_frame_ranges", None) if _train else None
        if _gfrs is not None:
            self._goal_frame_ranges = [(int(r[0]), int(r[1])) for r in _gfrs]
        elif _gfr is not None:
            self._goal_frame_ranges = [(int(_gfr[0]), int(_gfr[1]))]
        else:
            self._goal_frame_ranges = None
        self._n_goal_deltas = len(self._goal_frame_ranges) if self._goal_frame_ranges is not None else 0
        if self._goal_frame_ranges is not None and self._future_offset_fixed <= 0:
            raise ValueError(
                f"[{Path(dataset_path).name}] goal_frame_range(s) needs future_frame_idx > 0 — "
                f"the goal frame is appended after the fixed video-target future slot."
            )
        if self._attach_future_state or self._attach_future_state_traj:
            if "state" not in modality_configs:
                raise ValueError(
                    f"[{Path(dataset_path).name}] attach_future_state(_traj) needs a 'state' "
                    f"modality in the robot's data config."
                )
        # Per-sample scratch (set by get_step_data, consumed by get_video/_pack_sample).
        self._future_offset: int | None = None
        self._future_video_override: int | None = None
        self._goal_offsets: list[int] | None = None
        self._goal_video_overrides: list[int] | None = None
        self._pending_future_state: np.ndarray | None = None
        self._pending_future_state_traj: np.ndarray | None = None
        self._pending_cond_h: int = 0
        # Lazy per-worker RNG for the H draw (numpy's global state is fork-duplicated
        # across DataLoader workers; torch.initial_seed() is already per-worker).
        self._future_rng: np.random.Generator | None = None
        self._future_rng_worker: int | None = None

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = transforms if transforms is not None else ComposedModalityTransform(transforms=[])

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._init_action_mode()
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None
        # Load-and-keep cache for lerobot v3 chunk parquets. Each `pd.read_parquet`
        # costs ~700ms, so under shuffled DataLoader access we hold every chunk
        # in memory rather than re-reading. Preloaded eagerly below.
        self._loaded_files: dict = {}

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self._apply_eval_holdout(eval_split)
        self._apply_episode_subsample(eval_split)
        self._load_advantage_sidecar()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        # Preload all v3 chunk parquets in the parent process so DataLoader
        # workers inherit them via fork() COW (one shared copy, not per-worker).
        self._preload_v3_chunks()

        # Fail loudly here if v3 `data/file_index` disagrees with the actual
        # parquet contents, instead of much later as a cryptic size-0 index error.
        self._check_v3_file_index_integrity()

        print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(le_modality_meta, modality)
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [le_state_action_meta[subkey].end - le_state_action_meta[subkey].start],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert le_info_path.exists(), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                # channels = le_video_meta["shape"][le_video_meta["names"].index("channels")]
                channels = le_video_meta["info"]["video.channels"]
                fps = le_video_meta["info"]["video.fps"]
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }

        # 2. Dataset statistics
        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")

        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        stats_cache_config = _build_stats_cache_config(
            action_mode=action_mode,
        )
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        parquet_files_filtered = [pf for pf in parquet_files if "episode_033675.parquet" not in pf.name]

        if is_main():
            le_statistics = _load_or_compute_statistics(
                stats_path,
                stats_cache_config=stats_cache_config,
                parquet_paths=parquet_files_filtered,
                dataset_name=self.dataset_name,
                action_mode=action_mode,
                lerobot_modality_meta=le_modality_meta,
                action_keys_full=action_keys_full,
                state_keys_full=state_keys_full,
                action_indices=action_indices,
                state_indices=state_indices,
                action_mode_apply_keys=apply_keys,
                action_mode_state_map=normalized_state_map,
            )
        else:
            le_statistics = None

        if dist.is_initialized():
            dist.barrier()

        if le_statistics is None:
            le_statistics = _load_stats_cache(
                stats_path,
                stats_cache_config,
                invalidate_legacy=False,
            )
            if le_statistics is None:
                raise RuntimeError(f"Dataset statistics cache is missing or invalid after sync: {stats_path}")

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)

        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            n_dropped_unsuccessful = 0
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                # Per-episode success filter: datasets that carry a `success` column
                # (e.g. the cosmos-policy `all_episodes` conversion, which stores failed
                # replays too) only expose successful episodes for training. Datasets
                # without the column are unaffected — every episode is kept. No config
                # knob: presence of the column is the trigger. EXCEPTION: when this
                # dataset has an advantage sidecar (rung-3 AWR), failed episodes are the
                # signal — keep them and let the loss weight exp(A/beta) down-weight.
                has_success_col = "success" in episodes_data.columns
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    if self._holdout_fold is not None:
                        _key = f"{self.dataset_path.name}/{int(episode['episode_index'])}"
                        if zlib.crc32(_key.encode()) % self._holdout_num_folds == self._holdout_fold:
                            self._n_holdout_dropped += 1
                            continue
                    if has_success_col and not bool(episode["success"]):
                        if not (self._advantage_keep_failed or self._keep_failed_episodes):
                            n_dropped_unsuccessful += 1
                            continue
                        if self._advantage_keep_failed:
                            self._advantage_n_failed += 1
                        if self._keep_failed_episodes:
                            self._n_failed_kept += 1
                    if self._episode_success is not None:
                        self._episode_success[int(episode["episode_index"])] = (
                            float(bool(episode["success"])) if has_success_col else 1.0
                        )
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        # videos/{video_key}/from_timestamp -> {video_key}
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)

                    # TODO auto map key
                    # Collect video file indices for each video key
                    # 已修改的lerobotv3.0的视频索引（提取视频和文件的索引）
                    video_file_indices = {}
                    for col in timestamp_cols:
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        if chunk_col in episode and file_col in episode:
                            video_file_indices[video_key] = {
                                "chunk_index": int(episode[chunk_col]),
                                "file_index": int(episode[file_col]),
                            }
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                        "videos/file_indices": video_file_indices,
                    }
                    # episode_meta = {
                    #     "data/chunk_index": episode["data/chunk_index"],
                    #     "data/file_index": episode["data/file_index"],
                    #     "data/file_from_index": index,
                    #     "videos/from_timestamps": from_timestamps,
                    # }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            if n_dropped_unsuccessful:
                print(
                    f"[{self.dataset_name}] success-filter: kept {len(trajectory_ids)} successful "
                    f"episodes, dropped {n_dropped_unsuccessful} unsuccessful"
                )
            if self._advantage_keep_failed:
                print(
                    f"[{self.dataset_name}] advantage mode ({self._advantage_label_id}): success filter "
                    f"bypassed, kept all {len(trajectory_ids)} episodes ({self._advantage_n_failed} failed)"
                )
            elif self._advantage_has_sidecar:
                print(
                    f"[{self.dataset_name}] advantage mode ({self._advantage_label_id}, success_only): "
                    f"success filter kept, {len(trajectory_ids)} successful episodes carry sidecar A"
                )
            if self._keep_failed_episodes:
                print(
                    f"[{self.dataset_name}] keep_failed_episodes: success filter bypassed, kept all "
                    f"{len(trajectory_ids)} episodes ({self._n_failed_kept} failed)"
                )
            if self._holdout_fold is not None:
                print(
                    f"[{self.dataset_name}] holdout fold {self._holdout_fold}/{self._holdout_num_folds}: "
                    f"excluded {self._n_holdout_dropped} eval episodes, kept {len(trajectory_ids)}"
                )
            # Should be able to directly read the saved index info here
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """

        def is_main():
            return (not dist.is_initialized()) or dist.get_rank() == 0

        config_key = self._get_steps_config_key()
        steps_filename = "steps_data_index.pkl"
        steps_path = self.dataset_path / "meta" / steps_filename

        # ---------- try to read from cache  ----------
        # NOTE: we must NOT return early on a cache hit. Every rank has to execute
        # the dist.barrier() below exactly once per call, otherwise a rank that
        # cache-hits (e.g. it arrives just after rank0 wrote the file) skips the
        # barrier, drifts one barrier ahead, and desyncs every subsequent
        # collective (manifests as a missing stats_gr00t.json on the next dataset).
        cached_steps = None
        if steps_path.exists():
            try:
                with open(steps_path, "rb") as f:
                    cached_data = pickle.load(f)
                # Reject a stale cache whose config or trajectory count no longer matches
                # (e.g. after the success-filter changed which episodes are exposed). Without
                # this guard the load path returns the old unfiltered step list verbatim.
                stale = cached_data.get("config_key") != config_key or cached_data.get("num_trajectories") != len(
                    self.trajectory_ids
                )
                if stale:
                    print(
                        f"[RANK {os.environ.get('RANK', 'NA')}] cached steps stale "
                        f"(config/num_trajectories mismatch), will rebuild."
                    )
                else:
                    cached_steps = cached_data["steps"]
            except Exception as e:
                # include EOFError / PickleError / KeyError
                print(f"[RANK {os.environ.get('RANK', 'NA')}] " f"Failed to load cached steps ({e}), will rebuild.")

        # ---------- only build by rank0 (when no valid cache) ----------
        if cached_steps is None and is_main():
            all_steps = self._get_all_steps_single_process()

            cache_data = {
                "config_key": config_key,
                "steps": all_steps,
                "num_trajectories": len(self.trajectory_ids),
                "total_steps": len(all_steps),
                "computed_timestamp": pd.Timestamp.now().isoformat(),
                "delete_pause_frame": self.delete_pause_frame,
            }

            steps_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = steps_path.with_suffix(".tmp")

            with open(tmp_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, steps_path)

            print(f"[RANK 0] Cached steps saved to {steps_path}")

        # ---------- sync after rank0  ----------
        if dist.is_initialized():
            dist.barrier()

        # ---------- read by all rank (reuse our own valid cache load) ----------
        if cached_steps is not None:
            return cached_steps
        with open(steps_path, "rb") as f:
            cached_data = pickle.load(f)

        return cached_data["steps"]

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            "dataset_name": self.dataset_name,
        }
        # Failed episodes included (advantage mode) -> different trajectory set than the
        # success-filtered runs, so the cached steps index must not be shared with them.
        # Sidecar-less datasets keep their baseline trajectory set and cache key.
        if self._advantage_keep_failed:
            config_dict["advantage_keep_failed"] = True
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #

    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0

        # Check if language modality is configured
        has_language_modality = "language" in self.modality_keys and len(self.modality_keys["language"]) > 0
        # TODO why trajectory_length here, why not use data length?
        for trajectory_id, trajectory_length in tqdm(
            zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"
        ):
            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)

                trajectory_skipped = False

                # Check if trajectory has valid language instruction (if language modality is configured)
                if has_language_modality:
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language(trajectory_id, self.modality_keys["language"][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                trajectory_skipped = True
                continue

            if not trajectory_skipped:
                processed_trajectories += 1

            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))

        # Print summary statistics
        print(
            f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories"
        )
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")

        return all_steps

    def _apply_eval_holdout(self, eval_split: str) -> None:
        """Episode-level train/eval split for the held-out action-MSE monitor.

        Controlled by `data_cfg.eval_holdout_episodes` (per-dataset count; 0 =
        disabled, the default → no-op, behaviour unchanged). A deterministic,
        seed-stable set of whole episodes is held out so the same episodes are
        excluded on every rank and across resumes (episode-level → no frame
        leakage). Filters the in-memory `_all_steps` only, leaving the on-disk
        steps cache (`steps_data_index.pkl`) shared between the train and eval
        datasets.

        - eval_split="train": drop every step belonging to a held-out episode.
        - eval_split="eval":  keep ONLY held-out episodes, tiled into
          non-overlapping windows (stride = `eval_chunk_stride`, default =
          `action_horizon`), dropping the final partial window so every scored
          chunk is a full `action_horizon` within the episode.
        """
        cfg = self.data_cfg or {}
        n_holdout = int(cfg.get("eval_holdout_episodes", 0))
        if n_holdout <= 0:
            return

        seed = int(cfg.get("eval_holdout_seed", 42))
        ids = np.sort(self._trajectory_ids)
        n_holdout = min(n_holdout, len(ids))
        rng = np.random.default_rng(seed)
        holdout = set(int(t) for t in rng.choice(ids, size=n_holdout, replace=False).tolist())

        before = len(self._all_steps)
        if eval_split == "eval":
            horizon = int(cfg.get("action_horizon", 16))
            stride = int(cfg.get("eval_chunk_stride", horizon))
            lengths = {int(t): int(L) for t, L in zip(self._trajectory_ids, self._trajectory_lengths)}
            kept = [
                (tid, base)
                for (tid, base) in self._all_steps
                if int(tid) in holdout and base % stride == 0 and base + horizon <= lengths[int(tid)]
            ]
        elif eval_split == "train":
            kept = [s for s in self._all_steps if int(s[0]) not in holdout]
        else:
            raise ValueError(f"eval_split must be 'train' or 'eval', got {eval_split!r}")

        self._all_steps = kept
        print(
            f"[{self.dataset_name}] eval_split={eval_split}: held out {len(holdout)} episodes; "
            f"steps {before} -> {len(kept)}"
        )

    def _apply_episode_subsample(self, eval_split: str) -> None:
        """Data-efficiency ablation: train on only N episodes per dataset.

        Controlled by `data_cfg.max_episodes_per_dataset` (0 = disabled, the
        default -> no-op, behaviour bit-identical to before). A deterministic,
        seed-stable subset of whole episodes is KEPT per dataset, so with the
        24-task kitchen mixture `max_episodes_per_dataset: 25` trains on 25
        demos per task instead of the full ~50.

        Applied AFTER `_apply_eval_holdout`, and the candidate pool is taken from
        the surviving `_all_steps`, so held-out eval episodes are never drawn
        back in and the eval split itself is untouched (the MSE monitor stays
        comparable across data-scale arms). Filters the in-memory arrays only —
        the on-disk steps cache (`steps_data_index.pkl`) and the dataset
        statistics (computed from the parquet files, not from this episode set)
        are shared with the full-data runs, so normalization is identical across
        arms and no cache is invalidated.

        Both `_trajectory_ids`/`_trajectory_lengths` (what the mixture sampler
        draws from) and `_all_steps` (what `__len__`/dataset weights use) are
        filtered — filtering only the latter would leave dropped episodes
        reachable through `LeRobotMixtureDataset.sample_step`.
        """
        cfg = self.data_cfg or {}
        n_keep = int(cfg.get("max_episodes_per_dataset", 0) or 0)
        if n_keep <= 0 or eval_split == "eval":
            return

        pool = np.array(sorted({int(t) for (t, _b) in self._all_steps}))
        if len(pool) <= n_keep:
            print(
                f"[{self.dataset_name}] max_episodes_per_dataset={n_keep}: dataset has only "
                f"{len(pool)} episodes, keeping all"
            )
            return

        # Per-dataset stream so each task keeps a different (but deterministic,
        # rank-invariant and resume-stable) subset rather than the same indices.
        seed = int(cfg.get("episode_subsample_seed", 42))
        rng = np.random.default_rng([seed, zlib.crc32(self.dataset_name.encode())])
        keep = set(int(t) for t in rng.choice(pool, size=n_keep, replace=False).tolist())

        before_steps = len(self._all_steps)
        mask = np.array([int(t) in keep for t in self._trajectory_ids], dtype=bool)
        self._trajectory_ids = self._trajectory_ids[mask]
        self._trajectory_lengths = self._trajectory_lengths[mask]
        self._all_steps = [s for s in self._all_steps if int(s[0]) in keep]
        print(
            f"[{self.dataset_name}] max_episodes_per_dataset={n_keep} (seed {seed}): "
            f"episodes {len(pool)} -> {len(self._trajectory_ids)}; "
            f"steps {before_steps} -> {len(self._all_steps)}"
        )

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get("action", [])

        # Extract position data
        delta_position_values = None
        position_candidates = ["delta_eef_position"]
        coordinate_candidates = ["x", "y", "z"]

        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue

        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == "x":
                                    x_data = coord_data
                                elif coord == "y":
                                    y_data = coord_data
                                elif coord == "z":
                                    z_data = coord_data
                    except Exception:
                        continue

            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()

        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.delta_eef_position" in data.columns:
                delta_position_values = data["action.delta_eef_position"].to_numpy().tolist()
            elif all(col in data.columns for col in ["action.x", "action.y", "action.z"]):
                x_vals = data["action.x"].to_numpy()
                y_vals = data["action.y"].to_numpy()
                z_vals = data["action.z"].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")

        # Extract gripper data
        gripper_values = None
        gripper_candidates = ["gripper_close", "gripper"]

        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue

        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if "action.gripper_close" in data.columns:
                gripper_values = data["action.gripper_close"].to_numpy().tolist()
            elif "action.gripper" in data.columns:
                gripper_values = data["action.gripper"].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")

        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        delta_indices: dict[str, np.ndarray] = {}
        for config in self.modality_configs.values():
            for key in config.modality_keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        if f"state.{base}" in self.modality_keys.get("state", []):
            return f"state.{base}"
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                print(f"[WARNING] Action key {action_key} not found in data")
                continue
            state_key = self._infer_state_key_for_action(action_key)

            # for safety, check if the state key is valid
            if state_key is None or state_key not in data:
                continue

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values

            data[action_key] = out

        return data

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert modality_meta_path.exists(), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        return modality_meta

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")

        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # convert index to a column, typically named 'index'
            df = df.rename(columns={"index": "task"})  # rename 'index' column to 'task'
            df = df[["task_index", "task"]]  # reorder columns
            return df

    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption."""
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for modality_config in self.modality_configs.values():
            for key in modality_config.modality_keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}")

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)
        sample = self._attach_vae_cache(sample, trajectory_id, base_index)
        sample = self._attach_goal_depth(sample, trajectory_id, base_index)
        sample = self._attach_advantage(sample, trajectory_id, base_index)
        return self._attach_success(sample, trajectory_id)

    def _load_advantage_sidecar(self) -> None:
        """Load the rung-3 OOF advantage sidecar (label_advantages.py) into per-episode
        arrays for `_attach_advantage`. Sidecar-less datasets are exempt: they keep the
        baseline success filter and their samples get A = 0 (AWR weight 1 == plain BC).
        `get_vla_dataset` asserts that at least one dataset in the mix DID load a
        sidecar, so a typo'd label_id cannot silently degenerate AWR into filtered BC.
        """
        if not self._advantage_has_sidecar:
            if self._advantage_label_id:
                print(
                    f"[{self.dataset_name}] advantage mode ({self._advantage_label_id}): no sidecar — "
                    f"baseline success filter kept, all samples get A=0 (weight 1)"
                )
            return
        df = pd.read_parquet(self._advantage_sidecar)
        adv: dict[int, np.ndarray] = {}
        for ep, g in df.groupby("episode_index"):
            g = g.sort_values("frame_index")
            assert (
                g["frame_index"].to_numpy() == np.arange(len(g))
            ).all(), f"[{self.dataset_name}] advantage sidecar ep {ep}: frame_index not contiguous from 0"
            adv[int(ep)] = g["A"].to_numpy(np.float32)
        for tid, length in zip(self._trajectory_ids, self._trajectory_lengths, strict=True):
            assert int(tid) in adv and len(adv[int(tid)]) == int(length), (
                f"[{self.dataset_name}] advantage sidecar does not cover episode {tid} "
                f"(len {length}) — stale labels? re-run label_advantages.py"
            )
        self._advantage = adv

    def _attach_advantage(self, sample: dict, trajectory_id: int, base_index: int) -> dict:
        """When `advantage_label_id` is configured, attach the obs frame's out-of-fold
        advantage A(o_t); the whole action chunk starting at t inherits it. Same
        (episode_index, frame_index) == (trajectory_id, base_index) identity as the
        VAE cache below. Called from both the single-dataset and mixture paths."""
        if not self._advantage_label_id:
            return sample
        if self._advantage is None:
            sample["advantage"] = 0.0
        else:
            sample["advantage"] = float(self._advantage[int(trajectory_id)][int(base_index)])
        return sample

    def _attach_success(self, sample: dict, trajectory_id: int) -> dict:
        """keep_failed_episodes plumbing (phase-2 rungs 6/7/9): attach the episode's
        success bit so the framework can mask failed samples out of the action loss
        (framework.mask_failed_action_loss) or condition on the outcome
        (framework.cfgrl). 1.0 for episodes of datasets without a success column.
        Off (default) leaves samples untouched. Called from both the single-dataset
        and mixture paths."""
        if self._episode_success is None:
            return sample
        sample["success"] = self._episode_success.get(int(trajectory_id), 1.0)
        return sample

    def _attach_vae_cache(self, sample: dict, trajectory_id: int, base_index: int) -> dict:
        """If a VAE cache reader is attached, look up the precomputed latent
        for this (episode, frame) and add it to the sample. LeRobot v3
        guarantees ``trajectory_id == episode_index`` and ``base_index ==
        frame_index`` (frames within an episode are 0..length-1 in parquet
        row order), so no extra metadata is needed for the lookup. Called
        from both the single-dataset and mixture __getitem__ paths.
        """
        vae_cache = getattr(self, "_vae_cache", None)
        if vae_cache is None:
            return sample
        sample["vae_latent"] = vae_cache.lookup(int(trajectory_id), int(base_index))
        sample["vae_cond_frames"] = vae_cache.num_frames
        return sample

    def _attach_goal_depth(self, sample: dict, trajectory_id: int, base_index: int) -> dict:
        """Goal-image plan E4: ship precomputed depth for the OBS frame and/or the GOAL frame.

        `depth_source` (default "obs"):
          - "obs":  depth of the current frame at `base_index`.
          - "goal": depth of the hindsight goal at t+g, clamped to the episode end exactly
                    as `get_video` clamps `step_indices`, so it is the same frame the RGB
                    goal in `image[-1]` came from.
          - "both": ships both keys.

        **Obs is the default because of what inference can actually deliver** (user call,
        2026-08-11): the eval path never decodes the generated goal — the framework stays in
        latent space — so goal depth at inference would mean VAE-decoding a frame the DiT
        hallucinated and estimating geometry of something that does not exist. The obs frame
        is a real camera image that arrives as pixels already.

        Same (episode, frame) == (trajectory_id, base_index) identity as the VAE cache.
        """
        depth_cache = getattr(self, "_depth_cache", None)
        if depth_cache is None:
            return sample
        source = getattr(self, "_depth_source", "obs")
        if source in ("obs", "both"):
            sample["obs_depth"] = depth_cache.lookup(int(trajectory_id), int(base_index))
        if source in ("goal", "both"):
            if self._goal_offsets is None or len(self._goal_offsets) != 1:
                raise RuntimeError(
                    "depth_source includes 'goal' but no single goal offset was drawn — goal_depth "
                    "needs datasets.vla_data.goal_frame_range (E8.0 goal_frame_ranges is unsupported)"
                )
            ep_len = int(self.trajectory_lengths[self.get_trajectory_index(trajectory_id)])
            goal_frame = min(max(int(base_index) + int(self._goal_offsets[0]), 0), ep_len - 1)
            sample["goal_depth"] = depth_cache.lookup(int(trajectory_id), goal_frame)
        return sample

    def _pack_sample(self, data: dict) -> dict:
        """Pack transformed modality data into training sample format.

        When video delta_indices is widened past length 1 (multi-frame
        history) every video tensor has shape (N, H, W, C); all N chronological
        frames are kept and concatenated camera-by-camera so the downstream
        world model receives a single per-sample video clip. Mixing N>1 with
        more than one camera view is rejected here because the resulting
        interleave order (per-camera-then-time vs per-time-then-camera) is
        ambiguous and no current consumer expects it.
        """
        prim_images = []
        wrist_views = []
        camera_concat = self.data_cfg.get("camera_concat", "none") if self.data_cfg is not None else "none"
        video_keys_present = [vk for vk in self.modality_keys["video"] if vk in data]
        _multi_cam_modes = ("side_by_side", "primary_inset")
        if len(video_keys_present) > 1 and camera_concat not in _multi_cam_modes:
            # The default (list) packing has no interleave policy for N>1 history
            # across multiple cameras. The multi-cam modes resolve this by stitching
            # the cameras per-timestep (see below), so the guard only applies to the
            # default path.
            sample_frame_counts = {vk: len(data[vk]) for vk in video_keys_present}
            if any(n > 1 for n in sample_frame_counts.values()):
                raise NotImplementedError(
                    f"Multi-frame history (N>1) with multiple video keys is not "
                    f"supported yet. Got keys={video_keys_present} with frame "
                    f"counts={sample_frame_counts}. Either restrict the dataset "
                    f"to a single primary camera, set datasets.vla_data.camera_concat="
                    f"side_by_side, or extend _pack_sample with an explicit interleave policy."
                )
        if camera_concat == "primary_inset":
            # RoboCasa-kitchen 3-camera layout: full-res primary + smaller secondary &
            # wrist insets stitched per-timestep into one 336x224 frame for the
            # single-stream world model. Roles are keyed by substring of the modality key.
            by_role: dict[str, list] = {}
            for video_key in video_keys_present:
                role = "wrist" if "wrist" in video_key else ("secondary" if "secondary" in video_key else "primary")
                by_role[role] = [Image.fromarray(f) for f in data[video_key]]
            missing = {"primary", "secondary", "wrist"} - set(by_role)
            if missing:
                raise ValueError(
                    f"camera_concat=primary_inset needs primary/secondary/wrist video keys; "
                    f"missing {sorted(missing)} (present: {video_keys_present})"
                )
            T = len(by_role["primary"])
            all_images = [
                stitch_primary_with_insets(by_role["primary"][t], [by_role["secondary"][t], by_role["wrist"][t]])
                for t in range(T)
            ]
        elif camera_concat == "side_by_side" and len(video_keys_present) > 1:
            # `side_by_side`: pack each timestep's cameras full-res into one wide
            # frame, ordered by role (primary | secondary | wrist) so the stitch
            # matches the eval client's training-order view list. Covers both the
            # LIBERO 2-camera recipe (primary | wrist, DiT4DiT parity) and the
            # RoboCasa-kitchen 3-camera layout (224x224 each -> 672x224).
            def _role_rank(vk: str) -> int:
                return 2 if "wrist" in vk else (1 if "secondary" in vk else 0)

            ordered_keys = sorted(video_keys_present, key=_role_rank)
            per_key = [[Image.fromarray(f) for f in data[vk]] for vk in ordered_keys]
            T = len(per_key[0])
            if any(len(frames) != T for frames in per_key):
                raise ValueError(
                    f"camera_concat=side_by_side needs equal frame counts across cameras; "
                    f"got {[(vk, len(data[vk])) for vk in ordered_keys]}"
                )
            all_images = [stitch_views_side_by_side([frames[t] for frames in per_key]) for t in range(T)]
        else:
            for video_key in video_keys_present:
                # Video keys may be absent when `_skip_video_decode` is on (VAE
                # cache supplies latents directly). `sample["image"]` is then an
                # empty list — the framework's cached path doesn't read it.
                frames = data[video_key]  # (N, H, W, C); N==1 for the default path
                imgs = [Image.fromarray(f).resize((224, 224)) for f in frames]
                if "wrist" not in video_key:
                    prim_images.extend(imgs)
                else:
                    wrist_views.extend(imgs)
            all_images = prim_images + wrist_views

        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": all_images,
            "lang": language,
            "language": language,
        }

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys["state"]:
                state.append(data[state_key])
            state = np.concatenate(state, axis=1).astype(np.float16)
            sample["state"] = state

        # §8 state-conditioning: raw achieved proprio at t+H (+ H itself) and/or the
        # per-step hindsight rows at t+1..t+Hp, stashed by _fetch_future_state. Always
        # preceded by get_step_data in both the single-dataset and mixture paths.
        if self._pending_future_state is not None:
            sample["future_state"] = self._pending_future_state
            sample["cond_horizon_rows"] = self._pending_cond_h
        if self._pending_future_state_traj is not None:
            sample["future_state_traj"] = self._pending_future_state_traj
        # Goal-image cotrain: the trailing frame of `image` is the goal @t+g; ship g (native
        # rows) for logging / an optional horizon embedding. The goal frame itself already
        # rides in `image` as the last video delta (GlanceWAM reads im[-1]).
        if self._goal_offsets is not None:
            # int for the single-goal cotrain (bit-compatible with existing consumers),
            # list for the E8.0 multi-horizon window (one offset per goal band).
            sample["goal_offset_rows"] = (
                int(self._goal_offsets[0]) if len(self._goal_offsets) == 1 else [int(g) for g in self._goal_offsets]
            )

        # Optional text cache. T5 (Cosmos-Predict2) and Gemma2 (SANA-Video)
        # both return (embed, mask); Qwen2.5-VL postproj (Cosmos-Predict2.5)
        # returns embed only — no mask because the released DiT's cross-attn
        # was trained without one. The framework reads `lang_embed` (and
        # optionally `lang_mask`) from the sample; we attach whichever cache
        # the dataset was configured with. At most one should be enabled per
        # dataset; the precedence here is Qwen > Gemma2 > T5 if multiple
        # somehow get attached. Hard-errors on miss so silently mismatched
        # caches can't corrupt training.
        qwenvl_cache = getattr(self, "_qwenvl_cache", None)
        gemma2_cache = getattr(self, "_gemma2_cache", None)
        t5_cache = getattr(self, "_t5_cache", None)
        if qwenvl_cache is not None:
            sample["lang_embed"] = qwenvl_cache.lookup(language)
        elif gemma2_cache is not None:
            embed, mask = gemma2_cache.lookup(language)
            sample["lang_embed"] = embed
            sample["lang_mask"] = mask
        elif t5_cache is not None:
            embed, mask = t5_cache.lookup(language)
            sample["lang_embed"] = embed
            sample["lang_mask"] = mask

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # §8: draw this sample's future offset H BEFORE the modality walk so get_video's
        # future-frame fetch and the future-state rows below agree on the same H. This is
        # the only point shared by the single-dataset and mixture __getitem__ paths.
        self._sample_future_offset()
        # When `_skip_video_decode` is set (auto-enabled by the VAE cache),
        # bypass torchcodec entirely and drop video keys from the modality
        # walk. `make_LeRobotSingleDataset` also strips Video* transforms
        # from the chain so nothing downstream references the missing keys;
        # `_pack_sample` returns an empty `image` list, which the framework
        # discards on the cached path. Net effect: per-sample work drops
        # from ~50 ms (torchcodec seek+decode) to ~50 µs.
        skip_video = getattr(self, "_skip_video_decode", False)
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            if modality == "video" and skip_video:
                continue
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        self._fetch_future_state(trajectory_id, base_index)
        return data

    def _sample_future_offset(self) -> None:
        """§8: set this sample's future offset H (native rows). With `future_frame_range`
        active, H ~ U{min..max} from a per-worker RNG; otherwise the fixed
        `future_frame_idx` (no RNG consumed — stream parity with the baseline loader).
        `_future_video_override` is only set when the range is active, so the fixed-H
        path leaves get_video's index arithmetic bit-identical to current behavior."""
        if self._future_frame_range is not None:
            lo, hi = self._future_frame_range
            self._future_offset = int(self._worker_rng().integers(lo, hi + 1))
            self._future_video_override = self._future_offset
        else:
            self._future_offset = self._future_offset_fixed if self._future_offset_fixed > 0 else None
            self._future_video_override = None
        # Goal-image cotrain: draw this sample's goal offset g ~ U{min..max} (the (0, H_g]
        # cadence-rule distribution) for the trailing goal video delta. Independent draw
        # from the future offset; only consumes the RNG when goal mode is active.
        if self._goal_frame_ranges is not None:
            self._goal_offsets = [
                int(self._worker_rng().integers(g_lo, g_hi + 1)) for g_lo, g_hi in self._goal_frame_ranges
            ]
            self._goal_video_overrides = list(self._goal_offsets)
        else:
            self._goal_offsets = None
            self._goal_video_overrides = None

    def _worker_rng(self) -> np.random.Generator:
        """Lazy per-worker numpy Generator seeded from torch.initial_seed() (already
        per-worker) AND the dataset name. Re-seeds after fork: workers inherit the
        parent's cached generator, so the cache is keyed on the worker id.

        The dataset-name component matters under LeRobotMixtureDataset: every per-task
        dataset object in a worker would otherwise seed the same stream, making H draws
        IDENTICAL across tasks at equal draw counts — measured as ~6 distinct H per
        16-sample batch instead of ~14 (marginals stay uniform, but within-batch H
        clusters). crc32, not hash(): the latter is PYTHONHASHSEED-salted per process.
        """
        info = torch.utils.data.get_worker_info()
        wid = int(info.id) if info is not None else -1
        if self._future_rng is None or self._future_rng_worker != wid:
            self._future_rng = np.random.default_rng(
                [torch.initial_seed() % (2**32), zlib.crc32(self._dataset_name.encode())]
            )
            self._future_rng_worker = wid
        return self._future_rng

    def _fetch_future_state(self, trajectory_id: int, base_index: int) -> None:
        """§8: read the raw achieved-proprio rows this sample's conditioning/targets
        need from the already-loaded episode data (O(1) — no extra IO). Stashes on
        self; `_pack_sample` emits them. Runs after the modality walk so
        `curr_traj_data` is guaranteed loaded."""
        self._pending_future_state = None
        self._pending_future_state_traj = None
        self._pending_cond_h = 0
        if self._attach_future_state and self._future_offset:
            self._pending_cond_h = int(self._future_offset)
            self._pending_future_state = self.get_future_state_rows(
                trajectory_id, base_index, np.array([self._future_offset])
            )
        if self._attach_future_state_traj:
            action_key = self.modality_keys["action"][0]
            horizon = len(self.delta_indices[action_key])
            self._pending_future_state_traj = self.get_future_state_rows(
                trajectory_id, base_index, np.arange(1, horizon + 1)
            )

    def get_future_state_rows(self, trajectory_id: int, base_index: int, offsets: np.ndarray) -> np.ndarray:
        """Raw state rows at base_index + offsets, concatenated across the state keys in
        modality order — the same layout as sample["state"]. Past-episode-end offsets
        clamp to the last row ("first_last"), matching get_video's last-frame clamp, so
        a shipped future state always describes the actually-shipped future frame.
        """
        trajectory_index = self.get_trajectory_index(trajectory_id)
        max_length = self.trajectory_lengths[trajectory_index]
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        parts = []
        for key in self.modality_keys["state"]:
            sub_key = key.replace("state.", "")
            le_cfg = self.lerobot_modality_meta.state[sub_key]
            le_key = le_cfg.original_key if le_cfg.original_key is not None else sub_key
            data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
            data_array = data_array[:, le_cfg.start : le_cfg.end]
            parts.append(
                self.retrieve_data_and_pad(
                    array=data_array,
                    step_indices=base_index + offsets,
                    max_length=max_length,
                    padding_strategy="first_last",
                )
            )
        return np.concatenate(parts, axis=1).astype(np.float32)

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":

            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                return pd.read_parquet(parquet_path)
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)

    def _preload_v3_chunks(self) -> None:
        # Load all chunk parquets in the parent so DataLoader workers share
        # them via fork() COW instead of each loading their own copy.
        if getattr(self, "_lerobot_version", None) != "v3.0":
            return
        if not getattr(self, "trajectory_ids_to_metadata", None):
            return
        unique_keys = {(m["data/chunk_index"], m["data/file_index"]) for m in self.trajectory_ids_to_metadata.values()}
        t0 = time.perf_counter()
        for ci, fi in sorted(unique_keys):
            parquet_path = self.dataset_path / self.data_path_pattern.format(chunk_index=ci, file_index=fi)
            self._loaded_files[(ci, fi)] = pd.read_parquet(parquet_path)
        print(
            f"  preloaded {len(self._loaded_files)} chunk parquet(s) for "
            f"{self.dataset_name} in {time.perf_counter() - t0:.1f}s"
        )

    def _check_v3_file_index_integrity(self) -> None:
        """Assert metadata `data/file_index` matches where episodes actually live.

        LeRobot's v2->v3 converter can emit an off-by-one `data/file_index` at
        chunk-file boundaries: the episode straddling a parquet split is routed
        to the wrong file-NNN.parquet, so `get_trajectory_data_lerobot_v3` slices
        an empty frame and the failure only surfaces later, deep in indexing, as
        the cryptic "index N is out of bounds for axis 0 with size 0". Catch it
        once at init instead. Set GLANCEWAM_SKIP_V3_FILEINDEX_CHECK=1 to bypass.
        """
        if getattr(self, "_lerobot_version", None) != "v3.0":
            return
        if os.environ.get("GLANCEWAM_SKIP_V3_FILEINDEX_CHECK") == "1":
            return
        meta = getattr(self, "trajectory_ids_to_metadata", None)
        if not meta:
            return

        # episode -> actual (chunk, file), from real parquet contents. Reuse the
        # preloaded frames; read episode_index for any data files they missed.
        actual: dict = {}
        for (ci, fi), df in self._loaded_files.items():
            for e in df["episode_index"].unique():
                actual[int(e)] = (ci, fi)
        for path in sorted(self.dataset_path.glob("data/*/file-*.parquet")):
            ci = int(path.parent.name.split("-")[1])
            fi = int(path.stem.split("-")[1])
            if (ci, fi) in self._loaded_files:
                continue
            for e in pd.read_parquet(path, columns=["episode_index"])["episode_index"].unique():
                actual.setdefault(int(e), (ci, fi))

        mismatches, missing = [], []
        for tid, m in meta.items():
            want = (m["data/chunk_index"], m["data/file_index"])
            got = actual.get(int(tid))
            if got is None:
                missing.append(int(tid))
            elif got != want:
                mismatches.append((int(tid), want[1], got[1]))

        if not mismatches and not missing:
            return

        lines = [
            f"v3 dataset integrity check FAILED for '{self.dataset_name}': metadata "
            f"`data/file_index` disagrees with the actual data parquet contents."
        ]
        if mismatches:
            preview = "; ".join(f"ep {t}: meta=file {w} but rows in file {g}" for t, w, g in mismatches[:5])
            lines.append(
                f"  {len(mismatches)} mis-mapped episode(s) [{preview}{', ...' if len(mismatches) > 5 else ''}]"
            )
        if missing:
            lines.append(f"  {len(missing)} episode(s) absent from every data parquet [{missing[:5]}]")
        lines.append(
            "  This is the LeRobot v2->v3 off-by-one bug. Correct `data/file_index` in "
            "meta/episodes/*.parquet to match the actual file each episode lives in "
            "(run tools/fix_lerobot_v3_fileindex.py --apply) before training."
        )
        raise ValueError("\n".join(lines))

    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3."""
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data

        episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
        chunk_index = episode_meta["data/chunk_index"]
        file_index = self.get_episode_file_index(trajectory_id)

        file_key = (chunk_index, file_index)
        file_data = self._loaded_files.get(file_key)
        if file_data is None:
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            file_data = pd.read_parquet(parquet_path)
            self._loaded_files[file_key] = file_data

        episode_data = file_data.loc[file_data["episode_index"] == trajectory_id].copy()
        return episode_data

    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}")
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size

    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]

    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]

    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]

            video_file_indices = episode_meta.get("videos/file_indices", {})
            # print(f"{video_file_indices=}")
            # 已修改的lerobotv3.0的视频索引
            if original_key in video_file_indices:
                video_chunk_index = video_file_indices[original_key]["chunk_index"]
                video_file_index = video_file_indices[original_key]["file_index"]
            else:
                video_chunk_index = episode_meta["data/chunk_index"]
                video_file_index = episode_meta["data/file_index"]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=video_chunk_index,
                file_index=video_file_index,
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # §8 random-H: with future_frame_range active, this sample's future frame moves
        # from the fixed future_frame_idx slot to t+H (drawn in _sample_future_offset).
        # The future video slot is the last delta, OR the second-to-last when the
        # goal-image cotrain appends a trailing goal delta; the goal slot, when present,
        # is the last one and is overridden to t+g every sample.
        n_goal = self._n_goal_deltas
        fut_slot = -1 - n_goal
        if self._future_video_override is not None and self.delta_indices[key][fut_slot] > 0:
            step_indices = step_indices.copy()
            step_indices[fut_slot] = base_index + self._future_video_override
        if n_goal and self._goal_video_overrides is not None and self.delta_indices[key][-1] > 0:
            step_indices = step_indices.copy()
            for j, g in enumerate(self._goal_video_overrides):
                step_indices[-n_goal + j] = base_index + g
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key].original_key
            if original_video_key is None:
                original_video_key = key
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,  # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
            # padding_strategy="zero",           # HACK for realdata
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith("annotation."), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)):  #
            # task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            value = self.curr_traj_data[original_key].iloc[step_indices[i]]  # TODO check v2.0
            task_indices.append(value if isinstance(value, (int, float)) else value.item())

        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)

        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}

        # Process action statistics (only for used keys)
        if hasattr(self.metadata.statistics, "action") and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action

            # Filter to only include used action keys and reorder: non-gripper first, gripper last
            non_gripper_keys = []
            gripper_keys = []

            for key in action_stats.keys():
                if key in used_action_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_action_stats = {}
            for key in reordered_keys:
                filtered_action_stats[key] = action_stats[key]

            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)

                # Add mask field based on whether it's gripper or not
                mask = generate_action_mask_for_used_keys(self.metadata.modalities.action, filtered_action_stats.keys())
                combined_action_stats["mask"] = mask

                tag_stats["action"] = combined_action_stats

        # Process state statistics (only for used keys)
        if hasattr(self.metadata.statistics, "state") and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state

            # Filter to only include used state keys, optionally reorder gripper to end
            non_gripper_keys = []
            gripper_keys = []

            for key in state_stats.keys():
                if key in used_state_keys:
                    if "gripper" in key.lower():
                        gripper_keys.append(key)
                    else:
                        non_gripper_keys.append(key)

            # Reorder: non-gripper first, gripper last
            reordered_keys = non_gripper_keys + gripper_keys

            filtered_state_stats = {}
            for key in reordered_keys:
                filtered_state_stats[key] = state_stats[key]

            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats

        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)

        statistics_data[tag] = tag_stats

        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(used_action_keys)}")
        print(f"Used state keys (reordered): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        if (
            self._future_frame_range is not None
            or self._attach_future_state
            or self._attach_future_state_traj
            or self._goal_frame_ranges is not None
        ):
            # This subclass overrides get_step_data/get_video without the §8 hooks
            # (_sample_future_offset / _fetch_future_state / the per-sample video
            # override) — the flags would silently no-op.
            raise ValueError(
                "future_frame_range / attach_future_state(_traj) / goal_frame_range are not "
                "supported by CachedLeRobotSingleDataset — use the base LeRobotSingleDataset."
            )
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"

                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed

                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics


def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.

    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.

    Returns:
        dict: Combined statistics
    """
    combined_stats = {"mean": [], "std": [], "max": [], "min": [], "q01": [], "q99": []}

    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object

        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, "tolist"):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))

    return combined_stats


def generate_action_mask_for_used_keys(action_modalities: dict, used_action_keys_ordered) -> list[bool]:
    """
    Generate mask based on action modalities, but only for used keys.
    Gripper-related are False, others are True.

    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys in the correct order.

    Returns:
        list[bool]: List of mask values
    """
    mask = []

    # Generate mask in the same order as the statistics were combined
    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, "shape") and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # Check if it's gripper-related
            is_gripper = "gripper" in subkey.lower()

            # Generate mask value for each dimension
            for _ in range(dim_count):
                mask.append(not is_gripper)  # gripper is False, others are True

    return mask


def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []

    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)

    # Extract state keys (remove "state." prefix)
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)

    return used_action_keys, used_state_keys


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
        """
        datasets: list[LeRobotSingleDataset] = []
        dataset_sampling_weights: list[float] = []
        for dataset, weight in data_mixture:
            # Check if dataset is valid and has data
            if len(dataset) == 0:
                print(f"Warning: Skipping empty dataset {dataset.dataset_name}")
                continue
            datasets.append(dataset)
            dataset_sampling_weights.append(weight)

        if len(datasets) == 0:
            raise ValueError("No valid datasets found in the mixture. All datasets are empty.")

        self.datasets = datasets
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")
        self._getitem_count = 0
        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)

        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        # Check for zero or negative weights before normalization
        if np.any(self._dataset_sampling_weights <= 0):
            print(f"Warning: Found zero or negative sampling weights: {self._dataset_sampling_weights}")
            # Set minimum weight to prevent division issues
            self._dataset_sampling_weights = np.maximum(self._dataset_sampling_weights, 1e-8)

        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print("Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum

        # 3. Trajectory sampling weights
        self._trajectory_sampling_weights: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths))
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= dataset.trajectory_lengths

            # Check for zero or negative weights before normalization
            if np.any(trajectory_sampling_weights <= 0):
                print(f"Warning: Dataset {i} has zero or negative trajectory weights")
                trajectory_sampling_weights = np.maximum(trajectory_sampling_weights, 1e-8)

            # Normalize weights
            weights_sum = trajectory_sampling_weights.sum()
            if weights_sum == 0 or np.isnan(weights_sum):
                print(f"Error: Dataset {i} has invalid trajectory weights sum: {weights_sum}")
                # Fallback to equal weights
                trajectory_sampling_weights = np.ones(len(dataset.trajectory_lengths)) / len(dataset.trajectory_lengths)
            else:
                trajectory_sampling_weights /= weights_sum

            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            print(f"Warning: No dataset with weight 1.0 found. Original weights: {dataset_sampling_weights}")
            # Fallback: use the dataset(s) with maximum weight as primary
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            print(f"Using datasets with maximum weight {max_weight} as primary: {self._primary_dataset_indices}")

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Set the epoch and sample the first epoch
        self.set_epoch(0)

        self._sequential_step_sampling = True
        if self.data_cfg is not None:
            seq_cfg = self.data_cfg.get("sequential_step_sampling", True)
            self._sequential_step_sampling = seq_cfg not in ["False", False]

        self._step_order: list[np.ndarray] = []
        self._step_pos: list[int] = []
        if self._sequential_step_sampling:
            for dataset in self.datasets:
                self._step_order.append(np.arange(len(dataset.all_steps)))
                if self.mode == "train":
                    rng = np.random.default_rng(self.seed)
                    rng.shuffle(self._step_order[-1])
                self._step_pos.append(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
        # self.sampled_steps = self.sample_epoch()

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]

        # Set seed
        seed = index if self.mode != "train" else safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory
        trajectory_index = rng.choice(len(dataset.trajectory_ids), p=self.trajectory_sampling_weights[dataset_index])
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index
        if len(dataset.all_steps) == 0:
            raise ValueError(f"Dataset {dataset.dataset_name} has no steps.")

        if not self._sequential_step_sampling:
            single_step_index = rng.choice(len(dataset.all_steps))
        else:
            step_pos = self._step_pos[dataset_index]
            if step_pos >= len(dataset.all_steps):
                order = np.arange(len(dataset.all_steps))
                if self.mode == "train":
                    seed = safe_hash((self.epoch, dataset_index, self.seed, step_pos))
                    rng = np.random.default_rng(seed)
                    rng.shuffle(order)
                self._step_order[dataset_index] = order
                step_pos = 0

            single_step_index = self._step_order[dataset_index][step_pos]
            self._step_pos[dataset_index] = step_pos + 1
        trajectory_id, base_index = dataset.all_steps[single_step_index]
        return dataset, trajectory_id, base_index

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        self._getitem_count += 1
        if self._getitem_count % 1000 == 0:
            gc.collect()

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                while True:  # @DUG
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                sample = dataset._attach_vae_cache(sample, trajectory_id, step)
                sample = dataset._attach_goal_depth(sample, trajectory_id, step)
                sample = dataset._attach_advantage(sample, trajectory_id, step)
                sample = dataset._attach_success(sample, trajectory_id)
                sample["robot_tag"] = dataset.tag
                return sample

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print("Retrying with new sample...")
                    # For retry, we can use a slightly different index to get a new sample
                    # This helps avoid getting stuck on the same problematic sample
                    index = random.randint(0, len(self) - 1)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        # Check for potential issues
        if len(self.datasets) == 0:
            return 0

        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)

        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0

        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(
                f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}"
            )
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0

        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]

        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()

        result = int(max_ratio)
        if result == 0:
            print("Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

        return overall_stats

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one."""
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs
        modality_configs = defaultdict(set)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                modality_configs[modality].add(json.dumps(configs))
        merged_metadata["modalities"] = {}
        for modality, configs in modality_configs.items():
            # Check that all modality configs correspond to the same tag matches
            assert len(configs) == 1, f"Multiple modality configs for modality {modality}: {list(configs)}"
            merged_metadata["modalities"][modality] = json.loads(configs.pop())

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        for dataset in self.datasets:
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
            all_metadatas[dataset.tag].append(dataset.metadata)
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=self.dataset_sampling_weights.tolist(),
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.

        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the data structure to save
        statistics_data = {}

        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []

        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)

        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}

            # Process action statistics
            if hasattr(merged_metadata.statistics, "action") and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action

                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]

                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)

                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys()
                    )
                    combined_action_stats["mask"] = mask

                    tag_stats["action"] = combined_action_stats

            # Process state statistics
            if hasattr(merged_metadata.statistics, "state") and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state

                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []

                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)

                reordered_keys = non_gripper_keys + gripper_keys

                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]

                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats

            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))

            statistics_data[tag] = tag_stats

        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith(".json"):
                save_path = save_path.with_suffix(".json")
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")

        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")

    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.

        Args:
            tag (str): embodiment tag

        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0

        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)

        return {"num_transitions": num_transitions, "num_trajectories": num_trajectories}

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.

        Args:
            load_path (Path | str): Path to the statistics file

        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")

        if load_path.suffix.lower() == ".json":
            with open(load_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif load_path.suffix.lower() == ".pkl":
            import pickle

            with open(load_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.

        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)

            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return

        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue

            # Convert back to DatasetMetadata format
            metadata_dict = {"embodiment_tag": tag, "statistics": {"action": {}, "state": {}}, "modalities": {}}

            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data

            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data

            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)

        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")
